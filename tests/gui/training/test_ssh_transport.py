# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Loopback tests for SshClient — every Protocol method exercised against
``ssh feit@127.0.0.1`` with a throwaway key (set up by the ``ssh_loopback``
fixture in conftest.py).

Skipped automatically when sshd / tmux / ssh-keygen aren't available
locally — see the ``requires_ssh_loopback`` marker in pyproject.toml.
CI runners need ``sudo systemctl start ssh`` before pytest; absent that
these tests skip cleanly.

These are the **unit-level** SSH tests (one method at a time). The
contract-level orchestrator-over-SSH tests live in
``test_orchestrator_ssh.py``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from lerobot.gui.training.ssh_transport import SshClient
from lerobot.gui.training.transport import SshTransport

pytestmark = pytest.mark.requires_ssh_loopback


@pytest.fixture
def ssh_client(ssh_loopback: dict) -> Iterator[SshClient]:
    """Build a fresh SshClient for one test + tear down the control
    socket at the end. Each test gets its own ControlMaster socket via
    distinct pid+host segments, but we explicitly close() to keep the
    /tmp dir tidy and force the next test to re-handshake (avoids
    cross-test state bleed)."""
    t = SshTransport(
        host=ssh_loopback["host"],
        port=ssh_loopback["port"],
        user=ssh_loopback["user"],
    )
    client = SshClient(t)
    yield client
    client.close()


@pytest.fixture
def remote_dir(request: pytest.FixtureRequest) -> Iterator[Path]:
    """A unique /tmp directory on the remote, cleaned up after the test.
    Loopback shares the local FS so we use shutil.rmtree to clean up."""
    name = request.node.name.replace("[", "_").replace("]", "")
    d = Path(f"/tmp/lerobot-pytest-ssh-{os.getpid()}-{name}")
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ── File ops ──────────────────────────────────────────────────────────────────


def test_read_text_missing_file_returns_none(ssh_client: SshClient, remote_dir: Path) -> None:
    assert ssh_client.read_text(remote_dir / "nope") is None


def test_append_text_creates_parent_and_writes(ssh_client: SshClient, remote_dir: Path) -> None:
    target = remote_dir / "a" / "b" / "out.txt"
    ssh_client.append_text(target, "hello\n")
    ssh_client.append_text(target, "world\n")
    assert ssh_client.read_text(target) == "hello\nworld\n"


def test_read_bytes_from_offset_returns_tail(ssh_client: SshClient, remote_dir: Path) -> None:
    target = remote_dir / "events.jsonl"
    ssh_client.append_text(target, "abcdefghij")
    data, new_offset = ssh_client.read_bytes_from_offset(target, 3)
    assert data == b"defghij"
    assert new_offset == 10
    # Reading from offset==EOF returns empty + same offset
    data2, off2 = ssh_client.read_bytes_from_offset(target, 10)
    assert data2 == b""
    assert off2 == 10


def test_read_bytes_from_offset_missing_file(ssh_client: SshClient, remote_dir: Path) -> None:
    data, off = ssh_client.read_bytes_from_offset(remote_dir / "nope", 0)
    assert data == b""
    assert off == 0


def test_read_tail_returns_last_n_bytes(ssh_client: SshClient, remote_dir: Path) -> None:
    target = remote_dir / "stderr.log"
    ssh_client.append_text(target, "0123456789")
    assert ssh_client.read_tail(target, 4) == b"6789"
    # Asking for more bytes than the file has returns the whole file
    assert ssh_client.read_tail(target, 100) == b"0123456789"


def test_read_tail_missing_file_returns_empty(ssh_client: SshClient, remote_dir: Path) -> None:
    assert ssh_client.read_tail(remote_dir / "nope", 100) == b""


def test_list_dir_returns_children(ssh_client: SshClient, remote_dir: Path) -> None:
    (remote_dir / "a").mkdir()
    (remote_dir / "b").mkdir()
    ssh_client.append_text(remote_dir / "c.txt", "x")
    children = sorted(p.name for p in ssh_client.list_dir(remote_dir))
    assert children == ["a", "b", "c.txt"]


def test_list_dir_missing_returns_empty(ssh_client: SshClient, remote_dir: Path) -> None:
    assert ssh_client.list_dir(remote_dir / "nope") == []


def test_sha256_of_matches_local(ssh_client: SshClient, remote_dir: Path) -> None:
    import hashlib

    target = remote_dir / "blob.bin"
    payload = "the quick brown fox\n" * 100
    ssh_client.append_text(target, payload)
    expected = hashlib.sha256(payload.encode()).hexdigest()
    assert ssh_client.sha256_of(target) == expected


def test_sha256_of_missing_returns_none(ssh_client: SshClient, remote_dir: Path) -> None:
    assert ssh_client.sha256_of(remote_dir / "nope") is None


def test_fetch_file_copies_to_local_dst(ssh_client: SshClient, remote_dir: Path, tmp_path: Path) -> None:
    src = remote_dir / "src.bin"
    ssh_client.append_text(src, "checkpoint contents")
    dst = tmp_path / "nested" / "dst.bin"
    ssh_client.fetch_file(src, dst)
    assert dst.read_text() == "checkpoint contents"


# ── Lifecycle ─────────────────────────────────────────────────────────────────


def test_launch_runs_to_completion_with_zero_exit(ssh_client: SshClient, remote_dir: Path) -> None:
    log_path = remote_dir / "stderr.log"
    sid = ssh_client.launch(
        command=["bash", "-c", "echo hi"],
        env={},
        workdir=remote_dir,
        log_path=log_path,
    )
    # Wait up to 5 s for completion (loopback should be << 1 s).
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and ssh_client.is_alive(sid):
        time.sleep(0.1)
    assert not ssh_client.is_alive(sid)
    assert ssh_client.exit_code(sid) == 0
    assert ssh_client.read_text(log_path) == "hi\n"


def test_launch_propagates_nonzero_exit(ssh_client: SshClient, remote_dir: Path) -> None:
    log_path = remote_dir / "stderr.log"
    sid = ssh_client.launch(
        command=["bash", "-c", "echo failing >&2; exit 17"],
        env={},
        workdir=remote_dir,
        log_path=log_path,
    )
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and ssh_client.is_alive(sid):
        time.sleep(0.1)
    assert ssh_client.exit_code(sid) == 17
    # stderr redirected into the log, not into stdout
    assert "failing" in ssh_client.read_text(log_path)


def test_launch_passes_env_to_command(ssh_client: SshClient, remote_dir: Path) -> None:
    log_path = remote_dir / "stderr.log"
    sid = ssh_client.launch(
        command=["bash", "-c", "echo X=$X Y=$Y"],
        env={"X": "hello", "Y": "with space"},
        workdir=remote_dir,
        log_path=log_path,
    )
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and ssh_client.is_alive(sid):
        time.sleep(0.1)
    assert ssh_client.exit_code(sid) == 0
    # space-containing env value survives the shlex.quote roundtrip
    assert ssh_client.read_text(log_path).strip() == "X=hello Y=with space"


def test_stop_terminates_running_worker(ssh_client: SshClient, remote_dir: Path) -> None:
    log_path = remote_dir / "stderr.log"
    sid = ssh_client.launch(
        command=["bash", "-c", "echo running; sleep 30; echo never"],
        env={},
        workdir=remote_dir,
        log_path=log_path,
    )
    # Give the worker a moment to start the sleep.
    time.sleep(0.5)
    assert ssh_client.is_alive(sid)
    ssh_client.stop(sid)
    # SIGTERM → tmux session torn down within a poll cycle.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and ssh_client.is_alive(sid):
        time.sleep(0.1)
    assert not ssh_client.is_alive(sid)
    # The wrapper never got to `echo $?` because tmux was killed mid-cmd,
    # so exit_code is None — matches SubprocessClient post-stop behavior.
    assert ssh_client.exit_code(sid) is None
    # Only the first echo made it
    log = ssh_client.read_text(log_path) or ""
    assert "running" in log
    assert "never" not in log


def test_stop_is_idempotent(ssh_client: SshClient, remote_dir: Path) -> None:
    log_path = remote_dir / "stderr.log"
    sid = ssh_client.launch(
        command=["bash", "-c", "sleep 10"],
        env={},
        workdir=remote_dir,
        log_path=log_path,
    )
    ssh_client.stop(sid)
    # Second stop must NOT raise even though the session is already gone.
    ssh_client.stop(sid)
    ssh_client.stop(sid, force=True)


def test_is_alive_unknown_session_returns_false(ssh_client: SshClient) -> None:
    assert ssh_client.is_alive("nonexistent-session|/tmp/nope") is False


def test_exit_code_unknown_session_returns_none(ssh_client: SshClient) -> None:
    assert ssh_client.exit_code("nonexistent-session|/tmp/nope") is None


def test_exit_code_unparsable_session_id_returns_none(ssh_client: SshClient) -> None:
    """A session_id with no separator (e.g. a stringified PID from a
    prior SubprocessClient launch that somehow leaked into an SSH host's
    Run record) returns None instead of raising."""
    assert ssh_client.exit_code("12345") is None


# ── Docker ops ────────────────────────────────────────────────────────────────


def test_image_inspect_unknown_returns_false(ssh_client: SshClient) -> None:
    # Docker may or may not be installed on the test host; if it isn't,
    # `docker image inspect` exits non-zero either way → image_inspect=False.
    # We only assert the False outcome; never True for a name that can't exist.
    assert ssh_client.image_inspect("lerobot-pytest-does-not-exist:never") is False


@pytest.mark.skipif(not shutil.which("docker"), reason="docker not installed locally")
def test_image_inspect_known_returns_true_when_pulled(ssh_client: SshClient) -> None:
    # Use the smallest possible image we can assume locally available;
    # don't pull anything. If `hello-world` isn't there, skip — we don't
    # want this test to incur network cost.
    r = subprocess.run(["docker", "image", "inspect", "hello-world"], capture_output=True)
    if r.returncode != 0:
        pytest.skip("hello-world image not present locally; skip to avoid network cost")
    assert ssh_client.image_inspect("hello-world") is True


def test_image_size_unknown_returns_none(ssh_client: SshClient) -> None:
    assert ssh_client.image_size("lerobot-pytest-does-not-exist:never") is None


def test_host_identity_rejects_relative_home(ssh_client, monkeypatch):
    """host_identity must fail loudly (user-facing message) on a remote
    whose $HOME comes back non-absolute, not assert-crash deep in launch."""
    import subprocess as sp

    def fake_exec(remote_cmd, **kw):
        return sp.CompletedProcess(args=[], returncode=0, stdout=b"1001 1001 relative/home\n", stderr=b"")

    monkeypatch.setattr(ssh_client, "_exec", fake_exec)
    with pytest.raises(RuntimeError, match="not an absolute path"):
        ssh_client.host_identity()
