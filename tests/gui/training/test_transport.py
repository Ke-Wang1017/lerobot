# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for :class:`SubprocessClient` and the transport factory.

The SSH transport client is stubbed; its tests land with the real
implementation in the SSH follow-up.
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import pytest

from lerobot.gui.training.transport import (
    SshTransport,
    SubprocessClient,
    SubprocessTransport,
    make_client,
)


@pytest.fixture
def transport(tmp_path: Path) -> SubprocessTransport:
    return SubprocessTransport(workdir=tmp_path)


@pytest.fixture
def client(transport: SubprocessTransport) -> SubprocessClient:
    return SubprocessClient(transport)


# ── Process lifecycle ─────────────────────────────────────────────────────────


def test_launch_then_alive_then_dead(tmp_path: Path, client: SubprocessClient) -> None:
    """A launched process is alive, then dead after exiting."""
    log = tmp_path / "log"
    pid = client.launch([sys.executable, "-c", "import time; time.sleep(0.3)"], {}, tmp_path, log)
    assert client.is_alive(pid)
    # Wait past the sleep with margin
    deadline = time.monotonic() + 3.0
    while client.is_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not client.is_alive(pid)


def test_launch_writes_log(tmp_path: Path, client: SubprocessClient) -> None:
    log = tmp_path / "log"
    pid = client.launch(
        [sys.executable, "-c", "print('hello'); import sys; sys.stderr.write('world\\n')"],
        {},
        tmp_path,
        log,
    )
    deadline = time.monotonic() + 3.0
    while client.is_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not client.is_alive(pid)
    contents = log.read_text()
    assert "hello" in contents
    assert "world" in contents  # stderr merged into stdout/log


def test_launch_creates_workdir_and_log_parent(tmp_path: Path, client: SubprocessClient) -> None:
    workdir = tmp_path / "nested" / "workdir"
    log = tmp_path / "deeper" / "logs" / "out.log"
    assert not workdir.exists()
    assert not log.parent.exists()
    pid = client.launch([sys.executable, "-c", ""], {}, workdir, log)
    deadline = time.monotonic() + 3.0
    while client.is_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert workdir.exists()
    assert log.parent.exists()


def test_launch_uses_env_overrides(tmp_path: Path, client: SubprocessClient) -> None:
    """Env vars passed at launch are visible to the subprocess and override OS env."""
    log = tmp_path / "log"
    out_file = tmp_path / "out.txt"
    code = f"import os; open({str(out_file)!r}, 'w').write(os.environ.get('LEROBOT_PROBE', ''))"
    pid = client.launch(
        [sys.executable, "-c", code],
        {"LEROBOT_PROBE": "set-by-launcher"},
        tmp_path,
        log,
    )
    deadline = time.monotonic() + 3.0
    while client.is_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert out_file.read_text() == "set-by-launcher"


def test_stop_graceful(tmp_path: Path, client: SubprocessClient) -> None:
    """SIGTERM (default) should terminate a sleeping process."""
    log = tmp_path / "log"
    pid = client.launch([sys.executable, "-c", "import time; time.sleep(30)"], {}, tmp_path, log)
    assert client.is_alive(pid)
    client.stop(pid)
    deadline = time.monotonic() + 3.0
    while client.is_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not client.is_alive(pid)


def test_stop_force(tmp_path: Path, client: SubprocessClient) -> None:
    """SIGKILL (force=True) kills a process that ignores SIGTERM."""
    log = tmp_path / "log"
    ready = tmp_path / "ready"
    # Subprocess installs SIGTERM handler, signals readiness via a file, then sleeps.
    # We wait for `ready` before stopping so SIG_IGN is definitely installed —
    # otherwise SIGTERM can arrive before the handler is set.
    code = (
        "import signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"open({str(ready)!r}, 'w').close(); "
        "time.sleep(30)"
    )
    pid = client.launch([sys.executable, "-c", code], {}, tmp_path, log)
    # Wait for the child to install SIG_IGN
    deadline = time.monotonic() + 3.0
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert ready.exists(), "child failed to signal readiness"
    assert client.is_alive(pid)
    # SIGTERM should be ignored
    client.stop(pid)
    time.sleep(0.3)
    assert client.is_alive(pid)
    # SIGKILL forces exit
    client.stop(pid, force=True)
    deadline = time.monotonic() + 3.0
    while client.is_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not client.is_alive(pid)


def test_stop_idempotent_on_dead_pid(tmp_path: Path, client: SubprocessClient) -> None:
    """Stop after the process has already exited should not raise."""
    log = tmp_path / "log"
    pid = client.launch([sys.executable, "-c", ""], {}, tmp_path, log)
    deadline = time.monotonic() + 3.0
    while client.is_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not client.is_alive(pid)
    client.stop(pid)
    client.stop(pid, force=True)


# ── File reads ────────────────────────────────────────────────────────────────


def test_read_text_missing(tmp_path: Path, client: SubprocessClient) -> None:
    assert client.read_text(tmp_path / "nope.json") is None


def test_read_text_present(tmp_path: Path, client: SubprocessClient) -> None:
    p = tmp_path / "file.txt"
    p.write_text("hello world")
    assert client.read_text(p) == "hello world"


def test_read_bytes_from_offset_basic(tmp_path: Path, client: SubprocessClient) -> None:
    p = tmp_path / "log"
    p.write_bytes(b"abcdef")
    data, new_offset = client.read_bytes_from_offset(p, 0)
    assert data == b"abcdef"
    assert new_offset == 6


def test_read_bytes_from_offset_incremental(tmp_path: Path, client: SubprocessClient) -> None:
    """Subsequent reads from the previous offset return only the new bytes."""
    p = tmp_path / "log"
    p.write_bytes(b"abcdef")
    data, off = client.read_bytes_from_offset(p, 0)
    assert (data, off) == (b"abcdef", 6)
    with p.open("ab") as f:
        f.write(b"ghij")
    data, off = client.read_bytes_from_offset(p, 6)
    assert (data, off) == (b"ghij", 10)


def test_read_bytes_from_offset_past_eof(tmp_path: Path, client: SubprocessClient) -> None:
    """Reading past EOF returns empty bytes; offset unchanged."""
    p = tmp_path / "log"
    p.write_bytes(b"abc")
    data, off = client.read_bytes_from_offset(p, 10)
    assert data == b""
    assert off == 10


def test_read_bytes_from_offset_missing(tmp_path: Path, client: SubprocessClient) -> None:
    data, off = client.read_bytes_from_offset(tmp_path / "missing", 0)
    assert data == b""
    assert off == 0


def test_read_bytes_from_offset_negative_offset_asserts(tmp_path: Path, client: SubprocessClient) -> None:
    p = tmp_path / "log"
    p.write_bytes(b"abc")
    with pytest.raises(AssertionError):
        client.read_bytes_from_offset(p, -1)


# ── fetch_file ────────────────────────────────────────────────────────────────


def test_fetch_file_basic(tmp_path: Path, client: SubprocessClient) -> None:
    src = tmp_path / "src.bin"
    src.write_bytes(b"payload")
    dst = tmp_path / "outdir" / "dst.bin"
    client.fetch_file(src, dst)
    assert dst.read_bytes() == b"payload"


def test_fetch_file_creates_dst_parent(tmp_path: Path, client: SubprocessClient) -> None:
    src = tmp_path / "src"
    src.write_bytes(b"x")
    dst = tmp_path / "a" / "b" / "c" / "dst"
    assert not dst.parent.exists()
    client.fetch_file(src, dst)
    assert dst.exists()


def test_fetch_file_missing_src_raises(tmp_path: Path, client: SubprocessClient) -> None:
    with pytest.raises(FileNotFoundError):
        client.fetch_file(tmp_path / "missing", tmp_path / "dst")


# ── Transport dataclasses + factory ────────────────────────────────────────────


def test_subprocess_transport_immutable() -> None:
    t = SubprocessTransport(workdir=Path("/tmp"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.workdir = Path("/elsewhere")  # type: ignore[misc]


def test_ssh_transport_defaults() -> None:
    t = SshTransport(host="example.com")
    assert t.port == 22
    assert t.user == "root"


def test_ssh_transport_immutable() -> None:
    t = SshTransport(host="example.com")
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.host = "other.com"  # type: ignore[misc]


def test_make_client_subprocess(transport: SubprocessTransport) -> None:
    client = make_client(transport)
    assert isinstance(client, SubprocessClient)


def test_make_client_ssh_returns_ssh_client() -> None:
    """Regression for the SshClient landing: factory now lazily imports
    and returns SshClient instead of raising. The actual SSH ops are
    exercised in ``test_ssh_transport.py`` under the
    ``requires_ssh_loopback`` marker."""
    from lerobot.gui.training.ssh_transport import SshClient

    ssh = SshTransport(host="localhost")
    client = make_client(ssh)
    assert isinstance(client, SshClient)


def test_make_client_unknown_type_raises() -> None:
    class BogusTransport:
        pass

    with pytest.raises(TypeError, match="Unknown transport"):
        make_client(BogusTransport())  # type: ignore[arg-type]


def test_subprocess_client_workdir_property(tmp_path: Path) -> None:
    client = SubprocessClient(SubprocessTransport(workdir=tmp_path))
    assert client.workdir == tmp_path


# ── ControlMaster socket identity (GPU smoke finding #10) ────────────────────


def test_ssh_control_path_distinct_per_user_and_port():
    """Regression: the control socket was keyed by host only, so two saved
    hosts on the same address with different users shared one
    authenticated ControlMaster — OpenSSH executed the second user's
    commands over the FIRST user's session (verified live: a smoketest@vm
    run silently ran as feit). The socket key must cover the full
    connection identity."""
    from lerobot.gui.training.ssh_transport import SshClient

    base = SshClient(SshTransport(host="1.2.3.4", port=22, user="alice"))
    other_user = SshClient(SshTransport(host="1.2.3.4", port=22, user="bob"))
    other_port = SshClient(SshTransport(host="1.2.3.4", port=2222, user="alice"))
    same = SshClient(SshTransport(host="1.2.3.4", port=22, user="alice"))

    assert base._control_path != other_user._control_path
    assert base._control_path != other_port._control_path
    assert base._control_path == same._control_path  # same identity still shares (that's the point of CM)
    # AF_UNIX limit headroom even for long hostnames
    long = SshClient(SshTransport(host="a" * 64 + ".example.com", port=22, user="someuser"))
    assert len(str(long._control_path)) < 108
