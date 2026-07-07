"""Unit tests for SshClient.wait_until_ready — the post-spawn boot-race guard.

A freshly-spawned cloud VM reports RUNNING before sshd/cloud-init are up, so the
first remote op used to race the boot and die on a single 30 s attempt (the
round-2 ephemeral smoke-test failure). wait_until_ready polls until SSH answers.
``_exec`` is stubbed so these never touch a real network; sleep/clock injected.
"""

from __future__ import annotations

import subprocess

import pytest

from lerobot.gui.training.ssh_transport import SshClient
from lerobot.gui.training.transport import SshTransport


def _client(tmp_path):
    return SshClient(SshTransport(host="1.2.3.4", port=22, user="bot"), control_path_dir=tmp_path)


def _ok():
    return subprocess.CompletedProcess(args=["ssh"], returncode=0, stdout=b"", stderr=b"")


class _Clock:
    """Monotonic clock that only advances when the injected sleep is called."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_retries_through_boot_then_succeeds(tmp_path, monkeypatch):
    clock = _Clock()
    slept: list[float] = []
    calls = {"n": 0}

    def fake_exec(remote_cmd, *, timeout, stdin=None):
        calls["n"] += 1
        if calls["n"] < 3:  # sshd not up yet
            raise subprocess.TimeoutExpired(cmd="ssh", timeout=timeout)
        return _ok()

    client = _client(tmp_path)
    monkeypatch.setattr(client, "_exec", fake_exec)

    def fake_sleep(s):
        slept.append(s)
        clock.t += s

    client.wait_until_ready(timeout_s=300, poll_interval_s=5, sleep=fake_sleep, clock=clock)
    assert calls["n"] == 3
    assert slept == [5, 5]  # slept after each of the two failed attempts


def test_succeeds_on_first_attempt_when_host_is_up(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_exec(remote_cmd, *, timeout, stdin=None):
        calls["n"] += 1
        return _ok()

    client = _client(tmp_path)
    monkeypatch.setattr(client, "_exec", fake_exec)
    client.wait_until_ready(sleep=lambda _s: pytest.fail("should not sleep"), clock=lambda: 0.0)
    assert calls["n"] == 1


def test_raises_after_deadline_naming_boot_or_security_group(tmp_path, monkeypatch):
    clock = _Clock()

    def always_timeout(remote_cmd, *, timeout, stdin=None):
        raise subprocess.TimeoutExpired(cmd="ssh", timeout=timeout)

    client = _client(tmp_path)
    monkeypatch.setattr(client, "_exec", always_timeout)

    def fake_sleep(s):
        clock.t += s

    with pytest.raises(RuntimeError, match="security group"):
        client.wait_until_ready(timeout_s=20, poll_interval_s=5, sleep=fake_sleep, clock=clock)


def test_ensure_prereqs_runs_script_over_sudo_and_resets_control(tmp_path, monkeypatch):
    """ensure_prereqs pipes the setup script to `sudo bash -s`, skips the
    redundant container smoke, and drops the control master so a freshly-added
    docker group takes effect."""
    captured = {}
    closed = {"n": 0}

    def fake_exec(remote_cmd, *, timeout, stdin=None):
        captured["cmd"] = remote_cmd
        captured["stdin"] = stdin
        return _ok()

    client = _client(tmp_path)
    monkeypatch.setattr(client, "_exec", fake_exec)
    monkeypatch.setattr(client, "close", lambda: closed.__setitem__("n", closed["n"] + 1))
    client.ensure_prereqs()
    assert "bash -s" in captured["cmd"]
    assert "sudo" in captured["cmd"]
    assert "LEROBOT_PREREQS_SKIP_CONTAINER_SMOKE=1" in captured["cmd"]
    assert captured["stdin"] and b"install" in captured["stdin"].lower()  # the script text
    assert closed["n"] == 1  # control master reset so the new group applies


def test_ensure_prereqs_raises_on_setup_failure(tmp_path, monkeypatch):
    def fail_exec(remote_cmd, *, timeout, stdin=None):
        return subprocess.CompletedProcess(
            args=["ssh"], returncode=1, stdout=b"", stderr=b"boom: held packages"
        )

    client = _client(tmp_path)
    monkeypatch.setattr(client, "_exec", fail_exec)
    monkeypatch.setattr(client, "close", lambda: pytest.fail("must not reset control on failure"))
    with pytest.raises(RuntimeError, match="prereqs setup failed"):
        client.ensure_prereqs()


def test_local_ensure_prereqs_is_noop():
    """SubprocessClient must not try to apt-install Docker on the user's box."""
    from pathlib import Path

    from lerobot.gui.training.transport import SubprocessClient, SubprocessTransport

    client = SubprocessClient(SubprocessTransport(workdir=Path(".")))
    assert client.ensure_prereqs() is None
