# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for src/lerobot/gui/training/probe.py.

The probe module is the one place that parses ssh stderr into a UI error
class. Tests stub ``asyncio.create_subprocess_exec`` with a fake process
that yields canned stdout/stderr/returncode triples — no real SSH.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from lerobot.gui.training import probe as probe_mod
from lerobot.gui.training.probe import probe_ssh


@dataclass
class _FakeProc:
    stdout_b: bytes = b""
    stderr_b: bytes = b""
    returncode: int = 0
    raise_timeout: bool = False
    killed: bool = False

    async def communicate(self):
        if self.raise_timeout:
            await asyncio.sleep(60)  # let wait_for kill us
        return self.stdout_b, self.stderr_b

    def kill(self):
        self.killed = True

    async def wait(self):
        return self.returncode


def _patch_subprocess(monkeypatch, fake: _FakeProc):
    async def fake_create(*args, **kwargs):
        return fake

    monkeypatch.setattr(probe_mod.asyncio, "create_subprocess_exec", fake_create)


@pytest.mark.asyncio
async def test_probe_full_success_parses_all_four_checks(monkeypatch):
    fake = _FakeProc(
        stdout_b=(
            b"/usr/bin/docker\n/usr/bin/tmux\nGPU 0: NVIDIA L40S (UUID: GPU-xxxxx)\n__LEROBOT_PROBE_OK__\n"
        ),
        stderr_b=b"",
        returncode=0,
    )
    _patch_subprocess(monkeypatch, fake)
    result = await probe_ssh("user@host")
    assert result.ok is True
    assert result.error_class is None
    names = [c.name for c in result.checks]
    assert names == ["ssh", "docker", "tmux", "nvidia"]
    assert all(c.ok for c in result.checks)
    assert "L40S" in result.checks[3].detail


@pytest.mark.asyncio
async def test_probe_ssh_only_no_docker_no_tmux(monkeypatch):
    fake = _FakeProc(
        stdout_b=b"__NO_DOCKER__\n__NO_TMUX__\n__NO_NVIDIA__\n__LEROBOT_PROBE_OK__\n",
        returncode=0,
    )
    _patch_subprocess(monkeypatch, fake)
    result = await probe_ssh("alias")
    assert result.ok is False
    assert result.error_class == "command_missing"
    assert result.checks[0].ok is True  # ssh worked
    assert result.checks[1].ok is False  # docker missing
    assert result.checks[2].ok is False  # tmux missing
    assert result.checks[3].ok is False  # nvidia missing


@pytest.mark.asyncio
async def test_probe_auth_failure_returns_auth_class(monkeypatch):
    fake = _FakeProc(
        stdout_b=b"",
        stderr_b=b"user@host: Permission denied (publickey).\r\n",
        returncode=255,
    )
    _patch_subprocess(monkeypatch, fake)
    result = await probe_ssh("user@host")
    assert result.ok is False
    assert result.error_class == "auth"
    assert "auth" in result.message.lower()
    # First check (ssh) reflects the failure; the rest are "not reached".
    assert result.checks[0].name == "ssh" and result.checks[0].ok is False
    assert [c.detail for c in result.checks[1:]] == ["not reached"] * 3


@pytest.mark.asyncio
async def test_probe_dns_failure(monkeypatch):
    fake = _FakeProc(
        stderr_b=b"ssh: Could not resolve hostname nope.invalid: Name or service not known\r\n",
        returncode=255,
    )
    _patch_subprocess(monkeypatch, fake)
    result = await probe_ssh("nope.invalid")
    assert result.error_class == "dns"


@pytest.mark.asyncio
async def test_probe_connection_refused(monkeypatch):
    fake = _FakeProc(stderr_b=b"ssh: connect to host port 22: Connection refused\n", returncode=255)
    _patch_subprocess(monkeypatch, fake)
    result = await probe_ssh("host")
    assert result.error_class == "refused"


@pytest.mark.asyncio
async def test_probe_host_key_changed(monkeypatch):
    fake = _FakeProc(stderr_b=b"Host key verification failed.\n", returncode=255)
    _patch_subprocess(monkeypatch, fake)
    result = await probe_ssh("host")
    assert result.error_class == "unknown_host"
    assert "known_hosts" in result.message or "terminal" in result.message


@pytest.mark.asyncio
async def test_probe_wallclock_timeout(monkeypatch):
    fake = _FakeProc(raise_timeout=True)
    _patch_subprocess(monkeypatch, fake)
    result = await probe_ssh("host", wallclock_timeout_s=1)
    assert result.ok is False
    assert result.error_class == "timeout"
    assert fake.killed is True
    # All four rows still rendered so the dialog has a stable checklist.
    assert len(result.checks) == 4


@pytest.mark.asyncio
async def test_probe_ssh_binary_missing(monkeypatch):
    async def fake_create(*args, **kwargs):
        raise FileNotFoundError("ssh: command not found")

    monkeypatch.setattr(probe_mod.asyncio, "create_subprocess_exec", fake_create)
    result = await probe_ssh("host")
    assert result.ok is False
    assert result.error_class == "config"
    assert "ssh" in result.message.lower()


@pytest.mark.asyncio
async def test_probe_basename_match_not_substring(monkeypatch):
    """Regression: a tmux under /home/docker-admin/... must not count as
    docker (and vice versa). `command -v` output is matched on the exact
    basename, not substring-anywhere-in-path."""
    fake = _FakeProc(
        stdout_b=(
            b"__NO_DOCKER__\n/home/docker-admin/.local/bin/tmux\n__NO_NVIDIA__\n__LEROBOT_PROBE_OK__\n"
        ),
        returncode=0,
    )
    _patch_subprocess(monkeypatch, fake)
    result = await probe_ssh("user@host")
    checks = {c.name: c for c in result.checks}
    assert checks["docker"].ok is False, "path containing 'docker' must not satisfy the docker check"
    assert checks["tmux"].ok is True
    assert checks["tmux"].detail == "/home/docker-admin/.local/bin/tmux"


@pytest.mark.asyncio
async def test_probe_argv_has_end_of_options_sentinel(monkeypatch):
    """Regression: host_spec is user-typed; without `--` before it, a value
    like '-oProxyCommand=...' would parse as an ssh option."""
    captured: dict = {}

    async def fake_create(*args, **kwargs):
        captured["argv"] = list(args)
        return _FakeProc(stdout_b=b"__LEROBOT_PROBE_OK__\n", returncode=0)

    monkeypatch.setattr(probe_mod.asyncio, "create_subprocess_exec", fake_create)
    await probe_ssh("-oProxyCommand=evil")
    argv = captured["argv"]
    assert "--" in argv
    assert argv.index("--") == argv.index("-oProxyCommand=evil") - 1


def test_remote_cmd_prepends_local_bin_path():
    """Regression: non-interactive SSH skips ~/.profile, so without the
    PATH prepend a tmux under ~/.local/bin probes as missing on hosts
    where SshClient._exec (which does prepend) would find it fine. The
    probe and the transport must agree on PATH or the Test button lies."""
    assert probe_mod._REMOTE_CMD.startswith('export PATH="$HOME/.local/bin:$PATH"; ')


@pytest.mark.asyncio
async def test_probe_records_latency(monkeypatch):
    fake = _FakeProc(
        stdout_b=b"/usr/bin/docker\n/usr/bin/tmux\nGPU 0: NVIDIA\n__LEROBOT_PROBE_OK__\n",
        returncode=0,
    )
    _patch_subprocess(monkeypatch, fake)
    result = await probe_ssh("host")
    assert result.latency_ms >= 0
    assert result.latency_ms < 5000


@pytest.mark.asyncio
async def test_probe_docker_unusable_distinct_from_missing(monkeypatch):
    """Regression (GPU smoke finding #1, verified live): a docker binary on
    PATH with no docker-group membership passed `command -v` and probed
    green, while every actual docker call failed on the socket. The remote
    command now probes `docker info` as the user and emits a distinct
    sentinel; the row must be red with the group-membership hint."""
    fake = _FakeProc(
        stdout_b=b"__DOCKER_UNUSABLE__\n/usr/bin/tmux\nGPU 0: NVIDIA L40S\n__LEROBOT_PROBE_OK__\n",
        returncode=0,
    )
    _patch_subprocess(monkeypatch, fake)
    result = await probe_ssh("user@host")
    checks = {c.name: c for c in result.checks}
    assert checks["docker"].ok is False
    assert "docker group" in checks["docker"].detail
    assert result.ok is False


def test_remote_cmd_probes_docker_usability():
    """The remote command must run `docker info` (socket access as the ssh
    user), not just `command -v` (binary presence)."""
    assert "docker info" in probe_mod._REMOTE_CMD
    assert "__DOCKER_UNUSABLE__" in probe_mod._REMOTE_CMD


@pytest.mark.asyncio
async def test_probe_forces_c_locale(monkeypatch):
    """Regression: _classify_stderr matches English ssh messages; without
    LC_ALL=C a non-English locale turns every error into 'unknown'."""
    captured: dict = {}

    async def fake_create(*args, **kwargs):
        captured["env"] = kwargs.get("env") or {}
        return _FakeProc(stdout_b=b"__LEROBOT_PROBE_OK__\n", returncode=0)

    monkeypatch.setattr(probe_mod.asyncio, "create_subprocess_exec", fake_create)
    await probe_ssh("host")
    assert captured["env"].get("LC_ALL") == "C"


@pytest.mark.asyncio
async def test_probe_classifies_openssh9_timeout_variant(monkeypatch):
    """OpenSSH 9.x phrases timeouts as 'ssh: connect ... Timeout' rather
    than the classic 'connection timed out' — both must classify."""
    fake = _FakeProc(stderr_b=b"ssh: connect to host example.com port 22: Timeout\n", returncode=255)
    _patch_subprocess(monkeypatch, fake)
    result = await probe_ssh("example.com")
    assert result.error_class == "timeout"
