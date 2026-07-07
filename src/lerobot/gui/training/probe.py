# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""SSH host probe — the "Test" button in the Add SSH host dialog.

Runs one short SSH command against the host the user typed and reports
back four checks: SSH reachable, docker present, tmux present, NVIDIA
GPU visible. The first failing check short-circuits the rest with
``ok=False, detail="not reached"`` so the dialog always renders a stable
4-row checklist.

``BatchMode=yes`` forces key auth — no interactive password prompt can
leak into the GUI server's address space. ``ConnectTimeout=5`` caps the
TCP/handshake phase; the outer ``asyncio.wait_for`` caps the whole
operation at 10 s. ``StrictHostKeyChecking=accept-new`` is TOFU: the
first probe to a new host writes it to ``~/.ssh/known_hosts`` (same
behavior as ``ssh`` from a terminal); subsequent connects verify it.

This module is the only place that parses ``ssh`` stderr into a UI error
class — keep error_class strings stable, the frontend's status row keys
off them.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Literal

# The closed set of probe error classes. The frontend's status row and the
# tests key off these strings — extend the Literal when adding one so mypy
# flags any branch that emits an unknown class.
ProbeErrorClass = Literal[
    "auth",
    "dns",
    "timeout",
    "refused",
    "unknown_host",
    "config",
    "command_missing",
    "unknown",
]

# Caps. The SSH connect timeout (5s) is enforced by `-o ConnectTimeout=5`
# inside the ssh client; the wallclock cap (10s) is enforced by
# `asyncio.wait_for` around the whole subprocess so a remote command that
# blocks post-connect can't hang the API request.
SSH_CONNECT_TIMEOUT_S = 5
PROBE_WALLCLOCK_TIMEOUT_S = 10

# Sentinel emitted at end of the remote command — its absence in stdout
# means the command was killed (timeout) or never ran (auth failure).
_OK_SENTINEL = "__LEROBOT_PROBE_OK__"

# The remote shell. Each check is independent: failure of one does not
# short-circuit the rest, so the user sees every gap in one round-trip.
# ``2>/dev/null`` on nvidia-smi silences "command not found" noise on
# hosts without the binary. The PATH prepend mirrors SshClient._exec —
# non-interactive SSH skips ~/.profile, so user-installed binaries under
# ~/.local/bin (a common tmux install location) would otherwise probe as
# missing on hosts where training would actually work.
_REMOTE_CMD = (
    'export PATH="$HOME/.local/bin:$PATH"; '
    # docker: presence is not usability — a binary on PATH with no docker-
    # group membership passes `command -v` but every actual `docker` call
    # fails on the socket (verified live on a fresh Nebius VM). Probe the
    # daemon as this user; distinguish "not installed" from "not usable".
    "if command -v docker >/dev/null 2>&1; then "
    "docker info >/dev/null 2>&1 && command -v docker || echo __DOCKER_UNUSABLE__; "
    "else echo __NO_DOCKER__; fi; "
    "command -v tmux || echo __NO_TMUX__; "
    "nvidia-smi -L 2>/dev/null || echo __NO_NVIDIA__; "
    f"echo {_OK_SENTINEL}"
)


@dataclass(frozen=True)
class CheckItem:
    """One row in the probe results table."""

    name: str  # one of: ssh, docker, tmux, nvidia
    ok: bool
    detail: str


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    latency_ms: int
    checks: list[CheckItem] = field(default_factory=list)
    error_class: ProbeErrorClass | None = None  # None = all checks passed
    message: str | None = None


def _not_reached(name: str) -> CheckItem:
    return CheckItem(name=name, ok=False, detail="not reached")


def _classify_stderr(stderr: str) -> tuple[ProbeErrorClass, str]:
    """Map ssh's stderr to ``(error_class, message)`` for UI display."""
    s = stderr.lower()
    if "permission denied" in s or "publickey" in s:
        return (
            "auth",
            "SSH authentication failed — check your ssh-agent / ~/.ssh/config",
        )
    if "could not resolve hostname" in s or "name or service not known" in s:
        return "dns", "Hostname not resolvable"
    if "timed out" in s or "timeout" in s:
        return "timeout", f"Connection timed out after {SSH_CONNECT_TIMEOUT_S}s"
    if "connection refused" in s:
        return "refused", "Connection refused — is sshd running on that port?"
    if "host key verification failed" in s or "remote host identification has changed" in s:
        return (
            "unknown_host",
            "Host key verification failed — run `ssh <host>` once in a terminal first",
        )
    if "bad configuration option" in s or "no such file" in s:
        return "config", "ssh config error — see GUI server stderr for details"
    # Unmapped — surface the raw last line so the user has something to act on.
    tail = stderr.strip().splitlines()[-1] if stderr.strip() else "ssh failed with no stderr"
    return "unknown", tail[:200]


async def probe_ssh(
    host_spec: str,
    *,
    connect_timeout_s: int = SSH_CONNECT_TIMEOUT_S,
    wallclock_timeout_s: int = PROBE_WALLCLOCK_TIMEOUT_S,
) -> ProbeResult:
    """Run a short SSH check against ``host_spec``.

    Pre: ``host_spec`` is whatever the user typed in the dialog — a
    ``~/.ssh/config`` alias, ``user@host``, ``user@host:port``, or bare
    hostname. Passed verbatim to ``ssh``; we don't parse it.
    Post: returns a :class:`ProbeResult` whose ``checks`` list always has
    exactly four entries in order: ssh, docker, tmux, nvidia. The four
    rows let the UI render a stable checklist instead of a dynamic table.
    """
    argv = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={connect_timeout_s}",
        "-o",
        "StrictHostKeyChecking=accept-new",
        # End-of-options sentinel: host_spec is user-typed; without this a
        # value like "-oProxyCommand=..." would parse as an ssh option.
        "--",
        host_spec,
        _REMOTE_CMD,
    ]

    t0 = time.monotonic()
    proc = None
    try:
        # LC_ALL=C: _classify_stderr matches English ssh messages; on a
        # non-English locale every error would fall through to "unknown".
        env = {**os.environ, "LC_ALL": "C"}
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=wallclock_timeout_s)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return ProbeResult(
                ok=False,
                latency_ms=int((time.monotonic() - t0) * 1000),
                checks=[
                    _not_reached("ssh"),
                    _not_reached("docker"),
                    _not_reached("tmux"),
                    _not_reached("nvidia"),
                ],
                error_class="timeout",
                message=f"Probe timed out after {wallclock_timeout_s}s",
            )
        rc = proc.returncode
        stdout = (stdout_b or b"").decode("utf-8", errors="replace")
        stderr = (stderr_b or b"").decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ProbeResult(
            ok=False,
            latency_ms=int((time.monotonic() - t0) * 1000),
            checks=[
                _not_reached("ssh"),
                _not_reached("docker"),
                _not_reached("tmux"),
                _not_reached("nvidia"),
            ],
            error_class="config",
            message="ssh binary not found on PATH",
        )

    latency_ms = int((time.monotonic() - t0) * 1000)

    # SSH itself didn't connect — classify and bail.
    if rc != 0 or _OK_SENTINEL not in stdout:
        err_class, msg = _classify_stderr(stderr)
        return ProbeResult(
            ok=False,
            latency_ms=latency_ms,
            checks=[
                CheckItem(name="ssh", ok=False, detail=msg),
                _not_reached("docker"),
                _not_reached("tmux"),
                _not_reached("nvidia"),
            ],
            error_class=err_class,
            message=msg,
        )

    # SSH OK — parse the per-tool lines.
    out_lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    out_lines = out_lines[: out_lines.index(_OK_SENTINEL)] if _OK_SENTINEL in out_lines else out_lines

    docker_path: str | None = None
    docker_unusable = False
    tmux_path: str | None = None
    nvidia_lines: list[str] = []
    for ln in out_lines:
        if ln == "__NO_DOCKER__":
            docker_path = None
        elif ln == "__DOCKER_UNUSABLE__":
            docker_unusable = True
        elif ln == "__NO_TMUX__":
            tmux_path = None
        elif ln == "__NO_NVIDIA__":
            nvidia_lines = []
        elif ln.startswith("GPU "):
            nvidia_lines.append(ln)
        elif "/" in ln and not ln.startswith("__"):
            # `command -v` output is an absolute path. Match on the exact
            # basename, not substring — a path like
            # /home/docker-admin/.local/bin/tmux must not count as docker.
            basename = ln.rsplit("/", 1)[-1]
            if docker_path is None and basename == "docker":
                docker_path = ln
            elif tmux_path is None and basename == "tmux":
                tmux_path = ln

    if docker_unusable:
        docker_detail = (
            "installed but not usable by this user — add yourself to the "
            "docker group (install_prereqs.sh does this), then reconnect"
        )
    elif docker_path:
        docker_detail = docker_path
    else:
        docker_detail = (
            "not installed — from the lerobot checkout, run: "
            f"ssh {host_spec} 'sudo bash -s' < scripts/training/install_prereqs.sh"
        )

    checks = [
        CheckItem(name="ssh", ok=True, detail=f"connected in {latency_ms} ms"),
        CheckItem(
            name="docker",
            ok=docker_path is not None and not docker_unusable,
            detail=docker_detail,
        ),
        CheckItem(
            name="tmux",
            ok=tmux_path is not None,
            detail=tmux_path or "not installed — `sudo apt install tmux` on the host",
        ),
        CheckItem(
            name="nvidia",
            ok=bool(nvidia_lines),
            detail=nvidia_lines[0]
            if nvidia_lines
            else "no NVIDIA GPU detected (nvidia-smi missing or empty)",
        ),
    ]
    all_ok = all(c.ok for c in checks)
    return ProbeResult(
        ok=all_ok,
        latency_ms=latency_ms,
        checks=checks,
        error_class=None if all_ok else "command_missing",
        message="All checks passed"
        if all_ok
        else "SSH ok — install the missing tools on the host, then Test again",
    )
