# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Transport layer for training runs: how the GUI server reaches the training process.

Two transports, matching ``src/lerobot/gui/training/DESIGN.md`` § Components:

- :class:`SubprocessTransport` — training runs as a subprocess of the GUI server
  (workstation case). "Host" is the GUI server's own filesystem; all operations
  are local file I/O.
- :class:`SshTransport` — training runs over SSH on a remote host
  (Persistent / Ephemeral cases). Lands in the SSH transport follow-up; stub here.

The :class:`TransportClient` protocol is what the run orchestrator drives. The
operations it needs are intentionally narrow:

- :meth:`launch` — start a detached training process; returns a session id
- :meth:`is_alive` / :meth:`stop` — process lifecycle
- :meth:`read_text` / :meth:`read_bytes_from_offset` — pull the structured
  state files (progress.json, events.jsonl, checkpoints.jsonl, stderr.log) from
  the host without holding an open pipe
- :meth:`fetch_file` — copy a checkpoint file from the host to local storage
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import shutil
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

# ── Transport configurations ───────────────────────────────────────────────────


@dataclass(frozen=True, kw_only=True)
class SubprocessTransport:
    """Run training as a subprocess of the GUI server, on the same machine.

    The "host" is the GUI server's own filesystem; no SSH dance needed.
    Files the training process writes are directly readable; no SCP for
    checkpoint pull either (the GUI server opens them in place).
    """

    workdir: Path


@dataclass(frozen=True, kw_only=True)
class SshTransport:
    """Run training over SSH on a remote host.

    Authentication inherits from the user's existing SSH setup —
    ``~/.ssh/config`` Host blocks, ssh-agent, and the default-path
    ``~/.ssh/id_*`` keys. The GUI server never reads or stores private
    keys; ``ssh`` does its own auth from the host this process runs on.
    The Host field accepts either a ``~/.ssh/config`` alias or
    ``user@host`` (or just a hostname with separate ``user`` / ``port``
    fields here). See :doc:`DESIGN.md § Host setup UX`.

    Same trust posture as VS Code Remote-SSH / JetBrains Gateway /
    ``gh codespace ssh``: if the user can ``ssh <host>`` from a
    terminal, they can use it here.
    """

    host: str
    port: int = 22
    user: str = "root"


HostTransport = SubprocessTransport | SshTransport


# ── TransportClient protocol ──────────────────────────────────────────────────


class TransportClient(Protocol):
    """Operations the run orchestrator drives, regardless of transport."""

    def launch(
        self,
        command: list[str],
        env: dict[str, str],
        workdir: Path,
        log_path: Path,
    ) -> str:
        """Start a detached process running ``command``. Returns a session id.

        Pre: ``workdir`` exists or can be created. ``log_path``'s parent
        exists or can be created.
        Post: returned process is detached from the caller's process group
        (survives the GUI server restarting). stdout + stderr are merged
        and written to ``log_path``.

        The session id is **opaque to the orchestrator** — only the client
        that launched it knows how to interpret it. SubprocessClient returns
        the stringified PID; SshClient returns ``<tmux-name>|<workdir>`` so
        that ``exit_code`` can recover the path to the exit-code file
        without an extra parameter. Persistence layer treats it as opaque.
        """
        ...

    def is_alive(self, session_id: str) -> bool:
        """Is the launched process still running?"""
        ...

    def exit_code(self, session_id: str) -> int | None:
        """Process exit status, or None if still running / unknown.

        Returns 0 for clean exit, non-zero for crash. Returns None when
        the process is still alive OR when we don't know (e.g., the
        process was launched in a prior GUI server lifetime and the
        Popen handle is gone). Callers that get None should fall back
        to artifact-based heuristics (checkpoint count, terminal event,
        etc.).
        """
        ...

    def stop(self, session_id: str, *, force: bool = False) -> None:
        """Send SIGTERM (graceful) or SIGKILL (``force=True``). Idempotent —
        no error if the process is already dead."""
        ...

    def read_text(self, path: Path) -> str | None:
        """Read a file's full text content from the host. Returns None if
        the file doesn't exist."""
        ...

    def read_bytes_from_offset(self, path: Path, offset: int) -> tuple[bytes, int]:
        """Read from ``offset`` to EOF. Returns ``(data, new_offset)``.

        For a missing file, returns ``(b"", offset)``. Used for incremental
        tailing of stderr.log without redownloading the whole thing each poll.
        """
        ...

    def fetch_file(self, src: Path, dst: Path) -> None:
        """Copy a file from the host's filesystem to a local destination.

        For SubprocessClient, both paths are local; this is just ``shutil.copy``.
        For SshClient (later), this is SCP. ``dst``'s parent is created if missing.
        """
        ...

    def read_tail(self, path: Path, n_bytes: int) -> bytes:
        """Return up to the last ``n_bytes`` of ``path``. Empty bytes if
        the file doesn't exist. Cheap for both transports: subprocess
        seeks; SSH does ``ssh remote 'tail -c N <path>'``."""
        ...

    def list_dir(self, path: Path) -> list[Path]:
        """List immediate children of ``path``. Empty if the dir doesn't
        exist. Names are returned as full ``Path`` joined to ``path`` (so
        the caller doesn't have to know the host's separator)."""
        ...

    def sha256_of(self, path: Path) -> str | None:
        """Hex digest of ``path``'s sha256. None if the file doesn't
        exist. Host-side hash so SSH transports don't have to SCP the
        file just to checksum it."""
        ...

    def append_text(self, path: Path, text: str) -> None:
        """Append ``text`` to ``path`` (created if missing, parent dir
        too). Used by the orchestrator for its own event-emit writes
        (started / image_* / aborted_by_user / etc.) so they land on the
        same events.jsonl the worker writes to."""
        ...

    def host_identity(self) -> tuple[int, int, str]:
        """``(uid, gid, home)`` of the user commands run as ON THIS HOST.

        Recipes are composed on the GUI server but execute wherever the
        transport points; host-dependent values in the command (the
        ``--user`` flag, ``$HOME``-derived mount sources) are emitted as
        placeholders and resolved through this method at launch time.
        The GUI server's own uid is NOT a valid substitute — the first
        remote VM whose user wasn't uid 1000 proved it.

        Post: ``home`` is an absolute path on the target host.
        """
        ...

    def wait_until_ready(self, *, timeout_s: float = 300.0) -> None:
        """Block until the host is reachable for remote ops, or raise.

        For a freshly-spawned cloud VM the transport target exists before
        sshd does; callers invoke this once after spawn so the first real
        op doesn't race the boot. A no-op for already-up transports.
        """
        ...

    def ensure_prereqs(self) -> None:
        """Make the host able to run training: Docker + nvidia-container-toolkit
        installed, the transport user in the docker group, GPU reachable.

        Idempotent — fast when the host is already provisioned, installs what's
        missing otherwise. Raises if the host can't be made ready (e.g. no GPU,
        or it can't reach the package repos). A no-op for the local transport,
        where docker availability is probed separately at run start.
        """
        ...

    def ensure_dir(self, path: Path) -> None:
        """``mkdir -p`` on the host, as the transport's user. Used to
        pre-create bind-mount sources before ``docker run`` — a missing
        source is auto-created by the docker daemon AS ROOT, after which
        a non-root container can never write into it."""
        ...

    # ── Image (docker) ops, host-side ──────────────────────────────────────────
    #
    # Folded into the TransportClient Protocol so the host abstraction is
    # complete — there used to be a separate `ImageRunner` shim that only
    # the orchestrator's image-prep path used, but it had the same shape
    # as a transport: "do something on the host." The SSH transport will
    # implement these as ``ssh remote 'docker ...'``; the subprocess
    # transport shells out locally.

    def image_inspect(self, tag: str) -> bool:
        """Whether the image is present in this host's docker cache.
        Cheap (~50 ms). Used as the cache-hit check before pull."""
        ...

    def image_pull(self, tag: str) -> tuple[bool, str]:
        """Pull ``tag`` on this host. Returns ``(ok, stderr_tail)``;
        ``stderr_tail`` is non-empty only on failure. Can be long — the
        caller already handles this in a background thread (C5)."""
        ...

    def image_size(self, tag: str) -> int | None:
        """On-disk size of the image in bytes (post-pull). None if
        unknown / image absent. Used for the ``image_pulled`` event's
        ``size_bytes`` field."""
        ...


# ── SubprocessClient ──────────────────────────────────────────────────────────


class SubprocessClient:
    """TransportClient impl for :class:`SubprocessTransport`.

    All operations are local file I/O — the "host" is the GUI server itself.
    Processes are launched detached (``start_new_session=True``) so they
    survive the GUI server restarting.
    """

    def __init__(self, transport: SubprocessTransport) -> None:
        self._transport = transport
        # PID → Popen, populated on launch. Lets exit_code() return the
        # actual returncode after the process exits, rather than guessing
        # from artifacts. Bounded by the number of concurrent runs the GUI
        # spawns in one server lifetime (small); GC'd on server restart.
        self._popens: dict[int, subprocess.Popen] = {}

    @property
    def workdir(self) -> Path:
        return self._transport.workdir

    def launch(
        self,
        command: list[str],
        env: dict[str, str],
        workdir: Path,
        log_path: Path,
    ) -> str:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        workdir.mkdir(parents=True, exist_ok=True)
        # Merge our env on top of the GUI server's environment (the worker
        # needs PATH, HOME, etc.).
        full_env = {**os.environ, **env}
        # Open the log file in write-binary mode; ownership transfers to the
        # subprocess (its stdout/stderr handles point to it). The parent's
        # handle leaks intentionally — closed when the subprocess exits.
        log_f = log_path.open("wb")  # noqa: SIM115 - intentional handover to subprocess
        proc = subprocess.Popen(
            command,
            cwd=str(workdir),
            env=full_env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # detach from GUI server's process group
        )
        self._popens[proc.pid] = proc
        # Protocol session_id is `str`; we use the stringified PID. The
        # internal _popens key stays an int (it's what subprocess.Popen
        # uses natively). _to_pid() handles the round-trip.
        return str(proc.pid)

    @staticmethod
    def _to_pid(session_id: str) -> int:
        """Parse the stringified PID. Raises ValueError on malformed input
        — that's the right fail-fast: SubprocessClient should never see a
        session_id from any other transport."""
        return int(session_id)

    def exit_code(self, session_id: str) -> int | None:
        # If we have the Popen, .poll() reaps the zombie and sets
        # returncode. Returns None if still alive.
        proc = self._popens.get(self._to_pid(session_id))
        if proc is not None:
            return proc.poll()
        # No Popen — this process was launched in a prior GUI lifetime,
        # or by a different client. Can't recover the exit code post-hoc
        # (it was delivered to whatever process originally reaped it).
        return None

    def is_alive(self, session_id: str) -> bool:
        pid = self._to_pid(session_id)
        # Two cases:
        # (1) The process is still our direct child (we launched it, haven't
        #     restarted). waitpid(WNOHANG) reaps zombies and returns (pid, _);
        #     if still running it returns (0, 0). Without this reap, kill(pid,0)
        #     returns True for zombies — would deadlock the alive check.
        # (2) After GUI server restart, the process is no longer our child
        #     (reparented to init). waitpid raises ChildProcessError; we fall
        #     back to kill(pid, 0).
        try:
            reaped, _ = os.waitpid(pid, os.WNOHANG)
            # waitpid returns (pid, status) if reaped (process is dead),
            # (0, 0) if our child is still running.
            return reaped != pid
        except ChildProcessError:
            pass  # not our child; fall through to signal probe
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def stop(self, session_id: str, *, force: bool = False) -> None:
        pid = self._to_pid(session_id)
        sig = signal.SIGKILL if force else signal.SIGTERM
        # Signal the whole process group so detached children are caught,
        # not just the leader. Idempotent — already-dead PID is a no-op.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(pid), sig)

    def read_text(self, path: Path) -> str | None:
        try:
            return path.read_text()
        except FileNotFoundError:
            return None

    def read_bytes_from_offset(self, path: Path, offset: int) -> tuple[bytes, int]:
        assert offset >= 0, f"offset must be non-negative, got {offset}"
        try:
            with path.open("rb") as f:
                f.seek(offset)
                data = f.read()
                return data, offset + len(data)
        except FileNotFoundError:
            return b"", offset

    def fetch_file(self, src: Path, dst: Path) -> None:
        # Same path = already local (the orchestrator calls fetch_file
        # transport-blind to localize run artifacts; on the subprocess
        # transport they were never anywhere else). shutil.copy would
        # raise SameFileError.
        if src == dst:
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)

    def read_tail(self, path: Path, n_bytes: int) -> bytes:
        assert n_bytes >= 0, f"n_bytes must be non-negative, got {n_bytes}"
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return b""
        with path.open("rb") as f:
            f.seek(max(0, size - n_bytes))
            return f.read()

    def list_dir(self, path: Path) -> list[Path]:
        try:
            return list(path.iterdir())
        except FileNotFoundError:
            return []
        except OSError as e:
            # PermissionError / NotADirectoryError mid-poll must not crash
            # the request thread; an empty listing + a log line matches the
            # SSH client's behavior (its find 2>/dev/null swallows these).
            logger.warning("list_dir(%s) failed: %s", path, e)
            return []

    def sha256_of(self, path: Path) -> str | None:
        if not path.exists():
            return None
        h = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                block = f.read(1 << 20)
                if not block:
                    break
                h.update(block)
        return h.hexdigest()

    def append_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(text)

    def host_identity(self) -> tuple[int, int, str]:
        # Local transport: the GUI server's own identity IS the host truth.
        return os.getuid(), os.getgid(), str(Path.home())

    def wait_until_ready(self, *, timeout_s: float = 300.0) -> None:
        # Local transport is always ready — nothing to boot.
        return None

    def ensure_prereqs(self) -> None:
        # The GUI server is the user's own machine — don't apt-install Docker on
        # it. Docker availability for local docker recipes is probed at run
        # start (recipes.docker_available); nothing to do here.
        return None

    def ensure_dir(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    def image_inspect(self, tag: str) -> bool:
        try:
            r = subprocess.run(
                ["docker", "image", "inspect", tag],
                capture_output=True,
                timeout=30,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
        return r.returncode == 0

    def image_pull(self, tag: str) -> tuple[bool, str]:
        try:
            r = subprocess.run(
                ["docker", "pull", tag],
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            return False, f"docker binary not found: {exc}"
        if r.returncode != 0:
            return False, (r.stderr or r.stdout or "")[-1000:]
        return True, ""

    def image_size(self, tag: str) -> int | None:
        try:
            r = subprocess.run(
                ["docker", "image", "inspect", "-f", "{{.Size}}", tag],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None
        if r.returncode != 0:
            return None
        try:
            return int(r.stdout.strip())
        except ValueError:
            return None


# ── Factory ────────────────────────────────────────────────────────────────────


def make_client(transport: HostTransport) -> TransportClient:
    """Construct the appropriate :class:`TransportClient` for a given transport.

    Pre: ``transport`` is one of the known transport types.
    Post: returns a client ready to drive a training run on that transport.

    ``SshClient`` is lazy-imported so callers that only use the subprocess
    transport don't pay the SSH module's import cost (shlex / subprocess /
    paramiko-free but still adds a small import surface).
    """
    if isinstance(transport, SubprocessTransport):
        return SubprocessClient(transport)
    if isinstance(transport, SshTransport):
        from lerobot.gui.training.ssh_transport import SshClient  # lazy

        return SshClient(transport)
    raise TypeError(f"Unknown transport: {type(transport).__name__}")
