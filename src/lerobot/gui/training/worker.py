# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Subprocess worker that owns one remote-SSH training run end-to-end.

Spawned by the GUI server via `python -m lerobot.gui.training.worker`,
never directly by the user. Reads its config from
``LEROBOT_TRAIN_WORKER_CONFIG`` (a JSON blob) at startup and writes
progress + events + log files the server polls.

Architectural shape mirrors :mod:`lerobot.gui.hub_worker`:

  GUI server (FastAPI)                training_worker.py (this file)
  ────────────────────                ───────────────────────────────
  spawn()                              ─► reads JobConfig from env
                                       ─► loads HostProfile
                                       ─► opens SSH session to pod
                                       ─► docker pull on pod (idempotent)
                                       ─► docker run lerobot-train in tmux
                                       ─►
                                       ┌── every POD_POLL_INTERVAL_S:
                                       │     ssh pod 'cat progress.json'
                                       │     ssh pod 'tail -c +N stderr.log'
                                       │     PollScheduler.schedule_next()
                                       │     atomic_write_json(local progress.json)
                                       │     append_event(...) on state change
                                       └──
                                       ─► on SIGTERM: kill remote training
                                       ─► on terminal: final progress write, exit

  reads progress.json/events.jsonl    ◄── (file IPC, same dir layout
   on its own polling cycle               as Hub Transfers worker)

This file is the SKELETON for the worker; the actual "launch training in
tmux on the pod" + "parse lerobot-train stdout into structured progress"
logic is staged for follow-up commits. The polling-loop machinery,
PollScheduler integration, SSH wrapper, error classification, and event-
log writer all live here — these are the resilience-design pieces from
the design conversation.

For the full design see ``src/lerobot/gui/training/DESIGN.md`` (colocated
with this package).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lerobot.gui.training.jobs import (
    PROGRESS_WRITE_INTERVAL_S,
    HostProfile,
    PollScheduler,
    TrainingJobConfig,
    TrainingJobPaths,
    append_event,
    atomic_write_json,
    classify_ssh_error,
)

logger = logging.getLogger(__name__)


# ── SSH wrapper ─────────────────────────────────────────────────────────────


@dataclass
class SshExecResult:
    """Outcome of a single `ssh ... '<cmd>'` invocation."""

    success: bool
    stdout: bytes = b""
    stderr: bytes = b""
    exit_code: int | None = None
    error_class: str | None = None  # set if success=False
    error_message: str | None = None


class SshConnection:
    """Stateless wrapper around shelling out to `ssh`.

    We use ssh-the-binary (not paramiko) for v1 to avoid the extra
    dependency and to inherit the user's ssh_config / agent / known_hosts
    semantics for free. ControlMaster/ControlPersist makes repeated calls
    cheap — the first `ssh ...` opens a control socket; subsequent calls
    reuse it for up to 10 minutes.

    Errors are classified via :func:`classify_ssh_error` so the caller
    can distinguish transient (worth a backoff retry) from permanent
    (give up immediately) failures.
    """

    def __init__(self, host: HostProfile, control_path_dir: Path | None = None) -> None:
        self.host = host
        # ControlMaster socket lives in a per-job temp dir so we don't
        # collide with the user's ambient SSH connections. Use the system
        # temp dir from tempfile rather than a hardcoded /tmp so this works
        # on platforms that put temp elsewhere (e.g., $TMPDIR set in CI).
        base_dir = control_path_dir or Path(tempfile.gettempdir())
        self._control_path = base_dir / f"lerobot-train-ssh-{os.getpid()}"

    # ---- public surface ----------------------------------------------------

    def exec(self, cmd: str, *, timeout_s: float = 15.0) -> SshExecResult:
        """Run a single shell command on the remote host.

        Pre: ``cmd`` is a shell command string the remote will execute.
        Post: result captures stdout/stderr; on failure, error_class is
        one of the strings from classify_ssh_error.
        """
        return self._run(["bash", "-lc", cmd], timeout_s=timeout_s)

    def exec_bin(self, cmd: str, *, timeout_s: float = 15.0) -> SshExecResult:
        """Same as exec but for commands whose stdout is binary (no decode)."""
        return self._run(["bash", "-lc", cmd], timeout_s=timeout_s, binary=True)

    def fetch_file_size(self, remote_path: str) -> int | None:
        """Return the size in bytes of a remote file, or None if the call failed.

        Used by the incremental log-tail loop to detect "new bytes since
        last poll" without re-fetching the whole file each time.
        """
        res = self.exec(f"stat -c %s {self._shquote(remote_path)} 2>/dev/null")
        if not res.success or not res.stdout.strip():
            return None
        try:
            return int(res.stdout.strip())
        except ValueError:
            return None

    def fetch_bytes_from(self, remote_path: str, start_offset: int) -> bytes | None:
        """Stream bytes from ``start_offset`` onwards in ``remote_path``.

        Returns the chunk, or None on failure. Used to append-only mirror
        the pod's stderr log to the GUI server's local copy.
        """
        # tail -c +N is 1-indexed; we want bytes after byte `start_offset`,
        # which means starting AT byte (start_offset + 1) in tail's counting.
        cmd = f"tail -c +{start_offset + 1} {self._shquote(remote_path)}"
        res = self.exec_bin(cmd, timeout_s=30.0)
        return res.stdout if res.success else None

    def close(self) -> None:
        """Tear down the ControlMaster socket so no SSH session lingers."""
        if self._control_path.exists():
            with contextlib.suppress(Exception):
                self._build_ssh_cmd("-O", "exit").run_silent()

    # ---- internals ---------------------------------------------------------

    def _run(self, remote_argv: list[str], *, timeout_s: float, binary: bool = False) -> SshExecResult:
        ssh_cmd = self._build_ssh_cmd(*remote_argv)
        try:
            completed = subprocess.run(
                ssh_cmd.argv,
                capture_output=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as e:
            return SshExecResult(
                success=False,
                error_class=classify_ssh_error(TimeoutError(str(e))),
                error_message=f"ssh timed out after {timeout_s}s",
            )
        except OSError as e:
            return SshExecResult(
                success=False,
                error_class=classify_ssh_error(e),
                error_message=f"ssh process failed: {e}",
            )

        if completed.returncode == 0:
            return SshExecResult(
                success=True,
                stdout=completed.stdout,
                stderr=completed.stderr,
                exit_code=0,
            )

        # ssh exits 255 on connection errors, otherwise it forwards the
        # remote command's exit code. 255 → SSH-layer failure (transient/permanent);
        # anything else → remote command failed (not our concern at the SSH layer).
        if completed.returncode == 255:
            return SshExecResult(
                success=False,
                stderr=completed.stderr,
                exit_code=255,
                error_class="permanent" if self._stderr_looks_permanent(completed.stderr) else "transient",
                error_message=completed.stderr.decode(errors="replace").strip()[:300],
            )

        # Remote command itself failed (e.g. `cat` returned 1 because file
        # missing). Treat as a non-fatal success at the SSH layer; caller
        # decides what to do with the non-zero exit code.
        return SshExecResult(
            success=True,
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
        )

    def _build_ssh_cmd(self, *remote: str) -> _SshArgvBuilder:
        argv = ["ssh"]
        # ControlMaster/ControlPersist: first call opens a multiplexed
        # control socket; subsequent calls reuse it for 10 minutes,
        # avoiding repeated TCP handshakes and auth.
        argv += [
            "-o",
            "ControlMaster=auto",
            "-o",
            f"ControlPath={self._control_path}",
            "-o",
            "ControlPersist=600",
            "-o",
            "BatchMode=yes",  # never prompt for password
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-p",
            str(self.host.ssh_port),
        ]
        # No ``-i <key>``. Identity resolution via ``~/.ssh/config`` /
        # ssh-agent / default-path keys — see DESIGN.md § Authentication.
        argv.append(f"{self.host.ssh_user}@{self.host.ssh_host}")
        argv.extend(remote)
        return _SshArgvBuilder(argv)

    @staticmethod
    def _shquote(s: str) -> str:
        """Single-quote a string for safe inclusion in a remote shell command."""
        return "'" + s.replace("'", "'\\''") + "'"

    @staticmethod
    def _stderr_looks_permanent(stderr: bytes) -> bool:
        """Heuristic: which ssh-layer errors should bypass backoff retry."""
        s = stderr.decode(errors="replace").lower()
        return any(
            pat in s
            for pat in (
                "permission denied",
                "host key verification failed",
                "no matching host key",
                "could not resolve hostname",  # DNS: bad config, not transient
            )
        )


@dataclass
class _SshArgvBuilder:
    """Tiny helper so SshConnection can build + optionally run an ssh argv."""

    argv: list[str]

    def run_silent(self) -> None:
        subprocess.run(self.argv, capture_output=True, check=False)


# ── Worker state + writer thread ────────────────────────────────────────────


class _WorkerState:
    """In-memory state of one training-worker process.

    Single instance per worker. The polling-loop writes to fields here;
    the writer thread periodically flushes the state to progress.json.
    """

    def __init__(self, config: TrainingJobConfig, paths: TrainingJobPaths) -> None:
        self.config = config
        self.paths = paths
        self._lock = threading.Lock()
        self._stop_writer = threading.Event()

        self.started_at = time.time()
        self.finished_at: float | None = None
        self.status: str = "pending"
        self.connection_state: str = "initial"
        self.step: int = 0
        self.total_steps: int = 0
        self.milestone: str = "starting"
        self.milestone_at: float = self.started_at
        self.loss_recent: list[float] = []
        self.current_file: str | None = None
        self.gpu_util_pct: int | None = None
        self.gpu_mem_used_gb: float | None = None
        self.error: str | None = None
        self.error_class: str | None = None
        self.consecutive_poll_failures: int = 0
        self.cancel_requested: bool = False

    # ---- snapshot + flush --------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "job_id": self.config.job_id,
                "host_name": self.config.host_name,
                "dataset_id": self.config.dataset_id,
                "recipe_name": self.config.recipe_name,
                "status": self.status,
                "connection_state": self.connection_state,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "step": self.step,
                "total_steps": self.total_steps,
                "milestone": self.milestone,
                "milestone_at": self.milestone_at,
                "loss_recent": list(self.loss_recent),
                "current_file": self.current_file,
                "gpu_util_pct": self.gpu_util_pct,
                "gpu_mem_used_gb": self.gpu_mem_used_gb,
                "error": self.error,
                "error_class": self.error_class,
                "consecutive_poll_failures": self.consecutive_poll_failures,
            }

    def write_progress(self) -> None:
        atomic_write_json(self.paths.progress, self.snapshot())

    def set_status(self, status: str) -> None:
        with self._lock:
            self.status = status
        self.write_progress()

    def set_connection(self, state: str, consecutive_failures: int = 0) -> None:
        with self._lock:
            self.connection_state = state
            self.consecutive_poll_failures = consecutive_failures
        self.write_progress()

    # ---- writer-thread heartbeat -------------------------------------------

    def start_writer_thread(self) -> threading.Thread:
        """Periodic flush. Catches field updates that don't trigger an
        explicit write_progress() call (e.g., loss appended mid-loop)."""

        def _run() -> None:
            while not self._stop_writer.wait(PROGRESS_WRITE_INTERVAL_S):
                with contextlib.suppress(Exception):
                    self.write_progress()

        t = threading.Thread(target=_run, name="train-worker-progress", daemon=True)
        t.start()
        return t

    def stop_writer_thread(self) -> None:
        self._stop_writer.set()


# ── Signal handling ─────────────────────────────────────────────────────────


def _install_signal_handlers(state: _WorkerState) -> None:
    """Set cancel flag on SIGTERM/SIGINT; the main loop observes it.

    Do NOT acquire state._lock here — Python delivers signals synchronously
    on the main thread, and the main thread may already hold the lock.
    Self-deadlock on a non-reentrant lock would wedge the worker. The
    individual field writes below are atomic in CPython; transient
    inconsistency in a snapshot is preferable to a frozen worker. (Same
    lesson as the Hub Transfers worker — see its signal-handler comment.)
    """

    def _on_sigterm(signum, frame):  # noqa: ARG001
        state.cancel_requested = True
        state.milestone = "cancelling"
        state.milestone_at = time.time()

    signal.signal(signal.SIGTERM, _on_sigterm)
    signal.signal(signal.SIGINT, _on_sigterm)


# ── Polling loop ────────────────────────────────────────────────────────────


def poll_once(
    ssh: SshConnection,
    paths: TrainingJobPaths,
    state: _WorkerState,
    remote_progress_path: str,
    remote_log_path: str,
    log_offset: int,
) -> tuple[bool, str | None, int]:
    """One iteration of the polling loop.

    Returns ``(success, error_class_if_failed, new_log_offset)``.

    Steps in order:
      1. Fetch the remote progress.json. If we can't, the SSH layer is
         broken → transient/permanent failure.
      2. Fetch any new bytes in the remote stderr log since last offset.
      3. Update local state from the remote snapshot.

    The caller (the polling loop) feeds the success/failure into
    PollScheduler to decide the next sleep + whether to give up.
    """
    # (1) progress.json
    prog_res = ssh.exec(f"cat {SshConnection._shquote(remote_progress_path)}")
    if not prog_res.success:
        return False, prog_res.error_class or "transient", log_offset

    # The remote file may not exist yet (training just started). Treat
    # "exit_code != 0" as a transient probe miss — file will appear soon.
    if prog_res.exit_code != 0:
        return True, None, log_offset

    # Parse + merge
    try:
        snapshot = json.loads(prog_res.stdout.decode(errors="replace"))
    except json.JSONDecodeError:
        # Mid-write read — try again next poll. Not a connection failure.
        return True, None, log_offset

    with state._lock:
        for k in (
            "status",
            "step",
            "total_steps",
            "milestone",
            "milestone_at",
            "loss_recent",
            "current_file",
            "gpu_util_pct",
            "gpu_mem_used_gb",
            "error",
            "error_class",
            "finished_at",
        ):
            if k in snapshot and snapshot[k] is not None:
                setattr(state, k, snapshot[k])

    # (2) incremental log tail. Best-effort; failure here doesn't fail
    # the poll (we still got the structured progress, which is more
    # important than the raw log).
    new_offset = log_offset
    size = ssh.fetch_file_size(remote_log_path)
    if size is not None and size > log_offset:
        chunk = ssh.fetch_bytes_from(remote_log_path, log_offset)
        if chunk is not None:
            paths.base.mkdir(parents=True, exist_ok=True)
            with open(paths.log, "ab") as f:
                f.write(chunk)
            new_offset = size
            paths.log_offset.write_text(str(new_offset))
    elif size is not None and size < log_offset:
        # File was truncated or rotated — start fresh from 0.
        new_offset = 0

    return True, None, new_offset


def run_polling_loop(
    ssh: SshConnection,
    paths: TrainingJobPaths,
    state: _WorkerState,
    remote_progress_path: str,
    remote_log_path: str,
) -> None:
    """The main loop. Polls until terminal state or cancel.

    Resilience design (from src/lerobot/gui/training/DESIGN.md):
    - Backoff via PollScheduler on transient failures
    - Immediate give-up on permanent failures (auth, host key, DNS)
    - State transitions logged to events.jsonl for audit
    - On max-retries-exceeded: mark lost_contact + status=failed
    """
    scheduler = PollScheduler()
    log_offset = 0
    if paths.log_offset.exists():
        with contextlib.suppress(ValueError):
            log_offset = int(paths.log_offset.read_text().strip())

    state.set_connection("initial")

    while not state.cancel_requested:
        ok, err_class, log_offset = poll_once(
            ssh, paths, state, remote_progress_path, remote_log_path, log_offset
        )
        is_permanent = err_class == "permanent"

        if ok:
            if state.connection_state != "connected":
                # Recovered (or first successful poll)
                if state.consecutive_poll_failures > 0:
                    append_event(
                        paths.events,
                        "reconnected",
                        missed_polls=state.consecutive_poll_failures,
                    )
                else:
                    append_event(paths.events, "connected")
            state.set_connection("connected", 0)
        else:
            attempt = state.consecutive_poll_failures + 1
            append_event(
                paths.events,
                "poll_failed",
                error_class=err_class,
                attempt=attempt,
            )

        delay = scheduler.schedule_next(success=ok, permanent=is_permanent)
        if delay is None:
            # Give up: either too many retries, or hit a permanent error.
            reason = "permanent_error" if is_permanent else "max_retries_exceeded"
            append_event(paths.events, "marked_lost_contact", reason=reason)
            state.set_connection("lost_contact", state.consecutive_poll_failures)
            with state._lock:
                state.status = "failed"
                state.error = state.error or ("SSH auth failed" if is_permanent else "Pod unreachable")
                state.error_class = state.error_class or "pod_unreachable"
                state.finished_at = time.time()
            state.write_progress()
            return

        if not ok:
            state.set_connection("reconnecting", state.consecutive_poll_failures + 1)

        # Stop polling if remote training reached terminal state
        if state.status in ("complete", "failed", "cancelled"):
            return

        # Sleep with periodic wake to honor cancel_requested promptly
        deadline = time.monotonic() + delay
        while time.monotonic() < deadline and not state.cancel_requested:
            time.sleep(min(0.5, deadline - time.monotonic()))


# ── Worker main entry point ─────────────────────────────────────────────────


def _load_config() -> tuple[TrainingJobConfig, TrainingJobPaths, HostProfile]:
    raw = os.environ.get("LEROBOT_TRAIN_WORKER_CONFIG")
    if raw is None:
        print("training_worker: missing LEROBOT_TRAIN_WORKER_CONFIG env var", file=sys.stderr)
        sys.exit(2)
    cfg = TrainingJobConfig.from_json(raw)
    paths = TrainingJobPaths.for_job(cfg.job_id, cfg.jobs_dir)
    paths.ensure_dir()
    # Load host profile by name. The GUI server is the source of truth
    # for these; the worker only reads them.
    from lerobot.gui.training.jobs import HOSTS_DIR

    host_path = HOSTS_DIR / f"{cfg.host_name}.json"
    if not host_path.exists():
        print(f"training_worker: host profile not found: {host_path}", file=sys.stderr)
        sys.exit(2)
    host = HostProfile.load(host_path)
    return cfg, paths, host


def main() -> int:
    cfg, paths, host = _load_config()
    state = _WorkerState(cfg, paths)
    _install_signal_handlers(state)

    # Write our PID so the server can verify identity for cancel.
    atomic_write_json(paths.pid, {"pid": os.getpid(), "started_at": state.started_at})

    state.set_status("starting")
    append_event(paths.events, "worker_started", pid=os.getpid())
    writer = state.start_writer_thread()

    ssh = SshConnection(host, control_path_dir=paths.base)

    rc = 0
    try:
        # The actual "spawn lerobot-train in tmux on the pod" step is
        # staged for a follow-up commit — this skeleton focuses on the
        # resilience-design pieces (SshConnection, PollScheduler,
        # events.jsonl, polling loop). The spawn step would look like:
        #
        #   remote_run_dir = f"/workspace/runs/{cfg.job_id}"
        #   ssh.exec(f"mkdir -p {remote_run_dir}")
        #   docker_cmd = build_docker_run_cmd(host, cfg, remote_run_dir)
        #   ssh.exec(
        #     f"tmux new-session -d -s lerobot-train-{cfg.job_id} "
        #     f"'exec {docker_cmd} > {remote_run_dir}/stderr.log 2>&1'"
        #   )
        #   state.set_status("running")
        #
        # For now the worker just polls whatever progress.json is at the
        # remote path — useful for testing the polling/resilience layer
        # against a hand-spawned remote process.
        remote_run_dir = f"/workspace/runs/{cfg.job_id}"
        remote_progress = f"{remote_run_dir}/progress.json"
        remote_log = f"{remote_run_dir}/stderr.log"

        run_polling_loop(ssh, paths, state, remote_progress, remote_log)

        if state.cancel_requested:
            with state._lock:
                state.status = "cancelled"
                state.error = "Cancelled by user"
                state.error_class = "cancelled"
                state.finished_at = time.time()
            append_event(paths.events, "cancelled_by_user")
    except Exception as e:  # noqa: BLE001 — terminal-error catch is intentional
        with state._lock:
            state.status = "failed"
            state.error = f"{type(e).__name__}: {e}"
            state.error_class = "other"
            state.finished_at = time.time()
        append_event(paths.events, "worker_crashed", error=str(e))
        rc = 1
    finally:
        state.stop_writer_thread()
        writer.join(timeout=1.0)
        state.write_progress()  # final flush AFTER writer is stopped
        ssh.close()
        append_event(paths.events, "worker_exited", returncode=rc)
        # Leave pid file for server's startup sweep; server reaps it.

    return rc


if __name__ == "__main__":
    sys.exit(main())
