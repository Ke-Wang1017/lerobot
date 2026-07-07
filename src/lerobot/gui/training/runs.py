# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Run data model + filesystem registry.

A "run" is one invocation of the training pipeline: pick a host, pick a
recipe + dataset, click Start. The Run lives in
``~/.cache/lerobot/runs/<run_id>/`` (configurable via ``RUNS_DIR``):

- ``run.json`` — metadata (recipe, dataset, host, state machine state, pid)
- ``progress.json`` — atomically rewritten on each training step (worker writes)
- ``events.jsonl`` — append-only state transitions (worker writes)
- ``checkpoints.jsonl`` — manifest: one line per completed checkpoint (worker writes)
- ``stderr.log`` — merged stdout + stderr of the training process (worker writes)
- ``checkpoints/<step>/`` — actual checkpoint files (worker writes)

The state machine matches DESIGN.md § Concurrency:
``pending → running → completing → completed / failed / aborted``.

Transitions are gated; you can't go back from a terminal state. Concurrency
defense: the registry's per-host index makes the per-host single-active-run
lock enforceable in one place.
"""

from __future__ import annotations

import contextlib
import enum
import json
import os
import secrets
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# ── Constants ──────────────────────────────────────────────────────────────────


# Configurable via env so tests can use a tmp dir. Default mirrors HF / standard
# user cache convention.
def _default_runs_dir() -> Path:
    env = os.environ.get("LEROBOT_RUNS_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "lerobot" / "runs"


RUNS_DIR = _default_runs_dir()


# ── State machine ─────────────────────────────────────────────────────────────


class RunState(str, enum.Enum):
    """Lifecycle states. Stored as strings for JSON round-tripping."""

    PENDING = "pending"  # created, worker not yet launched
    RUNNING = "running"  # worker process is alive
    COMPLETING = "completing"  # SIGTERM sent, awaiting final events
    COMPLETED = "completed"  # worker exited cleanly (worker wrote completed_naturally)
    FAILED = "failed"  # worker exited unexpectedly (crash)
    ABORTED = "aborted"  # user-initiated stop, completed cleanly


TERMINAL_STATES = frozenset({RunState.COMPLETED, RunState.FAILED, RunState.ABORTED})

# Allowed transitions. Anything not in this set raises in advance() — the
# state machine prevents stale duplicate "start" requests re-running a finished
# run (DESIGN.md § Concurrency).
_ALLOWED: dict[RunState, frozenset[RunState]] = {
    RunState.PENDING: frozenset({RunState.RUNNING, RunState.FAILED, RunState.ABORTED}),
    RunState.RUNNING: frozenset({RunState.COMPLETING, RunState.COMPLETED, RunState.FAILED, RunState.ABORTED}),
    RunState.COMPLETING: frozenset({RunState.COMPLETED, RunState.FAILED, RunState.ABORTED}),
    RunState.COMPLETED: frozenset(),
    RunState.FAILED: frozenset(),
    RunState.ABORTED: frozenset(),
}


# ── Run dataclass ─────────────────────────────────────────────────────────────


@dataclass
class Run:
    """Metadata for one training run. Persisted as ``run.json`` in the run dir."""

    run_id: str  # opaque short id (uuid-derived)
    host_id: str  # which host profile / "this-server" etc.
    recipe_name: str  # display + reuse
    dataset_id: str  # display + reuse
    args: dict[str, Any]  # arbitrary kwargs forwarded to the training runner
    state: RunState
    created_at: float  # unix timestamp
    started_at: float | None = None
    finished_at: float | None = None
    # Opaque to the orchestrator — only the client that launched the run
    # knows how to interpret it. SubprocessClient: stringified PID.
    # SshClient: ``<tmux-session-name>|<remote-workdir>``. Persisted as
    # a JSON string; clients parse on use.
    session_id: str | None = None
    idempotency_key: str | None = None  # client-supplied, defends against double-clicks
    error: str | None = None  # short reason for FAILED state
    # Ephemeral (provider-spawned) runs only. Persisted so the orchestrator
    # can tear the VM down on every terminal transition — including after a
    # GUI-server restart, where the in-memory handle would be lost. Stored as
    # a plain dict (the serialized HostHandle); None for workstation/SSH runs.
    ephemeral_handle: dict[str, Any] | None = None
    # Set once the provider has destroyed the spawned VM, so teardown is
    # idempotent (we never double-destroy, and verify only runs once).
    ephemeral_destroyed: bool = False

    def to_json(self) -> str:
        d = asdict(self)
        d["state"] = self.state.value
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> Run:
        d = json.loads(raw)
        d["state"] = RunState(d["state"])
        # Legacy records (pre-SSH transport) persisted session_id as the
        # raw subprocess PID (int). The Protocol widened it to str; coerce
        # on load so old run history doesn't poison list_runs().
        if isinstance(d.get("session_id"), int):
            d["session_id"] = str(d["session_id"])
        return cls(**d)

    def advance(self, to: RunState) -> None:
        """Transition the state machine. Asserts the transition is allowed."""
        allowed = _ALLOWED[self.state]
        assert to in allowed, (
            f"illegal state transition: {self.state.value} → {to.value}; "
            f"allowed from {self.state.value}: {sorted(s.value for s in allowed)}"
        )
        self.state = to
        if to == RunState.RUNNING and self.started_at is None:
            self.started_at = time.time()
        if to in TERMINAL_STATES:
            self.finished_at = time.time()


# ── Per-run filesystem layout ─────────────────────────────────────────────────


@dataclass(frozen=True)
class RunPaths:
    """File locations both the orchestrator and the worker agree on."""

    root: Path
    run_id: str

    @classmethod
    def for_run(cls, run_id: str, runs_dir: Path | None = None) -> RunPaths:
        base = (runs_dir or RUNS_DIR) / run_id
        return cls(root=base, run_id=run_id)

    @property
    def run_json(self) -> Path:
        return self.root / "run.json"

    @property
    def progress_json(self) -> Path:
        return self.root / "progress.json"

    @property
    def events_jsonl(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def checkpoints_jsonl(self) -> Path:
        return self.root / "checkpoints.jsonl"

    @property
    def stderr_log(self) -> Path:
        return self.root / "stderr.log"

    @property
    def metrics_jsonl(self) -> Path:
        """Training-signal series parsed from stderr — one JSON row per logged
        step. Distinct from progress.json (position): see DESIGN.md § Polling."""
        return self.root / "metrics.jsonl"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    def ensure_exists(self) -> None:
        """Create the run dir if missing. Does NOT pre-create
        ``checkpoints/`` — the worker (fake or real lerobot-train) creates
        it on first checkpoint write. Pre-creating an empty checkpoints/
        confuses scanners that use its existence as the "is this a training
        run?" signal (e.g., the Models tab scanner)."""
        self.root.mkdir(parents=True, exist_ok=True)


# ── Registry: load / save / list runs ─────────────────────────────────────────


def _atomic_write(path: Path, content: str) -> None:
    """Write content to ``path`` atomically via temp-file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)  # safe-destruct: our own mkstemp tmp file, failed-write cleanup
        raise


def new_run_id() -> str:
    """Short, URL-safe, sortable-ish run id. UUID4 hex prefix — collision-free
    in practice for any single user's lifetime of runs."""
    return uuid.uuid4().hex[:12]


def new_idempotency_key() -> str:
    """For tests / programmatic callers; clients normally supply their own."""
    return secrets.token_hex(16)


class RunRegistry:
    """File-backed run registry.

    One ``run.json`` per run dir. The whole registry is just the filesystem;
    no in-memory cache (read each call, cheap enough). Adding a database later
    would be a local change to this class.
    """

    def __init__(self, runs_dir: Path | None = None) -> None:
        self.runs_dir = runs_dir or RUNS_DIR

    def save(self, run: Run) -> None:
        paths = RunPaths.for_run(run.run_id, self.runs_dir)
        paths.ensure_exists()
        _atomic_write(paths.run_json, run.to_json())

    def load(self, run_id: str) -> Run | None:
        paths = RunPaths.for_run(run_id, self.runs_dir)
        if not paths.run_json.exists():
            return None
        return Run.from_json(paths.run_json.read_text())

    def list_all(self) -> list[Run]:
        if not self.runs_dir.exists():
            return []
        runs: list[Run] = []
        for child in self.runs_dir.iterdir():
            if not child.is_dir():
                continue
            run_json = child / "run.json"
            if not run_json.exists():
                continue
            try:
                runs.append(Run.from_json(run_json.read_text()))
            except (json.JSONDecodeError, KeyError, ValueError):
                # Skip malformed entries; better than failing the whole list.
                continue
        # Newest first.
        runs.sort(key=lambda r: r.created_at, reverse=True)
        return runs

    def find_by_idempotency_key(self, key: str) -> Run | None:
        """Look for an existing run with the given idempotency key.

        Used to deflect double-Start clicks: the same key returns the same
        run_id instead of creating a duplicate (DESIGN.md § Concurrency).
        """
        if not key:
            return None
        for run in self.list_all():
            if run.idempotency_key == key:
                return run
        return None

    def active_run_on_host(self, host_id: str) -> Run | None:
        """Return the active (non-terminal) run on ``host_id``, if any.

        Per-host single-active-run lock (DESIGN.md § Concurrency, v1 single-GPU model).
        """
        for run in self.list_all():
            if run.host_id == host_id and run.state not in TERMINAL_STATES:
                return run
        return None


# ── Append helper for events.jsonl (used by both orchestrator + runner) ────────


def append_event(events_path: Path, type_: str, **fields: Any) -> None:
    """Append a JSON line to ``events.jsonl``. Caller-defined ``type_`` + fields.

    Standard event types (DESIGN.md § Health):

    - ``started``                 — worker launched (orchestrator)
    - ``completed_naturally``     — recipe finished (worker)
    - ``aborted_by_user``         — Stop button (worker, after SIGTERM)
    - ``crashed``                 — unexpected exit (orchestrator on detection)
    - ``connect`` / ``disconnect`` / ``retry`` / ``lost_contact`` — transport
    """
    events_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"type": type_, "ts": time.time(), **fields}) + "\n"
    with events_path.open("a") as f:
        f.write(line)
