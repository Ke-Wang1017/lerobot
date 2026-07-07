# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Training-job types + shared helpers (boundary between GUI server and worker).

Mirrors the architectural shape of :mod:`lerobot.gui.hub_jobs`:

* :class:`TrainingJobConfig` — JSON payload the GUI server hands to the
  worker (via ``LEROBOT_TRAIN_WORKER_CONFIG`` env var). Frozen +
  JSON-roundtripped.
* :class:`TrainingJobPaths` — per-job file locations under
  ``~/.cache/lerobot/gui/training/jobs``. Both sides derive the same paths
  from the same ``job_id``.
* :class:`TrainingJobState` — server-side in-memory mirror, populated by
  merging the worker's ``progress.json`` with server bookkeeping.
* :class:`HostProfile` — saved SSH endpoint + image_ref + capabilities for
  a training host. Lives under ``~/.config/lerobot/training_hosts/<name>.json``.
* :class:`PollScheduler` — exponential backoff with cap + max-attempts for
  the worker's SSH polling loop. Distinguishes transient vs permanent
  errors so auth failures fail-fast instead of burning all 10 retries.
* :func:`classify_ssh_error` — buckets a Python exception into one of the
  five error classes the UI surfaces (transient / permanent / etc.).

No transfer logic here; see :mod:`lerobot.gui.training.worker`.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


# ── Constants ───────────────────────────────────────────────────────────────


# Per-job IPC files. One subdir per running/recent training run; server
# cleans up terminal entries older than STALE_TERMINAL_RETENTION_S.
JOBS_DIR = Path.home() / ".cache" / "lerobot" / "gui" / "training_jobs"

# Host profile storage — mirror of robot profile pattern, one JSON per host.
HOSTS_DIR = Path.home() / ".config" / "lerobot" / "training_hosts"

# How often the worker refreshes progress.json + polls the remote pod.
# Same rationale as Hub Transfers' 0.5s tick (write rate) and 5s pod-poll
# rate: writes are cheap, network probes are not.
PROGRESS_WRITE_INTERVAL_S = 0.5
POD_POLL_INTERVAL_S = 5.0

# After a job reaches a terminal state, hold its entry + per-job files for
# this long so the GUI can show "recently failed / completed" + offer
# Resume, then GC.
STALE_TERMINAL_RETENTION_S = 1800.0  # 30 min

# Default training extras to pass to lerobot-train when none specified.
# Smoke runs use this set.
DEFAULT_TRAIN_EXTRA_ARGS: tuple[str, ...] = (
    "--policy.device=cuda",
    "--wandb.enable=false",
)


# Narrow string types — small, fixed, makes UI dispatch easy.
HostKind = Literal["local", "permanent", "temporary"]
TrainingStatus = Literal[
    "pending",  # job created, worker not started
    "starting",  # worker spawned, SSH being established
    "running",  # training actively progressing
    "complete",  # training finished cleanly
    "failed",  # training crashed or pod unreachable
    "cancelled",  # user cancelled
]
ConnectionState = Literal[
    "initial",  # no successful poll yet
    "connected",  # last poll succeeded
    "reconnecting",  # transient failures, in backoff
    "lost_contact",  # exceeded max retries OR permanent error
]
TrainingErrorClass = Literal[
    "transient",  # network drop, timeout, refused — retry with backoff
    "permanent",  # auth failed, host key changed — give up immediately
    "pod_unreachable",  # pod was terminated / never reachable again
    "training_crashed",  # lerobot-train errored inside the container
    "cancelled",  # user-initiated stop
    "other",
]


# ── HostProfile: saved SSH host with capabilities ──────────────────────────


@dataclass
class HostProfile:
    """A saved training-host endpoint. Mirrors robot-profile JSON pattern.

    Stored as ``~/.config/lerobot/training_hosts/<name>.json``. The GUI's
    "Add training host" dialog creates / edits these; the training worker
    reads them to know where to SSH.
    """

    name: str
    # SSH coordinates — present for workstation/persistent (user-added SSH)
    # hosts; empty for Ephemeral hosts, whose VM (and thus endpoint) doesn't
    # exist until a run spawns it.
    ssh_user: str = ""
    ssh_host: str = ""
    ssh_port: int = 22
    # No private-key fields by design. Authentication uses the user's
    # local SSH setup (``~/.ssh/config`` aliases, ssh-agent, default-path
    # keys) — the GUI server never reads or stores key bytes. See
    # DESIGN.md § Host setup UX + § Authentication.
    kind: HostKind = "temporary"
    display_name: str = ""  # user-facing label; defaults to name in __post_init__
    workdir: str = "/workspace/lerobot"
    image_ref: str = "ghcr.io/thewisp/lerobot-training:latest"
    persistent_volume: str | None = None  # for HF cache survival across pods
    # ── Ephemeral (provider-spawned) fields ──────────────────────────────
    # Set iff this is an Ephemeral cloud host. ``provider_id`` selects the
    # HostProvider; the rest describe the SpawnSpec the orchestrator hands it.
    provider_id: str | None = None
    gpu: str = "L40S"
    gpu_count: int = 1
    disk_gib: int = 100
    preemptible: bool = True
    region_hint: str | None = None
    ttl_hours: int = 24  # hard-TTL runaway backstop (see DESIGN.md § Cost discipline)
    # Capabilities discovered at host-profile-setup time. Mostly informational.
    capabilities: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_connected_at: float | None = None

    def __post_init__(self) -> None:
        if not self.display_name:
            object.__setattr__(self, "display_name", self.name)

    @property
    def is_ephemeral(self) -> bool:
        return self.provider_id is not None

    @classmethod
    def load(cls, path: Path) -> HostProfile:
        d = json.loads(path.read_text())
        # Drop unknown keys defensively so old saved files survive schema
        # additions and new saved files survive schema removals.
        field_names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})

    @classmethod
    def load_all(cls, dir_: Path = HOSTS_DIR) -> list[HostProfile]:
        """All saved profiles under ``dir_``. Missing dir → empty list.
        Files that fail to parse are logged and skipped — one corrupt
        entry must not break the registry for the rest."""
        if not dir_.exists():
            return []
        out: list[HostProfile] = []
        for path in sorted(dir_.glob("*.json")):
            try:
                out.append(cls.load(path))
            except Exception as e:
                logger.warning("skipping unreadable host profile %s: %s", path, e)
        return out

    @classmethod
    def delete(cls, name: str, dir_: Path = HOSTS_DIR) -> bool:
        """Remove ``<dir>/<name>.json``. Returns True if a file was removed,
        False if it didn't exist (no-op, idempotent).

        Pre: ``name`` is a bare profile name, not a path. Callers (the
        DELETE endpoint) gate on registry membership, but assert here too
        so no future caller can turn this into a traversal primitive."""
        assert "/" not in name and "\\" not in name and ".." not in name, f"unsafe profile name: {name!r}"
        path = dir_ / f"{name}.json"
        if not path.exists():
            return False
        path.unlink()  # safe-destruct: host profile removal, user-initiated via the GUI's DELETE endpoint
        return True

    def save(self, dir_: Path = HOSTS_DIR) -> Path:
        dir_.mkdir(parents=True, exist_ok=True)
        path = dir_ / f"{self.name}.json"
        path.write_text(json.dumps(self.__dict__, indent=2, default=str))
        return path

    @property
    def ssh_endpoint(self) -> str:
        return f"{self.ssh_user}@{self.ssh_host}:{self.ssh_port}"


# ── TrainingJobConfig: server → worker payload ──────────────────────────────


@dataclass(frozen=True)
class TrainingJobConfig:
    """Immutable config the GUI server hands the worker via env var.

    Roundtripped through JSON so the worker can be a clean subprocess
    that doesn't need to share Python objects with the parent.

    Pre: ``host`` exists in ~/.config/lerobot/training_hosts/. ``args`` is
    a list of CLI flags forwarded verbatim to ``lerobot-train`` inside the
    container.
    """

    job_id: str
    host_name: str  # which HostProfile to use; resolved at worker startup
    dataset_id: str  # for display + reuse-PR lookup; matches lerobot dataset id
    recipe_name: str  # e.g. "act-default"; for display + run-history reuse
    args: list[str]  # lerobot-train CLI args, fully resolved (no defaults applied later)
    image_ref: str  # full image ref to docker-pull; usually copy of host.image_ref
    bind_local_src: bool = False  # rsync src/lerobot into the pod (dev iteration)
    jobs_dir: str = str(JOBS_DIR)

    def to_json(self) -> str:
        d = dict(self.__dict__)
        return json.dumps(d)

    @classmethod
    def from_json(cls, raw: str) -> TrainingJobConfig:
        return cls(**json.loads(raw))


# ── TrainingJobPaths: file locations both sides agree on ───────────────────


@dataclass(frozen=True)
class TrainingJobPaths:
    """Per-job file paths. Both the GUI server and the worker derive these
    from the same ``job_id``."""

    jobs_dir: Path
    job_id: str

    @classmethod
    def for_job(cls, job_id: str, jobs_dir: Path | str = JOBS_DIR) -> TrainingJobPaths:
        return cls(jobs_dir=Path(jobs_dir), job_id=job_id)

    @property
    def base(self) -> Path:
        return self.jobs_dir / self.job_id

    @property
    def progress(self) -> Path:
        """Atomically-written JSON snapshot of current run state."""
        return self.base / "progress.json"

    @property
    def events(self) -> Path:
        """Append-only JSONL of state transitions (connect, fail, retry)."""
        return self.base / "events.jsonl"

    @property
    def log(self) -> Path:
        """Mirror of the remote pod's stderr (continuously tailed via SSH)."""
        return self.base / "stderr.log"

    @property
    def log_offset(self) -> Path:
        """Last-fetched byte offset for incremental log tail."""
        return self.base / "log.offset"

    @property
    def pid(self) -> Path:
        """Worker PID file (for cancel + dead-worker reconciliation)."""
        return self.base / "worker.pid"

    def ensure_dir(self) -> None:
        self.base.mkdir(parents=True, exist_ok=True)


# ── TrainingJobState: server-side in-memory mirror ──────────────────────────


@dataclass
class TrainingJobState:
    """Mutable in-memory representation. Populated by merging the worker's
    progress.json with server-side bookkeeping (worker PID, etc.).

    Mirrors the shape of :class:`lerobot.gui.hub_jobs.HubJobState`.
    """

    job_id: str
    host_name: str
    dataset_id: str
    recipe_name: str
    status: TrainingStatus = "pending"
    connection_state: ConnectionState = "initial"
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    # Live progress fields written by the worker as it parses lerobot-train stdout.
    step: int = 0
    total_steps: int = 0
    loss_recent: list[float] = field(default_factory=list)  # rolling window, ~100 vals
    milestone: str = "starting"
    milestone_at: float | None = None
    current_file: str | None = None  # for upload-style progress display
    gpu_util_pct: int | None = None  # last sampled
    gpu_mem_used_gb: float | None = None
    # Error / outcome
    error: str | None = None
    error_class: TrainingErrorClass | None = None
    # Worker side
    pid: int | None = None
    # Connection bookkeeping
    last_poll_success_at: float | None = None
    consecutive_poll_failures: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def merge_progress(self, snapshot: dict[str, Any]) -> None:
        """Overwrite locally-mirrored fields with worker's latest progress.json.

        Terminal-state monotonicity: once status is complete/failed/cancelled,
        we don't roll back to running on a stale read. Same invariant as
        HubJobState.merge_progress.
        """
        if self.status in ("complete", "failed", "cancelled"):
            return
        for field_name in (
            "status",
            "connection_state",
            "step",
            "total_steps",
            "loss_recent",
            "milestone",
            "milestone_at",
            "current_file",
            "gpu_util_pct",
            "gpu_mem_used_gb",
            "error",
            "error_class",
            "finished_at",
        ):
            if field_name in snapshot and snapshot[field_name] is not None:
                setattr(self, field_name, snapshot[field_name])


def make_training_job(
    *,
    host_name: str,
    dataset_id: str,
    recipe_name: str,
) -> TrainingJobState:
    """Build a fresh server-side TrainingJobState in `pending`."""
    return TrainingJobState(
        job_id=uuid.uuid4().hex,
        host_name=host_name,
        dataset_id=dataset_id,
        recipe_name=recipe_name,
    )


# ── Atomic JSON write (reuse the Hub Transfers helper) ──────────────────────


def atomic_write_json(path: Path, data: Any) -> None:
    """Atomically write JSON to ``path`` via ``.tmp + os.replace``.

    A concurrent reader sees either the previous coherent content or the
    new, never a partial write. Used for progress.json (rewritten ~2 Hz)
    and the worker's PID file.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, default=str))
    os.replace(tmp, path)


def append_event(events_path: Path, kind: str, **fields: Any) -> None:
    """Append a JSON line to events.jsonl.

    Each line: ``{"ts": <float>, "kind": <str>, ...fields}``. Append-only,
    no truncation — the file is the audit trail for the run.
    """
    events_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": time.time(), "kind": kind, **fields}
    with open(events_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


# ── Error classification ────────────────────────────────────────────────────


def classify_ssh_error(exc: BaseException) -> TrainingErrorClass:
    """Bucket an SSH/network exception into the UI-visible error class.

    Distinguishes transient errors (worth a backoff retry) from permanent
    errors (give up immediately — no number of retries will fix an
    AuthenticationException). The worker uses this directly in the polling
    loop to decide whether to retry or escalate to lost_contact.
    """
    msg = f"{type(exc).__name__}: {exc}".lower()

    # paramiko exceptions — match by class name to avoid hard-importing paramiko here
    type_name = type(exc).__name__
    if type_name in ("AuthenticationException", "BadHostKeyException", "PartialAuthentication"):
        return "permanent"
    if type_name in ("PasswordRequiredException", "SSHException") and (
        "password" in msg or "host key" in msg or "no acceptable" in msg
    ):
        return "permanent"

    # Standard library network errors
    if isinstance(exc, (TimeoutError, ConnectionResetError, ConnectionRefusedError)):
        return "transient"
    if isinstance(exc, OSError) and getattr(exc, "errno", None) in (
        110,  # ETIMEDOUT
        113,  # EHOSTUNREACH (No route to host)
        111,  # ECONNREFUSED — already covered above but defensive
        101,  # ENETUNREACH
        104,  # ECONNRESET
    ):
        return "transient"
    if "timed out" in msg or "connection refused" in msg or "no route" in msg:
        return "transient"

    # Fall-through: assume transient (worth a retry). The worker still
    # caps retries via PollScheduler, so misclassifying as transient costs
    # at most ~5 minutes of backoff before escalating to lost_contact.
    return "transient"


# ── PollScheduler: backoff for the worker's SSH polling loop ────────────────


@dataclass
class PollScheduler:
    """Exponential backoff with cap + max-attempts, distinguishing transient
    from permanent errors.

    Used by the training worker around each SSH probe. Successful polls
    reset the failure counter; transient failures double the delay (capped
    at ``MAX_INTERVAL_S``); permanent failures or hitting ``MAX_ATTEMPTS``
    return None from :meth:`schedule_next` so the worker escalates to
    ``lost_contact``.

    Concrete retry timeline (5s base, 60s cap, 10 attempts):

        attempt 1: 5s
        attempt 2: 10s
        attempt 3: 20s
        attempt 4: 40s
        attempt 5: 60s (cap)
        ...
        attempt 10: 60s
        attempt 11 → None (give up)

    Total time before giving up: ~5 minutes.
    """

    BASE_INTERVAL_S: float = 5.0
    MAX_INTERVAL_S: float = 60.0
    MAX_ATTEMPTS: int = 10

    consecutive_failures: int = 0
    last_success_at: float = field(default_factory=time.time)
    last_failure_at: float | None = None

    def schedule_next(self, success: bool, *, permanent: bool = False) -> float | None:
        """Compute seconds-until-next-poll. Returns None to give up.

        Pre: ``permanent`` is meaningful only when ``success=False``.
        Post: if returns None, the caller should mark connection_state
        as ``lost_contact`` and stop polling.
        """
        if success:
            self.consecutive_failures = 0
            self.last_success_at = time.time()
            return self.BASE_INTERVAL_S

        # Failure path
        self.last_failure_at = time.time()
        if permanent:
            return None
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.MAX_ATTEMPTS:
            return None

        # Exponential backoff: 5, 10, 20, 40, 60 (cap), 60, ...
        delay = self.BASE_INTERVAL_S * (2 ** (self.consecutive_failures - 1))
        return min(delay, self.MAX_INTERVAL_S)

    @property
    def is_failing(self) -> bool:
        """True iff we're currently in the retry-with-backoff state."""
        return self.consecutive_failures > 0
