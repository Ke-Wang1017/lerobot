# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Hub transfer types + shared helpers.

This module is the boundary between the GUI server (FastAPI) and the
worker subprocess that runs one transfer:

* :class:`JobConfig` — the JSON payload the server passes to the worker via
  the ``LEROBOT_HUB_WORKER_CONFIG`` env var. Frozen + JSON-roundtripped.
* :class:`JobPaths` — per-job file paths under ``~/.cache/lerobot/gui/hub_jobs``.
  Both sides compute the same paths from the same job_id.
* :class:`HubJobState` — the in-memory mirror the server keeps, populated
  from the per-job progress JSON file plus server-side bookkeeping.
* helpers — process-identity checks, error classification, file enumeration.

No transfer logic lives here; see :mod:`lerobot.gui.hub_worker` for that.
For the full design see :doc:`gui/docs/hub_transfers.md`.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


# ── Constants ───────────────────────────────────────────────────────────────


# Where per-job IPC files live. One subdir per running/recent worker; the
# server cleans up entries after `STALE_TERMINAL_RETENTION_S`.
JOBS_DIR = Path.home() / ".cache" / "lerobot" / "gui" / "hub_jobs"

# Worker writes its progress file at this interval. Server polls at the
# same rate via /hub/jobs; faster polling sees the same snapshot, slower
# polling loses a fraction of a second of freshness.
PROGRESS_WRITE_INTERVAL_S = 0.5

# Soft cap for super_squash_history's HTTP call. If HF takes longer than
# this server-side (typical for multi-TB uploads with hundreds of batch
# commits), the worker treats it as a failure and falls back to merging
# the unsquashed branch.
SUPER_SQUASH_TIMEOUT_S = 60.0

# After a job reaches a terminal state, we hold its entry + files for this
# long so the GUI can show "recently failed" + retry, then GC them.
STALE_TERMINAL_RETENTION_S = 1800.0  # 30 min

# Paths under the upload root that we never push: GUI-local metadata + HF
# cache lock files + temp artifacts left by interrupted writes. These
# match what HF itself refuses to commit anyway (it rejects `.cache/`).
DEFAULT_UPLOAD_IGNORES: tuple[str, ...] = (
    ".cache/",
    ".lerobot_gui_edits.json",
    ".huggingface/",
    ".DS_Store",
)


# Narrow string types — keeps the GUI renderer's dispatch trivial.
HubDirection = Literal["upload", "download"]
HubStatus = Literal["pending", "running", "complete", "failed", "cancelled"]
HubRepoType = Literal["dataset", "model"]

# Error classifications — used by the tray to surface a specific
# remediation hint rather than just dumping the exception string.
HubErrorClass = Literal["auth", "rate_limit", "network", "cancelled", "other"]


# ── JobConfig: server → worker payload ──────────────────────────────────────


@dataclass(frozen=True)
class JobConfig:
    """Immutable config passed from server to worker.

    The server constructs this, serializes to JSON, sets it in
    ``LEROBOT_HUB_WORKER_CONFIG``, and execs the worker module. The worker
    parses it at startup and never re-reads.

    Pre: ``local_path`` exists (for uploads) or its parent exists (for
    downloads); ``repo_id`` is a valid ``owner/name`` string.
    """

    job_id: str
    dataset_id: str  # The GUI's identifier for the local dataset; not always == repo_id.
    direction: HubDirection
    repo_id: str
    repo_type: HubRepoType
    local_path: str  # Resolved absolute path.
    jobs_dir: str  # Where per-job IPC files live; usually JOBS_DIR.
    private: bool = True
    commit_message: str | None = None
    allow_patterns: list[str] | None = None
    ignore_patterns: tuple[str, ...] | None = None
    # On retry of a failed/cancelled upload, the server passes the existing
    # PR number to resume into instead of creating a new one.
    reuse_pr_num: int | None = None

    def __post_init__(self) -> None:
        if self.direction not in ("upload", "download"):
            raise ValueError(f"bad direction: {self.direction!r}")
        if self.repo_type not in ("dataset", "model"):
            raise ValueError(f"bad repo_type: {self.repo_type!r}")

    def to_json(self) -> str:
        return json.dumps(
            {
                "job_id": self.job_id,
                "dataset_id": self.dataset_id,
                "direction": self.direction,
                "repo_id": self.repo_id,
                "repo_type": self.repo_type,
                "local_path": self.local_path,
                "jobs_dir": self.jobs_dir,
                "private": self.private,
                "commit_message": self.commit_message,
                "allow_patterns": self.allow_patterns,
                "ignore_patterns": list(self.ignore_patterns) if self.ignore_patterns else None,
                "reuse_pr_num": self.reuse_pr_num,
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> JobConfig:
        d = json.loads(raw)
        ig = d.get("ignore_patterns")
        return cls(
            job_id=d["job_id"],
            dataset_id=d["dataset_id"],
            direction=d["direction"],
            repo_id=d["repo_id"],
            repo_type=d["repo_type"],
            local_path=d["local_path"],
            jobs_dir=d["jobs_dir"],
            private=d.get("private", True),
            commit_message=d.get("commit_message"),
            allow_patterns=d.get("allow_patterns"),
            ignore_patterns=tuple(ig) if ig else None,
            reuse_pr_num=d.get("reuse_pr_num"),
        )


# ── JobPaths: per-job file locations ────────────────────────────────────────


@dataclass(frozen=True)
class JobPaths:
    """All file paths owned by one job. Both server and worker compute this.

    All three files live under ``jobs_dir / job_id``-prefixed names so a
    glob like ``<jobs_dir>/*.pid`` enumerates active workers without
    cross-talk with other transfers.
    """

    jobs_dir: Path
    job_id: str

    @property
    def progress(self) -> Path:
        return self.jobs_dir / f"{self.job_id}.json"

    @property
    def log(self) -> Path:
        return self.jobs_dir / f"{self.job_id}.log"

    @property
    def pid(self) -> Path:
        return self.jobs_dir / f"{self.job_id}.pid"

    @classmethod
    def for_job(cls, job_id: str, jobs_dir: str | os.PathLike[str]) -> JobPaths:
        return cls(jobs_dir=Path(jobs_dir), job_id=job_id)


# ── HubJobState: server-side in-memory mirror ───────────────────────────────


@dataclass
class HubJobState:
    """Server-side mirror of a worker's transfer. Polled by the GUI tray.

    The server reads each running job's progress JSON file and merges its
    contents into this dataclass on every ``/hub/jobs`` request. Fields
    here are a superset of the worker's snapshot — they also carry server-
    only state (the worker's PID, the user_mode_choice, etc.) that the
    worker doesn't know or need.
    """

    job_id: str
    dataset_id: str
    direction: HubDirection
    repo_id: str
    repo_type: HubRepoType
    status: HubStatus
    started_at: float
    finished_at: float | None = None
    # Live progress mirrored from the worker.
    stage: str = "starting"
    milestone: str = "starting"
    milestone_at: float = 0.0
    files_total: int = 0
    files_done_estimate: int = 0
    bytes_total: int = 0
    bytes_done_estimate: int = 0
    current_file: str | None = None
    error: str | None = None
    error_class: HubErrorClass | None = None
    # Upload-only: HF PR number + URL.
    pr_num: int | None = None
    pr_url: str | None = None
    # Server-side worker tracking. None until the worker has spawned.
    pid: int | None = None
    process_start_time: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "dataset_id": self.dataset_id,
            "direction": self.direction,
            "repo_id": self.repo_id,
            "repo_type": self.repo_type,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "stage": self.stage,
            "milestone": self.milestone,
            "milestone_at": self.milestone_at,
            "files_total": self.files_total,
            "files_done_estimate": self.files_done_estimate,
            "bytes_total": self.bytes_total,
            "bytes_done_estimate": self.bytes_done_estimate,
            "current_file": self.current_file,
            "error": self.error,
            "error_class": self.error_class,
            "pr_num": self.pr_num,
            "pr_url": self.pr_url,
        }

    def merge_progress(self, snapshot: dict[str, Any]) -> None:
        """Pull worker-owned fields from a progress JSON snapshot.

        The server's status field is authoritative until the worker writes
        a terminal value (``complete`` / ``failed`` / ``cancelled``); we
        accept whatever the worker said in those cases since the worker
        knows the truth. For non-terminal states we trust the worker for
        progress numbers but never let it un-terminalize.
        """
        if self.status in ("complete", "failed", "cancelled"):
            # Once terminal, the snapshot can't drag us back.
            return
        for key in (
            "status",
            "stage",
            "milestone",
            "milestone_at",
            "finished_at",
            "files_total",
            "files_done_estimate",
            "bytes_total",
            "bytes_done_estimate",
            "current_file",
            "error",
            "error_class",
            "pr_num",
            "pr_url",
        ):
            if key in snapshot and snapshot[key] is not None:
                setattr(self, key, snapshot[key])


def make_job(
    *,
    dataset_id: str,
    direction: HubDirection,
    repo_id: str,
    repo_type: HubRepoType = "dataset",
) -> HubJobState:
    """Build a fresh server-side ``HubJobState`` in ``pending``."""
    now = time.time()
    return HubJobState(
        job_id=uuid.uuid4().hex,
        dataset_id=dataset_id,
        direction=direction,
        repo_id=repo_id,
        repo_type=repo_type,
        status="pending",
        started_at=now,
        milestone=f"Starting {direction}",
        milestone_at=now,
    )


# ── Process identity (PID + start_time tuple) ──────────────────────────────
#
# Key safety property: PIDs are recycled by the OS. Verifying both the PID
# AND the process start time before sending signals catches the "the PID
# we recorded is now a different process" case.


def _process_start_time(pid: int) -> float | None:
    """Read the process start time for ``pid``, or None if the process is gone.

    Linux: ``/proc/<pid>/stat`` field 22 (starttime in jiffies since boot).
    Other platforms: best-effort via ``psutil`` if available; otherwise
    fall back to ``None`` (the (pid, start_time) check degrades to a
    plain pid-exists check on those platforms).
    """
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            # The process name in field 2 can contain spaces, so we can't
            # just split on whitespace. The name is enclosed in parens —
            # everything after the closing paren is space-separated.
            data = f.read()
        rparen = data.rfind(b")")
        if rparen < 0:
            return None
        fields = data[rparen + 2 :].split()
        # Field index 22 in the man page is index 19 in the post-name list
        # (fields 1-2 = pid + comm; comm is what we stripped).
        return float(fields[19])
    except FileNotFoundError:
        return None
    except (OSError, IndexError, ValueError):
        # Fall back to None — the caller will treat absence as "can't
        # verify; assume alive if process exists at all."
        return None


def pid_file_payload(pid: int) -> dict[str, Any]:
    """Construct the dict written into ``<job_id>.pid``.

    Pre: ``pid`` is the worker's own PID at the moment of writing.
    Post: the returned dict contains the worker's PID, its start_time
    (Linux-specific; ``None`` on other platforms), and a wall-clock
    ``started_at`` for human-readable debugging.
    """
    return {
        "pid": pid,
        "start_time": _process_start_time(pid),
        "started_at": time.time(),
    }


def read_pid_file(path: Path) -> dict[str, Any] | None:
    """Parse a ``<job_id>.pid`` file. Returns ``None`` if missing or invalid."""
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def atomic_write_json(path: Path, data: Any) -> None:
    """Write ``data`` as JSON to ``path`` atomically.

    ``.tmp`` + ``os.replace``: a concurrent reader sees either the previous
    coherent contents or the new ones, never a partial write. Used for
    the progress file (rewritten ~2 Hz by the worker, polled at 1 Hz by
    the server) and the PID file (written once at worker startup).
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


def is_worker_alive(pid_payload: dict[str, Any]) -> bool:
    """True iff the PID in ``pid_payload`` is still the worker we spawned.

    Verification path:
      1. Send signal 0 to the PID — raises ProcessLookupError if dead.
      2. If the file recorded a ``start_time`` (Linux), compare against the
         current start_time of that PID. Mismatch ⇒ PID was recycled ⇒
         the worker is dead even though *some* process owns the PID.

    On platforms without ``/proc``, step 2 degrades to "no verification";
    that's correct given we have no cheap alternative there.
    """
    pid = pid_payload.get("pid")
    if not isinstance(pid, int):
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    expected_start = pid_payload.get("start_time")
    if expected_start is None:
        # File predates recording, or we're on a non-Linux platform.
        # Best we can do is "process exists", which is what os.kill told us.
        return True
    actual_start = _process_start_time(pid)
    if actual_start is None:
        # PID exists but we can't read its start_time — odd, but treat as
        # alive to avoid spuriously marking jobs failed.
        return True
    # /proc clock resolution is jiffies (~10ms); equality is the right
    # check, not "close to."
    return abs(actual_start - expected_start) < 1.0


# ── Error classification ────────────────────────────────────────────────────


def classify_error(exc: BaseException) -> HubErrorClass:
    """Map an HF exception to a tray-surfaced error class.

    The class drives a specific remediation hint in the GUI tray rather
    than dumping the exception string. Unknown exceptions fall into
    ``"other"`` and the user sees the raw message.
    """
    msg = str(exc).lower()
    typename = type(exc).__name__

    # huggingface_hub specifics. Import lazily so we don't depend on it
    # in non-Hub code paths.
    try:
        from huggingface_hub.errors import (  # type: ignore[import-not-found]
            HfHubHTTPError,
            RepositoryNotFoundError,
        )

        if isinstance(exc, RepositoryNotFoundError):
            return "auth"
        if isinstance(exc, HfHubHTTPError):
            # Status is on the response if attached.
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (401, 403):
                return "auth"
            if status == 429:
                return "rate_limit"
            if status is not None and 500 <= status < 600:
                return "network"
    except ImportError:  # pragma: no cover
        pass

    # Heuristic fallbacks if the type-based check missed.
    if "401" in msg or "403" in msg or "unauthorized" in msg or "forbidden" in msg:
        return "auth"
    if "429" in msg or "rate limit" in msg or "too many requests" in msg:
        return "rate_limit"
    if typename in ("ConnectionError", "ReadTimeoutError", "TimeoutError"):
        return "network"
    if "connection" in msg or "timeout" in msg or "timed out" in msg:
        return "network"
    return "other"


# ── Upload-side file enumeration ────────────────────────────────────────────


def _is_ignored(rel_path: str, ignores: tuple[str, ...]) -> bool:
    """Match ``rel_path`` (posix-style) against ignore prefixes/names."""
    base = rel_path.rsplit("/", 1)[-1]
    for pat in ignores:
        if pat.endswith("/"):
            if rel_path.startswith(pat) or f"/{pat}" in f"/{rel_path}":
                return True
        elif base == pat or rel_path == pat:
            return True
    return False


def enumerate_upload_files(
    root: Path,
    *,
    ignore_patterns: tuple[str, ...] = DEFAULT_UPLOAD_IGNORES,
) -> list[Path]:
    """List every regular file under ``root`` not matching an ignore pattern.

    Used by the upload-time completeness check and by tests. The worker
    itself doesn't enumerate — it hands the directory to
    ``upload_large_folder`` which does its own enumeration. Keeping this
    function ensures both sides use the same ignore semantics.

    Pre: ``root`` exists and is a directory.
    Post: returned paths are absolute and sorted.
    """
    assert root.exists() and root.is_dir(), f"upload root missing or not a dir: {root}"
    out: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if _is_ignored(rel, ignore_patterns):
            continue
        out.append(path)
    return out


# ── Completeness check: defends against download-fail-upload corruption ────


def check_upload_completeness(
    local_root: Path,
    repo_id: str,
    repo_type: HubRepoType = "dataset",
    *,
    api: Any = None,
) -> dict[str, list[str]]:
    """Compare local files against remote siblings; flag what's missing.

    Returns a dict with two keys:
      * ``missing_locally`` — files present on remote ``main`` but absent
        from ``local_root`` (or only present as ``.incomplete``). If
        non-empty, the local copy is partial — uploading from it would
        push a worse-than-remote state.
      * ``incomplete_locally`` — files with an HF ``.incomplete`` sibling
        marker still present, meaning the prior download didn't finish.

    Both lists empty ⇒ the local root is at least as complete as remote.
    The caller decides what to do with non-empty results (warn the user,
    refuse the upload, etc.).

    A repo that doesn't exist yet on HF (first push) is treated as
    "nothing to compare against" — empty lists. Worker-time HF auth
    errors are surfaced as ``RuntimeError`` so the upload pipeline can
    classify them via :func:`classify_error`.
    """
    from huggingface_hub.errors import RepositoryNotFoundError  # type: ignore[import-not-found]

    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    try:
        info = api.repo_info(repo_id, repo_type=repo_type, files_metadata=True)
    except RepositoryNotFoundError:
        # Fresh upload — nothing remote to compare against.
        return {"missing_locally": [], "incomplete_locally": []}

    missing: list[str] = []
    incomplete: list[str] = []
    for sib in info.siblings or []:
        rel = sib.rfilename
        local = local_root / rel
        incomp = local.with_name(local.name + ".incomplete")
        if incomp.exists():
            incomplete.append(rel)
        if not local.exists():
            missing.append(rel)
    return {"missing_locally": missing, "incomplete_locally": incomplete}
