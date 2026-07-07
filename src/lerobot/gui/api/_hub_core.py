# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Shared business logic for Hugging Face Hub read-only operations.

Mirrors the pattern of ``_edits_core.py``: sync pure-Python helpers that
both the FastAPI ``/api/hub/*`` routes and the new MCP ``hub_*`` tools
call. No HTTP self-call, no cross-surface duplication.

Only one typed exception is needed:

- ``HubJobNotFoundError`` → FastAPI 404 / MCP error (caller passed an
  unknown ``job_id``).

Auth and repo-info probes deliberately catch all exceptions and return
a transparent ``{"logged_in": False, ...}`` / ``{"exists": False, ...}``
shape so the agent can branch on the result without parsing error text.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lerobot.gui.state import AppState

logger = logging.getLogger(__name__)


# ── Typed exceptions ──────────────────────────────────────────────────────


class HubJobNotFoundError(KeyError):
    """Hub job id not present in ``AppState.hub_jobs``."""


# ── Public helpers — called by both FastAPI handlers and MCP tools ────────


def get_auth_status() -> dict[str, Any]:
    """Probe HF Hub auth via ``whoami()``.

    Returns ``{"logged_in": bool, "username": str | None}``. Cheap
    single GET; not cached because auth state changes rarely and
    callers want freshness. Any exception path collapses to
    ``logged_in=False`` — the agent should treat absence of a valid
    token, expired tokens, and network failures uniformly (re-run
    ``huggingface-cli login`` or set ``HF_TOKEN``).
    """
    try:
        from huggingface_hub import HfApi

        info = HfApi().whoami()
        return {
            "logged_in": True,
            "username": info.get("name", info.get("fullname", "unknown")),
        }
    except Exception:  # noqa: BLE001 — probe collapses all failures
        return {"logged_in": False, "username": None}


def get_repo_info(repo_id: str) -> dict[str, Any]:
    """Look up a dataset repo on the Hub.

    Returns ``{"exists": bool, ...}``. When ``exists=False`` (repo
    missing, private with no access, network down) only ``repo_id``
    is filled — fields like ``total_episodes`` are omitted rather than
    nulled so an agent can branch unambiguously on ``"exists"``.

    Best-effort enriches with remote ``meta/info.json`` (episode and
    frame counts, fps) when available; that fetch is wrapped in its
    own try/except so a missing or unreadable info.json doesn't fail
    the whole call.
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.dataset_info(repo_id, files_metadata=True)
    except Exception as e:  # noqa: BLE001 — repo missing / network / auth
        return {"exists": False, "repo_id": repo_id, "error": f"{type(e).__name__}: {e}"}

    siblings = info.siblings or []
    total_size = sum(s.size for s in siblings if s.size)
    remote_episodes = None
    remote_frames = None
    remote_fps = None
    try:
        import json as _json
        from pathlib import Path

        from huggingface_hub import hf_hub_download

        info_path = hf_hub_download(repo_id, "meta/info.json", repo_type="dataset")
        remote_info = _json.loads(Path(info_path).read_text())
        remote_episodes = remote_info.get("total_episodes")
        remote_frames = remote_info.get("total_frames")
        remote_fps = remote_info.get("fps")
    except Exception:  # noqa: BLE001 — best-effort enrichment
        pass

    return {
        "exists": True,
        "repo_id": info.id,
        "private": info.private,
        "last_modified": str(info.last_modified) if info.last_modified else None,
        "downloads": info.downloads,
        "files": len(siblings),
        "total_size_mb": round(total_size / 1e6, 1),
        "sha": info.sha[:12] if info.sha else None,
        "total_episodes": remote_episodes,
        "total_frames": remote_frames,
        "fps": remote_fps,
    }


def list_hub_jobs(app_state: AppState) -> dict[str, Any]:
    """All Hub transfers known to the server, newest-first.

    Reads ``app_state.hub_jobs``, refreshes pending/running jobs
    from each worker's progress JSON, and opportunistically GCs
    terminal jobs older than 30 minutes. Same code path as the
    existing ``GET /api/hub/jobs`` route the GUI's Transfers tray
    polls.

    Returns ``{"jobs": [...], "total": N, "active": N_active}`` —
    the ``total`` / ``active`` summary is the outcome-transparent
    bit so the agent doesn't have to count the array.
    """
    # Lazy import to avoid a circular gui.api.datasets → _hub_core cycle.
    from lerobot.gui.api.datasets import _refresh_progress_from_file

    app_state.gc_finished_hub_jobs()
    for j in app_state.hub_jobs.values():
        if j.status in ("pending", "running"):
            _refresh_progress_from_file(j)
    jobs = sorted(
        (j.to_dict() for j in app_state.hub_jobs.values()),
        key=lambda d: d["started_at"],
        reverse=True,
    )
    active = sum(1 for d in jobs if d["status"] in ("pending", "running"))
    return {"jobs": jobs, "total": len(jobs), "active": active}


def get_job_progress(app_state: AppState, job_id: str) -> dict[str, Any]:
    """Snapshot of one Hub job's state + latest progress merge.

    Raises ``HubJobNotFoundError`` if ``job_id`` is unknown. For
    active jobs, refreshes from the worker's progress file before
    returning so the snapshot is current at call time.
    """
    from lerobot.gui.api.datasets import _refresh_progress_from_file

    job = app_state.hub_jobs.get(job_id)
    if job is None:
        raise HubJobNotFoundError(f"Hub job not found: {job_id}")
    if job.status in ("pending", "running"):
        _refresh_progress_from_file(job)
    return job.to_dict()
