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
"""Shared read-only helpers for the run-tracking surface.

Unlike ``_edits_core.py`` (which centers on ``AppState.pending_edits``)
and ``_hub_core.py`` (which centers on ``AppState.hub_jobs``), the
run-state lives in module globals on ``lerobot.gui.api.run``
(``_active_process``, ``_active_command``, ``_active_config``,
``_output_lines``). The subprocess lifecycle is tightly coupled to the
asyncio event loop, which makes storing the state on an immutable
dataclass awkward; the helpers below read those globals through their
public proxies on ``run`` instead.

Read-only by design: no helper here mutates state, sends signals, or
spawns subprocesses. Stopping / starting runs is a separate operate-tier
surface that lands behind explicit operator sign-off.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_EMPTY_LATENCY_SNAPSHOT: dict[str, Any] = {
    "n_records": 0,
    "dropped_records": 0,
    "overrun_ratio": 0.0,
    "stages": {},
    "series": {},
}

_EMPTY_RLT_METRICS: dict[str, Any] = {
    "episode": 0,
    "step_count": 0,
    "buffer_size": 0,
    "total_updates": 0,
    "mode": "IDLE",
    "success_rate": 0,
    "total_successes": 0,
    "total_episodes": 0,
    "series": {},
}


def get_run_status() -> dict[str, Any]:
    """Current state of the GUI-managed subprocess.

    Three-way response — the AI branches on ``running``:

    - ``{"running": False, "command": None}`` — no subprocess at all.
    - ``{"running": False, "command": "<kind>", "returncode": int}`` —
      a subprocess existed but has since exited; ``returncode`` is
      its exit status (0 = clean, non-zero = error or signal).
    - ``{"running": True, "command": "<kind>", "pid": int}`` —
      subprocess is currently active.

    ``command`` is one of the launcher kinds (``'teleoperate'``,
    ``'record'``, ``'replay'``, ``'hvla'``) — the same value the GUI's
    "Run" tab shows.
    """
    from lerobot.gui.api import run as run_mod

    proc = run_mod._active_process
    if proc is None:
        return {"running": False, "command": None}
    returncode = proc.returncode
    if returncode is not None:
        return {
            "running": False,
            "command": run_mod._active_command,
            "returncode": returncode,
        }
    return {
        "running": True,
        "command": run_mod._active_command,
        "pid": proc.pid,
    }


def get_run_output_tail(last_n: int = 200) -> dict[str, Any]:
    """Snapshot of the tail of the active subprocess's captured output.

    The FastAPI ``/api/run/output`` route streams via Server-Sent Events
    for the GUI's live log panel; this helper exists for callers that
    want a discrete snapshot (the MCP tool, ad-hoc scripts) without
    holding a streaming connection.

    Args:
        last_n: Max number of lines to include. The buffer is capped at
            2000 lines on the GUI side; asking for more silently caps
            at the buffer size.

    Returns ``{"lines": [...], "total_buffered": N, "truncated": bool}``.
    ``truncated=True`` when the buffer held more lines than were returned.
    """
    from lerobot.gui.api import run as run_mod

    if last_n < 0:
        raise ValueError(f"last_n must be >= 0; got {last_n}")
    buf = list(run_mod._output_lines)  # snapshot in case of concurrent append
    total = len(buf)
    if last_n == 0:
        return {"lines": [], "total_buffered": total, "truncated": total > 0}
    take = min(last_n, total)
    return {
        "lines": buf[-take:],
        "total_buffered": total,
        "truncated": total > take,
    }


def get_latency_metrics(source: str = "teleop") -> dict[str, Any]:
    """Latest latency snapshot for the requested loop source.

    Each subprocess (teleop / record / HVLA inference) atomically
    replaces ``<source_dir>/latency_snapshot.json`` once per second; this
    function reads the matching file. Returns the empty-snapshot stub
    when no snapshot exists yet (fresh session, source not running,
    unknown source) so callers can branch on ``n_records`` without
    error-handling.

    Args:
        source: One of the keys in ``LATENCY_SOURCES`` on the GUI side
            — currently ``'teleop'``, ``'record'``, or ``'hvla'``.
            Unknown sources return the empty stub.

    Returns ``{"n_records": N, "dropped_records": N, "overrun_ratio":
    float, "stages": {...}, "series": {...}}``. ``stages`` are per-stage
    latency stats (mean/p95/max); ``series`` is the windowed time-series
    the dashboard plots.
    """
    from lerobot.gui.api.run import LATENCY_SOURCES

    source_dir = LATENCY_SOURCES.get(source)
    if source_dir is None:
        return dict(_EMPTY_LATENCY_SNAPSHOT)
    snapshot_path = Path(source_dir) / "latency_snapshot.json"
    try:
        if snapshot_path.exists():
            with open(snapshot_path, encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:  # noqa: BLE001
        # Atomic-replace race: a concurrent rename can leave the path
        # momentarily missing or partially-readable. Harmless — the next
        # poll succeeds.
        logger.debug("latency snapshot read failed (source=%s): %s", source, e)
    return dict(_EMPTY_LATENCY_SNAPSHOT)


def get_rlt_metrics() -> dict[str, Any]:
    """RLT training metrics for the currently-active RLT run.

    Reads the ``metrics.json`` that belongs to the active RLT session
    (resolved from ``_active_config["rlt_output_dir"]``). Returns the
    empty-metrics stub when no RLT run is active, so callers can branch
    on ``mode == "IDLE"`` instead of error-handling.

    Returns ``{"episode", "step_count", "buffer_size", "total_updates",
    "mode", "success_rate", "total_successes", "total_episodes",
    "series"}`` — ``mode == "IDLE"`` means no live RLT session.
    """
    from lerobot.gui.api import run as run_mod

    try:
        from lerobot.policies.hvla.rlt.metrics import load_metrics_from_file

        path = None
        if run_mod._active_config and run_mod._active_config.get("rlt_output_dir"):
            path = str(Path(run_mod._active_config["rlt_output_dir"]) / "metrics.json")
        data = load_metrics_from_file(path)
        if data:
            return data
    except Exception as e:  # noqa: BLE001
        logger.warning("RLT metrics read failed: %s", e)
    return dict(_EMPTY_RLT_METRICS)
