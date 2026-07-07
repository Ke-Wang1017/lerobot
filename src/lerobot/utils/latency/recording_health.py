#!/usr/bin/env python

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

"""Post-recording quality summaries.

Two surfaces share the same primitives:

  - **End-of-episode warning**: ``lerobot_record`` calls ``summarize`` and
    ``verdict`` after each ``save_episode`` so an inattentive operator
    gets a loud ``log_say`` if the episode had frame-rate violations —
    they can press the rerecord-episode key while the warning is still
    fresh, rather than discovering the problem post-hoc.

  - **Persistent per-episode metadata**: the same summary is appended to
    ``<dataset.root>/meta/episodes_health.jsonl`` so the data panel /
    visualizer can render quality badges next to each episode without
    re-computing anything.

These functions are pure — they take a list of records (which the caller
filters by episode if they want per-episode stats) and return a dict.
Pure makes them trivial to test and to call from anywhere a list of
records is available.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _percentile(values: list[float], p: float) -> float:
    """``np.percentile`` with a NaN result for empty input (instead of raising)."""
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def summarize(records: list[dict[str, Any]], target_period_ms: float | None) -> dict[str, Any]:
    """Compute a quality summary for a list of latency records.

    Pass all the records produced by a single episode for the per-episode
    summary, or all records from a whole recording session for the
    end-of-recording summary. The schema is identical.

    Postconditions:
      - Returns a dict with: ``n_records``, ``duration_s`` (wall-clock
        span of the records), ``target_period_ms``, ``overrun_ratio``,
        ``loop_dt_ms`` (p50/p95/p99/max), and per-camera blocks.
      - When ``records`` is empty, returns ``n_records=0`` and NaN
        percentiles. Callers are expected to check ``n_records > 0``
        before drawing conclusions.
    """
    n = len(records)
    if n == 0:
        return {
            "n_records": 0,
            "duration_s": 0.0,
            "target_period_ms": target_period_ms,
            "overrun_ratio": 0.0,
            "loop_dt_ms": {
                "p50": float("nan"),
                "p95": float("nan"),
                "p99": float("nan"),
                "max": float("nan"),
            },
            "cameras": {},
        }

    loop_dts = [r["loop_dt_ms"] for r in records if "loop_dt_ms" in r]
    if target_period_ms is not None:
        overruns = sum(1 for x in loop_dts if x > target_period_ms)
        overrun_ratio = overruns / len(loop_dts) if loop_dts else 0.0
    else:
        overrun_ratio = 0.0

    times = [r["t"] for r in records if "t" in r]
    duration_s = (max(times) - min(times)) if times else 0.0

    # Collect every camera that appears in ANY record (cameras can come
    # and go across iterations, e.g. if a grab thread hiccups).
    cam_names: set[str] = set()
    for r in records:
        for k in r:
            if k.startswith("cam_") and k.endswith("_stale_ms"):
                cam_names.add(k[len("cam_") : -len("_stale_ms")])

    cameras: dict[str, dict[str, float]] = {}
    for cam in sorted(cam_names):
        stale = [r[f"cam_{cam}_stale_ms"] for r in records if f"cam_{cam}_stale_ms" in r]
        period = [r[f"cam_{cam}_period_ms"] for r in records if f"cam_{cam}_period_ms" in r]
        # The previous guard `if period else nan` only handled empty lists.
        # If every period sample is 0 (rare; observed when the policy
        # blocks the loop for the entire episode budget, e.g. during a
        # cold torch.compile autotune pass), the median is 0 and the
        # naive ``1000.0 / 0`` raises ZeroDivisionError. NaN signals
        # "no useful FPS info" to verdict() / the GUI badge code, which
        # already treat NaN as missing/unknown.
        period_p50 = _percentile(period, 50)
        cameras[cam] = {
            "stale_p50": _percentile(stale, 50),
            "stale_p95": _percentile(stale, 95),
            "period_p50": period_p50,
            "fps_effective": (1000.0 / period_p50) if period_p50 and period_p50 > 0 else float("nan"),
        }

    return {
        "n_records": n,
        "duration_s": duration_s,
        "target_period_ms": target_period_ms,
        "overrun_ratio": overrun_ratio,
        "loop_dt_ms": {
            "p50": _percentile(loop_dts, 50),
            "p95": _percentile(loop_dts, 95),
            "p99": _percentile(loop_dts, 99),
            "max": max(loop_dts) if loop_dts else float("nan"),
        },
        "cameras": cameras,
    }


# Verdict thresholds — kept distinct from ``LoopHealthDetector``'s
# *live* thresholds because the post-hoc check covers the entire episode
# while the live one watches a 120-iter window. Live: "is this loop
# slipping right now?". Post-hoc: "was this episode acceptable?"
#
# An episode with one bad second out of 60 may not have fired the live
# overrun warning (the rolling window cleared) but is still arguably
# unhealthy for training. The post-hoc threshold is therefore stricter:
# 5% overrun (vs 10% live) means even a partial breach gets flagged.
DEFAULT_VERDICT_THRESHOLDS = {
    "overrun_ratio": 0.05,  # > 5% of frames over budget = bad episode
    "loop_p95_factor": 1.5,  # p95 loop_dt > 1.5x budget = bad tail
    "cam_stale_p95_factor": 2.0,  # any cam stale p95 > 2x its period = bad
}


def verdict(
    summary: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Convert a summary dict into a healthy/unhealthy verdict + issue list.

    Returns ``{"healthy": bool, "issues": [{"rule": str, "message": str}, ...]}``.
    ``healthy=True`` when ``issues`` is empty. Issue messages are
    pre-formatted for direct display in the data panel.
    """
    th = {**DEFAULT_VERDICT_THRESHOLDS, **(thresholds or {})}
    issues: list[dict[str, str]] = []

    target_ms = summary.get("target_period_ms")
    if summary.get("n_records", 0) == 0:
        return {"healthy": True, "issues": []}

    # Overrun ratio over the whole episode/session
    if target_ms is not None and summary["overrun_ratio"] > th["overrun_ratio"]:
        issues.append(
            {
                "rule": "overrun_high",
                "message": f"{summary['overrun_ratio'] * 100:.1f}% of frames over the "
                f"{target_ms:.1f} ms budget (threshold {th['overrun_ratio'] * 100:.0f}%).",
            }
        )

    # Tail behaviour
    p95 = summary["loop_dt_ms"]["p95"]
    if target_ms is not None and p95 == p95 and p95 > target_ms * th["loop_p95_factor"]:
        issues.append(
            {
                "rule": "loop_tail_high",
                "message": f"loop_dt p95 = {p95:.1f} ms (> {th['loop_p95_factor']:.1f}x "
                f"budget {target_ms:.1f} ms).",
            }
        )

    # Per-camera staleness
    for cam, stats in (summary.get("cameras") or {}).items():
        period = stats.get("period_p50")
        stale_p95 = stats.get("stale_p95")
        if period != period or stale_p95 != stale_p95:  # NaN guard
            continue
        if period < 16.0:  # implausibly small — first-frame artefacts; skip
            continue
        if stale_p95 > period * th["cam_stale_p95_factor"]:
            issues.append(
                {
                    "rule": "camera_stale",
                    "message": f"{cam}: stale p95 = {stale_p95:.0f} ms vs "
                    f"period {period:.0f} ms (> {th['cam_stale_p95_factor']:.1f}x).",
                }
            )

    return {"healthy": len(issues) == 0, "issues": issues}


def filter_by_episode(records: list[dict[str, Any]], ep_index: int) -> list[dict[str, Any]]:
    """Convenience: pull all records belonging to one episode.

    ``record_loop`` tags every iteration with ``ep`` via ``set_field``;
    this helper just filters that field. Records without an ``ep`` are
    skipped — they belong to setup/reset phases and shouldn't count
    toward any specific episode's verdict.
    """
    return [r for r in records if r.get("ep") == ep_index]
