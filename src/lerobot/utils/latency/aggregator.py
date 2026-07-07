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

"""Bounded cyclic buffer of per-iteration latency records.

Backs every live query (current percentiles, time-series, histograms,
overrun ratio) from a single ``collections.deque(maxlen=N)`` of records.
See ``src/lerobot/gui/docs/latency_monitoring.md`` for the full contract.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

DEFAULT_MAXLEN = 4096


class LatencyAggregator:
    """In-memory store for live latency monitoring.

    Preconditions:
      - Records are dicts with ``"t"`` (wall-clock seconds) as a top-level key.
      - Latency fields end in ``_ms`` and are floats.
      - Percentile/series queries assume ``ingest()`` is called from one thread
        (the loop calling ``LatencyTracer.commit()``); read methods are safe to
        call from a different thread but may see a stale snapshot of the deque.

    Postconditions:
      - Memory is bounded by ``maxlen`` regardless of run length.
      - Queries on an empty aggregator return ``float("nan")`` or empty
        collections; they never raise.
    """

    def __init__(self, maxlen: int = DEFAULT_MAXLEN):
        assert maxlen > 0, f"maxlen must be positive, got {maxlen}"
        self._records: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._maxlen = maxlen
        self.dropped_records: int = 0
        # Sticky picks for representative_iterations(): label -> step. Keeps
        # the chosen record stable across snapshots so the GUI Gantt stops
        # re-picking a different "median-ish" record every second (which made
        # the camera markers visibly jitter).
        self._sticky_picks: dict[str, int] = {}

    @property
    def maxlen(self) -> int:
        return self._maxlen

    def __len__(self) -> int:
        return len(self._records)

    def ingest(self, record: dict[str, Any]) -> None:
        """Append a record. Oldest record is evicted when ``maxlen`` is reached."""
        assert isinstance(record, dict), f"record must be dict, got {type(record).__name__}"
        self._records.append(record)

    def clear(self) -> None:
        self._records.clear()
        self.dropped_records = 0
        self._sticky_picks.clear()

    def stage_keys(self) -> list[str]:
        """All ``*_ms`` keys seen across the buffer, sorted."""
        keys: set[str] = set()
        for r in self._records:
            keys.update(k for k in r if k.endswith("_ms"))
        return sorted(keys)

    def values(self, key: str) -> list[float]:
        """Raw values for one stage across the current buffer."""
        return [r[key] for r in self._records if key in r and r[key] is not None]

    def percentile(self, key: str, p: float | list[float]) -> float | list[float]:
        """p in [0, 100]. NaN when the key is absent."""
        vals = self.values(key)
        if not vals:
            return float("nan") if not isinstance(p, list) else [float("nan")] * len(p)
        result = np.percentile(np.asarray(vals, dtype=np.float64), p)
        if isinstance(p, list):
            return [float(x) for x in result]
        return float(result)

    def overrun_ratio(self) -> float:
        """Fraction of records flagged as overrun. 0.0 when empty."""
        flags = [bool(r.get("overrun")) for r in self._records if "overrun" in r]
        if not flags:
            return 0.0
        return sum(flags) / len(flags)

    def time_series(self, key: str, last_n_seconds: float | None = None) -> list[tuple[float, float]]:
        """Return ``[(t, value), ...]`` for ``key``, optionally trimmed to recent."""
        if last_n_seconds is None:
            return [(r["t"], r[key]) for r in self._records if key in r and r[key] is not None]
        if not self._records:
            return []
        latest_t = self._records[-1]["t"]
        cutoff = latest_t - last_n_seconds
        return [
            (r["t"], r[key]) for r in self._records if key in r and r[key] is not None and r["t"] >= cutoff
        ]

    def snapshot(
        self,
        percentiles: tuple[float, ...] = (50, 95, 99),
        series_window_s: float | None = None,
        series_max_points: int = 60,
    ) -> dict[str, Any]:
        """Compact dict for stderr summary / GUI overlay polling.

        Args:
          percentiles: which percentiles to compute per stage.
          series_window_s: when set, attach a downsampled time-series per stage
            (suitable for sparklines), trimmed to the last ``series_window_s``
            seconds and capped at ``series_max_points`` points.

        Postconditions:
          - Always contains ``n_records``, ``dropped_records``, ``overrun_ratio``,
            ``stages``.
          - ``stages`` maps each ``*_ms`` key to a dict of
            ``{f"p{int(p)}": ms, ..., "count": n}``.
          - When ``series_window_s`` is set, also contains ``series`` —
            ``stage_key -> [[t, value], ...]`` (lists, not tuples, so the dict
            is JSON-serializable).
        """
        snap: dict[str, Any] = {
            "n_records": len(self._records),
            "dropped_records": self.dropped_records,
            "overrun_ratio": self.overrun_ratio(),
            "stages": {},
        }
        keys = self.stage_keys()
        for key in keys:
            vals = self.values(key)
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            stage_dict: dict[str, float | int] = {
                f"p{int(p)}": float(np.percentile(arr, p)) for p in percentiles
            }
            stage_dict["count"] = len(vals)
            snap["stages"][key] = stage_dict

        if series_window_s is not None:
            assert series_max_points > 0
            snap["series"] = {}
            for key in keys:
                ts = self.time_series(key, last_n_seconds=series_window_s)
                if not ts:
                    continue
                stride = max(1, len(ts) // series_max_points)
                downsampled = ts[::stride][-series_max_points:]
                snap["series"][key] = [[t, v] for t, v in downsampled]

        return snap

    # Stickiness band: we re-use the previously-chosen record for a label
    # as long as its loop_dt is within ±STICKY_BAND × target percentile.
    # 0.10 = ±10%, which is the "feels stable to the eye" threshold we
    # tuned against bimanual teleop's natural per-iteration jitter.
    _STICKY_BAND = 0.10

    def representative_iterations(self) -> dict[str, dict[str, Any]]:
        """Return a small set of "interesting" iterations for Gantt rendering.

        Picks one record near each of {p50, p95, p99} of ``loop_dt_ms``,
        the single max record, and the most recent record. The frontend
        lets the operator switch between them in a dropdown.

        Picks for percentile labels are **sticky**: once a record is chosen,
        it's reused on subsequent calls as long as it's still in the deque
        and its loop_dt is within ±10% of the new percentile target. This
        prevents the Gantt from jittering when the percentile target shifts
        slightly and a different record happens to be marginally closer.

        ``max`` is naturally stable (single highest record). ``latest``
        always returns the most recent record (no stickiness).

        Returns a dict ``{label: record_subset}`` where each subset contains
        only the timeline-relevant keys (``spans``, ``cam_events``,
        ``loop_dt_ms``, ``step``, ``t``). Returns an empty dict if the
        aggregator has no records.
        """
        if not self._records:
            return {}

        with_loop = [(r.get("loop_dt_ms", 0.0), r) for r in self._records if "loop_dt_ms" in r]
        if not with_loop:
            return {}
        with_loop.sort(key=lambda x: x[0])
        loops = np.asarray([x[0] for x in with_loop])
        # Quick lookup of records still alive in the deque, by step.
        by_step: dict[int, dict[str, Any]] = {r["step"]: r for _, r in with_loop if "step" in r}

        def sticky_pick(label: str, p: float) -> dict[str, Any]:
            target = float(np.percentile(loops, p))
            prev_step = self._sticky_picks.get(label)
            if prev_step is not None and prev_step in by_step:
                prev_loop = by_step[prev_step].get("loop_dt_ms", 0.0)
                if target == 0.0 or abs(prev_loop - target) <= self._STICKY_BAND * target:
                    # Previous pick is still close enough; reuse for stability.
                    return _gantt_record_subset(by_step[prev_step])
            # First time, or previous pick fell out of the deque / drifted.
            best_idx = min(range(len(with_loop)), key=lambda i: abs(with_loop[i][0] - target))
            chosen = with_loop[best_idx][1]
            if "step" in chosen:
                self._sticky_picks[label] = chosen["step"]
            return _gantt_record_subset(chosen)

        return {
            "median": sticky_pick("median", 50),
            "p95": sticky_pick("p95", 95),
            "p99": sticky_pick("p99", 99),
            "max": _gantt_record_subset(with_loop[-1][1]),
            "latest": _gantt_record_subset(self._records[-1]),
        }

    def aggregate_iteration(self, p: float = 50.0) -> dict[str, Any] | None:
        """Synthesize a Gantt where each stage independently shows the p-th
        percentile of *its own* distribution across all records.

        Why this exists alongside ``representative_iterations``: those
        return real captured iterations (so per-stage values are whatever
        happened in that one iteration — useful for debugging the
        correlation between stages in an outlier). The synthetic
        aggregate answers a different question: "what does a typical
        iteration look like, where each stage is at its typical value?"
        It's stable across snapshots because each percentile is
        computed independently — no jitter from picking different
        records.

        Layout: each span is placed at
        ``[percentile(start_offsets), percentile(start_offsets) + percentile(durations)]``.
        Independent percentiles of start and duration preserve nesting:
        if A contains B in every input record (e.g. ``get_observation``
        contains ``motor_read_left``), then percentile-of-start and
        percentile-of-end remain ordered the same way, so A still
        contains B in the aggregate. Earlier versions cascaded all
        spans end-to-end in dict-insertion order, which silently broke
        nesting whenever child spans were inserted into the record dict
        before their parent (every nested case, since the inner span's
        ``__exit__`` fires first).

        Uninstrumented work appears as the gap between the rightmost
        span's end and ``loop_dt_ms``.

        Camera capture/consume offsets use per-camera percentiles so
        each camera's typical staleness is independently represented.

        Returns ``None`` when the aggregator has no records.
        """
        if not self._records:
            return None

        # Collect per-span start offsets and durations across all
        # records. Preserve the order each name first appeared so the
        # legend/sidebar listing stays stable across snapshots.
        span_starts: dict[str, list[float]] = {}
        span_durations: dict[str, list[float]] = {}
        span_order: list[str] = []
        for r in self._records:
            spans = r.get("spans", {})
            for name, endpoints in spans.items():
                if name not in span_durations:
                    span_durations[name] = []
                    span_starts[name] = []
                    span_order.append(name)
                start_ms, end_ms = endpoints
                span_starts[name].append(start_ms)
                span_durations[name].append(end_ms - start_ms)

        synth_spans: dict[str, list[float]] = {}
        for name in span_order:
            durs = span_durations[name]
            starts = span_starts[name]
            if not durs:
                continue
            start_p = float(np.percentile(np.asarray(starts), p))
            dur_p = float(np.percentile(np.asarray(durs), p))
            synth_spans[name] = [start_p, start_p + dur_p]

        # Camera events: per-cam captured/consumed offset percentiles.
        cam_captured: dict[str, list[float]] = {}
        cam_consumed: dict[str, list[float]] = {}
        for r in self._records:
            for cam_key, ev in r.get("cam_events", {}).items():
                cam_captured.setdefault(cam_key, []).append(ev["captured_at_ms"])
                cam_consumed.setdefault(cam_key, []).append(ev["consumed_at_ms"])
        synth_cam_events: dict[str, dict[str, float]] = {}
        for cam_key, captured in cam_captured.items():
            if not captured:
                continue
            synth_cam_events[cam_key] = {
                "captured_at_ms": float(np.percentile(np.asarray(captured), p)),
                "consumed_at_ms": float(np.percentile(np.asarray(cam_consumed[cam_key]), p)),
            }

        # Loop_dt comes from its own percentile, NOT the sum of stage
        # percentiles — the gap between the rightmost stage end and
        # this loop_dt is the visible "uninstrumented work" indicator.
        loop_durations = [r["loop_dt_ms"] for r in self._records if "loop_dt_ms" in r]
        loop_dt = float(np.percentile(np.asarray(loop_durations), p)) if loop_durations else 0.0

        return {
            "step": None,  # synthetic — no real step
            "t": None,
            "loop_dt_ms": loop_dt,
            "spans": synth_spans,
            "cam_events": synth_cam_events,
            "synthetic": True,
            "n_aggregated": len(loop_durations),
            "percentile": p,
        }


def _gantt_record_subset(record: dict[str, Any]) -> dict[str, Any]:
    """Pull only the keys needed to render one iteration's Gantt timeline."""
    return {
        "step": record.get("step"),
        "t": record.get("t"),
        "loop_dt_ms": record.get("loop_dt_ms"),
        "spans": record.get("spans", {}),
        "cam_events": record.get("cam_events", {}),
    }
