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

"""Per-iteration latency tracer for real-time loops.

Each iteration records (start_ms, end_ms) offsets per span (relative to
``iter_start``), plus per-camera (captured_at_ms, consumed_at_ms) pairs
where ``captured_at_ms`` may be negative (cameras grab in a background
thread before the iteration begins). Together these reconstruct a Gantt
timeline of one iteration — see ``src/lerobot/gui/docs/latency_monitoring.md``.

Why timestamp pairs instead of just durations:
- Visual gap detection in the Gantt replaces the previous numeric
  ``residual_ms`` cross-check.
- Overlapping work (camera capture vs. main-loop work) is representable.
- The same primitive serves teleop and policy without loop-kind branching.

Per-stage duration scalars (``motor_read_ms``, etc.) are still emitted in
the record — derived from the span's end-start — so the dashboard's
sparklines and percentile queries are unchanged.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from lerobot.utils.latency.aggregator import LatencyAggregator


class LatencyTracer:
    """Build one latency record per loop iteration.

    Preconditions:
      - One tracer instance per loop. Not thread-safe across iterations.
      - ``start()`` is called at the top of each iteration before any
        ``span()`` / ``cam_consume()`` calls.
      - ``commit()`` is called exactly once at the end of each iteration.

    Postconditions:
      - Each ``commit()`` produces a fresh dict with:
        - ``spans``: { name: [start_ms, end_ms] } offsets from iter_start.
        - ``cam_events``: { cam_key: { captured_at_ms, consumed_at_ms } }.
        - ``<name>_ms``: duration scalar per span (derived).
        - ``cam_<key>_stale_ms``, ``cam_<key>_period_ms``: derived camera
          observability scalars.
        - ``loop_dt_ms``: total iteration time (commit minus iter_start).
        - ``overrun``: bool when ``target_fps`` is set.
      - When an aggregator was provided, the record is appended to its deque.
    """

    def __init__(
        self,
        aggregator: LatencyAggregator | None = None,
        loop_kind: str = "teleop",
        target_fps: float | None = None,
    ):
        self.aggregator = aggregator
        self.loop_kind = loop_kind
        self._target_period_ms: float | None = (1000.0 / float(target_fps)) if target_fps else None
        self._record: dict[str, Any] = {}
        self._spans: dict[str, list[float]] = {}
        self._cam_events: dict[str, dict[str, float]] = {}
        self._iter_start_perf: float = 0.0
        # Per-camera last latest_ts seen, for period jitter computation.
        self._cam_last_ts: dict[str, float] = {}
        self._step: int = 0
        self._started: bool = False

    @property
    def step(self) -> int:
        return self._step

    def start(self, ep: int | None = None) -> None:
        """Begin a new iteration. Resets the per-iter record."""
        self._iter_start_perf = time.perf_counter()
        self._spans = {}
        self._cam_events = {}
        self._record = {
            "loop_kind": self.loop_kind,
            "t": time.time(),
            # Wall-clock perf_counter at iteration start. Process-wide clock,
            # so dashboards comparing tracks across threads of the same
            # process can render them on a shared time axis without joining
            # records.
            "iter_start_perf": self._iter_start_perf,
            "step": self._step,
        }
        if ep is not None:
            self._record["ep"] = ep
        self._started = True

    def _offset_ms(self, perf_ts: float) -> float:
        """Convert an absolute perf_counter timestamp to ms-from-iter_start.

        Negative values are valid (e.g. a camera frame captured before the
        iteration started).
        """
        return (perf_ts - self._iter_start_perf) * 1000.0

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        """Time a code block; record both endpoints as ms offsets from iter_start.

        Reentrant within an iteration; nested spans are supported. Each
        ``name`` should be unique within an iteration (the second write
        overwrites the first).
        """
        assert self._started, "LatencyTracer.start() must be called before span()"
        t0 = time.perf_counter()
        try:
            yield
        finally:
            t1 = time.perf_counter()
            start_off = self._offset_ms(t0)
            end_off = self._offset_ms(t1)
            self._spans[name] = [start_off, end_off]
            # Duration scalar for sparklines / percentile queries.
            self._record[f"{name}_ms"] = end_off - start_off

    def add_span(self, name: str, start_perf: float, end_perf: float | None = None) -> None:
        """Record a span from already-captured ``perf_counter`` values.

        Use when the natural code structure (if/elif branches, early
        ``continue``, etc.) makes a ``with span(name):`` block awkward.
        Capture the start before the work and pass it here afterwards.

        ``end_perf=None`` uses ``time.perf_counter()`` at call time.
        """
        assert self._started, "LatencyTracer.start() must be called before add_span()"
        if end_perf is None:
            end_perf = time.perf_counter()
        start_off = self._offset_ms(start_perf)
        end_off = self._offset_ms(end_perf)
        self._spans[name] = [start_off, end_off]
        self._record[f"{name}_ms"] = end_off - start_off

    def cam_consume(self, cam_key: str, latest_ts: float) -> None:
        """Record a camera frame consumption event.

        ``latest_ts`` is the camera's ``latest_timestamp`` (perf_counter
        domain) — i.e. when the grab thread cached this frame. The
        ``captured_at_ms`` offset will typically be negative because the
        capture happened before the iteration began.
        """
        assert self._started, "LatencyTracer.start() must be called before cam_consume()"
        now = time.perf_counter()
        captured_at_ms = self._offset_ms(latest_ts)
        consumed_at_ms = self._offset_ms(now)
        self._cam_events[cam_key] = {
            "captured_at_ms": captured_at_ms,
            "consumed_at_ms": consumed_at_ms,
        }
        # Derived scalar: staleness at the moment of consumption.
        self._record[f"cam_{cam_key}_stale_ms"] = consumed_at_ms - captured_at_ms
        # Period: time between successive captures (camera-grab cadence).
        prev_ts = self._cam_last_ts.get(cam_key)
        if prev_ts is not None:
            self._record[f"cam_{cam_key}_period_ms"] = (latest_ts - prev_ts) * 1000.0
        self._cam_last_ts[cam_key] = latest_ts

    def set_field(self, key: str, value: Any) -> None:
        """Add a non-timing field to the record (e.g. dataset write counts)."""
        assert self._started, "LatencyTracer.start() must be called before set_field()"
        self._record[key] = value

    def commit(self) -> dict[str, Any]:
        """Finalize the record. Appends ``spans``/``cam_events``/``loop_dt_ms``/``overrun``.

        Postconditions:
          - Returns the finalized record dict.
          - When an aggregator was provided, also appends to its deque.
          - The tracer is reset; the next iteration must call ``start()``.
        """
        assert self._started, "LatencyTracer.commit() requires a prior start()"
        end = time.perf_counter()
        loop_dt_ms = (end - self._iter_start_perf) * 1000.0
        self._record["loop_dt_ms"] = loop_dt_ms
        self._record["spans"] = self._spans
        self._record["cam_events"] = self._cam_events

        if self._target_period_ms is not None:
            self._record["overrun"] = loop_dt_ms > self._target_period_ms

        if self.aggregator is not None:
            self.aggregator.ingest(self._record)

        record = self._record
        self._step += 1
        self._started = False
        return record
