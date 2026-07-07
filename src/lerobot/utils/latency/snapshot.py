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

"""1 Hz JSON snapshot writer for the GUI to read cross-process.

Mirrors the RLT pattern (``policies/hvla/rlt/metrics.py`` →
``outputs/rlt_online/metrics.json``): the loop process writes a small
``latency_snapshot.json`` to a known location at ~1 Hz; the GUI's
``/api/latency-metrics`` endpoint reads it and serves to the frontend.

**Threading**: ``maybe_write`` runs on the loop thread (cheap — just
takes a list copy of the aggregator's records and hands it to a
background daemon thread). The heavy work — percentile computation
across every stage, JSON serialisation, atomic file replace — runs
off the loop thread so the loop doesn't pay an ~8 ms spike once per
second. The earlier inline implementation caused visible periodic
jitter on teleop at 30 Hz; see the design doc's "Thread-safety
contract" section.

Tests: call ``flush()`` to block until the pending write completes
before asserting the file exists.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from lerobot.utils.latency.aggregator import LatencyAggregator

if TYPE_CHECKING:
    from lerobot.utils.latency.health import LoopHealthDetector

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_S = 1.0
DEFAULT_SERIES_WINDOW_S = 30.0
DEFAULT_SERIES_MAX_POINTS = 60


class LatencySnapshotWriter:
    """Writes a JSON snapshot of the aggregator to disk on a fixed interval.

    Preconditions:
      - ``maybe_write()`` is called from the loop thread, e.g. after
        ``LatencyTracer.commit()``.
      - The output directory is writable; created on first write.

    Postconditions:
      - Writes to ``<output_dir>/latency_snapshot.json`` atomically (write to
        a temp file, then ``os.replace``) so the GUI never reads a partial
        file. Stale data on read is fine — the GUI polls again.
    """

    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        interval_s: float = DEFAULT_INTERVAL_S,
        series_window_s: float = DEFAULT_SERIES_WINDOW_S,
        series_max_points: int = DEFAULT_SERIES_MAX_POINTS,
        loop_kind: str = "teleop",
        target_fps: float | None = None,
        process: str | None = None,
        track: str | None = None,
    ):
        assert interval_s >= 0, f"interval_s must be non-negative, got {interval_s}"
        self._dir = Path(output_dir)
        self._path = self._dir / "latency_snapshot.json"
        self._tmp_path = self._dir / "latency_snapshot.json.tmp"
        self._interval_s = interval_s
        self._series_window_s = series_window_s
        self._series_max_points = series_max_points
        self._loop_kind = loop_kind
        # ``process`` / ``track`` identify a snapshot in a multi-thread
        # process (e.g. HVLA writes one snapshot per thread: track=main vs
        # track=inference, both with process=hvla). Single-loop callers
        # can omit them — they default to ``loop_kind`` so existing
        # consumers see the same identifier in both fields.
        self._process = process if process is not None else loop_kind
        self._track = track if track is not None else loop_kind
        # The loop's nominal iteration budget in ms (1000/fps). Published in
        # every snapshot so the GUI can scale color thresholds to the actual
        # FPS instead of hardcoding for 60 Hz.
        self._target_period_ms: float | None = (1000.0 / float(target_fps)) if target_fps else None
        self._last_write_at: float = 0.0
        self._dir_ready: bool = False
        # Background writer thread — see module docstring. One in-flight
        # at a time; if a previous tick's writer is still running when
        # the next tick fires, we skip (the next tick will publish a
        # fresher snapshot anyway, no point queueing).
        self._write_thread: threading.Thread | None = None

    @property
    def path(self) -> Path:
        return self._path

    def maybe_write(
        self,
        aggregator: LatencyAggregator,
        now: float | None = None,
        detector: LoopHealthDetector | None = None,
    ) -> bool:
        """Schedule a snapshot write if ``interval_s`` has elapsed.

        Returns True when a write was scheduled, False when skipped. The
        actual snapshot computation + health detector evaluation + JSON
        serialisation + atomic file replace + digest log emission all
        happen on a background daemon thread; this call costs ~µs on
        the loop thread (a list snapshot of the aggregator's records +
        thread spawn). Errors are logged at WARNING from the background
        thread — never crashes the loop.

        ``detector`` is optional; when provided, its rules are evaluated
        on the snapshot copy from the background thread, and the
        resulting issues are embedded in the published JSON.

        If a previous tick's writer is still in flight (rare; only if the
        disk is briefly slow), this call skips rather than queueing.
        The next tick will publish a fresher snapshot.
        """
        t = time.time() if now is None else now
        if (t - self._last_write_at) < self._interval_s:
            return False
        if self._write_thread is not None and self._write_thread.is_alive():
            # Previous tick still running; skip this one. Don't update
            # _last_write_at so the next tick will fire immediately on
            # the next iteration past interval_s.
            return False
        self._last_write_at = t
        # Snapshot the aggregator's records on the loop thread (cheap;
        # list(deque) is thread-safe on CPython). The background thread
        # then operates on this immutable copy — no shared state with
        # the loop's ongoing appends.
        records_copy = list(aggregator._records)
        dropped = aggregator.dropped_records
        self._write_thread = threading.Thread(
            target=self._write_async,
            args=(records_copy, dropped, t, detector),
            daemon=True,
            name=f"latency-writer-{self._loop_kind}",
        )
        self._write_thread.start()
        return True

    def flush(self, timeout: float = 5.0) -> None:
        """Block until the most recently scheduled write completes.

        Tests call this before asserting the snapshot file exists. Loop
        code should never need it — the background-thread design means
        the loop doesn't wait for the writer.
        """
        if self._write_thread is not None:
            self._write_thread.join(timeout=timeout)

    def _write_async(
        self,
        records: list[dict],
        dropped: int,
        t: float,
        detector: LoopHealthDetector | None,
    ) -> None:
        try:
            self._write_records(records, dropped, t, detector)
        except Exception as e:  # noqa: BLE001
            logger.warning("latency snapshot write failed: %s", e)

    def _write_records(
        self,
        records: list[dict],
        dropped: int,
        t: float,
        detector: LoopHealthDetector | None,
    ) -> None:
        # Build a transient aggregator from the snapshot of records so
        # we can reuse its snapshot() / representative_iterations() /
        # aggregate_iteration() methods without touching the live one.
        transient = LatencyAggregator(maxlen=max(len(records), 1))
        for r in records:
            transient.ingest(r)
        transient.dropped_records = dropped
        # Run health detector against the snapshot copy. Off-loop —
        # rules can do as much percentile compute as they need.
        issues: list[dict[str, str]] = []
        if detector is not None:
            last_record = records[-1] if records else None
            try:
                active = detector.check(transient, last_record)
                issues = [i.to_dict() for i in active]
            except Exception as e:  # noqa: BLE001
                logger.warning("latency detector check failed: %s", e)
        snap = transient.snapshot(
            percentiles=(50, 95, 99),
            series_window_s=self._series_window_s,
            series_max_points=self._series_max_points,
        )
        snap["t"] = t
        snap["loop_kind"] = self._loop_kind
        snap["process"] = self._process
        snap["track"] = self._track
        if self._target_period_ms is not None:
            snap["target_period_ms"] = self._target_period_ms
        # Two flavors of timeline data, both cheap to produce:
        #
        # - representative_iterations(): real captured iterations (median /
        #   p95 / p99 / max / latest sample). Per-stage values inside each
        #   reflect THAT specific iteration. Useful for outlier debugging
        #   because correlations are real.
        #
        # - aggregate_iteration(p): synthetic timeline whose per-stage bars
        #   each show the p-th percentile of that stage *independently*.
        #   Stable across snapshots; answers "what does a typical iteration
        #   look like?" without picking one specific record.
        snap["iterations"] = transient.representative_iterations()
        snap["iterations"]["aggregate_median"] = transient.aggregate_iteration(50)
        snap["iterations"]["aggregate_p95"] = transient.aggregate_iteration(95)
        # Health envelope. Always present so the GUI can render the
        # badge logic without optional-field checks; empty issues list
        # signals "healthy".
        snap["health"] = {"issues": issues}
        # Emit the 1 Hz digest log from this thread too — same stats we
        # just computed, so it costs ~µs to format and saves the loop
        # thread from running a separate (~10 ms on a 4k-record deque)
        # snapshot pass. Importing locally to avoid a top-level
        # circular dependency with ``lerobot.utils.latency.__init__``.
        if snap["n_records"] > 0:
            from lerobot.utils.latency import format_latency_summary

            logger.info("[latency:%s] %s", self._loop_kind, format_latency_summary(snap))

        if not self._dir_ready:
            self._dir.mkdir(parents=True, exist_ok=True)
            self._dir_ready = True

        # Atomic publish: write tmp file, then rename.
        with open(self._tmp_path, "w", encoding="utf-8") as f:
            json.dump(snap, f, separators=(",", ":"))
        os.replace(self._tmp_path, self._path)
