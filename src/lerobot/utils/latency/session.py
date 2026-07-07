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

"""Per-loop latency monitoring lifecycle.

``LatencySession`` bundles the ``LatencyTracer`` / ``LatencyAggregator`` /
``LatencySnapshotWriter`` trio plus the 1-Hz stderr summary into one
object whose API is shaped around the loop's iteration boundary. Replaces
the ~25 lines of per-loop boilerplate that teleop / record / record-with-
policy / eval / inference would otherwise each have to repeat.

Two key properties:

1. **Disabled-by-default is free**. ``LatencySession.disabled()`` returns
   a session object whose methods are all no-ops. Loop code stays
   identical whether monitoring is enabled or not — no ``if tracer is
   None`` branches at the call site.

2. **Iteration is a context manager**. ``with session.iteration():`` does
   start, then runs the body, then commits + (throttled) snapshot +
   (throttled) stderr summary on exit. If the body raises, the iteration
   is still committed and the next one starts cleanly.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Any

from lerobot.utils.latency.aggregator import LatencyAggregator
from lerobot.utils.latency.snapshot import LatencySnapshotWriter
from lerobot.utils.latency.tracer import LatencyTracer

if TYPE_CHECKING:
    # Type-only import keeps the LoopHealthDetector annotation in __init__
    # honest for type checkers without paying for the import (and avoiding
    # a potential circular if health.py ever takes a session dependency).
    from lerobot.utils.latency.health import LoopHealthDetector

logger = logging.getLogger(__name__)

# Thread-local "current session" registry. ``start_iter()`` registers the
# session as current for the calling thread; ``end_iter()`` clears it. Code
# deep in a call stack (e.g. ``Robot.get_observation``) can then declare
# sub-spans via ``current_span("motor_read")`` without taking the session
# as a kwarg through every layer.
#
# Per-thread (not per-process) on purpose: HVLA's main control thread and
# inference thread each have their own session — the robot is only ever
# touched by the main thread, and sub-spans there should land in the main
# session, not in whatever the inference thread happens to be doing.
_thread_session = threading.local()


def current_span(name: str):
    """Span on the active session for the calling thread, else nullcontext.

    Use at the bottom of the call stack (e.g. inside ``Robot.get_observation``)
    so a robot can publish sub-spans without taking a session through its
    public signature. When monitoring is disabled (no session active on this
    thread), the cost is one nullcontext enter/exit (~50 ns).
    """
    session = getattr(_thread_session, "session", None)
    return session.span(name) if session is not None else nullcontext()


class LatencySession:
    """Owns one loop's tracer + aggregator + snapshot writer + summary loop.

    Preconditions:
      - ``iteration()`` is called exactly once per loop turn.
      - All ``span()`` / ``cam_consume()`` / ``add_span()`` calls occur
        inside an ``iteration()`` block.

    Postconditions:
      - On ``iteration()`` exit, the tracer commits, the snapshot writer
        (if configured) publishes when its interval has elapsed, and the
        stderr summary fires when its interval has elapsed.
      - When ``enabled=False`` (use ``disabled()``), every method is a
        cheap no-op; loop call sites need no conditional logic.
    """

    def __init__(
        self,
        aggregator: LatencyAggregator,
        writer: LatencySnapshotWriter | None,
        loop_kind: str,
        target_fps: float | None,
        summary_interval_s: float = 1.0,
        detector: LoopHealthDetector | None = None,
    ):
        self.aggregator = aggregator
        self.writer = writer
        self.loop_kind = loop_kind
        self.target_fps = target_fps
        self._tracer = LatencyTracer(aggregator, loop_kind=loop_kind, target_fps=target_fps)
        self._summary_interval_s = summary_interval_s
        self._last_summary_at: float = 0.0
        # Health detector. Runs at ``_health_interval_s`` cadence — NOT
        # every iteration — because rule evaluations (np.percentile over
        # the aggregator deque) cost ~50 µs each and add up to a 100× hot-
        # path regression at 30 Hz. At 1 Hz the detector still fires log
        # warnings promptly (rate-limited to once/min/rule anyway) and
        # the snapshot's health.issues stays fresh for the GUI (which
        # also polls at 1 Hz). ``None`` disables all checks.
        self._detector = detector
        self._active_issues: list[Any] = []
        self._health_interval_s = summary_interval_s
        self._last_health_check_at: float = 0.0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        *,
        enabled: bool,
        loop_kind: str,
        target_fps: float | None = None,
        output_dir: str | os.PathLike[str] | None = None,
        summary_interval_s: float = 1.0,
        process: str | None = None,
        track: str | None = None,
    ) -> LatencySession:
        """Build either a real session or a no-op (when ``enabled=False``).

        ``output_dir`` is the directory for ``latency_snapshot.json``.
        Pass ``None`` to skip the snapshot writer entirely (still
        keeps the in-memory aggregator and stderr summary).

        ``process`` / ``track`` are stamped into the snapshot envelope so
        the GUI can group multi-thread loops (e.g. HVLA's main + inference
        threads). Both default to ``loop_kind`` for single-track loops.
        """
        if not enabled:
            return _DisabledSession()
        agg = LatencyAggregator()
        writer = (
            LatencySnapshotWriter(
                output_dir,
                loop_kind=loop_kind,
                target_fps=target_fps,
                process=process,
                track=track,
            )
            if output_dir is not None
            else None
        )
        # Always wire up the default health detector when monitoring is
        # enabled. It's a tiny cost per iteration and gives the operator
        # a free safety net (overrun, slow tail, stale cameras, no data)
        # without requiring any per-loop opt-in.
        from lerobot.utils.latency.health import LoopHealthDetector

        target_period_ms = (1000.0 / float(target_fps)) if target_fps else None
        detector = LoopHealthDetector(loop_kind=loop_kind, target_period_ms=target_period_ms)
        return cls(
            aggregator=agg,
            writer=writer,
            loop_kind=loop_kind,
            target_fps=target_fps,
            summary_interval_s=summary_interval_s,
            detector=detector,
        )

    @classmethod
    def disabled(cls) -> LatencySession:
        """Return a no-op session for callers that want unconditional plumbing."""
        return _DisabledSession()

    # ------------------------------------------------------------------
    # Per-iteration API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return True

    @property
    def tracer(self) -> LatencyTracer | None:
        """Direct tracer access for callers needing ``add_span`` outside ``span()``."""
        return self._tracer

    @contextmanager
    def iteration(self, ep: int | None = None) -> Iterator[LatencySession]:
        """Context manager around one loop iteration.

        On entry: ``tracer.start(ep=ep)``.
        On exit: ``tracer.commit()`` + (throttled) snapshot publish +
        (throttled) stderr summary. Runs even if the body raises, so the
        next iteration starts from a clean state.

        Use this when the loop body has a clean structure that fits in
        a single ``with`` block. For loops with early ``break`` /
        ``continue`` paths that need to skip the commit, use the explicit
        ``start_iter()`` / ``end_iter()`` pair instead.
        """
        self.start_iter(ep=ep)
        try:
            yield self
        finally:
            self.end_iter()

    def start_iter(self, ep: int | None = None) -> None:
        """Start a new iteration explicitly. Pair with ``end_iter()``.

        Used by loops that can't cleanly wrap their body in
        ``with iteration():`` (e.g. record_loop has many ``continue``
        branches that should skip the commit). ``continue``-skipped
        iterations are simply discarded — the next ``start_iter()``
        resets the tracer's per-iteration state.

        Also registers self as the current session for this thread so
        sub-spans declared deep in the call stack (e.g. inside
        ``Robot.get_observation``) attach to this iteration.
        """
        self._tracer.start(ep=ep)
        _thread_session.session = self

    def end_iter(self) -> None:
        """Finalize the current iteration: commit + (throttled) publish/log.

        Clears the thread-local current-session registration so any
        ``current_span()`` calls between iterations are no-ops rather
        than attaching to a stale tracer.
        """
        self._tracer.commit()
        _thread_session.session = None
        self._after_commit()

    def span(self, name: str):
        """Context manager around a timed region. Forwards to the tracer."""
        return self._tracer.span(name)

    def add_span(self, name: str, start_perf: float, end_perf: float | None = None) -> None:
        """Record a span from already-captured perf_counter values."""
        self._tracer.add_span(name, start_perf, end_perf)

    def cam_consume(self, cam_key: str, latest_ts: float) -> None:
        """Record a single camera frame consumption event."""
        self._tracer.cam_consume(cam_key, latest_ts)

    def cam_consume_all(self, cameras: dict[str, Any] | None) -> None:
        """Convenience: read ``latest_timestamp`` from each camera and record it.

        Skips cameras that don't expose ``latest_timestamp`` (e.g. mock
        cameras in tests, or cameras that haven't grabbed any frame yet).
        Safe to call with ``None`` for robots that have no cameras.
        """
        if not cameras:
            return
        for cam_key, cam in cameras.items():
            ts = getattr(cam, "latest_timestamp", None)
            if ts is not None:
                self._tracer.cam_consume(cam_key, ts)

    def set_field(self, key: str, value: Any) -> None:
        """Add a non-timing field to the current iteration's record."""
        self._tracer.set_field(key, value)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _after_commit(self) -> None:
        """Schedule async snapshot publish; do no percentile work on the loop.

        Everything heavy — health detector rules, snapshot computation,
        digest formatting, JSON serialisation, atomic file replace —
        runs on the writer's background thread. The loop thread only
        pays for the maybe_write check + list-snapshot of the records
        deque + thread spawn, well under 1 ms even at 4096 records.

        Earlier designs ran health checks and the digest log on the
        loop thread, recomputing percentiles on a full deque every
        second. That caused a ~20 ms 1 Hz spike — visible as periodic
        jitter on real teleop at 30 Hz. The fold-into-writer move
        eliminated it.
        """
        if self.writer is not None:
            self.writer.maybe_write(self.aggregator, detector=self._detector)
            return
        # No writer (output_dir was None) — fallback: do the digest log
        # inline. Cost still applies on the loop, but a user who
        # explicitly disables the writer has accepted no GUI consumer,
        # and the digest is the only feedback channel left.
        now = time.time()
        if now - self._last_summary_at >= self._summary_interval_s:
            self._last_summary_at = now
            if self._detector is not None:
                last_record = self.aggregator._records[-1] if len(self.aggregator) > 0 else None
                self._active_issues = self._detector.check(self.aggregator, last_record)
            snap = self.aggregator.snapshot(percentiles=(50, 95))
            if snap["n_records"] > 0:
                from lerobot.utils.latency import format_latency_summary

                logger.info("[latency:%s] %s", self.loop_kind, format_latency_summary(snap))


class _DisabledSession(LatencySession):
    """No-op session for ``enabled=False`` paths.

    Every method either returns a no-op context manager or quietly
    discards its arguments. Call sites can use the same code regardless
    of whether monitoring is on; the only cost when disabled is one
    null context-manager enter+exit per iteration (~50 ns).
    """

    def __init__(self) -> None:
        # Skip the parent's __init__ — we don't allocate aggregator etc.
        self.aggregator = None  # type: ignore[assignment]
        self.writer = None
        self.loop_kind = ""
        self.target_fps = None

    @property
    def enabled(self) -> bool:
        return False

    @property
    def tracer(self) -> LatencyTracer | None:
        return None

    @contextmanager
    def iteration(self, ep: int | None = None) -> Iterator[LatencySession]:
        yield self

    def start_iter(self, ep: int | None = None) -> None:
        pass

    def end_iter(self) -> None:
        pass

    def span(self, name: str):
        return nullcontext()

    def add_span(self, name: str, start_perf: float, end_perf: float | None = None) -> None:
        pass

    def cam_consume(self, cam_key: str, latest_ts: float) -> None:
        pass

    def cam_consume_all(self, cameras: dict[str, Any] | None) -> None:
        pass

    def set_field(self, key: str, value: Any) -> None:
        pass

    def _after_commit(self) -> None:
        pass
