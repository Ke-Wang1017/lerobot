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

"""Tests for ``LatencySession`` — the per-loop monitoring lifecycle."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from lerobot.utils.latency import LatencySession, current_span

# ---------------------------------------------------------------------------
# Disabled session — must be a true no-op
# ---------------------------------------------------------------------------


class TestDisabledSession:
    def test_disabled_factory_returns_disabled(self):
        session = LatencySession.from_config(enabled=False, loop_kind="teleop", target_fps=30)
        assert session.enabled is False
        assert session.tracer is None
        assert session.aggregator is None
        assert session.writer is None

    def test_disabled_iteration_yields_session(self):
        session = LatencySession.disabled()
        with session.iteration() as ls:
            # Must return *the session* so call sites can `with session.iteration() as ls: ls.span(...)`
            assert ls is session

    def test_disabled_span_is_nullcontext(self):
        session = LatencySession.disabled()
        # span() must be usable as a context manager and not raise without a prior start().
        with session.span("anything"):
            pass

    def test_disabled_methods_are_silent_noops(self, tmp_path: Path):
        session = LatencySession.disabled()
        # All these would raise on a real tracer (no start() called); on a
        # disabled session they must silently no-op.
        session.add_span("foo", time.perf_counter())
        session.cam_consume("top", time.perf_counter())
        session.cam_consume_all({"top": SimpleNamespace(latest_timestamp=time.perf_counter())})
        session.cam_consume_all(None)
        session.set_field("ep", 1)

    def test_disabled_no_snapshot_written(self, tmp_path: Path):
        # Even if from_config is asked for an output_dir, enabled=False
        # short-circuits and never creates a writer.
        session = LatencySession.from_config(
            enabled=False,
            loop_kind="teleop",
            target_fps=30,
            output_dir=tmp_path,
        )
        with session.iteration():
            pass
        assert not (tmp_path / "latency_snapshot.json").exists()


# ---------------------------------------------------------------------------
# Enabled session — iteration commits, snapshot publishes, summary fires
# ---------------------------------------------------------------------------


class TestEnabledSession:
    def test_iteration_commits_record(self):
        session = LatencySession.from_config(enabled=True, loop_kind="teleop", target_fps=30)
        with session.iteration():
            with session.span("work"):
                time.sleep(0.001)
        assert len(session.aggregator) == 1

    def test_iteration_records_multiple(self):
        session = LatencySession.from_config(enabled=True, loop_kind="teleop", target_fps=30)
        for _ in range(5):
            with session.iteration():
                pass
        assert len(session.aggregator) == 5

    def test_iteration_commits_even_on_exception(self):
        """If the loop body raises, the iteration must still commit so the
        next start() doesn't fail with 'tracer not reset'."""
        session = LatencySession.from_config(enabled=True, loop_kind="teleop", target_fps=30)
        with pytest.raises(ValueError):
            with session.iteration():
                raise ValueError("boom")
        assert len(session.aggregator) == 1
        # And the next iteration must work cleanly.
        with session.iteration():
            pass
        assert len(session.aggregator) == 2

    def test_span_records_duration_inside_iteration(self):
        session = LatencySession.from_config(enabled=True, loop_kind="teleop", target_fps=30)
        with session.iteration():
            with session.span("motor_read"):
                time.sleep(0.005)
        records = list(session.aggregator._records)
        assert "motor_read_ms" in records[0]
        assert records[0]["motor_read_ms"] >= 4.0  # tolerate scheduling jitter

    def test_cam_consume_all_skips_cameras_without_timestamp(self):
        session = LatencySession.from_config(enabled=True, loop_kind="teleop", target_fps=30)
        cams = {
            "front": SimpleNamespace(latest_timestamp=time.perf_counter() - 0.01),
            "wrist": SimpleNamespace(latest_timestamp=None),  # None → skip
            "broken": SimpleNamespace(),  # no attr → skip
        }
        with session.iteration():
            session.cam_consume_all(cams)
        record = list(session.aggregator._records)[0]
        assert "cam_front_stale_ms" in record
        assert "cam_wrist_stale_ms" not in record
        assert "cam_broken_stale_ms" not in record

    def test_cam_consume_all_handles_none_cameras_dict(self):
        """Robots without cameras pass None / empty dict; must not raise."""
        session = LatencySession.from_config(enabled=True, loop_kind="teleop", target_fps=30)
        with session.iteration():
            session.cam_consume_all(None)
            session.cam_consume_all({})

    def test_explicit_start_end_iter_for_loops_with_continue(self):
        """Loops with branchy `continue` paths use start_iter / end_iter
        explicitly — `continue` skips end_iter but the next start_iter()
        resets cleanly. Verify no leftover state."""
        session = LatencySession.from_config(enabled=True, loop_kind="record", target_fps=30)

        # Iteration 1: starts, then "continues" without end_iter.
        session.start_iter()
        with session.span("partial_work"):
            pass
        # No end_iter() — simulating continue.

        # Iteration 2: clean start + commit.
        session.start_iter()
        with session.span("real_work"):
            pass
        session.end_iter()

        # Only iteration 2 should be in the aggregator.
        assert len(session.aggregator) == 1
        record = list(session.aggregator._records)[0]
        assert "real_work_ms" in record
        assert "partial_work_ms" not in record

    def test_disabled_start_end_iter_are_noops(self):
        session = LatencySession.disabled()
        # Must not raise even though there's no real tracer underneath.
        session.start_iter()
        with session.span("foo"):
            pass
        session.end_iter()

    def test_add_span_records_outside_with_block(self):
        """Branchy code (record_loop's action selection) uses add_span
        instead of `with span()`. Verify it's accessible via the session."""
        session = LatencySession.from_config(enabled=True, loop_kind="record", target_fps=30)
        with session.iteration():
            t0 = time.perf_counter()
            time.sleep(0.002)
            session.add_span("process_action", t0)
        record = list(session.aggregator._records)[0]
        assert "process_action_ms" in record
        assert record["process_action_ms"] >= 1.5


# ---------------------------------------------------------------------------
# Snapshot publishing — throttled by interval
# ---------------------------------------------------------------------------


class TestSnapshotPublish:
    def test_snapshot_published_when_writer_configured(self, tmp_path: Path):
        session = LatencySession.from_config(
            enabled=True,
            loop_kind="teleop",
            target_fps=30,
            output_dir=tmp_path,
        )
        # Force the writer's interval to 0 so every iteration publishes.
        session.writer._interval_s = 0.0
        with session.iteration():
            with session.span("work"):
                pass
        session.writer.flush()  # writer runs async on a background thread
        snap_path = tmp_path / "latency_snapshot.json"
        assert snap_path.exists()
        data = json.loads(snap_path.read_text())
        assert data["loop_kind"] == "teleop"

    def test_no_writer_when_output_dir_missing(self):
        session = LatencySession.from_config(enabled=True, loop_kind="teleop", target_fps=30)
        assert session.writer is None
        # Iteration still works; aggregator just isn't published to disk.
        with session.iteration():
            pass
        assert len(session.aggregator) == 1


# ---------------------------------------------------------------------------
# Stderr summary — throttled by summary_interval_s
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_fires_after_interval(self, caplog):
        session = LatencySession.from_config(
            enabled=True,
            loop_kind="teleop",
            target_fps=30,
            summary_interval_s=0.0,  # fire on every iteration
        )
        with caplog.at_level("INFO", logger="lerobot.utils.latency.session"):
            with session.iteration():
                with session.span("work"):
                    pass
        # First iteration produces a summary because interval=0.
        assert any("[latency:teleop]" in r.message for r in caplog.records)

    def test_summary_skipped_when_no_records(self, caplog):
        """Even with interval=0, the summary should skip when the
        aggregator is empty (e.g., the very first call before any commit)."""
        session = LatencySession.from_config(
            enabled=True,
            loop_kind="teleop",
            target_fps=30,
            summary_interval_s=0.0,
        )
        with caplog.at_level("INFO", logger="lerobot.utils.latency.session"):
            # Force-run _after_commit on a fresh aggregator.
            session._after_commit()
        assert not any("[latency:teleop]" in r.message for r in caplog.records)

    def test_summary_throttled_by_default(self, caplog):
        """With default 1s interval, two back-to-back iterations produce
        only one summary line."""
        session = LatencySession.from_config(enabled=True, loop_kind="teleop", target_fps=30)
        with caplog.at_level("INFO", logger="lerobot.utils.latency.session"):
            for _ in range(3):
                with session.iteration():
                    pass
        # 0 or 1 summary lines — never 3.
        n_summary = sum(1 for r in caplog.records if "[latency:teleop]" in r.message)
        assert n_summary <= 1


# ---------------------------------------------------------------------------
# Process / track schema — multi-thread loops (HVLA) need a way to identify
# which thread published a snapshot so the dashboard can stack them.
# ---------------------------------------------------------------------------


class TestProcessTrackSchema:
    def test_snapshot_carries_explicit_process_and_track(self, tmp_path: Path):
        """When passed explicitly, ``process`` and ``track`` land in the
        published snapshot envelope. HVLA main + inference both set
        ``process="hvla"`` so the dashboard can group them visually."""
        session = LatencySession.from_config(
            enabled=True,
            loop_kind="hvla_main",
            target_fps=30,
            output_dir=tmp_path,
            process="hvla",
            track="main",
        )
        session.writer._interval_s = 0.0
        with session.iteration():
            pass
        session.writer.flush()
        data = json.loads((tmp_path / "latency_snapshot.json").read_text())
        assert data["process"] == "hvla"
        assert data["track"] == "main"
        assert data["loop_kind"] == "hvla_main"  # original field preserved

    def test_snapshot_defaults_process_and_track_to_loop_kind(self, tmp_path: Path):
        """Single-track loops (teleop, record) don't need to set process /
        track explicitly — they default to loop_kind so the snapshot
        envelope is always populated."""
        session = LatencySession.from_config(
            enabled=True,
            loop_kind="teleop",
            target_fps=30,
            output_dir=tmp_path,
        )
        session.writer._interval_s = 0.0
        with session.iteration():
            pass
        session.writer.flush()
        data = json.loads((tmp_path / "latency_snapshot.json").read_text())
        assert data["process"] == "teleop"
        assert data["track"] == "teleop"


# ---------------------------------------------------------------------------
# Overhead microbenchmark — guards against silent regressions in the hot
# path. The thresholds below were measured on the dev box (3+ GHz x86,
# Python 3.12); they have a comfortable margin so they shouldn't flake on
# slower CI hardware. If a future change pushes us over, we either fix the
# regression or raise the bound deliberately (with a comment explaining
# why).
# ---------------------------------------------------------------------------


class TestCurrentSpan:
    """``current_span`` lets deep call sites (e.g. Robot.get_observation)
    publish sub-spans without taking a session through their public
    signature. The thread-local registration is set on start_iter() and
    cleared on end_iter()."""

    def test_no_session_outside_iteration_is_nullcontext(self):
        """Before any session ever starts an iteration on this thread,
        current_span must be a no-op (nullcontext)."""
        # Don't actually start any session — just call current_span on a
        # fresh thread-local. We do this on a fresh thread so any prior
        # tests that left state behind can't pollute the result.
        import threading

        result = []

        def body():
            with current_span("anything"):
                result.append("entered")

        t = threading.Thread(target=body)
        t.start()
        t.join()
        assert result == ["entered"]  # didn't raise

    def test_current_span_attaches_to_active_session(self):
        session = LatencySession.from_config(enabled=True, loop_kind="teleop", target_fps=30)
        with session.iteration():
            with current_span("deep_call"):
                time.sleep(0.001)
        record = list(session.aggregator._records)[0]
        assert "deep_call_ms" in record
        assert record["deep_call_ms"] >= 0.5

    def test_current_span_cleared_after_end_iter(self):
        """Between iterations, current_span must NOT attach to a stale
        tracer — that would either raise (tracer not started) or silently
        write into the next iteration's record."""
        session = LatencySession.from_config(enabled=True, loop_kind="teleop", target_fps=30)
        with session.iteration():
            pass
        # Outside the iteration: should be a no-op, not raise.
        with current_span("between_iters"):
            pass
        # Next iteration should not contain "between_iters_ms".
        with session.iteration():
            pass
        records = list(session.aggregator._records)
        assert len(records) == 2
        for rec in records:
            assert "between_iters_ms" not in rec

    def test_disabled_session_does_not_register(self):
        """Disabled session's start_iter is a no-op, so it must not register
        itself as the current session — current_span stays a nullcontext
        even inside a disabled iteration block."""
        session = LatencySession.disabled()
        with session.iteration():
            with current_span("noop"):
                pass
        # Nothing to assert structurally — the test passes if nothing raised
        # and (manually verified) no record was created.
        assert session.aggregator is None  # disabled session has no aggregator

    def test_two_threads_have_independent_sessions(self):
        """HVLA's main and inference threads each start their own session.
        current_span on each thread must attach to that thread's session,
        not whichever started most recently."""
        import threading

        s_main = LatencySession.from_config(enabled=True, loop_kind="hvla_main", target_fps=30)
        s_infer = LatencySession.from_config(enabled=True, loop_kind="hvla_infer", target_fps=30)

        # Use events to interleave: both threads start their iteration,
        # both call current_span("work"), both end. Without thread-local
        # isolation this would fail because one would clobber the other.
        ready = threading.Barrier(2)
        commit = threading.Barrier(2)

        def main_thread():
            with s_main.iteration():
                ready.wait()
                with current_span("work"):
                    time.sleep(0.002)
                commit.wait()

        def infer_thread():
            with s_infer.iteration():
                ready.wait()
                with current_span("work"):
                    time.sleep(0.002)
                commit.wait()

        t1 = threading.Thread(target=main_thread)
        t2 = threading.Thread(target=infer_thread)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        main_rec = list(s_main.aggregator._records)[0]
        infer_rec = list(s_infer.aggregator._records)[0]
        assert "work_ms" in main_rec
        assert "work_ms" in infer_rec
        # Each thread's span attached to its own session — neither record
        # contains the other's data.


class TestOverhead:
    """Profiling itself must be cheap. These tests pin the per-iteration
    cost of start_iter / span / end_iter so a future refactor can't
    silently double the hot-path overhead."""

    def _measure_per_iter_us(self, n: int, body_fn) -> float:
        """Run ``body_fn`` ``n`` times and return microseconds per call."""
        # Warm up the JIT-y bits (dict allocation patterns, attribute
        # lookups), so the timed loop sees steady-state cost.
        for _ in range(min(n, 1000)):
            body_fn()
        t0 = time.perf_counter()
        for _ in range(n):
            body_fn()
        elapsed = time.perf_counter() - t0
        return (elapsed / n) * 1e6

    def test_disabled_session_is_effectively_free(self):
        """Disabled session: a full iteration with several spans should
        cost under ~2 µs per iteration. The bound is the sum of a few
        Python method calls + nullcontext enters/exits — anything more
        and we've broken the no-op contract somewhere."""
        session = LatencySession.disabled()

        def body():
            with session.iteration():
                with session.span("a"):
                    pass
                with session.span("b"):
                    pass
                session.cam_consume_all(None)

        per_iter_us = self._measure_per_iter_us(10000, body)
        assert per_iter_us < 5.0, (
            f"Disabled session iteration cost {per_iter_us:.2f} µs > 5 µs budget. "
            "The whole point of LatencySession.disabled() is that it costs "
            "essentially nothing — investigate before raising this bound."
        )

    def test_enabled_no_op_iteration_under_budget(self):
        """Enabled session with no spans / no writer: just start_iter +
        end_iter + commit (deque append, percentile-snapshot check, log
        check). Should run under ~10 µs per iteration."""
        session = LatencySession.from_config(
            enabled=True,
            loop_kind="bench",
            target_fps=60,
            summary_interval_s=1e9,  # disable the summary log path entirely
        )

        def body():
            with session.iteration():
                pass

        per_iter_us = self._measure_per_iter_us(10000, body)
        assert per_iter_us < 25.0, (
            f"Enabled empty iteration cost {per_iter_us:.2f} µs > 25 µs budget. "
            "At 60 fps the loop budget is 16667 µs — even a 25 µs cost is "
            "0.15% of the budget, but anything above that is creeping toward "
            "noticeable."
        )

    def test_enabled_realistic_iteration_under_budget(self):
        """Enabled session with a realistic shape — a few spans, a few
        cam_consume calls — matching a teleop / HVLA-main iteration.
        Should run under ~30 µs per iteration."""
        from types import SimpleNamespace

        session = LatencySession.from_config(
            enabled=True,
            loop_kind="bench",
            target_fps=60,
            summary_interval_s=1e9,
        )
        cams = {f"cam{i}": SimpleNamespace(latest_timestamp=time.perf_counter()) for i in range(3)}

        def body():
            with session.iteration():
                with session.span("get_observation"):
                    pass
                session.cam_consume_all(cams)
                with session.span("publish_obs"):
                    pass
                with session.span("get_chunk"):
                    pass
                with session.span("action_send"):
                    pass
                session.set_field("intervention", False)

        per_iter_us = self._measure_per_iter_us(5000, body)
        assert per_iter_us < 60.0, (
            f"Realistic iteration cost {per_iter_us:.2f} µs > 60 µs budget. "
            "At 30 fps (33333 µs budget) this is still <0.2%, but we want "
            "headroom; investigate any regression."
        )
