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

"""Unit tests for LatencyAggregator, LatencyTracer, LatencySnapshotWriter."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pytest

from lerobot.utils.latency import (
    LatencyAggregator,
    LatencySnapshotWriter,
    LatencyTracer,
)

# ---------------------------------------------------------------------------
# LatencyAggregator
# ---------------------------------------------------------------------------


class TestAggregator:
    def test_empty_aggregator(self):
        agg = LatencyAggregator()
        assert len(agg) == 0
        assert agg.stage_keys() == []
        assert math.isnan(agg.percentile("missing_ms", 50))
        assert agg.overrun_ratio() == 0.0
        assert agg.time_series("missing_ms") == []
        snap = agg.snapshot()
        assert snap["n_records"] == 0
        assert snap["dropped_records"] == 0
        assert snap["overrun_ratio"] == 0.0
        assert snap["stages"] == {}
        assert agg.representative_iterations() == {}

    def test_ingest_and_percentile(self):
        agg = LatencyAggregator()
        for i in range(100):
            agg.ingest({"t": float(i), "loop_dt_ms": float(i)})
        assert len(agg) == 100
        p50, p95 = agg.percentile("loop_dt_ms", [50, 95])
        assert 49.0 <= p50 <= 50.0
        assert 94.0 <= p95 <= 95.0

    def test_maxlen_eviction(self):
        agg = LatencyAggregator(maxlen=10)
        for i in range(100):
            agg.ingest({"t": float(i), "v_ms": float(i)})
        assert len(agg) == 10
        assert agg.values("v_ms") == [float(i) for i in range(90, 100)]

    def test_invalid_maxlen_rejected(self):
        with pytest.raises(AssertionError):
            LatencyAggregator(maxlen=0)
        with pytest.raises(AssertionError):
            LatencyAggregator(maxlen=-1)

    def test_overrun_ratio(self):
        agg = LatencyAggregator()
        for i in range(10):
            agg.ingest({"t": float(i), "overrun": (i % 2 == 0)})
        assert agg.overrun_ratio() == pytest.approx(0.5)
        agg.ingest({"t": 10.0})
        assert agg.overrun_ratio() == pytest.approx(0.5)

    def test_time_series_filters_by_window(self):
        agg = LatencyAggregator()
        for i in range(10):
            agg.ingest({"t": float(i), "v_ms": float(i)})
        ts = agg.time_series("v_ms", last_n_seconds=3.0)
        assert len(ts) == 4
        assert ts[0] == (6.0, 6.0)
        assert ts[-1] == (9.0, 9.0)

    def test_snapshot_shape(self):
        agg = LatencyAggregator()
        for i in range(20):
            agg.ingest({"t": float(i), "loop_dt_ms": float(i), "motor_read_ms": float(i) / 2})
        snap = agg.snapshot(percentiles=(50, 95))
        assert snap["n_records"] == 20
        assert set(snap["stages"].keys()) == {"loop_dt_ms", "motor_read_ms"}
        loop = snap["stages"]["loop_dt_ms"]
        assert {"p50", "p95", "count"} <= set(loop.keys())
        assert loop["count"] == 20

    def test_clear_resets_state(self):
        agg = LatencyAggregator()
        for i in range(5):
            agg.ingest({"t": float(i), "v_ms": float(i)})
        agg.dropped_records = 3
        agg.clear()
        assert len(agg) == 0
        assert agg.dropped_records == 0

    def test_representative_iterations_are_sticky(self):
        """Median/p95/p99 picks should reuse the previous record across calls
        as long as it's still in the deque and close to the target.
        Otherwise the GUI Gantt re-picks a different "median-ish" iteration
        every snapshot and the camera markers visibly jitter."""
        agg = LatencyAggregator(maxlen=500)
        # Tight cluster around p50 = ~50: many records qualify as "median".
        for i in range(200):
            agg.ingest(
                {
                    "t": float(i),
                    "step": i,
                    # Loop_dt clusters tightly so several records compete for
                    # "closest to p50" — exactly the case where stickiness matters.
                    "loop_dt_ms": 50.0 + (i % 5) * 0.5,
                    "spans": {"work": [0.0, 50.0 + (i % 5) * 0.5]},
                    "cam_events": {},
                }
            )
        first = agg.representative_iterations()["median"]
        # Append more records that don't change the percentile much.
        for i in range(200, 250):
            agg.ingest(
                {
                    "t": float(i),
                    "step": i,
                    "loop_dt_ms": 50.0 + (i % 5) * 0.5,
                    "spans": {"work": [0.0, 50.0 + (i % 5) * 0.5]},
                    "cam_events": {},
                }
            )
        second = agg.representative_iterations()["median"]
        # Same step should be returned both times — sticky.
        assert first["step"] == second["step"]

    def test_aggregate_iteration_uses_per_stage_percentiles(self):
        """The synthesized Gantt's per-stage durations should equal the
        per-stage percentiles, NOT the per-stage values from a single
        median-loop record (which is what representative_iterations
        returns). This is the whole reason aggregate_iteration exists."""
        agg = LatencyAggregator()
        # 100 records where motor_read drifts 1..100 ms but action_send
        # is constant at 5 ms. Median of motor_read = ~50; median of
        # action_send = 5. A real median-loop record would happen to
        # have whatever motor_read it had; the aggregate must show 50.
        for i in range(1, 101):
            agg.ingest(
                {
                    "t": float(i),
                    "step": i,
                    "loop_dt_ms": float(i + 5),
                    "spans": {
                        "motor_read": [0.0, float(i)],
                        "action_send": [float(i), float(i + 5)],
                    },
                    "cam_events": {},
                }
            )

        synth = agg.aggregate_iteration(50)
        assert synth is not None
        assert synth["synthetic"] is True
        assert synth["n_aggregated"] == 100
        assert synth["percentile"] == 50
        # motor_read p50 ~= 50; action_send p50 = 5.
        m_start, m_end = synth["spans"]["motor_read"]
        a_start, a_end = synth["spans"]["action_send"]
        assert m_start == 0.0
        assert 49.0 <= (m_end - m_start) <= 51.0
        # action_send is laid out *after* motor_read (cumulative cursor).
        assert a_start == pytest.approx(m_end)
        assert (a_end - a_start) == pytest.approx(5.0)

    def test_aggregate_iteration_camera_events_use_per_cam_percentiles(self):
        """Each camera's captured/consumed offsets get its own percentile.
        A noisy wrist cam should not contaminate the front cam's number."""
        agg = LatencyAggregator()
        # front cam: stable -10 ms capture; wrist cam: noisy -5..-50 ms.
        for i in range(50):
            agg.ingest(
                {
                    "t": float(i),
                    "step": i,
                    "loop_dt_ms": 20.0,
                    "spans": {},
                    "cam_events": {
                        "front": {"captured_at_ms": -10.0, "consumed_at_ms": 0.0},
                        "wrist": {"captured_at_ms": -(5.0 + i), "consumed_at_ms": 0.0},
                    },
                }
            )
        synth = agg.aggregate_iteration(50)
        assert synth["cam_events"]["front"]["captured_at_ms"] == pytest.approx(-10.0)
        # Wrist median ~= -29.5 (i in 0..49 → 5..54, p50 ~ 29.5)
        assert -32.0 <= synth["cam_events"]["wrist"]["captured_at_ms"] <= -27.0

    def test_aggregate_iteration_empty_returns_none(self):
        agg = LatencyAggregator()
        assert agg.aggregate_iteration() is None

    def test_aggregate_iteration_preserves_nesting(self):
        """Regression test: spans that nest in every input record must
        also nest in the aggregate.

        The earlier cascade layout placed spans end-to-end in dict
        insertion order. Child spans (whose ``__exit__`` fires first)
        landed in the dict before their parents, so the aggregate showed
        children laid out BEFORE their parents — visually breaking the
        nested structure operators see in real-iteration samples.

        Concrete shape used here: ``get_observation`` contains
        ``motor_read_left`` in every record (motor_read_left's start
        offset is always >= get_observation's, end <= get_observation's).
        After aggregation the same containment must hold.
        """
        agg = LatencyAggregator()
        for i in range(50):
            # Outer span [0, 10ms]; inner span starts at 1-3ms and ends
            # well before 10ms. Crucially the child is inserted FIRST in
            # dict order — the tracer always closes the inner span before
            # the outer, so the inner gets dict-inserted first. The
            # aggregator must NOT use dict order to decide layout.
            spans = {}
            inner_start = 1.0 + (i % 3)
            inner_end = inner_start + 3.0
            spans["motor_read_left"] = [inner_start, inner_end]
            spans["get_observation"] = [0.0, 10.0]
            agg.ingest({"t": float(i), "step": i, "loop_dt_ms": 12.0, "spans": spans, "cam_events": {}})

        synth = agg.aggregate_iteration(50)
        outer_s, outer_e = synth["spans"]["get_observation"]
        inner_s, inner_e = synth["spans"]["motor_read_left"]
        assert outer_s <= inner_s <= inner_e <= outer_e, (
            f"motor_read_left {[inner_s, inner_e]} not contained in get_observation {[outer_s, outer_e]}"
        )

    @pytest.mark.parametrize("percentile", [50, 75, 95])
    def test_aggregate_iteration_containment_invariant(self, percentile):
        """Property-based test: for every pair of spans (A, B) where A
        contains B in *every* input record, A must contain B in the
        aggregate at any percentile.

        Catches future regressions where someone changes the layout
        algorithm — no need to enumerate per-span expectations. Add a
        new span anywhere in the codebase and this test still covers it
        as long as the new span has a deterministic parent relationship
        with at least one existing span.
        """
        agg = LatencyAggregator()
        # Build records that mirror the real teleop nesting:
        #   get_observation:    contains motor_read_*, camera_read_*
        #   process_obs, action_send: siblings of get_observation
        import random

        rng = random.Random(0)
        records: list[dict] = []
        for i in range(40):
            obs_dur = rng.uniform(4.0, 8.0)
            ml_start = rng.uniform(0.0, 0.5)
            ml_end = ml_start + rng.uniform(0.8, 1.5)
            mr_start = ml_end + rng.uniform(0.0, 0.3)
            mr_end = mr_start + rng.uniform(0.8, 1.5)
            cam_start = mr_end + rng.uniform(0.0, 0.5)
            cam_end = min(cam_start + rng.uniform(0.5, 1.0), obs_dur - 0.1)
            proc_dur = rng.uniform(0.5, 2.0)
            send_dur = rng.uniform(0.1, 0.4)
            spans = {
                # Insert children before parents (matches __exit__ order).
                "motor_read_left": [ml_start, ml_end],
                "motor_read_right": [mr_start, mr_end],
                "camera_read_top": [cam_start, cam_end],
                "get_observation": [0.0, obs_dur],
                "process_obs": [obs_dur, obs_dur + proc_dur],
                "action_send": [obs_dur + proc_dur, obs_dur + proc_dur + send_dur],
            }
            rec = {"t": float(i), "step": i, "loop_dt_ms": 15.0, "spans": spans, "cam_events": {}}
            agg.ingest(rec)
            records.append(rec)

        synth = agg.aggregate_iteration(percentile)
        spans = synth["spans"]

        # For each pair, if A contained B in every input record, A must
        # contain B in the aggregate.
        names = list(spans.keys())
        checked_pairs = 0
        for a in names:
            for b in names:
                if a == b:
                    continue
                a_contains_b_always = all(
                    r["spans"][a][0] <= r["spans"][b][0] and r["spans"][b][1] <= r["spans"][a][1]
                    for r in records
                )
                if not a_contains_b_always:
                    continue
                checked_pairs += 1
                a_s, a_e = spans[a]
                b_s, b_e = spans[b]
                assert a_s <= b_s <= b_e <= a_e, (
                    f"[p={percentile}] {a} {[a_s, a_e]} should contain {b} {[b_s, b_e]}"
                )
        # Sanity: at least the get_observation containments must have fired.
        assert checked_pairs >= 3, f"expected get_observation to contain ≥3 children, saw {checked_pairs}"

    def test_aggregate_iteration_loop_dt_independent_of_span_sum(self):
        """loop_dt_ms in the synthetic record uses its OWN percentile,
        not the sum of stage percentiles. The visible gap on the right
        of the Gantt indicates uninstrumented work — important signal."""
        agg = LatencyAggregator()
        for i in range(20):
            agg.ingest(
                {
                    "t": float(i),
                    "step": i,
                    "loop_dt_ms": 30.0,  # constant 30 ms total
                    "spans": {"work": [0.0, 10.0]},  # only 10 ms accounted for
                    "cam_events": {},
                }
            )
        synth = agg.aggregate_iteration(50)
        assert synth["loop_dt_ms"] == pytest.approx(30.0)
        # Span sum is only 10, so there's a 20 ms gap visible on the Gantt.
        rightmost_end = max(end for _, end in synth["spans"].values())
        assert rightmost_end == pytest.approx(10.0)
        # The gap is loop_dt - rightmost = 20 ms.
        assert (synth["loop_dt_ms"] - rightmost_end) == pytest.approx(20.0)

    def test_representative_iterations_picks_percentiles(self):
        agg = LatencyAggregator()
        # Ingest records with linearly increasing loop_dt and a span dict
        # that varies by step so we can check the right one was picked.
        for i in range(100):
            agg.ingest(
                {
                    "t": float(i),
                    "step": i,
                    "loop_dt_ms": float(i),
                    "spans": {"work": [0.0, float(i)]},
                    "cam_events": {},
                }
            )
        reps = agg.representative_iterations()
        assert set(reps.keys()) == {"median", "p95", "p99", "max", "latest"}
        # Each entry should be a small subset (no other keys leak).
        assert set(reps["median"].keys()) == {"step", "t", "loop_dt_ms", "spans", "cam_events"}
        # Sanity: the chosen records should have loop_dt_ms close to the
        # corresponding percentile of the input distribution (0..99).
        assert 45.0 <= reps["median"]["loop_dt_ms"] <= 55.0
        assert 90.0 <= reps["p95"]["loop_dt_ms"] <= 99.0
        assert reps["max"]["loop_dt_ms"] == 99.0
        assert reps["latest"]["loop_dt_ms"] == 99.0


# ---------------------------------------------------------------------------
# LatencyTracer
# ---------------------------------------------------------------------------


class TestTracer:
    def test_span_records_endpoint_pair_and_duration(self):
        tracer = LatencyTracer()
        tracer.start()
        with tracer.span("work"):
            time.sleep(0.005)
        record = tracer.commit()
        # Duration scalar still emitted for sparkline / percentile queries.
        assert "work_ms" in record
        assert 3.0 <= record["work_ms"] <= 30.0
        # Endpoint pair available for Gantt rendering.
        assert "spans" in record
        assert "work" in record["spans"]
        start, end = record["spans"]["work"]
        # Endpoints are offsets from iter_start; both >= 0 and end >= start.
        assert start >= 0.0
        assert end >= start
        # Duration matches end - start.
        assert end - start == pytest.approx(record["work_ms"])

    def test_commit_without_aggregator(self):
        tracer = LatencyTracer()
        tracer.start()
        record = tracer.commit()
        assert record["loop_kind"] == "teleop"
        assert "loop_dt_ms" in record
        assert "step" in record
        # Even with no spans, the spans/cam_events keys exist (empty dicts).
        assert record["spans"] == {}
        assert record["cam_events"] == {}

    def test_commit_with_aggregator_ingests(self):
        agg = LatencyAggregator()
        tracer = LatencyTracer(agg)
        for _ in range(3):
            tracer.start()
            tracer.commit()
        assert len(agg) == 3

    def test_step_counter_increments(self):
        tracer = LatencyTracer()
        steps = []
        for _ in range(5):
            tracer.start()
            steps.append(tracer.commit()["step"])
        assert steps == [0, 1, 2, 3, 4]

    def test_iter_start_perf_recorded(self):
        """The wall-clock perf_counter at iteration start lands in the
        record so multi-track dashboards (HVLA main + inference) can
        align spans on a shared time axis without joining records."""
        tracer = LatencyTracer()
        before = time.perf_counter()
        tracer.start()
        record = tracer.commit()
        after = time.perf_counter()
        assert "iter_start_perf" in record
        assert before <= record["iter_start_perf"] <= after

    def test_cam_consume_records_capture_and_consume_offsets(self):
        tracer = LatencyTracer()
        tracer.start()
        # Frame captured 20 ms ago — captured_at_ms should be negative.
        t0 = time.perf_counter() - 0.020
        tracer.cam_consume("top", t0)
        record_first = tracer.commit()
        assert "cam_top_stale_ms" in record_first
        # No period yet (only one consume).
        assert "cam_top_period_ms" not in record_first
        ev = record_first["cam_events"]["top"]
        assert ev["captured_at_ms"] < 0  # captured before iter_start
        assert ev["consumed_at_ms"] >= 0  # consumed inside the iter
        # Stale matches consumed - captured.
        assert record_first["cam_top_stale_ms"] == pytest.approx(ev["consumed_at_ms"] - ev["captured_at_ms"])

        # Second consume with frame 33 ms after the previous → period appears.
        tracer.start()
        t1 = t0 + 0.033
        tracer.cam_consume("top", t1)
        record_second = tracer.commit()
        assert record_second["cam_top_period_ms"] == pytest.approx(33.0, abs=0.5)

    def test_overrun_flag_when_target_fps_set(self):
        tracer = LatencyTracer(target_fps=120.0)
        tracer.start()
        time.sleep(0.020)
        record = tracer.commit()
        assert record["overrun"] is True

        tracer.start()
        record_fast = tracer.commit()
        assert record_fast["overrun"] is False

    def test_no_overrun_field_when_no_target_fps(self):
        tracer = LatencyTracer()
        tracer.start()
        record = tracer.commit()
        assert "overrun" not in record

    def test_assertion_when_no_start(self):
        tracer = LatencyTracer()
        with pytest.raises(AssertionError):
            tracer.commit()
        with pytest.raises(AssertionError):
            with tracer.span("foo"):
                pass
        with pytest.raises(AssertionError):
            tracer.cam_consume("top", time.perf_counter())

    def test_set_field_extends_record(self):
        tracer = LatencyTracer()
        tracer.start()
        tracer.set_field("ep", 7)
        tracer.set_field("note", "smoke")
        record = tracer.commit()
        assert record["ep"] == 7
        assert record["note"] == "smoke"

    def test_gantt_can_be_reconstructed_from_spans(self):
        """An iteration's spans should describe a coherent timeline that the
        GUI can render: every span's [start, end] is contained within
        [0, loop_dt_ms]; cameras can sit before zero (background grab)."""
        tracer = LatencyTracer()
        tracer.start()
        # Camera captured 30 ms ago.
        cam_ts = time.perf_counter() - 0.030
        tracer.cam_consume("top", cam_ts)
        with tracer.span("motor_read"):
            time.sleep(0.002)
        with tracer.span("process_action"):
            time.sleep(0.001)
        record = tracer.commit()

        loop_dt = record["loop_dt_ms"]
        for name, (start, end) in record["spans"].items():
            assert 0.0 <= start <= end <= loop_dt + 0.5, f"span {name} out of bounds"
        ev = record["cam_events"]["top"]
        assert ev["captured_at_ms"] < 0  # before iter
        assert 0.0 <= ev["consumed_at_ms"] <= loop_dt


# ---------------------------------------------------------------------------
# Hot-path overhead — synthetic loop benchmark
# ---------------------------------------------------------------------------


class TestOverhead:
    """Capture overhead must stay well below 1 ms / iter at 60 Hz."""

    @pytest.mark.parametrize("with_aggregator", [True, False])
    def test_per_iteration_overhead(self, with_aggregator):
        agg = LatencyAggregator() if with_aggregator else None
        tracer = LatencyTracer(agg, target_fps=60.0)

        n_iters = 5000
        for _ in range(100):  # warmup
            tracer.start()
            with tracer.span("a"):
                pass
            with tracer.span("b"):
                pass
            tracer.cam_consume("top", time.perf_counter())
            tracer.commit()

        t0 = time.perf_counter()
        for _ in range(n_iters):
            tracer.start()
            with tracer.span("motor_read"):
                pass
            tracer.cam_consume("top", time.perf_counter())
            with tracer.span("process_obs"):
                pass
            with tracer.span("process_action"):
                pass
            with tracer.span("action_send"):
                pass
            tracer.commit()
        elapsed = time.perf_counter() - t0
        per_iter_us = (elapsed / n_iters) * 1e6

        # Strict goal is ~10 µs; CI is noisy. 200 µs catches a 20× regression.
        assert per_iter_us < 200.0, f"per-iter overhead {per_iter_us:.1f} µs > 200 µs"


# ---------------------------------------------------------------------------
# Snapshot series + LatencySnapshotWriter
# ---------------------------------------------------------------------------


class TestSnapshotSeries:
    def test_snapshot_with_series_window(self):
        agg = LatencyAggregator()
        for i in range(60):
            agg.ingest({"t": float(i), "loop_dt_ms": float(i)})
        snap = agg.snapshot(series_window_s=10.0, series_max_points=10)
        assert "series" in snap
        series = snap["series"]["loop_dt_ms"]
        assert len(series) <= 10
        for t, _v in series:
            assert t >= 49.0

    def test_snapshot_without_series(self):
        agg = LatencyAggregator()
        agg.ingest({"t": 0.0, "v_ms": 1.0})
        snap = agg.snapshot()
        assert "series" not in snap


class TestSnapshotWriter:
    def test_first_write_creates_file(self, tmp_path: Path):
        agg = LatencyAggregator()
        agg.ingest(
            {
                "t": 0.0,
                "step": 0,
                "loop_dt_ms": 12.5,
                "spans": {"work": [0.0, 12.0]},
                "cam_events": {},
            }
        )
        writer = LatencySnapshotWriter(tmp_path, interval_s=0.0)
        wrote = writer.maybe_write(agg)
        assert wrote
        writer.flush()  # snapshot writer runs the heavy work in a background thread
        assert writer.path.exists()
        data = json.loads(writer.path.read_text())
        assert data["loop_kind"] == "teleop"
        assert "stages" in data
        assert "loop_dt_ms" in data["stages"]
        assert "iterations" in data
        # Every label points at a record subset with timeline keys.
        for label, rec in data["iterations"].items():
            assert "spans" in rec
            assert "cam_events" in rec
            assert "loop_dt_ms" in rec, f"missing loop_dt_ms in {label}"

    def test_throttle_skips_within_interval(self, tmp_path: Path):
        agg = LatencyAggregator()
        agg.ingest({"t": 0.0, "loop_dt_ms": 1.0})
        writer = LatencySnapshotWriter(tmp_path, interval_s=10.0)
        assert writer.maybe_write(agg, now=100.0) is True
        writer.flush()
        assert writer.maybe_write(agg, now=101.0) is False
        assert writer.maybe_write(agg, now=111.0) is True
        writer.flush()

    def test_invalid_interval_rejected(self, tmp_path: Path):
        with pytest.raises(AssertionError):
            LatencySnapshotWriter(tmp_path, interval_s=-1)

    def test_atomic_replace_no_partial_file(self, tmp_path: Path):
        agg = LatencyAggregator()
        agg.ingest({"t": 0.0, "v_ms": 1.0})
        writer = LatencySnapshotWriter(tmp_path, interval_s=0.0)
        writer.maybe_write(agg)
        writer.flush()
        assert writer.path.exists()
        tmp = tmp_path / "latency_snapshot.json.tmp"
        assert not tmp.exists()
