#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for the post-recording quality summary helpers.

The functions under test are pure (no I/O, no global state), so the
tests just drive them with hand-crafted record lists and assert the
returned dict shape and values. The persistent file writes live in
``lerobot_record._write_episode_health`` / ``_write_recording_health``
— those wrappers are thin and their behaviour is the responsibility of
an integration test on the recording loop itself."""

from __future__ import annotations

import math

from lerobot.utils.latency.recording_health import (
    DEFAULT_VERDICT_THRESHOLDS,
    filter_by_episode,
    summarize,
    verdict,
)

# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


class TestSummarize:
    def test_empty_records_returns_zero_n_records(self):
        out = summarize([], target_period_ms=33.3)
        assert out["n_records"] == 0
        assert out["overrun_ratio"] == 0.0
        assert math.isnan(out["loop_dt_ms"]["p50"])
        assert out["cameras"] == {}

    def test_basic_loop_dt_percentiles(self):
        records = [{"loop_dt_ms": v, "t": float(i)} for i, v in enumerate([10, 20, 30, 40, 50])]
        out = summarize(records, target_period_ms=33.3)
        assert out["n_records"] == 5
        assert out["loop_dt_ms"]["p50"] == 30.0
        assert out["loop_dt_ms"]["max"] == 50.0

    def test_overrun_ratio_uses_target_period(self):
        records = [{"loop_dt_ms": v, "t": float(i)} for i, v in enumerate([10, 20, 30, 40, 50])]
        # 2 of 5 over 35ms → 0.4
        out = summarize(records, target_period_ms=35.0)
        assert out["overrun_ratio"] == 0.4

    def test_no_target_period_means_zero_overrun(self):
        """target_period_ms=None means we don't know the budget. Don't
        guess — return 0 overrun_ratio so downstream verdict skips the
        overrun rule."""
        records = [{"loop_dt_ms": 100.0, "t": 0.0} for _ in range(10)]
        out = summarize(records, target_period_ms=None)
        assert out["overrun_ratio"] == 0.0

    def test_duration_from_record_timestamps(self):
        records = [
            {"loop_dt_ms": 10.0, "t": 100.0},
            {"loop_dt_ms": 10.0, "t": 105.0},
            {"loop_dt_ms": 10.0, "t": 130.0},
        ]
        out = summarize(records, target_period_ms=None)
        assert out["duration_s"] == 30.0

    def test_camera_block_per_camera(self):
        records = [
            {
                "loop_dt_ms": 10.0,
                "t": float(i),
                "cam_front_stale_ms": 15.0 + i,
                "cam_front_period_ms": 33.0,
                "cam_top_stale_ms": 50.0,
                "cam_top_period_ms": 66.0,
            }
            for i in range(10)
        ]
        out = summarize(records, target_period_ms=33.3)
        assert set(out["cameras"]) == {"front", "top"}
        # period_p50 should equal 33.0 for front, 66.0 for top
        assert out["cameras"]["front"]["period_p50"] == 33.0
        assert out["cameras"]["top"]["period_p50"] == 66.0
        # fps_effective is 1000/period_p50
        assert out["cameras"]["front"]["fps_effective"] == 1000.0 / 33.0
        assert math.isclose(out["cameras"]["top"]["fps_effective"], 1000.0 / 66.0)

    def test_camera_appearing_only_in_some_records_still_aggregated(self):
        """Cameras can come and go (grab thread hiccup, late connect).
        summarize should aggregate over whichever records have the key."""
        records = [
            {"loop_dt_ms": 10.0, "t": 0.0, "cam_front_stale_ms": 20.0, "cam_front_period_ms": 33.0},
            {"loop_dt_ms": 10.0, "t": 1.0},  # cam missing this iter
            {"loop_dt_ms": 10.0, "t": 2.0, "cam_front_stale_ms": 40.0, "cam_front_period_ms": 33.0},
        ]
        out = summarize(records, target_period_ms=33.3)
        assert "front" in out["cameras"]
        # Only 2 stale values → median = 30
        assert out["cameras"]["front"]["stale_p50"] == 30.0

    def test_zero_period_does_not_divide_by_zero(self):
        """Regression: when the policy blocks the loop for the entire
        episode (cold torch.compile autotune, network stall, etc.) the
        cam_*_period_ms samples can degenerate to all-zero. The naive
        ``1000.0 / median`` then raises ZeroDivisionError and aborts
        recording at episode boundary. fps_effective must fall back to
        NaN — same sentinel as the empty-list case — and the function
        must complete normally for the rest of the stats to land in
        episodes_health.jsonl."""
        records = [
            {
                "loop_dt_ms": 165049.3,
                "t": 0.0,
                "cam_front_stale_ms": 17.0,
                "cam_front_period_ms": 0.0,
            },
            {
                "loop_dt_ms": 165049.3,
                "t": 165.0,
                "cam_front_stale_ms": 33.0,
                "cam_front_period_ms": 0.0,
            },
        ]
        out = summarize(records, target_period_ms=33.3)
        # No exception — that's the structural guarantee.
        assert out["n_records"] == 2
        assert out["cameras"]["front"]["period_p50"] == 0.0
        # fps_effective is NaN (the "no useful FPS" sentinel), not inf or 0.
        import math

        assert math.isnan(out["cameras"]["front"]["fps_effective"])
        # Other stats still computed normally — the guard is narrow.
        assert out["cameras"]["front"]["stale_p50"] == 25.0

    def test_single_record_does_not_divide_by_zero(self):
        """Single-record episode (e.g., the recording terminated after one
        frame because the compile ate the budget): cam period is 0 by
        construction (no second sample to diff against). Same NaN guard
        applies."""
        records = [
            {"loop_dt_ms": 60000.0, "t": 0.0, "cam_top_stale_ms": 5.0, "cam_top_period_ms": 0.0},
        ]
        out = summarize(records, target_period_ms=33.3)
        import math

        assert math.isnan(out["cameras"]["top"]["fps_effective"])


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------


class TestVerdict:
    def test_healthy_episode_no_issues(self):
        summary = summarize(
            [{"loop_dt_ms": 20.0, "t": float(i)} for i in range(100)],
            target_period_ms=33.3,
        )
        out = verdict(summary)
        assert out["healthy"] is True
        assert out["issues"] == []

    def test_high_overrun_flagged(self):
        # 30/100 over 33.3ms = 30% overrun, well above the 5% default
        records = [{"loop_dt_ms": 50.0 if i < 30 else 20.0, "t": float(i)} for i in range(100)]
        summary = summarize(records, target_period_ms=33.3)
        out = verdict(summary)
        assert out["healthy"] is False
        rule_names = {i["rule"] for i in out["issues"]}
        assert "overrun_high" in rule_names

    def test_loop_tail_flagged(self):
        # 95th percentile far over 1.5× budget
        records = [{"loop_dt_ms": 20.0 if i < 90 else 70.0, "t": float(i)} for i in range(100)]
        summary = summarize(records, target_period_ms=33.3)
        out = verdict(summary)
        rule_names = {i["rule"] for i in out["issues"]}
        assert "loop_tail_high" in rule_names

    def test_camera_stale_flagged_per_camera(self):
        # front: stale 80 > 2 × period 33 → flag
        # top:   stale 30 < 2 × period 33 → no flag
        records = [
            {
                "loop_dt_ms": 10.0,
                "t": float(i),
                "cam_front_stale_ms": 80.0,
                "cam_front_period_ms": 33.0,
                "cam_top_stale_ms": 30.0,
                "cam_top_period_ms": 33.0,
            }
            for i in range(30)
        ]
        summary = summarize(records, target_period_ms=33.3)
        out = verdict(summary)
        cam_issues = [i for i in out["issues"] if i["rule"] == "camera_stale"]
        assert len(cam_issues) == 1
        assert "front" in cam_issues[0]["message"]
        assert "top" not in cam_issues[0]["message"]

    def test_empty_summary_is_healthy(self):
        """Don't flag an empty record set as unhealthy — there's nothing
        to evaluate. The data panel decides separately how to render
        "no data" episodes."""
        out = verdict(summarize([], target_period_ms=33.3))
        assert out["healthy"] is True
        assert out["issues"] == []

    def test_custom_thresholds_loosen_default(self):
        """A user who explicitly relaxes thresholds gets fewer issues —
        proves the threshold knob is wired through. Carefully chosen
        values: 30% over budget (trips overrun_high default 5%), but
        p95 stays under 1.5× budget (so loop_tail_high doesn't fire and
        we can isolate the overrun_ratio threshold behaviour)."""
        records = [{"loop_dt_ms": 35.0 if i < 30 else 30.0, "t": float(i)} for i in range(100)]
        summary = summarize(records, target_period_ms=33.3)
        # Default: 5% threshold → fires (overrun is 30%)
        assert verdict(summary)["healthy"] is False
        # Loose: 50% threshold → does not fire
        assert verdict(summary, thresholds={"overrun_ratio": 0.50})["healthy"] is True

    def test_default_thresholds_are_documented_values(self):
        """Guard against silent drift of the defaults. If we change these
        thresholds intentionally, this test forces us to update the
        accompanying docs at the same time."""
        assert DEFAULT_VERDICT_THRESHOLDS["overrun_ratio"] == 0.05
        assert DEFAULT_VERDICT_THRESHOLDS["loop_p95_factor"] == 1.5
        assert DEFAULT_VERDICT_THRESHOLDS["cam_stale_p95_factor"] == 2.0


# ---------------------------------------------------------------------------
# filter_by_episode
# ---------------------------------------------------------------------------


class TestFilterByEpisode:
    def test_filters_records_to_one_episode(self):
        records = [
            {"loop_dt_ms": 10.0, "t": 0.0, "ep": 0},
            {"loop_dt_ms": 10.0, "t": 1.0, "ep": 0},
            {"loop_dt_ms": 10.0, "t": 2.0, "ep": 1},
            {"loop_dt_ms": 10.0, "t": 3.0, "ep": 1},
            {"loop_dt_ms": 10.0, "t": 4.0, "ep": 2},
        ]
        ep0 = filter_by_episode(records, 0)
        assert len(ep0) == 2
        assert all(r["ep"] == 0 for r in ep0)

    def test_records_without_ep_field_excluded(self):
        """Reset / setup-phase records have no ``ep`` tag. They must not
        count toward any episode's verdict."""
        records = [
            {"loop_dt_ms": 10.0, "t": 0.0},  # no ep — reset phase
            {"loop_dt_ms": 10.0, "t": 1.0, "ep": 0},
            {"loop_dt_ms": 10.0, "t": 2.0, "ep": 0},
        ]
        ep0 = filter_by_episode(records, 0)
        assert len(ep0) == 2

    def test_unknown_episode_returns_empty(self):
        records = [{"loop_dt_ms": 10.0, "t": 0.0, "ep": 0}]
        assert filter_by_episode(records, 99) == []


# ---------------------------------------------------------------------------
# Integration: per-episode summary mirrors session-wide summary on the
# subset of records belonging to that episode.
# ---------------------------------------------------------------------------


class TestPerEpisodeFlowEndToEnd:
    def test_per_episode_summary_matches_session_summary_for_that_ep(self):
        """If we summarize one episode's records, we should get the same
        numbers as summarize(session_records).filter(ep)."""
        ep0 = [{"loop_dt_ms": 20.0, "t": float(i), "ep": 0} for i in range(50)]
        ep1 = [{"loop_dt_ms": 60.0, "t": float(50 + i), "ep": 1} for i in range(50)]
        all_records = ep0 + ep1
        session_summary_for_ep1 = summarize(filter_by_episode(all_records, 1), target_period_ms=33.3)
        # ep1 has 100% overrun (60 > 33.3 every time)
        assert session_summary_for_ep1["overrun_ratio"] == 1.0
        assert session_summary_for_ep1["loop_dt_ms"]["p50"] == 60.0
        # And the corresponding verdict flags it
        v = verdict(session_summary_for_ep1)
        assert v["healthy"] is False
        assert any(i["rule"] == "overrun_high" for i in v["issues"])
