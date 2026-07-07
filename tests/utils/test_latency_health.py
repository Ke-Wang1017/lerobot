#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for ``LoopHealthDetector`` and the built-in rule set.

The detector is the always-on guardrail surfaced both as rate-limited
WARNING log lines and as health badges on the GUI dashboard. These tests
pin (1) each rule's threshold semantics so a future tweak can't silently
change when the badge fires, (2) the rate-limit behaviour so we don't
spam the log at 30 Hz, and (3) the detector's robustness to a buggy
rule (a single bad rule must not crash the loop)."""

from __future__ import annotations

import time

from lerobot.utils.latency.aggregator import LatencyAggregator
from lerobot.utils.latency.health import (
    CameraStalenessRule,
    HealthIssue,
    LoopBudgetRule,
    LoopHealthDetector,
    NoDataRule,
    OverrunRatioRule,
    Rule,
    default_rules,
)


def _ingest_records(agg: LatencyAggregator, records: list[dict]) -> None:
    """Push records into the aggregator in order, simulating a loop."""
    for r in records:
        agg.ingest(r)


# ---------------------------------------------------------------------------
# OverrunRatioRule
# ---------------------------------------------------------------------------


class TestOverrunRatioRule:
    def test_quiet_when_below_threshold(self):
        """8% overrun over 100 iters at threshold=10% must not fire — the
        threshold is "sustained", not "any overrun ever."""
        agg = LatencyAggregator()
        _ingest_records(
            agg,
            [{"loop_dt_ms": 50.0 if i < 8 else 20.0, "t": time.time()} for i in range(100)],
        )
        rule = OverrunRatioRule(threshold=0.10, window_n=100)
        assert rule.evaluate(agg, agg._records[-1], target_period_ms=33.3) is None

    def test_fires_above_threshold(self):
        agg = LatencyAggregator()
        # 50/100 = 50% over budget, well above 10%
        _ingest_records(
            agg,
            [{"loop_dt_ms": 50.0 if i < 50 else 20.0, "t": time.time()} for i in range(100)],
        )
        rule = OverrunRatioRule(threshold=0.10, window_n=100)
        issue = rule.evaluate(agg, agg._records[-1], target_period_ms=33.3)
        assert issue is not None
        assert issue.rule == "overrun_high"
        assert "50%" in issue.message
        assert "33.3 ms" in issue.message

    def test_quiet_without_target_period(self):
        """No target_period_ms means we don't know the budget — can't
        compute overrun, so the rule stays silent."""
        agg = LatencyAggregator()
        _ingest_records(agg, [{"loop_dt_ms": 100.0, "t": time.time()} for _ in range(100)])
        rule = OverrunRatioRule()
        assert rule.evaluate(agg, agg._records[-1], target_period_ms=None) is None

    def test_quiet_when_window_too_small(self):
        """Don't fire on the first handful of records — wait for the
        window to have enough data for a stable estimate."""
        agg = LatencyAggregator()
        _ingest_records(agg, [{"loop_dt_ms": 100.0, "t": time.time()} for _ in range(3)])
        rule = OverrunRatioRule(threshold=0.10, window_n=100)
        assert rule.evaluate(agg, agg._records[-1], target_period_ms=33.3) is None

    def test_window_size_limits_lookback(self):
        """If we just had a sustained problem but the most-recent window
        is clean, the rule should clear (recovery semantics)."""
        agg = LatencyAggregator()
        # 100 bad, then 30 good
        _ingest_records(agg, [{"loop_dt_ms": 100.0, "t": time.time()} for _ in range(100)])
        _ingest_records(agg, [{"loop_dt_ms": 20.0, "t": time.time()} for _ in range(30)])
        # window=20 → only the last 20 (all good) → no issue
        rule = OverrunRatioRule(threshold=0.10, window_n=20)
        assert rule.evaluate(agg, agg._records[-1], target_period_ms=33.3) is None


# ---------------------------------------------------------------------------
# LoopBudgetRule
# ---------------------------------------------------------------------------


class TestLoopBudgetRule:
    def test_fires_when_p95_exceeds_factor_x_budget(self):
        """All 100 iters at 60 ms; p95 ≈ 60 ms. Budget 33.3 ms × 1.5 = 50.
        60 > 50 → fire."""
        agg = LatencyAggregator()
        _ingest_records(agg, [{"loop_dt_ms": 60.0, "t": time.time()} for _ in range(100)])
        rule = LoopBudgetRule(p95_factor=1.5)
        issue = rule.evaluate(agg, agg._records[-1], target_period_ms=33.3)
        assert issue is not None
        assert "p95" in issue.message and "60" in issue.message

    def test_quiet_when_p95_within_factor(self):
        agg = LatencyAggregator()
        _ingest_records(agg, [{"loop_dt_ms": 40.0, "t": time.time()} for _ in range(100)])
        # 33.3 × 1.5 = 50, 40 < 50 → no fire
        rule = LoopBudgetRule(p95_factor=1.5)
        assert rule.evaluate(agg, agg._records[-1], target_period_ms=33.3) is None

    def test_quiet_without_target_period(self):
        agg = LatencyAggregator()
        _ingest_records(agg, [{"loop_dt_ms": 1000.0, "t": time.time()} for _ in range(100)])
        rule = LoopBudgetRule()
        assert rule.evaluate(agg, agg._records[-1], target_period_ms=None) is None


# ---------------------------------------------------------------------------
# CameraStalenessRule
# ---------------------------------------------------------------------------


class TestCameraStalenessRule:
    def _records(self, stale_ms: float, period_ms: float, n: int = 30, cam: str = "front") -> list[dict]:
        return [
            {
                "loop_dt_ms": 10.0,
                "t": time.time(),
                f"cam_{cam}_stale_ms": stale_ms,
                f"cam_{cam}_period_ms": period_ms,
            }
            for _ in range(n)
        ]

    def test_fires_when_stale_p95_exceeds_factor_x_period(self):
        """Stale 80 ms with period 33 ms → 80 > 33 × 2.0 → fire."""
        agg = LatencyAggregator()
        _ingest_records(agg, self._records(stale_ms=80.0, period_ms=33.0))
        rule = CameraStalenessRule(p95_factor=2.0)
        issue = rule.evaluate(agg, agg._records[-1], target_period_ms=33.3)
        assert issue is not None
        assert "front" in issue.message

    def test_quiet_when_stale_within_factor(self):
        """Stale 40 ms with period 33 ms → 40 < 33 × 2.0 → no fire."""
        agg = LatencyAggregator()
        _ingest_records(agg, self._records(stale_ms=40.0, period_ms=33.0))
        rule = CameraStalenessRule(p95_factor=2.0)
        assert rule.evaluate(agg, agg._records[-1], target_period_ms=33.3) is None

    def test_handles_multiple_cameras_aggregated_into_one_issue(self):
        """Three offending cameras must produce one issue listing all
        three, not three separate issues (keeps the badge compact)."""
        agg = LatencyAggregator()
        for _ in range(30):
            agg.ingest(
                {
                    "loop_dt_ms": 10.0,
                    "t": time.time(),
                    "cam_front_stale_ms": 100.0,
                    "cam_front_period_ms": 33.0,
                    "cam_top_stale_ms": 90.0,
                    "cam_top_period_ms": 33.0,
                    "cam_wrist_stale_ms": 200.0,
                    "cam_wrist_period_ms": 33.0,
                }
            )
        rule = CameraStalenessRule(p95_factor=2.0)
        issue = rule.evaluate(agg, agg._records[-1], target_period_ms=33.3)
        assert issue is not None
        for cam in ("front", "top", "wrist"):
            assert cam in issue.message

    def test_ignores_implausibly_small_period(self):
        """In the first few iterations a camera's period reading can be
        garbage (e.g. 1 ms because only one consume happened). Don't
        false-alarm on that."""
        agg = LatencyAggregator()
        _ingest_records(agg, self._records(stale_ms=100.0, period_ms=2.0))
        rule = CameraStalenessRule(p95_factor=2.0, min_period_ms=16.0)
        assert rule.evaluate(agg, agg._records[-1], target_period_ms=33.3) is None

    def test_quiet_with_no_cameras(self):
        agg = LatencyAggregator()
        _ingest_records(agg, [{"loop_dt_ms": 10.0, "t": time.time()} for _ in range(30)])
        rule = CameraStalenessRule()
        assert rule.evaluate(agg, agg._records[-1], target_period_ms=33.3) is None


# ---------------------------------------------------------------------------
# NoDataRule
# ---------------------------------------------------------------------------


class TestNoDataRule:
    def test_quiet_when_data_recent(self):
        agg = LatencyAggregator()
        agg.ingest({"loop_dt_ms": 10.0, "t": time.time()})
        rule = NoDataRule(timeout_s=5.0)
        assert rule.evaluate(agg, agg._records[-1], target_period_ms=33.3) is None

    def test_fires_when_data_stale(self):
        agg = LatencyAggregator()
        agg.ingest({"loop_dt_ms": 10.0, "t": time.time() - 30.0})
        rule = NoDataRule(timeout_s=5.0)
        issue = rule.evaluate(agg, agg._records[-1], target_period_ms=33.3)
        assert issue is not None
        assert issue.severity == "error"
        assert "30" in issue.message or "29" in issue.message

    def test_quiet_when_aggregator_empty(self):
        """Empty aggregator → there's nothing TO be stale. The rule
        shouldn't fire just because the session started a moment ago."""
        agg = LatencyAggregator()
        rule = NoDataRule()
        assert rule.evaluate(agg, None, target_period_ms=33.3) is None


# ---------------------------------------------------------------------------
# LoopHealthDetector
# ---------------------------------------------------------------------------


class TestLoopHealthDetector:
    def test_default_rules_are_installed(self):
        d = LoopHealthDetector("teleop", target_period_ms=33.3)
        assert len(d.rules) == 4
        names = {r.name for r in d.rules}
        assert names == {"overrun_high", "loop_tail_high", "camera_stale", "no_data"}

    def test_check_returns_active_issues(self):
        agg = LatencyAggregator()
        _ingest_records(agg, [{"loop_dt_ms": 100.0, "t": time.time()} for _ in range(100)])
        d = LoopHealthDetector("teleop", target_period_ms=33.3)
        issues = d.check(agg, agg._records[-1])
        # At minimum overrun_high + loop_tail_high should fire
        assert len(issues) >= 2
        rule_names = {i.rule for i in issues}
        assert "overrun_high" in rule_names
        assert "loop_tail_high" in rule_names

    def test_rate_limited_logging(self, caplog):
        """Even when a rule fires on every iteration, the log should be
        rate-limited per-rule. Otherwise a 30 Hz loop with a chronic
        problem would spam 30 lines/second."""
        agg = LatencyAggregator()
        _ingest_records(agg, [{"loop_dt_ms": 100.0, "t": time.time()} for _ in range(100)])
        d = LoopHealthDetector("teleop", target_period_ms=33.3, alert_interval_s=60.0)
        with caplog.at_level("WARNING", logger="lerobot.utils.latency.health"):
            # Call check 10x in quick succession; should log each rule at
            # most once thanks to the rate limit.
            for _ in range(10):
                d.check(agg, agg._records[-1])
        overrun_lines = [r for r in caplog.records if "overrun_high" in r.message]
        tail_lines = [r for r in caplog.records if "loop_tail_high" in r.message]
        assert len(overrun_lines) == 1
        assert len(tail_lines) == 1

    def test_buggy_rule_does_not_crash_check(self, caplog):
        """A rule that raises an exception must not crash the loop —
        log the error, skip the rule, continue with the rest."""

        class BadRule(Rule):
            name = "bad"
            severity = "warn"

            def evaluate(self, aggregator, record, target_period_ms):
                raise ValueError("boom")

        agg = LatencyAggregator()
        _ingest_records(agg, [{"loop_dt_ms": 100.0, "t": time.time()} for _ in range(100)])
        # Put the bad rule first; the good rules after it must still run.
        d = LoopHealthDetector(
            "teleop",
            target_period_ms=33.3,
            rules=[BadRule(), OverrunRatioRule(), LoopBudgetRule()],
        )
        with caplog.at_level("WARNING", logger="lerobot.utils.latency.health"):
            issues = d.check(agg, agg._records[-1])
        rule_names = {i.rule for i in issues}
        assert "overrun_high" in rule_names
        assert "loop_tail_high" in rule_names
        # Bad rule did NOT contribute an issue, but DID emit a log line.
        assert "bad" not in rule_names
        assert any("rule 'bad' raised" in r.message for r in caplog.records)

    def test_healthy_loop_produces_no_issues(self):
        agg = LatencyAggregator()
        _ingest_records(agg, [{"loop_dt_ms": 15.0, "t": time.time()} for _ in range(100)])
        d = LoopHealthDetector("teleop", target_period_ms=33.3)
        issues = d.check(agg, agg._records[-1])
        # Loop is at 15ms vs budget 33ms — no overrun, no tail issue, no
        # camera data so no staleness rule, fresh data so no no_data rule.
        assert issues == []


# ---------------------------------------------------------------------------
# Session integration — the detector must run on every commit and the
# active issues must end up in the snapshot envelope.
# ---------------------------------------------------------------------------


class TestSessionIntegration:
    def test_session_publishes_health_in_snapshot(self, tmp_path):
        """An overrunning session should publish a snapshot whose
        health.issues array contains overrun_high."""
        import json

        from lerobot.utils.latency import LatencySession

        session = LatencySession.from_config(
            enabled=True,
            loop_kind="teleop",
            target_fps=30,
            output_dir=tmp_path,
        )
        # Force the writer interval to 0 so every commit publishes.
        session.writer._interval_s = 0.0
        # Ingest 100 records that each exceed the 33ms budget. We can't
        # use real iterations easily (timing), so build records and
        # commit via the tracer's path: but tracer measures real time.
        # Simpler: ingest directly + manually run _after_commit.
        for _ in range(100):
            session.aggregator.ingest({"loop_dt_ms": 100.0, "t": time.time(), "overrun": True})
        session._after_commit()
        session.writer.flush()
        data = json.loads((tmp_path / "latency_snapshot.json").read_text())
        assert "health" in data
        assert "issues" in data["health"]
        rule_names = {i["rule"] for i in data["health"]["issues"]}
        assert "overrun_high" in rule_names

    def test_disabled_session_has_no_detector(self):
        """Disabled session is a no-op — no detector, no health overhead."""
        from lerobot.utils.latency import LatencySession

        session = LatencySession.disabled()
        # The disabled session inherits the parent's __init__ default-None
        # for _detector through skipping the parent init; the test verifies
        # there's no detector wired up.
        assert getattr(session, "_detector", None) is None


class TestHealthIssueDataclass:
    def test_to_dict_roundtrip(self):
        issue = HealthIssue(rule="x", severity="warn", message="hi")
        d = issue.to_dict()
        assert d == {"rule": "x", "severity": "warn", "message": "hi"}


class TestDefaultRulesHelper:
    def test_default_rules_returns_four(self):
        rules = default_rules()
        assert len(rules) == 4
        assert {r.name for r in rules} == {"overrun_high", "loop_tail_high", "camera_stale", "no_data"}
