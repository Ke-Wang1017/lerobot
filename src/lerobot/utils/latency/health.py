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

"""Always-on health checks for real-time loops.

``LoopHealthDetector`` watches a ``LatencyAggregator`` and emits structured
health issues when a configurable rule trips. Rules are intentionally
simple — boolean predicates over the aggregator's current state — so they
can run cheaply on every iteration and produce both:

  - a list of currently-active issues for the snapshot (the GUI badges),
  - a stream of rate-limited WARNING log lines (so operators running
    without the dashboard still hear about problems).

Why this lives next to ``LatencySession`` rather than the dashboard: the
in-memory aggregator already has every number a rule needs. Polling the
on-disk snapshot from a separate process would add a second copy of every
rule's thresholding logic and could miss transient breaches between
snapshot writes. The detector is single-threaded — it runs on the loop's
own thread after each ``end_iter()`` — so there are no synchronization
concerns.

The default rule set is deliberately small. We surface only the conditions
that *change what the operator should do next* (slow loop → look at the
breakdown; stale cameras → check producers; no data → something crashed).
Loud-by-default beats configurable-but-quiet.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lerobot.utils.latency.aggregator import LatencyAggregator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Issue + Rule types
# ---------------------------------------------------------------------------


@dataclass
class HealthIssue:
    """One active health problem on a loop. Carried in the snapshot so the
    GUI can render badges; also emitted as a WARNING log line (rate-limited
    per rule).

    Postconditions:
      - ``rule`` is a stable identifier (e.g. ``"overrun_high"``) — the
        GUI may use it as a CSS class or i18n key.
      - ``message`` is human-readable and includes the relevant number.
      - ``severity`` is one of ``"warn"`` | ``"error"``. Errors render
        red; warnings render yellow.
    """

    rule: str
    severity: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"rule": self.rule, "severity": self.severity, "message": self.message}


class Rule:
    """A health rule. Subclasses implement ``evaluate`` and return either
    None (everything fine) or a ``HealthIssue`` (rule is active).

    Subclasses must set ``name`` (stable identifier; doubles as the
    log-rate-limit key) and ``severity``. Keep evaluations cheap — they
    run on every iteration's ``end_iter``.
    """

    name: str = ""
    severity: str = "warn"

    def evaluate(
        self,
        aggregator: LatencyAggregator,
        record: dict[str, Any],
        target_period_ms: float | None,
    ) -> HealthIssue | None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Built-in rules
# ---------------------------------------------------------------------------


class OverrunRatioRule(Rule):
    """Fraction of recent iterations whose work time exceeded the target
    period. A sustained > 10% overrun means the loop cannot meet its
    target FPS; the policy / dataset / dashboard numbers are then based on
    a degraded loop and need explanation.
    """

    name = "overrun_high"
    severity = "warn"

    def __init__(self, threshold: float = 0.10, window_n: int = 120):
        # window_n iterations ≈ 4 s at 30 fps; long enough to filter
        # one-off hiccups, short enough that the badge clears quickly
        # once the loop recovers.
        assert 0.0 < threshold <= 1.0, f"threshold must be in (0, 1], got {threshold}"
        assert window_n >= 1, f"window_n must be >= 1, got {window_n}"
        self.threshold = threshold
        self.window_n = window_n

    def evaluate(self, aggregator, record, target_period_ms):
        if target_period_ms is None:
            return None
        # Use the aggregator's tail; deque has cheap negative indexing.
        records = list(aggregator._records)[-self.window_n :]
        if len(records) < min(10, self.window_n):
            return None
        overruns = sum(1 for r in records if r.get("loop_dt_ms", 0) > target_period_ms)
        ratio = overruns / len(records)
        if ratio < self.threshold:
            return None
        return HealthIssue(
            rule=self.name,
            severity=self.severity,
            message=(
                f"Overrun {ratio * 100:.0f}% over last {len(records)} iters "
                f"(loop > {target_period_ms:.1f} ms budget). "
                f"Check per-stage breakdown."
            ),
        )


class LoopBudgetRule(Rule):
    """``p95(loop_dt_ms)`` exceeds ``p95_factor`` × target period. Tail
    behaviour matters even when the median is fine — a 5% tail at 2×
    budget will still cause occasional missed actions / dropped frames.
    """

    name = "loop_tail_high"
    severity = "warn"

    def __init__(self, p95_factor: float = 1.5):
        assert p95_factor > 1.0, f"p95_factor must be > 1, got {p95_factor}"
        self.p95_factor = p95_factor

    def evaluate(self, aggregator, record, target_period_ms):
        if target_period_ms is None:
            return None
        if len(aggregator) < 20:
            return None
        p95 = aggregator.percentile("loop_dt_ms", 95)
        if p95 != p95:  # NaN guard — aggregator returns NaN when the key is absent
            return None
        threshold = target_period_ms * self.p95_factor
        if p95 <= threshold:
            return None
        return HealthIssue(
            rule=self.name,
            severity=self.severity,
            message=(
                f"loop_dt p95 = {p95:.1f} ms (> {self.p95_factor:.1f}× budget "
                f"{target_period_ms:.1f} ms). Tail iterations are slow."
            ),
        )


class CameraStalenessRule(Rule):
    """Per-camera staleness exceeds ``p95_factor`` × that camera's own
    observed period. Scaling to the camera's actual period (not the loop
    target) makes the rule meaningful for cameras running below their
    configured FPS — the warning fires when staleness is high *relative
    to what this camera can produce*, not relative to an arbitrary target.
    """

    name = "camera_stale"
    severity = "warn"

    def __init__(self, p95_factor: float = 2.0, min_period_ms: float = 16.0):
        assert p95_factor > 1.0, f"p95_factor must be > 1, got {p95_factor}"
        # When the camera period is implausibly small (e.g. first few
        # iterations before the consumer has cycled), don't issue an
        # alarm based on a thin estimate. 16 ms ≈ 60 Hz, a sane floor.
        self.p95_factor = p95_factor
        self.min_period_ms = min_period_ms

    def evaluate(self, aggregator, record, target_period_ms):
        if len(aggregator) < 20:
            return None
        # Find every camera that's been seen.
        stale_keys: list[str] = []
        for k in record or {}:
            if k.startswith("cam_") and k.endswith("_stale_ms"):
                stale_keys.append(k)
        if not stale_keys:
            return None
        # Report ONE issue covering all offending cameras so the badge
        # stays compact. Each camera's contribution carries its name +
        # numbers so the operator can drill in directly.
        offenders = []
        for stale_key in stale_keys:
            cam = stale_key[len("cam_") : -len("_stale_ms")]
            period_p50 = aggregator.percentile(f"cam_{cam}_period_ms", 50)
            stale_p95 = aggregator.percentile(stale_key, 95)
            # NaN guard — aggregator returns NaN when the key is absent.
            if period_p50 != period_p50 or stale_p95 != stale_p95:
                continue
            if period_p50 < self.min_period_ms:
                continue
            if stale_p95 > period_p50 * self.p95_factor:
                offenders.append(f"{cam} ({stale_p95:.0f}ms vs period {period_p50:.0f}ms)")
        if not offenders:
            return None
        return HealthIssue(
            rule=self.name,
            severity=self.severity,
            message=(f"Camera staleness p95 > {self.p95_factor:.1f}× period: " + ", ".join(offenders)),
        )


class NoDataRule(Rule):
    """The aggregator hasn't seen a record in ``timeout_s`` seconds — the
    loop has likely stalled, the subprocess crashed, or the writer is
    hung. Higher-severity than the other rules because the data the user
    is looking at is now stale.
    """

    name = "no_data"
    severity = "error"

    def __init__(self, timeout_s: float = 5.0):
        assert timeout_s > 0, f"timeout_s must be > 0, got {timeout_s}"
        self.timeout_s = timeout_s

    def evaluate(self, aggregator, record, target_period_ms):
        if len(aggregator) == 0:
            return None
        last_t = aggregator._records[-1].get("t")
        if last_t is None:
            return None
        age_s = time.time() - last_t
        if age_s < self.timeout_s:
            return None
        return HealthIssue(
            rule=self.name,
            severity=self.severity,
            message=(f"No records for {age_s:.1f}s. Loop may have stalled or the subprocess exited."),
        )


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


def default_rules() -> list[Rule]:
    """The rule set installed by ``LatencySession`` when monitoring is on.

    Conservative thresholds — they should fire only when something is
    obviously wrong (sustained > 10% overrun, tail > 1.5× budget, camera
    staleness > 2× the camera's own period). Easier to tighten later
    than to walk back a noisy default.
    """
    return [
        OverrunRatioRule(),
        LoopBudgetRule(),
        CameraStalenessRule(),
        NoDataRule(),
    ]


class LoopHealthDetector:
    """Runs a set of rules after every committed iteration.

    Preconditions:
      - ``check()`` is called once per iteration commit, on the loop's
        own thread. Not safe for concurrent calls on the same instance.

    Postconditions:
      - Returns the list of currently-active ``HealthIssue``s.
      - For each issue whose rule hasn't logged in ``alert_interval_s``
        seconds, a WARNING log line is emitted. (Rules that clear and
        re-fire later will re-log; the interval is per-rule, not per-fire.)
    """

    def __init__(
        self,
        loop_kind: str,
        target_period_ms: float | None,
        rules: Iterable[Rule] | None = None,
        alert_interval_s: float = 60.0,
    ):
        self.loop_kind = loop_kind
        self.target_period_ms = target_period_ms
        self.rules: list[Rule] = list(rules) if rules is not None else default_rules()
        self.alert_interval_s = alert_interval_s
        self._last_alert_at: dict[str, float] = {}

    def check(
        self,
        aggregator: LatencyAggregator,
        record: dict[str, Any] | None,
    ) -> list[HealthIssue]:
        """Evaluate every rule against the current aggregator state.

        ``record`` is the most recent committed record; rules may use it
        to enumerate dynamic fields (e.g. which cameras are present).
        Returns the list of active issues. The caller (LatencySession)
        decides what to do with them; this method also emits per-rule
        rate-limited WARNING log lines so headless operators see them.
        """
        active: list[HealthIssue] = []
        now = time.time()
        for rule in self.rules:
            try:
                issue = rule.evaluate(aggregator, record or {}, self.target_period_ms)
            except Exception as e:  # noqa: BLE001
                # A buggy rule must not crash the loop. Log once and skip.
                logger.warning(
                    "[health] rule %r raised %s; skipping for this iteration",
                    rule.name,
                    e,
                )
                continue
            if issue is None:
                continue
            active.append(issue)
            last_at = self._last_alert_at.get(rule.name, 0.0)
            if now - last_at >= self.alert_interval_s:
                self._last_alert_at[rule.name] = now
                logger.warning("[health:%s:%s] %s", self.loop_kind, issue.rule, issue.message)
        return active
