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

"""Config fields for the predictive lookahead controller.

A dataclass mixin: any robot config that wants the predictive
controller inherits both its base ``RobotConfig`` subclass AND this
class, e.g.::

    @RobotConfig.register_subclass("so107_follower_predictive")
    @dataclass
    class SO107FollowerPredictiveRobotConfig(PredictiveControllerConfig, SOFollowerConfig):
        pass

All fields here carry validated defaults plus a ``metadata.description``
that the GUI surfaces as a hover tooltip on the profile editor — so the
user can hover over "velocity_estimator" or "max_lookahead_ms" and see
what each knob actually does without reading source.

Single source of truth for predictive controller tunables — bi-arm
configs inherit from the same class instead of redeclaring fields
(which previously meant new fields added to one config silently went
missing from the other).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class PredictiveControllerConfig:
    """Controller tunables. Inherit alongside the base robot config."""

    lookahead_ms: float = field(
        default=80.0,
        metadata={
            "description": (
                "Initial lookahead horizon in milliseconds — how far ahead of "
                '"now" the controller targets when writing Goal_Position. With '
                "adaptive=True the controller refines this online from the "
                "observed action→state lag. With adaptive=False, this value is "
                "the final lookahead. 80 ms is a good starting point for "
                "Feetech STS3215."
            ),
        },
    )

    max_lookahead_ms: float = field(
        default=150.0,
        metadata={
            "description": (
                "Hard cap on the adaptive lookahead. Set ABOVE your physical "
                "motor τ (~147 ms on bi_so107 P=16) so the adaptive loop can "
                "fully compensate; if set below motor τ the loop saturates "
                "with residual state-vs-intent lag. Going too far past motor τ "
                "introduces overshoot at direction reversals."
            ),
        },
    )

    corrector_alpha: float = field(
        default=1.0,
        metadata={
            "description": (
                "Predictor-corrector blend factor. 1.0 = pure leader-based "
                "prediction (raw_shifted only). <1.0 mixes in a velocity-"
                "extrapolated predictor from the action history → smoother "
                "motor stream at the cost of slightly more lag. 0.3 is the "
                "validated value on bi_so107."
            ),
        },
    )

    velocity_window_ms: float = field(
        default=70.0,
        metadata={
            "description": (
                "Time window (ms) over which leader velocity is estimated. Too "
                "short → noise amplified. Too long → phase lag for centered "
                "estimators. Window-based estimators (quad/linear/forward_diff/"
                "amp_gated_lp) read samples that fall inside this window each "
                "tick; stateful_lp uses it only as a stale-publish timeout."
            ),
        },
    )

    velocity_estimator: Literal["quad", "linear", "forward_diff", "amp_gated_lp", "stateful_lp"] = field(
        default="stateful_lp",
        metadata={
            "description": (
                "How the controller computes intent velocity for lookahead "
                "extrapolation. stateful_lp (default) is rate-invariant and "
                "recommended for any low-rate (≤60 Hz) dict-publish source "
                "such as a chunked policy. The window-based estimators "
                "(quad / linear / forward_diff / amp_gated_lp) work fine for "
                "high-rate teleop pull-paths but degrade at low publish rates "
                "where the window holds <3 samples — see per-option tooltips."
            ),
            "choice_descriptions": {
                "quad": (
                    "LSQ quadratic fit, slope evaluated at the most-recent "
                    "sample. Unbiased v(now) under acceleration. Window-based "
                    "→ noisy at <3 samples in window."
                ),
                "linear": (
                    "LSQ linear slope (original prototype behaviour). Centered "
                    "velocity estimate, biased under acceleration by ~a·w/2. "
                    "Kept for regression-debugging only."
                ),
                "forward_diff": (
                    "Two-sample causal difference at window tail "
                    "(v ≈ (p[-1] − p[-2])/dt). Unbiased but high noise — fine "
                    "for smooth high-rate sources, marginal at ≤30 Hz publish."
                ),
                "amp_gated_lp": (
                    "EMA-lowpass V_est with per-joint amplitude gating, "
                    "rebuilt per tick. WARNING: degrades to ungated raw "
                    "forward_diff when the window holds <3 samples (e.g. 90% "
                    "of ticks at 30 Hz × 70 ms window). Use stateful_lp at "
                    "low publish rates instead."
                ),
                "stateful_lp": (
                    "Per-publish 1st-order EMA velocity held as controller "
                    "state, α keyed on actual publish dt → filter cutoff is "
                    "velocity_lowpass_hz regardless of publish rate. "
                    "Rate-INVARIANT. Recommended for dict-publish policies / "
                    "teleops."
                ),
            },
        },
    )

    velocity_lowpass_hz: float = field(
        default=4.0,
        metadata={
            "description": (
                "Cutoff frequency (Hz) of the 1st-order EMA applied to the "
                "velocity estimate before extrapolation. Only consulted by "
                "amp_gated_lp and stateful_lp. Default 4 Hz attenuates the "
                "human 8-12 Hz tremor band while passing real motion "
                "(≤2 Hz dominant). Lower → more smoothing, more phase lag → "
                "less effective lookahead."
            ),
        },
    )

    amp_gate_lo: float = field(
        default=1.0,
        metadata={
            "description": (
                "Lower amplitude gate threshold (deg p2p, in joint units). "
                "ONLY consulted by amp_gated_lp. Below this peak-to-peak "
                "motion over the velocity window, lookahead is fully "
                "suppressed (motor_cmd = intent). stateful_lp has no gate."
            ),
        },
    )

    amp_gate_hi: float = field(
        default=3.0,
        metadata={
            "description": (
                "Upper amplitude gate threshold (deg p2p). Above this "
                "peak-to-peak motion, full lookahead is applied. Linear "
                "ramp between amp_gate_lo and amp_gate_hi. ONLY consulted "
                "by amp_gated_lp."
            ),
        },
    )

    control_rate_hz: float = field(
        default=200.0,
        metadata={
            "description": (
                "Background controller thread tick rate. Feetech bus "
                "comfortably sustains 170+ Hz at P=48 under sync_write load. "
                "200 Hz is the validated default. Going higher risks serial "
                "timeouts."
            ),
        },
    )

    adaptive: bool = field(
        default=True,
        metadata={
            "description": (
                "Adapt lookahead_ms online via cross-correlation of observed "
                "(intent, state) every 2 s. Recommended. Set False for a "
                "fixed lookahead (e.g. when you've calibrated the value for "
                "your hardware and want it pinned)."
            ),
        },
    )

    max_step_deg: float = field(
        default=3.0,
        metadata={
            "description": (
                "Safety clamp on the controller's per-step Δ-position vs the "
                "last sent action. Extrapolation can spike during sharp "
                "leader reversals; this caps the worst-case motor jolt. "
                "Normalized units (or degrees when use_degrees=True) per "
                "control tick. 3 units/tick at 200 Hz ≈ 600 units/s."
            ),
        },
    )
