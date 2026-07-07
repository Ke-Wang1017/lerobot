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

"""SO-107 follower with the predictive-lookahead controller built in.

This is treated as a SEPARATE EMBODIMENT from the plain ``so107_follower``
even though the hardware is identical, because the trained-policy contract
is materially different:

  * ``so107_follower``: ``state(t) ≈ leader(t − τ)`` — motor lag visible in
    every observation. Policy implicitly compensates τ via action chunks.

  * ``so107_follower_predictive``: ``state(t) ≈ intent(t)`` — controller
    transparently compensates motor τ. Dataset has aligned action/state
    pairs at matching timestamps. Policy never observes τ.

Mixing datasets across the two regimes in training is a real hazard:
training-time state-action alignment would be inconsistent and the policy
would output actions that lurch when deployed. The robot_type
separation makes this prevention automatic — recording / training /
inference all key off the same ``robot_type`` string.

Implementation: this class is now a thin composition over
:class:`PredictiveLookaheadMixin` and :class:`SO107Follower`. All
controller logic + lifecycle plumbing lives in the mixin / controller
modules under ``lerobot.robots.predictive``. Adding a new
``XXFollowerPredictive`` variant for different motor counts (e.g. SO-100
6-motor) follows the same pattern with no code duplication — see
``SOFollowerPredictive``.
"""

from __future__ import annotations

from ..predictive.mixin import PredictiveLookaheadMixin
from ..so_follower.so_follower import SO107Follower
from .config_so107_follower_predictive import SO107FollowerPredictiveRobotConfig


class SO107FollowerPredictive(PredictiveLookaheadMixin, SO107Follower):
    """SO-107 follower with predictive-lookahead controller always on.

    Treated as a distinct embodiment (own ``name`` / ``config_class``) so
    the dataset / policy contract is unambiguous. All behavior comes from
    :class:`PredictiveLookaheadMixin`; this class only nails the variant
    name + config type to the SO-107 motor layout.
    """

    config_class = SO107FollowerPredictiveRobotConfig
    name = "so107_follower_predictive"

    def configure(self) -> None:
        super().configure()
        with self.bus.torque_disabled():
            # Zero out the small-error responsiveness floor on every motor.
            # Factory defaults are Minimum_Startup_Force=0 and
            # CW/CCW_Dead_Zone=1. At ~0.088°/step resolution the 1-step
            # deadband sits below sensor noise, but write zeros explicitly
            # so a fresh motor matches every other predictive SO-107 setup
            # — otherwise the value silently persists in EEPROM from
            # whatever the last configurator (or stock firmware) wrote.
            # Applied here (not on plain SO107Follower) so existing
            # non-predictive users keep their defaults; opt-in only for
            # the predictive variant where the high control rate is more
            # sensitive to micro-deadband behavior.
            for motor in self.bus.motors:
                self.bus.write("Minimum_Startup_Force", motor, 0)
                self.bus.write("CW_Dead_Zone", motor, 0)
                self.bus.write("CCW_Dead_Zone", motor, 0)
            if "gripper" in self.bus.motors:
                # Per gripper register sweep on SO-107 hardware:
                #   * P=24 — 12% faster settle than the loop default 16,
                #     with zero overshoot at all tested reversal times
                #     (P=32 introduces ~0.4-unit overshoot; P=48 ~1 unit).
                #   * Acceleration=0 (no profile cap) was decisive: Acc=4
                #     collapsed peak velocity from 248 → 24 units/s.
                #   * Maximum_Velocity_Limit=254 — above the motor's hard
                #     physical ceiling (~270 u/s on this hardware), so
                #     setting it lower starts cutting in around 150.
                self.bus.write("P_Coefficient", "gripper", 24)
                self.bus.write("Maximum_Velocity_Limit", "gripper", 254)
                self.bus.write("Acceleration", "gripper", 0)
