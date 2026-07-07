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

"""SO-100 / SO-101 follower with the predictive-lookahead controller built in.

Same hardware family as :class:`SO107FollowerPredictive`, just 6 motors
instead of 7 (no ``forearm_roll``). All controller logic + lifecycle
plumbing comes from :class:`PredictiveLookaheadMixin`; this class only
nails the variant name + config type to the 6-motor SO-100 / SO-101
motor layout (handled by the base :class:`SOFollower`).

SO-100 and SO-101 share this class — they're functionally identical from
the controller's perspective (both 6 STS3215 motors over Feetech bus).

Treated as a SEPARATE EMBODIMENT from the plain ``so_follower`` for the
same reasons documented on :class:`SO107FollowerPredictive`: predictive
datasets have ``state(t) ≈ intent(t)``, plain ones have
``state(t) ≈ leader(t − τ)``, and training-time mixing is a real hazard.
"""

from __future__ import annotations

from ..predictive.mixin import PredictiveLookaheadMixin
from ..so_follower.so_follower import SOFollower
from .config_so_follower_predictive import SOFollowerPredictiveRobotConfig


class SOFollowerPredictive(PredictiveLookaheadMixin, SOFollower):
    """6-motor SO follower with predictive-lookahead controller always on.

    Use for SO-100 and SO-101 hardware; both share this class. All
    behavior comes from :class:`PredictiveLookaheadMixin`. For 7-motor
    SO-107 hardware, use :class:`SO107FollowerPredictive` instead.
    """

    config_class = SOFollowerPredictiveRobotConfig
    name = "so_follower_predictive"

    def configure(self) -> None:
        super().configure()
        # Mirrors SO107FollowerPredictive's tuning — same sts3215 motor
        # family + similar gripper kinematics, so the SO-107 sweep values
        # transfer directly. See ``SO107FollowerPredictive.configure`` for
        # the rationale (deadband zeroing + gripper register sweep).
        with self.bus.torque_disabled():
            for motor in self.bus.motors:
                self.bus.write("Minimum_Startup_Force", motor, 0)
                self.bus.write("CW_Dead_Zone", motor, 0)
                self.bus.write("CCW_Dead_Zone", motor, 0)
            if "gripper" in self.bus.motors:
                self.bus.write("P_Coefficient", "gripper", 24)
                self.bus.write("Maximum_Velocity_Limit", "gripper", 254)
                self.bus.write("Acceleration", "gripper", 0)
