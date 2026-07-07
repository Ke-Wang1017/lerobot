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

from dataclasses import dataclass

from ..config import RobotConfig
from ..predictive.config import PredictiveControllerConfig
from ..so_follower.config_so_follower import SOFollowerConfig


@RobotConfig.register_subclass("so_follower_predictive")
@dataclass
class SOFollowerPredictiveRobotConfig(PredictiveControllerConfig, RobotConfig, SOFollowerConfig):
    """Config for the 6-motor SO follower (SO-100 / SO-101) with the
    predictive-lookahead controller.

    Mirrors :class:`SO107FollowerPredictiveRobotConfig` but targets the
    6-motor SO-100 / SO-101 hardware (no ``forearm_roll``). Same motor
    family (Feetech STS3215), same bus protocol — the only difference is
    motor count + IDs, handled in the base ``SOFollower`` class.

    Registered under its own ``robot_type`` for the same reason as the
    SO-107 predictive variant: dataset action / state alignment differs
    from the plain ``so_follower``, and treating predictive vs.
    pass-forward as separate embodiments prevents accidental mixing of
    recording regimes in training. See
    :class:`SO107FollowerPredictiveRobotConfig` for the long-form
    rationale.

    All controller fields live on :class:`PredictiveControllerConfig`.
    """
