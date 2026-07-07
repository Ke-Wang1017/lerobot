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


@RobotConfig.register_subclass("so107_follower_predictive")
@dataclass
class SO107FollowerPredictiveRobotConfig(PredictiveControllerConfig, RobotConfig, SOFollowerConfig):
    """Config for the SO-107 follower with predictive-lookahead controller.

    Registered under its own ``robot_type`` because the trained-policy
    contract differs from the plain ``so107_follower``: datasets recorded
    against this robot have ``action(t) ≈ state(t)`` (the operator's raw
    intent aligns with the motor's actual position because the controller
    transparently compensates motor τ), whereas plain so107_follower
    datasets have ``state(t) ≈ leader(t − τ)``. Treating predictive vs.
    pass-forward as separate embodiments prevents accidental mixing of
    incompatible recording regimes in training.

    All controller fields live on :class:`PredictiveControllerConfig`;
    inherit it here for the entire knob set with one line instead of
    redeclaring (which historically went silently out of sync with the
    bi-arm config when fields were added).
    """
