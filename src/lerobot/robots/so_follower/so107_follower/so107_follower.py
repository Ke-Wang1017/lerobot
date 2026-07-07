#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

from ..so_follower_base import SOFollowerBase
from .config_so107_follower import SO107FollowerConfig


class SO107Follower(SOFollowerBase):
    """
    SO-107 Follower Arm designed by TheRobotStudio and Hugging Face.
    7-axis arm with forearm_roll joint.
    """

    config_class = SO107FollowerConfig
    name = "so107_follower"

    def __init__(self, config: SO107FollowerConfig):
        # Call Robot.__init__ directly to skip SOFollowerBase.__init__
        # We need to define our own motor configuration
        super(SOFollowerBase, self).__init__(config)
        self.config = config

        # SO-107 specific: 7 motors including forearm_roll
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "forearm_roll": Motor(4, "sts3215", norm_mode_body),  # Extra motor for SO-107
                "wrist_flex": Motor(5, "sts3215", norm_mode_body),
                "wrist_roll": Motor(6, "sts3215", norm_mode_body),
                "gripper": Motor(7, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)
        # Cache for all motor positions to handle communication failures when motors can't reach goal
        self._cached_motor_positions: dict[str, float] = {}
