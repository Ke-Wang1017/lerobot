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

import logging
from functools import cached_property
from typing import Any

from lerobot.teleoperators.so_leader import SO107Leader, SO107LeaderConfig

from ..teleoperator import Teleoperator
from .config_bi_so107_leader import BiSO107LeaderConfig

logger = logging.getLogger(__name__)


class BiSO107Leader(Teleoperator):
    """
    Bimanual SO-107 Leader Arms designed by TheRobotStudio and Hugging Face.
    7-axis arms with forearm_roll joint for advanced teleoperation.
    """

    config_class = BiSO107LeaderConfig
    name = "bi_so107_leader"

    def __init__(self, config: BiSO107LeaderConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = SO107LeaderConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_port,
            gripper_bounce=config.gripper_bounce,
            intervention_enabled=config.intervention_enabled,  # Only left arm has keyboard listener
        )

        right_arm_config = SO107LeaderConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_port,
            gripper_bounce=config.gripper_bounce,
            intervention_enabled=False,  # No keyboard listener on right arm
        )

        self.left_arm = SO107Leader(left_arm_config)
        self.right_arm = SO107Leader(right_arm_config)

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"left_{motor}.pos": float for motor in self.left_arm.bus.motors} | {
            f"right_{motor}.pos": float for motor in self.right_arm.bus.motors
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {f"left_{motor}.pos": float for motor in self.left_arm.bus.motors} | {
            f"right_{motor}.pos": float for motor in self.right_arm.bus.motors
        }

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    def get_action(self) -> dict[str, float]:
        action_dict = {}

        # Add "left_" prefix
        left_action = self.left_arm.get_action()
        action_dict.update({f"left_{key}": value for key, value in left_action.items()})

        # Add "right_" prefix
        right_action = self.right_arm.get_action()
        action_dict.update({f"right_{key}": value for key, value in right_action.items()})

        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # Remove "left_" prefix
        left_feedback = {
            key.removeprefix("left_"): value for key, value in feedback.items() if key.startswith("left_")
        }
        # Remove "right_" prefix
        right_feedback = {
            key.removeprefix("right_"): value for key, value in feedback.items() if key.startswith("right_")
        }

        if left_feedback:
            self.left_arm.send_feedback(left_feedback)
        if right_feedback:
            self.right_arm.send_feedback(right_feedback)

    def setup_gripper_bounce(self) -> None:
        """Setup gripper bounce to neutral (50% open) for both arms with weak torque."""
        logger.info("Setting up LEFT arm gripper bounce...")
        self.left_arm.set_gripper_bounce()

        logger.info("Setting up RIGHT arm gripper bounce...")
        self.right_arm.set_gripper_bounce()

    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()

    def get_teleop_events(self) -> dict[str, Any]:
        """Forward intervention events from left arm (which has the keyboard listener)."""
        return self.left_arm.get_teleop_events()

    def enable_torque(self, num_retry: int = 5) -> None:
        """Enable torque on both leader arms (for inverse-follow mode)."""
        self.left_arm.enable_torque(num_retry=num_retry)
        self.right_arm.enable_torque(num_retry=num_retry)

    def disable_torque(self) -> None:
        """Disable torque on both leader arms (for human control)."""
        self.left_arm.disable_torque()
        self.right_arm.disable_torque()

    def reset_intervention(self) -> None:
        """Reset intervention state for new episode on both arms."""
        self.left_arm.reset_intervention()
        self.right_arm.reset_intervention()
