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

"""Removing concrete robot drivers must not drag hardware into the train/eval path.

``lerobot-train`` / ``lerobot-eval`` import ``lerobot.envs``, which references only the
``RobotConfig`` / ``TeleoperatorConfig`` *base* classes (for ``HILSerlRobotEnvConfig``).
No concrete device module should be imported as a side effect, and the relocated
``robot_kinematic_processor`` must resolve from its new home.
"""

import sys


def test_import_envs_does_not_pull_concrete_devices():
    import lerobot.envs  # noqa: F401

    leaked = [
        m
        for m in sys.modules
        if m.startswith(
            (
                "lerobot.robots.so_follower",
                "lerobot.motors.feetech",
                "lerobot.motors.dynamixel",
                "lerobot.cameras.opencv",
                "lerobot.cameras.realsense",
            )
        )
    ]
    assert leaked == [], f"concrete device modules leaked into the env import graph: {leaked}"


def test_hilserl_env_config_defaults_have_no_robot():
    from lerobot.envs import HILSerlRobotEnvConfig

    cfg = HILSerlRobotEnvConfig()
    assert cfg.robot is None
    assert cfg.teleop is None


def test_relocated_kinematic_processor_importable():
    # Moved out of robots/so_follower/ (deleted) into robots/ top-level; DAgger (rl/) needs it.
    from lerobot.robots.robot_kinematic_processor import (  # noqa: F401
        EEBoundsAndSafety,
        InverseKinematicsRLStep,
    )
