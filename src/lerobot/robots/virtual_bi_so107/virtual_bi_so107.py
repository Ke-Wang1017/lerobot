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

"""Motor-less bimanual SO-107 follower — a virtual "perfect tracker".

Implements the same ``Robot`` contract as ``BiSO107Follower`` but holds
joint state as "the last commanded joint position" in NumPy arrays
instead of driving a Feetech bus. Used to drive the Cartesian-IK
pipeline through the real GUI / record / replay codepaths without
hardware:

* ``connect()`` seeds each arm at the canonical ready pose (shoulder
  down, elbow at 60°, wrist at -40°) — same well-conditioned pose
  ``test_bimanual_tracks_shape`` and the trajectory-closure benchmark
  use.
* ``send_action()`` stores the commanded joint positions verbatim.
* ``get_observation()`` returns those joint positions (perfect tracker).
* ``attach_teleop()`` accepts a bimanual Cartesian teleop and installs
  the SO-107 IK transform synchronously via the shared helper. Workspace
  clip is disabled (the real-arm safety bounds don't apply to a
  motor-less follower) — anything the URDF can kinematically reach
  plays cleanly.

Cameras are not faked — ``cameras`` is an empty dict so
``hasattr(robot, "cameras")`` stays truthy while
``len(robot.cameras) == 0`` cleanly skips ``lerobot_record.py``'s
image-writer machinery.

Hardcoded to SO-107. Worth promoting to a generic
``VirtualURDFFollower`` once a second robot description needs it.
"""

from __future__ import annotations

import logging
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES
from lerobot.types import ActionChunk, action_first_frame

from ..robot import Robot
from .config_virtual_bi_so107 import VirtualBiSO107FollowerConfig

logger = logging.getLogger(__name__)


class VirtualSO107Arm:
    """Per-arm motor-less SO-107 — holds ``_q`` in MOTOR_NAMES order.

    Lightweight: just enough surface for ``VirtualBiSO107Follower`` to
    route per-arm send_action / get_observation through it identically
    to the real ``SO107Follower`` siblings. Not separately registered;
    only constructed by ``VirtualBiSO107Follower``.
    """

    def __init__(self) -> None:
        self._q: np.ndarray = np.zeros(len(MOTOR_NAMES), dtype=float)
        self._connected: bool = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, q_init: np.ndarray) -> None:
        # No calibration step — the virtual arm has no motor state.
        assert q_init.shape == (len(MOTOR_NAMES),), (
            f"q_init must be ({len(MOTOR_NAMES)},), got {q_init.shape}"
        )
        self._q = q_init.astype(float, copy=True)
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def get_observation(self) -> dict[str, float]:
        assert self._connected, "VirtualSO107Arm.get_observation requires connect()"
        return {f"{m}.pos": float(self._q[i]) for i, m in enumerate(MOTOR_NAMES)}

    def send_action(self, action: dict[str, Any]) -> dict[str, float]:
        """Store the commanded joint positions; return what was applied.

        Missing motor keys are filled from the current ``_q`` so partial
        actions (e.g. gripper-only) do not zero out the rest of the arm —
        consistent with the real ``SO107Follower`` behaviour.
        """
        assert self._connected, "VirtualSO107Arm.send_action requires connect()"
        for i, m in enumerate(MOTOR_NAMES):
            key = f"{m}.pos"
            if key in action:
                self._q[i] = float(action[key])
        return {f"{m}.pos": float(self._q[i]) for i, m in enumerate(MOTOR_NAMES)}


class VirtualBiSO107Follower(Robot):
    """Bimanual virtual SO-107: two ``VirtualSO107Arm`` instances stacked.

    Caller-facing contract matches plain ``BiSO107Follower``: connect /
    get_observation / send_action / attach_teleop / disconnect. The
    bimanual prefix (``left_*`` / ``right_*``) is applied here, identical
    to the real follower.
    """

    config_class = VirtualBiSO107FollowerConfig
    name = "virtual_bi_so107"

    def __init__(self, config: VirtualBiSO107FollowerConfig):
        super().__init__(config)
        self.config = config

        self.left_arm = VirtualSO107Arm()
        self.right_arm = VirtualSO107Arm()
        # Kept as an empty dict (not absent) so ``hasattr(robot, "cameras")``
        # stays truthy in ``lerobot_record.py``; downstream code gates the
        # image-writer machinery on ``len(robot.cameras) > 0``.
        self.cameras: dict[str, Any] = {}

        # IK kinematics — same lazy try/except as ``BiSO107Follower``;
        # None when pin-pink is unavailable (Cartesian teleop then no-ops
        # with a warning, identical behaviour to the real follower).
        self._ik_kinematics: dict[str, Any] | None = None
        try:
            from lerobot.robots.so107_description.cartesian_ik import (
                make_so107_arm_kinematics,
            )
            from lerobot.robots.so107_description.joint_alignment import (
                LEFT_ARM_ALIGNMENT,
                RIGHT_ARM_ALIGNMENT,
            )

            self._ik_kinematics = {
                "left": make_so107_arm_kinematics(
                    LEFT_ARM_ALIGNMENT,
                    posture_cost=config.ik_posture_cost,
                    max_iters=config.ik_max_iters,
                ),
                "right": make_so107_arm_kinematics(
                    RIGHT_ARM_ALIGNMENT,
                    posture_cost=config.ik_posture_cost,
                    max_iters=config.ik_max_iters,
                ),
            }
        except Exception:
            logger.exception("%s: Cartesian-IK kinematics unavailable; Cartesian teleop disabled", self.name)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"left_{m}.pos": float for m in MOTOR_NAMES} | {f"right_{m}.pos": float for m in MOTOR_NAMES}

    @cached_property
    def observation_features(self) -> dict[str, type]:
        # No cameras, so observation == action features.
        return self._motors_ft

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    def connect(self, calibrate: bool = True) -> None:
        # ``calibrate`` is accepted for parity with the real robot's signature
        # but unused — the virtual arms have no motor calibration state.
        del calibrate
        # Seed each arm at the canonical ready pose (see module docstring).
        from lerobot.robots.so107_description.joint_alignment import (
            LEFT_ARM_ALIGNMENT,
            READY_POSE_URDF_DEG,
            RIGHT_ARM_ALIGNMENT,
            motor_pose_from_urdf,
        )

        self.left_arm.connect(motor_pose_from_urdf(READY_POSE_URDF_DEG, LEFT_ARM_ALIGNMENT))
        self.right_arm.connect(motor_pose_from_urdf(READY_POSE_URDF_DEG, RIGHT_ARM_ALIGNMENT))

    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()

    @property
    def is_calibrated(self) -> bool:
        # No calibration state — always True so ``connect(calibrate=True)``
        # never tries to run a no-op calibration routine.
        return True

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    def get_observation(self) -> dict[str, Any]:
        left_obs = self.left_arm.get_observation()
        right_obs = self.right_arm.get_observation()
        return {f"left_{k}": v for k, v in left_obs.items()} | {f"right_{k}": v for k, v in right_obs.items()}

    def send_action(self, action: dict[str, Any] | ActionChunk) -> dict[str, Any]:
        # Chunk-unaware (perfect tracker, no lookahead): fall back to
        # frames[0] — same shape the dataset writer expects.
        action_dict = action_first_frame(action) if isinstance(action, ActionChunk) else action

        left_action = {k.removeprefix("left_"): v for k, v in action_dict.items() if k.startswith("left_")}
        right_action = {k.removeprefix("right_"): v for k, v in action_dict.items() if k.startswith("right_")}
        left_sent = self.left_arm.send_action(left_action)
        right_sent = self.right_arm.send_action(right_action)
        return {f"left_{k}": v for k, v in left_sent.items()} | {
            f"right_{k}": v for k, v in right_sent.items()
        }

    def attach_teleop(self, teleop: Any) -> None:
        """For a bimanual Cartesian VR teleop, install the SO-107 IK
        transform on it; for anything else (joint-space leader, None,
        chunk-aware policy), no-op — ``send_action`` from the loop driver
        handles intent delivery directly.

        Synchronous install (no background-thread adapter): the perfect
        tracker has no real-time motor constraints and the scripted /
        future Quest teleops drive at the loop rate.

        Workspace clip is disabled: the real ``BiSO107Follower``'s safety
        box is a belt-and-braces guard against a real arm + jittery
        hand-tracking input driving into the table or self-collision; a
        motor-less follower has neither concern.
        """
        from lerobot.robots.so107_description.cartesian_ik import (
            SO107_WORKSPACE_UNBOUNDED_MAX,
            SO107_WORKSPACE_UNBOUNDED_MIN,
            build_so107_bimanual_ik_transform,
            is_so107_bimanual_cartesian_teleop,
        )

        if teleop is None or not is_so107_bimanual_cartesian_teleop(teleop):
            return

        assert self.is_connected, "attach_teleop requires the robot to be connected"
        assert hasattr(teleop, "set_action_transform"), (
            "a Cartesian teleop must expose set_action_transform()"
        )

        if self._ik_kinematics is None:
            logger.warning(
                "%s: Cartesian teleop attached but IK kinematics are unavailable "
                "(is pin-pink installed?) — the arms will not be driven.",
                self.name,
            )
            return

        transform = build_so107_bimanual_ik_transform(
            self._ik_kinematics,
            self.left_arm,
            self.right_arm,
            workspace_min=SO107_WORKSPACE_UNBOUNDED_MIN,
            workspace_max=SO107_WORKSPACE_UNBOUNDED_MAX,
        )
        teleop.set_action_transform(transform)
        logger.info("%s: installed Cartesian-IK transform into %s", self.name, type(teleop).__name__)
