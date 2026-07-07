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

"""
Pink-based 6-DOF IK with posture regularization.

Drop-in alternative to lerobot.model.kinematics.RobotKinematics that adds a
PostureTask (regularize toward seed) on top of the FrameTask. The existing
placo-based RobotKinematics only configures the FrameTask, which causes
null-space drift in 7-DOF arms — joints can swing wildly while the EE stays
on target. PinkKinematics keeps joints close to the seed configuration,
which is what's needed for smooth teleop and stable per-tick step commands.

API mirrors RobotKinematics: degrees in, degrees out, 4x4 SE(3) targets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lerobot.utils.import_utils import _pin_pink_available

if TYPE_CHECKING:
    import pink
    import pinocchio as pin
else:
    pin = None
    pink = None


class PinkKinematics:
    """6-DOF IK using pink (QP-based) with FrameTask + PostureTask."""

    def __init__(
        self,
        urdf_path: str,
        target_frame_name: str = "gripper_frame_link",
        joint_names: list[str] | None = None,
        posture_cost: float = 0.05,
        max_iters: int = 10,
        converge_threshold: float = 1e-4,
        dt: float = 1.0 / 30,
        solver: str = "proxqp",
    ):
        """
        Args:
            urdf_path: Path to the URDF.
            target_frame_name: Name of the EE frame in the URDF.
            joint_names: Optional explicit list of joint names. If None, uses all URDF joints.
            posture_cost: Weight on the PostureTask. Higher = tighter regularization to seed.
                Default 0.05 vs FrameTask cost of 1.0 makes posture a tiebreaker for null-space.
            max_iters: Max QP iterations per solve.
            converge_threshold: Stop iterating when velocity norm * dt < this (radians).
            dt: Integration timestep for pink (affects how aggressive each iteration is).
            solver: QP solver name (proxqp, quadprog, daqp, ...).
        """
        if not _pin_pink_available:
            raise ImportError(
                "PinkKinematics requires pin-pink. Install with `uv pip install pin-pink qpsolvers[open_source_solvers]`."
            )
        # Imports deferred so the module can be loaded even without pink installed.
        global pin, pink
        import pink as _pink
        import pinocchio as _pin

        pin = _pin
        pink = _pink
        from pink.tasks import FrameTask, PostureTask
        from pinocchio.robot_wrapper import RobotWrapper

        self._FrameTask = FrameTask
        self._PostureTask = PostureTask
        self._solve_ik = _pink.solve_ik

        self.urdf_path = urdf_path
        self.target_frame_name = target_frame_name
        self.posture_cost = posture_cost
        self.max_iters = max_iters
        self.converge_threshold = converge_threshold
        self.dt = dt
        self.solver = solver
        # Last solve's iteration count (0 = converged immediately, max_iters = budget hit).
        # Inspected by callers for verbose debug logging.
        self.last_n_iters: int = 0

        import os

        mesh_dir = os.path.dirname(urdf_path)
        self.robot = RobotWrapper.BuildFromURDF(urdf_path, [mesh_dir])
        # Seed values can land microscopically past URDF joint limits due to
        # floating-point error in our deg->rad conversion; pink's solve_ik
        # strictly validates seeds, so we clamp with a small safety inset.
        self.q_lo = self.robot.model.lowerPositionLimit.copy()
        self.q_hi = self.robot.model.upperPositionLimit.copy()
        self._eps = 1e-4

        # joint_names: ordered list of URDF joint names. If None, pinocchio default order.
        if joint_names is None:
            # First joint is "universe"; skip it.
            joint_names = list(self.robot.model.names)[1:]
        self.joint_names = joint_names

    def _clamp_q(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.q_lo + self._eps, self.q_hi - self._eps)

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        """Return 4x4 SE(3) world->EE transform for given joint positions (degrees)."""
        q = self._clamp_q(np.deg2rad(joint_pos_deg[: len(self.joint_names)]))
        pin.forwardKinematics(self.robot.model, self.robot.data, q)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        fid = self.robot.model.getFrameId(self.target_frame_name)
        return self.robot.data.oMf[fid].homogeneous.copy()

    def inverse_kinematics(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 1.0,
    ) -> np.ndarray:
        """6-DOF IK from current joints toward desired EE pose.

        Args:
            current_joint_pos: Joint positions in degrees (initial guess + posture target).
                If longer than joint_names, trailing entries are preserved (gripper passthrough).
            desired_ee_pose: 4x4 SE(3) world->EE.
            position_weight, orientation_weight: FrameTask cost weights.

        Returns:
            Joint positions in degrees. Same length as input current_joint_pos
            (trailing entries preserved unchanged).
        """
        n = len(self.joint_names)
        seed_deg = np.asarray(current_joint_pos, dtype=float)
        seed_q = self._clamp_q(np.deg2rad(seed_deg[:n]))

        config = pink.Configuration(self.robot.model, self.robot.data, seed_q)
        target_se3 = pin.SE3(desired_ee_pose[:3, :3], desired_ee_pose[:3, 3])
        frame_task = self._FrameTask(
            self.target_frame_name,
            position_cost=position_weight,
            orientation_cost=orientation_weight,
        )
        frame_task.set_target(target_se3)
        posture_task = self._PostureTask(cost=self.posture_cost)
        posture_task.set_target(seed_q)
        tasks = [frame_task, posture_task]

        n_iters_used = 0
        for it in range(self.max_iters):
            v = self._solve_ik(config, tasks, self.dt, solver=self.solver)
            n_iters_used = it + 1
            if float(np.linalg.norm(v)) * self.dt < self.converge_threshold:
                break
            config.integrate_inplace(v, self.dt)
            # integrate may overshoot URDF limits by float error; next solve_ik
            # would raise NotWithinConfigurationLimits. Re-clamp by rebuilding.
            config = pink.Configuration(self.robot.model, self.robot.data, self._clamp_q(config.q))
        self.last_n_iters = n_iters_used

        result_deg = seed_deg.copy()
        result_deg[:n] = np.rad2deg(config.q)
        return result_deg
