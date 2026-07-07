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

"""End-to-end functional check for the Pink inverse kinematics.

A non-VR proof that ``PinkKinematics`` works: command a scripted Cartesian
shape, run IK for each waypoint (seeding each solve from the previous one,
the way live teleop drives it), forward-kinematics the result back, and
assert the achieved end-effector trace reproduces the commanded shape.

``FK(IK(target)) == target`` is the round trip. For a position-controlled
arm the observed joint state equals the commanded joints, so forward
kinematics of that state is exactly what a robot or sim would report — no
physics sim is needed (it would only add servo tracking-lag, irrelevant to
IK correctness).

Three shapes at two sizes — a 30 mm circle, a 60 mm circle and a 50 mm
square — exercise the IK across trajectory shape and scale, so a pass is
not an artifact of one tuned trajectory.

Two reference arms with different controllable task-space DOF:

- **SO-101** — a 5-DOF arm (``shoulder_pan, shoulder_lift, elbow_flex,
  wrist_flex, wrist_roll``; the 6th joint is the gripper jaw and does not
  move the EE frame). It cannot independently hold all three orientation
  axes, so its shapes are driven in position only.
- **SO-107** — a 6-DOF arm (SO-101's five plus ``forearm_roll``). Full
  SE(3) pose control, so its shapes are driven with a held orientation and
  both position and orientation are asserted.

``test_ee_jacobian_rank_proves_arm_dof`` is the definitional proof of those
DOF counts: the controllable task DOF at a pose equals the rank of the 6xN
end-effector Jacobian — 5 for SO-101, 6 for SO-107.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass

import numpy as np
import pytest

from lerobot.utils.import_utils import _pin_pink_available

pytestmark = pytest.mark.skipif(not _pin_pink_available, reason="pin-pink (optional) not installed")


@dataclass(frozen=True)
class _ArmCase:
    """A reference arm and a non-singular seed pose to exercise it from.

    ``task_dof`` is the controllable task-space DOF — 5 for the SO-101 arm,
    6 for SO-107. It drives both the Jacobian-rank assertion and whether the
    shapes are orientation-controlled (6-DOF) or position-only (5-DOF).
    """

    robot_id: str
    ee_frame: str
    seed_deg: tuple[float, ...]
    task_dof: int

    @property
    def controls_orientation(self) -> bool:
        return self.task_dof == 6


# Seeds are non-singular mid-range poses (the Jacobian-rank test would fail
# at a singular seed).
#
# KNOWN ISSUE: SO-107's end-effector frame ``L7_1`` is provisional. The
# CAD-exported URDF's tip frame does not yet sit exactly at the physical
# gripping point, so the *absolute* EE position needs further calibration.
# It does not affect these tests — they check that the EE *reaches commanded
# poses* (a relative property, around whatever point ``L7_1`` is).
_ARMS = [
    _ArmCase("so101", "gripper_frame_link", (0.0, 45.0, -90.0, 45.0, 0.0, 0.0), task_dof=5),
    _ArmCase("so107", "L7_1", (0.0, -90.0, 60.0, 0.0, -40.0, 0.0, 0.0), task_dof=6),
]


@dataclass(frozen=True)
class _Shape:
    """A scripted Cartesian shape. ``size_m`` is a circle radius or square side."""

    name: str
    kind: str  # "circle" | "square"
    size_m: float


_SHAPES = [
    _Shape("circle-30mm", "circle", 0.030),
    _Shape("circle-60mm", "circle", 0.060),
    _Shape("square-50mm", "square", 0.050),
]

# PinkKinematics is a velocity-resolved IK with a small per-call iteration
# budget (built for ~30 Hz teleop), so each shape is walked in many small
# steps starting from the seed pose — a large jump would not converge in one
# call. The achieved trace lags the command by ~1-2 mm (steady-state tracking
# lag of a velocity IK), comfortably inside the tolerance.
_N_WAYPOINTS = 256
_POS_TOL_M = 3.0e-3
_ROT_TOL_DEG = 1.5
_Z_AXIS = np.array([0.0, 0.0, 1.0])


def _plane_basis(t0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seed EE position + two in-plane unit vectors for the shape.

    Shapes are traced in the horizontal plane through the seed EE.
    ``inward`` points from the seed EE back toward the base; placing the
    shape there keeps it inside the reachable workspace for either arm.
    """
    p = t0[:3, 3].copy()
    flat = np.array([p[0], p[1], 0.0])
    inward = -flat / np.linalg.norm(flat)
    perp = np.cross(inward, _Z_AXIS)
    return p, inward, perp


def _shape_targets(shape: _Shape, t0: np.ndarray, n: int) -> list[np.ndarray]:
    """``n`` SE(3) targets tracing ``shape``, orientation held at the seed's.

    The loop starts at the seed EE position so the first IK solve has no
    jump to converge.
    """
    p, inward, perp = _plane_basis(t0)
    positions: list[np.ndarray] = []
    if shape.kind == "circle":
        r = shape.size_m
        center = p + r * inward  # the seed EE sits on the circle at angle 0
        for i in range(n):
            theta = 2.0 * math.pi * i / n
            positions.append(center + r * (math.cos(theta) * -inward + math.sin(theta) * perp))
    elif shape.kind == "square":
        s = shape.size_m
        corners = [p, p + s * inward, p + s * inward + s * perp, p + s * perp]
        per_edge = n // 4
        for e in range(4):
            a, b = corners[e], corners[(e + 1) % 4]
            for k in range(per_edge):
                positions.append(a + (b - a) * (k / per_edge))
    else:
        raise AssertionError(f"unknown shape kind {shape.kind!r}")

    targets = []
    for pos in positions:
        target = np.eye(4)
        target[:3, :3] = t0[:3, :3]
        target[:3, 3] = pos
        targets.append(target)
    return targets


def _kinematics(arm: _ArmCase):
    from lerobot.model.pink_kinematics import PinkKinematics

    mod = importlib.import_module(f"lerobot.robots.{arm.robot_id}_description")
    return PinkKinematics(urdf_path=str(mod.get_urdf_path()), target_frame_name=arm.ee_frame)


@pytest.mark.parametrize("arm", _ARMS, ids=[a.robot_id for a in _ARMS])
def test_ee_jacobian_rank_proves_arm_dof(arm: _ArmCase):
    """Controllable task-space DOF equals the rank of the EE Jacobian.

    This is the definitional proof that SO-101 is a 5-DOF arm and SO-107 a
    6-DOF arm — independent of any trajectory.
    """
    import pinocchio as pin

    kin = _kinematics(arm)
    model, data = kin.robot.model, kin.robot.data
    seed_q = np.deg2rad(np.asarray(arm.seed_deg, dtype=float))
    assert len(seed_q) == len(kin.joint_names), "seed length must match the URDF joint count"

    frame_id = model.getFrameId(arm.ee_frame)
    jac = pin.computeFrameJacobian(model, data, seed_q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    assert jac.shape == (6, len(kin.joint_names))

    rank = int(np.linalg.matrix_rank(jac, tol=1e-6))
    assert rank == arm.task_dof, (
        f"{arm.robot_id}: EE Jacobian rank {rank} != expected {arm.task_dof} task DOF "
        f"(is the seed pose singular?)"
    )


@pytest.mark.parametrize("arm", _ARMS, ids=[a.robot_id for a in _ARMS])
@pytest.mark.parametrize("shape", _SHAPES, ids=[s.name for s in _SHAPES])
def test_ik_traces_a_cartesian_shape(shape: _Shape, arm: _ArmCase):
    """Command a scripted Cartesian shape, solve IK per waypoint, FK back, and
    assert the achieved trace reproduces the commanded shape."""
    kin = _kinematics(arm)
    seed_deg = np.asarray(arm.seed_deg, dtype=float)
    t0 = kin.forward_kinematics(seed_deg)
    assert np.all(np.isfinite(t0)), "FK of the seed pose is non-finite"

    targets = _shape_targets(shape, t0, _N_WAYPOINTS)
    orientation_weight = 1.0 if arm.controls_orientation else 0.0

    q = seed_deg.copy()
    achieved: list[np.ndarray] = []
    for target in targets:
        q = kin.inverse_kinematics(q, target, position_weight=1.0, orientation_weight=orientation_weight)
        achieved.append(kin.forward_kinematics(q))

    # 1. Every commanded waypoint is reached in position — this is what proves
    #    the achieved trace reproduces the commanded shape, whatever the shape.
    for i, (cmd, got) in enumerate(zip(targets, achieved, strict=True)):
        pos_err_m = float(np.linalg.norm(got[:3, 3] - cmd[:3, 3]))
        assert pos_err_m < _POS_TOL_M, (
            f"{arm.robot_id}/{shape.name}: waypoint {i} position error "
            f"{pos_err_m * 1000:.3f} mm exceeds {_POS_TOL_M * 1000:.1f} mm"
        )

    # 2. Orientation is held across the trajectory — only a 6-DOF arm can.
    if arm.controls_orientation:
        from scipy.spatial.transform import Rotation

        for i, got in enumerate(achieved):
            rot_err = got[:3, :3] @ t0[:3, :3].T
            rot_err_deg = float(np.degrees(Rotation.from_matrix(rot_err).magnitude()))
            assert rot_err_deg < _ROT_TOL_DEG, (
                f"{arm.robot_id}/{shape.name}: waypoint {i} orientation drifted {rot_err_deg:.3f} deg"
            )

    # 3. For a circle, the achieved trace is provably circular: every point
    #    sits ~radius from the centroid.
    if shape.kind == "circle":
        pos = np.array([t[:3, 3] for t in achieved])
        radii = np.linalg.norm(pos - pos.mean(axis=0), axis=1)
        np.testing.assert_allclose(radii, shape.size_m, atol=_POS_TOL_M)
