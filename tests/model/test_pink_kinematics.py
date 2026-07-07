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

"""Tests for PinkKinematics.

Skipped when pin-pink is not installed (it's an optional dep for the
quest_vr / pink-IK paths). When installed, we test against the vendored
SO-107 URDF as the standard 7-DOF reference arm.
"""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.utils.import_utils import _pin_pink_available

pytestmark = pytest.mark.skipif(not _pin_pink_available, reason="pin-pink (optional) not installed")


@pytest.fixture
def kinematics():
    from lerobot.model.pink_kinematics import PinkKinematics
    from lerobot.robots.so107_description import get_urdf_path

    return PinkKinematics(urdf_path=str(get_urdf_path()), target_frame_name="L7_1")


def _typical_joints() -> np.ndarray:
    # Mid-trajectory pose, comfortably inside joint limits.
    return np.array([0.0, -90.0, 60.0, 0.0, -40.0, 0.0, 0.0])


def test_joint_names_match_so107_urdf(kinematics):
    assert kinematics.joint_names == ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]


def test_forward_kinematics_returns_homogeneous(kinematics):
    T = kinematics.forward_kinematics(_typical_joints())
    assert T.shape == (4, 4)
    np.testing.assert_allclose(T[3], [0, 0, 0, 1])  # last row of SE(3)
    np.testing.assert_allclose(np.linalg.det(T[:3, :3]), 1.0, atol=1e-6)


def test_ik_returns_seed_when_target_equals_fk_of_seed(kinematics):
    """When the seed is already a valid solution, pink should not move it."""
    q_in = _typical_joints()
    T = kinematics.forward_kinematics(q_in)
    q_out = kinematics.inverse_kinematics(q_in, T)
    # Posture task pulls toward seed, FrameTask is already satisfied -> identity.
    np.testing.assert_allclose(q_out, q_in, atol=1e-3)


def test_ik_reaches_perturbed_target(kinematics):
    """Move the target 1cm in X; verify pink reaches it within sub-millimeter."""
    q_seed = _typical_joints()
    T_target = kinematics.forward_kinematics(q_seed)
    T_target[0, 3] += 0.01  # +1cm in URDF X
    q_out = kinematics.inverse_kinematics(q_seed, T_target)
    T_achieved = kinematics.forward_kinematics(q_out)
    pos_err_mm = float(np.linalg.norm(T_achieved[:3, 3] - T_target[:3, 3])) * 1000
    assert pos_err_mm < 1.0, f"position error {pos_err_mm:.3f}mm too large"


def test_ik_preserves_orientation_under_position_perturbation(kinematics):
    """6-DOF IK should hold orientation when only position changes."""
    from scipy.spatial.transform import Rotation as R

    q_seed = _typical_joints()
    T_target = kinematics.forward_kinematics(q_seed)
    T_target[0, 3] += 0.01
    q_out = kinematics.inverse_kinematics(q_seed, T_target)
    T_achieved = kinematics.forward_kinematics(q_out)
    R_err = T_achieved[:3, :3] @ T_target[:3, :3].T
    rot_err_deg = np.degrees(R.from_matrix(R_err).magnitude())
    assert rot_err_deg < 1.0, f"orientation error {rot_err_deg:.3f}° too large"


def test_ik_preserves_trailing_entries(kinematics):
    """An 8-entry seed (e.g., gripper as 8th) should pass through the 8th entry."""
    q_in_8 = np.concatenate([_typical_joints(), [42.0]])
    T = kinematics.forward_kinematics(q_in_8)
    q_out = kinematics.inverse_kinematics(q_in_8, T)
    assert q_out.shape == q_in_8.shape
    assert q_out[-1] == 42.0  # trailing gripper preserved unchanged


def test_ik_respects_joint_limits_on_seed_clamp(kinematics):
    """A seed sitting microscopically past a URDF limit (float overshoot) shouldn't raise."""
    # Pinocchio limit for joint 1 is +/- pi rad. Construct a seed in degrees that maps
    # to slightly past pi rad after np.deg2rad to simulate the overshoot pink would reject.
    q_at_limit = _typical_joints()
    q_at_limit[0] = float(np.degrees(np.pi) + 1e-7)  # 180.0...01 deg -> overshoot
    T = kinematics.forward_kinematics(_typical_joints())
    # Should NOT raise pink.exceptions.NotWithinConfigurationLimits.
    q_out = kinematics.inverse_kinematics(q_at_limit, T)
    assert q_out.shape == q_at_limit.shape


def test_posture_regularization_resists_unnecessary_drift(kinematics):
    """For a target reachable by many configs, pink should pick the one near the seed.

    This is the property that prevents the null-space-drift seen with placo's
    wrapper (which has no PostureTask).
    """
    q_seed = _typical_joints()
    T_target = kinematics.forward_kinematics(q_seed)
    # Same target as FK(seed); should return seed (not some wildly-different valid IK).
    q_out = kinematics.inverse_kinematics(q_seed, T_target)
    max_joint_drift = float(np.max(np.abs(q_out - q_seed)))
    assert max_joint_drift < 0.5, (
        f"max drift {max_joint_drift:.3f}° too large; posture regularization not working"
    )
