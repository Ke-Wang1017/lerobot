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

"""Trajectory-closure visualization for the bimanual SO-107 Cartesian stack.

Plays three reference shapes (heart 50 mm, circle 60 mm, square 50 mm)
through this PR's full stack — ``CartesianIKController`` +
``make_bimanual_ik_transform`` + ``JointMappedKinematics`` +
``TipOffsetKinematics`` + ``PinkKinematics`` — and plots commanded vs
achieved EE traces side by side.

**On the position error**: PinkKinematics runs ONE QP iteration per
tick, minimising ``||FK(q) - target||² + posture_regularization``. On
a continuously moving target the QP makes a step toward the target
each tick but never fully converges, so the achieved trace lags the
commanded one by some fraction of the per-tick displacement. That lag
shows up here as ~5 mm on the 60 mm circle (the fastest of the three —
2π·60 / 256 ≈ 1.47 mm/tick) and <1 mm on the slower heart / square.
It is NOT exact-IK error and would not appear under a "settle to
convergence" mode that runs multiple QP iterations per tick.

Rotation tracks flat at the IK floor (drift ~0.01-0.03 deg,
indistinguishable from the reference orientation in the plot) thanks
to the L6 + ``TIP_OFFSET`` structural fix in the per-arm kinematics.

Run from the repo root::

    .venv/bin/python benchmarks/cartesian_ik_trajectory_viz.py

Writes:
    src/lerobot/robots/virtual_bi_so107/docs/cartesian_ik_trajectory.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lerobot.robots.so107_description.cartesian_ik import (
    CartesianIKController,
    make_bimanual_ik_transform,
    make_so107_arm_kinematics,
)
from lerobot.robots.so107_description.joint_alignment import (
    LEFT_ARM_ALIGNMENT,
    MOTOR_NAMES,
    READY_POSE_URDF_DEG,
    RIGHT_ARM_ALIGNMENT,
    motor_pose_from_urdf,
)

URDF_SEED = READY_POSE_URDF_DEG
N_WAYPOINTS = 256  # matches test_pink_ik_trajectory.py's _N_WAYPOINTS
WARMUP_TICKS = 20

SHAPES = [
    ("heart 50 mm wide (45 mm tall)", "heart", 0.050),
    ("circle 60 mm radius", "circle", 0.060),
    ("square 50 mm side", "square", 0.050),
]

OUT_PATH = Path("src/lerobot/robots/virtual_bi_so107/docs/cartesian_ik_trajectory.png")


def _motor_seed(alignment) -> np.ndarray:
    return motor_pose_from_urdf(URDF_SEED, alignment)


def _plane_basis(ref: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = ref[:3, 3]
    flat = np.array([p[0], p[1], 0.0])
    norm = float(np.linalg.norm(flat))
    inward = -flat / norm if norm > 1e-6 else np.array([-1.0, 0.0, 0.0])
    perp = np.cross(inward, np.array([0.0, 0.0, 1.0]))
    perp /= np.linalg.norm(perp)
    return p, inward, perp


def _heart_unit(n: int) -> np.ndarray:
    """Parametric heart, normalized to fit a unit bbox, starting at (0, 0).

    The classic ``16 sin³t`` / ``13 cos t - 5 cos 2t - 2 cos 3t - cos 4t``
    parametrization — bumps at the top of the (x, y) plane, point at the
    bottom, cusp between the bumps at ``t=0``. Translated so the t=0
    point sits at the origin (so the trace starts at the seed EE) and
    scaled so the bounding-box max extent is 1.

    The cusp is a stress point: ``dx/dt = dy/dt = 0`` there, so the curve
    speeds up out of it and back into it at the end of the loop — a
    sharper test of the IK's per-call closure than a smooth circle.
    """
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = 16.0 * np.sin(t) ** 3
    y = 13.0 * np.cos(t) - 5.0 * np.cos(2 * t) - 2.0 * np.cos(3 * t) - np.cos(4 * t)
    pts = np.stack([x, y], axis=1)
    pts -= pts[0]
    return pts / float((pts.max(axis=0) - pts.min(axis=0)).max())


def _shape_positions(ref: np.ndarray, shape: str, size_m: float, n: int) -> list[np.ndarray]:
    p, inward, perp = _plane_basis(ref)
    if shape == "circle":
        r = size_m
        center = p + r * inward
        return [
            center + r * (np.cos(2 * np.pi * i / n) * -inward + np.sin(2 * np.pi * i / n) * perp)
            for i in range(n)
        ]
    if shape == "heart":
        unit = _heart_unit(n)
        # x_unit -> inward, y_unit -> perp so the heart plots upright (bumps
        # up, point down) in the (inward, perp) canvas used by the viz.
        return [p + size_m * (u[0] * inward + u[1] * perp) for u in unit]
    s = size_m
    corners = [p, p + s * inward, p + s * inward + s * perp, p + s * perp]
    per_edge = n // 4
    out: list[np.ndarray] = []
    for e in range(4):
        a, b = corners[e], corners[(e + 1) % 4]
        for k in range(per_edge):
            out.append(a + (b - a) * (k / per_edge))
    return out


def _run_shape(shape: str, size_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (commanded_xy, achieved_xy, achieved_rot_deg, basis_uv).

    ``basis_uv`` is the in-plane (inward, perp) basis used to project the
    EE positions onto a 2-D plot; reporting in that basis makes the shape
    show up canonically (centered, axis-aligned) regardless of the arm.
    """
    q_left = _motor_seed(LEFT_ARM_ALIGNMENT)
    q_right = _motor_seed(RIGHT_ARM_ALIGNMENT)
    gripper_idx = MOTOR_NAMES.index("gripper")
    grip_left = float(q_left[gripper_idx])
    grip_right = float(q_right[gripper_idx])

    left_kin = make_so107_arm_kinematics(LEFT_ARM_ALIGNMENT)
    right_kin = make_so107_arm_kinematics(RIGHT_ARM_ALIGNMENT)
    ref_left = left_kin.forward_kinematics(q_left)
    ref_right = right_kin.forward_kinematics(q_right)

    def _ctrl(kin, q):
        return CartesianIKController(
            kinematics=kin,
            motor_names=list(MOTOR_NAMES),
            q_init=q,
            workspace_min=(-10.0, -10.0, -10.0),
            workspace_max=(10.0, 10.0, 10.0),
            max_ee_step_m=10.0,
        )

    transform = make_bimanual_ik_transform(_ctrl(left_kin, q_left), _ctrl(right_kin, q_right))
    cmd_left = _shape_positions(ref_left, shape, size_m, N_WAYPOINTS)
    cmd_right = _shape_positions(ref_right, shape, size_m, N_WAYPOINTS)
    p_left, u_left, v_left = _plane_basis(ref_left)

    achieved: list[np.ndarray] = []
    rot_err: list[float] = []
    for cl, cr in zip(cmd_left, cmd_right, strict=True):
        dl = cl - p_left
        dr = cr - ref_right[:3, 3]
        action = {
            "left_enabled": 1.0,
            "left_target_x": float(dl[0]),
            "left_target_y": float(dl[1]),
            "left_target_z": float(dl[2]),
            "left_target_wx": 0.0,
            "left_target_wy": 0.0,
            "left_target_wz": 0.0,
            "left_gripper_pos": grip_left,
            "right_enabled": 1.0,
            "right_target_x": float(dr[0]),
            "right_target_y": float(dr[1]),
            "right_target_z": float(dr[2]),
            "right_target_wx": 0.0,
            "right_target_wy": 0.0,
            "right_target_wz": 0.0,
            "right_gripper_pos": grip_right,
        }
        out = transform(action)
        q_left_out = np.array([out[f"left_{m}.pos"] for m in MOTOR_NAMES])
        fk = left_kin.forward_kinematics(q_left_out)
        achieved.append(fk[:3, 3])
        r = ref_left[:3, :3].T @ fk[:3, :3]
        cos_a = max(-1.0, min(1.0, (float(np.trace(r)) - 1.0) * 0.5))
        rot_err.append(float(np.degrees(np.arccos(cos_a))))

    # Project (cmd, achieved) onto (inward, perp) for canonical plotting.
    def _project(points: list[np.ndarray]) -> np.ndarray:
        pts = np.asarray(points)
        return np.stack([(pts - p_left) @ u_left, (pts - p_left) @ v_left], axis=1)

    cmd_xy = _project(cmd_left)
    ach_xy = _project(achieved)
    return cmd_xy, ach_xy, np.asarray(rot_err), np.stack([u_left, v_left])


def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), height_ratios=(2, 1))
    print(f"{'shape':>22}  {'max pos err':>11}  {'max rot err':>11}")
    print(f"{'-' * 22}  {'-' * 11}  {'-' * 11}")

    for col, (label, shape, size_m) in enumerate(SHAPES):
        cmd, ach, rot, _ = _run_shape(shape, size_m)
        pos_err_mm = np.linalg.norm(cmd - ach, axis=1) * 1000.0
        max_pos = float(pos_err_mm[WARMUP_TICKS:].max())
        max_rot = float(rot[WARMUP_TICKS:].max())
        print(f"{label:>22}  {max_pos:>8.2f} mm  {max_rot:>8.3f} deg")

        ax = axes[0, col]
        ax.plot(cmd[:, 0] * 1000, cmd[:, 1] * 1000, color="C0", linewidth=2.5, label="commanded")
        ax.plot(
            ach[:, 0] * 1000,
            ach[:, 1] * 1000,
            color="C3",
            linewidth=1.0,
            linestyle="--",
            label="achieved (FK ∘ IK)",
        )
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlabel("inward (mm)")
        if col == 0:
            ax.set_ylabel("perp (mm)")
        ax.set_title(f"{label}\nmax pos err: {max_pos:.2f} mm")
        ax.legend(loc="upper right", fontsize=9)

        rax = axes[1, col]
        rax.plot(np.arange(len(rot)), rot, color="C2", linewidth=1.5)
        rax.set_xlim(0, len(rot))
        rax.set_ylim(0, max(0.05, max_rot * 1.3))
        rax.grid(alpha=0.3)
        rax.set_xlabel("waypoint")
        if col == 0:
            rax.set_ylabel("rotation drift (deg)")
        rax.set_title(f"max rot err: {max_rot:.3f}°")

    fig.suptitle(
        "SO-107 bimanual stack — trajectory closure (left arm, with L6+TIP_OFFSET fix)",
        fontsize=12,
    )
    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=120)
    print(f"\nplot -> {OUT_PATH}")


if __name__ == "__main__":
    main()
