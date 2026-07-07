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

"""PinkKinematics tracking quality trade-off: shape × speed (bare-IK baseline).

Sweeps three trajectory shapes — a 30 mm-radius circle, a 50 mm-side
square, and a 60 mm linear back-and-forth — across a range of waypoint
counts (≡ peak EE speeds at a 30 Hz loop), and plots max position drift
against peak speed. Bare ``PinkKinematics`` — no per-arm joint-map, no
Cartesian controller, no bimanual transform — so this is the IK's
intrinsic operating envelope: the floor any downstream Cartesian-control
layer inherits.

Run on **SO-101** (5-DOF arm, the most-used SO-ARM variant), with
orientation weight = 0 because a 5-DOF arm cannot independently control
all three orientation axes. ``test_pink_ik_trajectory.py`` exercises both
SO-101 and SO-107; the bench picks SO-101 to characterise the IK on the
arm users actually have.

The shapes cover different stress regimes:
- **Circle**: smooth, continuous tangent — what the IK is "good at".
- **Square**: smooth sides, sharp 90° direction reversals at corners.
- **Line**: a 1-D back-and-forth (±30 mm along inward), with one direction
  reversal at each end.

Compare against ``benchmarks/cartesian_ik_tracking.py`` (on the VR teleop
branch) to see what each downstream layer adds.

Run from the repo root::

    .venv/bin/python benchmarks/pink_ik_tracking.py

Writes:
    src/lerobot/model/docs/pink_ik_tradeoff.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lerobot.model.pink_kinematics import PinkKinematics
from lerobot.robots.so101_description import get_urdf_path

# Match test_pink_ik_trajectory.py's SO-101 seed and target frame. SO-101's
# URDF has 6 joints but only 5 actuate the EE (the 6th is the gripper jaw
# on a branch off gripper_link, not on the path to gripper_frame_link).
URDF_TIP_FRAME = "gripper_frame_link"
SEED_DEG = np.array([0.0, 45.0, -90.0, 45.0, 0.0, 0.0])

LOOP_HZ = 30.0
WAYPOINT_COUNTS = (1024, 512, 256, 128, 64, 32)
WARMUP_TICKS = 20
TYPICAL_TELEOP_CM_S = (1.0, 10.0)

# Shape sizes — picked to be in-workspace at the SO-101 seed pose. The
# "size" semantics differ per shape (radius for circle, side for square,
# half-amplitude for line) but each produces a path of similar extent.
CIRCLE_RADIUS_M = 0.030
SQUARE_SIDE_M = 0.050
LINE_AMP_M = 0.030  # ±A from the seed along inward; must stay inside reach


def _plane_basis(t0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the seed EE point and ``(inward, perp)`` in-plane unit vectors."""
    p = t0[:3, 3]
    flat = np.array([p[0], p[1], 0.0])
    norm = float(np.linalg.norm(flat))
    inward = -flat / norm if norm > 1e-6 else np.array([-1.0, 0.0, 0.0])
    perp = np.cross(inward, np.array([0.0, 0.0, 1.0]))
    perp /= np.linalg.norm(perp)
    return p, inward, perp


def _se3(rot: np.ndarray, pos: np.ndarray) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = pos
    return t


def _circle_targets(t0: np.ndarray, n: int) -> list[np.ndarray]:
    p, inward, perp = _plane_basis(t0)
    center = p + CIRCLE_RADIUS_M * inward
    return [
        _se3(
            t0[:3, :3],
            center
            + CIRCLE_RADIUS_M * (np.cos(2 * np.pi * i / n) * -inward + np.sin(2 * np.pi * i / n) * perp),
        )
        for i in range(n)
    ]


def _square_targets(t0: np.ndarray, n: int) -> list[np.ndarray]:
    p, inward, perp = _plane_basis(t0)
    s = SQUARE_SIDE_M
    corners = [p, p + s * inward, p + s * inward + s * perp, p + s * perp]
    per_edge = n // 4
    out: list[np.ndarray] = []
    for e in range(4):
        a, b = corners[e], corners[(e + 1) % 4]
        for k in range(per_edge):
            out.append(_se3(t0[:3, :3], a + (b - a) * (k / per_edge)))
    return out


def _line_targets(t0: np.ndarray, n: int) -> list[np.ndarray]:
    """Back-and-forth along ``inward``: one cycle covers +A, back, -A, back."""
    p, inward, _perp = _plane_basis(t0)
    return [_se3(t0[:3, :3], p + LINE_AMP_M * np.sin(2 * np.pi * i / n) * inward) for i in range(n)]


SHAPES = {
    "circle 30 mm r": (_circle_targets, lambda n: 2 * np.pi * CIRCLE_RADIUS_M * LOOP_HZ / n),
    "square 50 mm s": (_square_targets, lambda n: 4 * SQUARE_SIDE_M * LOOP_HZ / n),
    "line 60 mm amp": (_line_targets, lambda n: 2 * np.pi * LINE_AMP_M * LOOP_HZ / n),
}


def _run_trajectory(targets: list[np.ndarray]) -> float:
    """Drive the targets through bare PinkKinematics; return max pos drift (mm).

    Position-only IK (``orientation_weight=0``) because a 5-DOF arm cannot
    independently control all three orientation axes.
    """
    kin = PinkKinematics(urdf_path=str(get_urdf_path()), target_frame_name=URDF_TIP_FRAME)
    q = SEED_DEG.copy()
    pos_err: list[float] = []
    for target in targets:
        q = kin.inverse_kinematics(q, target, position_weight=1.0, orientation_weight=0.0)
        fk = kin.forward_kinematics(q)
        pos_err.append(float(np.linalg.norm(fk[:3, 3] - target[:3, 3])))
    return max(pos_err[WARMUP_TICKS:]) * 1000.0


def main() -> None:
    kin_for_t0 = PinkKinematics(urdf_path=str(get_urdf_path()), target_frame_name=URDF_TIP_FRAME)
    t0 = kin_for_t0.forward_kinematics(SEED_DEG)

    results: dict[str, dict[str, list[float]]] = {name: {"speed": [], "pos": []} for name in SHAPES}

    print(f"{'shape':>16}  {'n_wp':>5}  {'speed':>11}  {'pos_drift':>10}")
    print(f"{'-' * 16}  {'-' * 5}  {'-' * 11}  {'-' * 10}")
    for name, (make_targets, speed_fn) in SHAPES.items():
        for n in WAYPOINT_COUNTS:
            targets = make_targets(t0, n)
            speed_cm_s = speed_fn(n) * 100.0
            pos_mm = _run_trajectory(targets)
            results[name]["speed"].append(speed_cm_s)
            results[name]["pos"].append(pos_mm)
            print(f"{name:>16}  {n:>5}  {speed_cm_s:>7.2f} cm/s  {pos_mm:>7.2f} mm")

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    colors = {"circle 30 mm r": "C0", "square 50 mm s": "C1", "line 60 mm amp": "C2"}
    markers = {"circle 30 mm r": "o", "square 50 mm s": "s", "line 60 mm amp": "^"}

    ax.axvspan(*TYPICAL_TELEOP_CM_S, alpha=0.12, color="green", label="typical teleop")
    ax.grid(alpha=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("Peak EE speed (cm/s, log)")

    for name, data in results.items():
        ax.plot(data["speed"], data["pos"], marker=markers[name], color=colors[name], label=name, linewidth=2)
    ax.set_ylabel("Max position drift (mm)")
    ax.set_title("Position tracking — bare PinkKinematics, SO-101 (5-DOF, position-only)")
    ax.legend(loc="upper left")

    plt.tight_layout()
    out_path = Path("src/lerobot/model/docs/pink_ik_tradeoff.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    print(f"\nplot -> {out_path}")


if __name__ == "__main__":
    main()
