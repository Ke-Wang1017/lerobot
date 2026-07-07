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

"""Cartesian end-effector teleop -> joint commands for one SO-107 arm.

A :class:`CartesianIKController` turns the eight EE-delta keys a Quest VR
controller emits (``enabled, target_x/y/z, target_wx/wy/wz, gripper_pos``)
into a ``{motor}.pos`` joint dict, by latching a reference pose on clutch
engage and running IK on the composed target.

It is deliberately a plain stateful callable, not a ``ProcessorStep``: a
robot installs it into a Cartesian teleop via ``attach_teleop`` so that
``teleop.get_action()`` returns joint commands, and the upstream teleop /
record / replay loops stay untouched. Because of that it must run with no
per-tick robot observation — so the reference pose is the forward
kinematics of the *last commanded* joints and the IK seed is likewise the
last command. Both are tracked internally; the only external input is the
one-time ``q_init`` seed handed in at construction (the arm's joint
configuration when the teleop is attached).

``JointMappedKinematics`` adapts a CAD-exported URDF whose joint-zero does
not match the motor calibration zero (see :mod:`joint_alignment`); it lets
the controller work entirely in motor-degree space while pinocchio runs in
URDF space. The upstream kinematic ProcessorSteps are not used or modified.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from lerobot.utils.rotation import Rotation

# Optional pin-pink dependency: bind the QP "no solution" exception at
# import-time, with a sentinel fallback so ``except _NoSolutionFound:``
# is syntactically valid even without pin-pink installed. The fallback
# never fires (it's only ever raised from inside Pink's solver), so the
# guard collapses to a no-op in the no-pink case.
try:
    from pink.exceptions import NoSolutionFound as _NoSolutionFound
except ImportError:

    class _NoSolutionFound(Exception):  # noqa: N818 — mirrors pink.exceptions.NoSolutionFound; sentinel never fires in no-pink envs
        pass


from .joint_alignment import (
    MOTOR_NAMES,
    TIP_OFFSET,
    URDF_ANCHOR_FRAME,
    URDF_JOINT_NAMES,
    JointAlignment,
)

logger = logging.getLogger(__name__)

# Reachable EE box for the SO-107, robot base frame, meters. Hand-tuned
# estimates from teleop trials, NOT measured — IK targets are clipped here
# so an out-of-reach Quest command can't drive the solver into garbage.
# TODO(so107): replace with a measured/calibrated workspace once the
# guided-calibration tool lands (see gui/TODO.md).
SO107_WORKSPACE_MIN: tuple[float, float, float] = (-0.20, -0.35, +0.03)
SO107_WORKSPACE_MAX: tuple[float, float, float] = (+0.25, +0.05, +0.36)

# Backstop cap on per-tick EE position change (m). The Quest teleop already
# bounds and de-glitches its own deltas; this only catches anything past that.
_MAX_EE_STEP_M: float = 0.10

# Safety backstop: IK can swing a joint wildly near a singular config even
# for a within-bounds EE target. A solve that moves any arm joint more than
# this in one tick is treated as a glitch and held — and logged, so a
# throttled arm shows up in the run log rather than being a silent mystery.
# 20 deg/tick is 600 deg/s at a 30 Hz loop — far above smooth teleop.
# TODO: provisional hardcoded limit; see teleoperators/quest_vr/TODO.md
# ("make safety limits explicit and configurable").
_MAX_JOINT_STEP_DEG: float = 20.0

# Per-second ramp rate for the operator-commanded ``reset`` motion. The
# reset button is hold-to-ramp: each call the controller advances every
# arm joint toward the attach-time seed by at most
# ``_RESET_RAMP_RATE_DEG_PER_SEC * dt`` degrees, where ``dt`` is wall-
# clock since the previous reset call. Rate-based (not per-call) because
# the IK transform is polled at different rates on different follower
# paths — ~30 Hz inline on the plain follower's main-loop
# ``get_action()`` vs 90 Hz on the predictive follower's
# ``BimanualCartesianIKAdapter`` thread. A per-call step would give the
# operator a 3× faster ramp on predictive, a noticeable feel difference.
# 30 °/s = 3 s for a 90° joint move, slow enough to release if the
# trajectory heads somewhere bad. Safe on a plain follower with
# unbounded ``max_relative_target``; the predictive follower's own rate
# limiter further constrains motor-side.
_RESET_RAMP_RATE_DEG_PER_SEC: float = 30.0
# Cap on the dt fed into the ramp. If the IK transform hasn't been
# called for a long time (paused thread, startup, debugger pause), don't
# let the first resumed reset tick fly across the whole arm — clamp the
# step the way the per-call version did. 100 ms gives 3° max per call,
# well under the implausible-jump backstop's 20°.
_RESET_RAMP_DT_CAP_S: float = 0.1


class _Kinematics(Protocol):
    """Minimal FK/IK surface the controller needs (degrees in, degrees out)."""

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray: ...

    def inverse_kinematics(self, seed_deg: np.ndarray, target: np.ndarray) -> np.ndarray: ...


class TipOffsetKinematics:
    """Wrap an anchor-frame kinematics with a fixed virtual-EE offset.

    The inner kinematics targets a URDF link (the anchor — for SO-107,
    ``L6_1``, upstream of the gripper joint ``S7``). This wrapper shifts
    the view to a virtual EE frame ``anchor @ tip_offset``, so callers
    work in the user's intuitive EE frame while the QP still optimizes
    against the anchor link.

    The point: when the anchor is upstream of ``S7``, the IK target is
    independent of S7, so the controller's post-IK overwrite of S7 with
    ``gripper_pos`` no longer fights the solver. Removes the 5-8 deg
    rotation drift on 2D paths (gripper-DOF cost — see
    ``teleoperators/quest_vr/TODO.md``).
    """

    def __init__(self, inner: _Kinematics, tip_offset: np.ndarray) -> None:
        assert tip_offset.shape == (4, 4), "tip_offset must be a 4x4 SE(3) matrix"
        self._inner = inner
        self._tip_offset = tip_offset.astype(float).copy()
        self._tip_offset_inv = np.linalg.inv(self._tip_offset)

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        return self._inner.forward_kinematics(joint_pos_deg) @ self._tip_offset

    def inverse_kinematics(self, seed_deg: np.ndarray, target: np.ndarray) -> np.ndarray:
        return self._inner.inverse_kinematics(seed_deg, target @ self._tip_offset_inv)


class JointMappedKinematics:
    """Wrap a URDF-space kinematics so callers can work in motor-degree space.

    The SO-107 URDF (CAD-exported) has per-joint ``(sign, offset_deg)``
    relative to the motors' calibrated zero — see :mod:`joint_alignment`.
    This wrapper applies ``urdf = sign * motor + offset`` on the way into
    the inner solver and the inverse on the way out, so the upstream IK
    code never needs a ``joint_map`` parameter and the upstream kinematic
    processors stay unmodified.
    """

    def __init__(
        self,
        inner: _Kinematics,
        motor_names: list[str],
        alignment: dict[str, JointAlignment],
    ) -> None:
        assert all(m in alignment for m in motor_names), "alignment missing a motor"
        self._inner = inner
        self._sign = np.array([alignment[m].sign for m in motor_names], dtype=float)
        self._offset = np.array([alignment[m].offset_deg for m in motor_names], dtype=float)
        assert np.all(np.abs(self._sign) == 1.0), "joint-alignment sign must be +/-1"

    def _motor_to_urdf(self, q_motor: np.ndarray) -> np.ndarray:
        return self._sign * np.asarray(q_motor, dtype=float) + self._offset

    def _urdf_to_motor(self, q_urdf: np.ndarray) -> np.ndarray:
        return (np.asarray(q_urdf, dtype=float) - self._offset) / self._sign

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        """Return the 4x4 SE(3) base->EE transform for motor-space joints (degrees)."""
        return self._inner.forward_kinematics(self._motor_to_urdf(joint_pos_deg))

    def inverse_kinematics(self, seed_deg: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Solve IK for a 4x4 SE(3) ``target``; seed and result are motor-space degrees."""
        q_urdf = self._inner.inverse_kinematics(self._motor_to_urdf(seed_deg), target)
        return self._urdf_to_motor(q_urdf)


class CartesianIKController:
    """Stateful EE-delta -> motor-joint transform for a single arm.

    Call once per tick with the eight EE-delta keys a Quest controller
    emits; returns a ``{motor}.pos`` dict in motor space.

    Preconditions:
        * ``q_init`` has one entry per ``motor_names`` entry, motor-space
          degrees, and ``motor_names`` contains ``"gripper"``.
        * Each call's ``action`` has keys ``enabled``, ``target_x/y/z``,
          ``target_wx/wy/wz``, ``gripper_pos`` (unprefixed).

    Postconditions:
        * Returns ``{name}.pos`` for every name in ``motor_names``.
        * While ``enabled`` is false the arm joints hold their last
          commanded values; ``gripper.pos`` still follows ``gripper_pos``
          (the trigger works while the clutch is released).
        * On the rising edge of ``enabled`` the reference pose is latched
          to ``FK(last commanded joints)``; targets are composed onto it.
        * The commanded EE position is clipped to the workspace box and
          its per-tick change capped, so IK never sees a wild target.
        * If IK still swings an arm joint implausibly far in one tick (e.g.
          near a singularity), that tick is held rather than executed.
    """

    def __init__(
        self,
        *,
        kinematics: _Kinematics,
        motor_names: list[str],
        q_init: np.ndarray,
        workspace_min: tuple[float, float, float],
        workspace_max: tuple[float, float, float],
        max_ee_step_m: float = _MAX_EE_STEP_M,
        label: str = "",
    ) -> None:
        assert "gripper" in motor_names, "motor_names must contain 'gripper'"
        assert len(q_init) == len(motor_names), "q_init length must match motor_names"
        self._kin = kinematics
        self._motor_names = list(motor_names)
        self._gripper_idx = self._motor_names.index("gripper")
        self._arm_idx = [i for i in range(len(self._motor_names)) if i != self._gripper_idx]
        self._q_last = np.asarray(q_init, dtype=float).copy()
        self._ws_min = np.asarray(workspace_min, dtype=float)
        self._ws_max = np.asarray(workspace_max, dtype=float)
        self._max_step = float(max_ee_step_m)
        # Tag for the per-tick hold-warning so a bimanual run log says
        # which arm hit the backstop (the bimanual transform sets "left" /
        # "right"). Empty default = no tag in the warning line.
        self._label = str(label)
        # Snapshot of the attach-time seed. The operator-commanded reset
        # button (handled in __call__) snaps _q_last back to this so the
        # arm goes "home" without needing a per-robot rest-pose config —
        # if you started at a good pose, you can always get back to it.
        self._q_init = self._q_last.copy()
        # Wall-clock timestamp of the previous reset-ramp call, used to
        # turn the per-second ramp rate into a per-call step. None when
        # the reset button is not held — re-armed on the rising edge so
        # a fresh hold starts with a zero-step warm-up rather than a
        # snap.
        self._reset_prev_t: float | None = None
        self._prev_enabled = False
        self._ref: np.ndarray | None = None  # latched reference pose (4x4)
        self._last_pos: np.ndarray | None = None  # last commanded EE position
        # True iff the last ``__call__`` ended up returning the held command
        # (NoSolutionFound or implausible-IK-jump branches). Inspected by
        # the bimanual-transform wrapper so the WebXR client can render a
        # tactile "you're pushing against an invisible wall" rumble while
        # the IK is held. Engagement gating (clutch released) is NOT a
        # hold — the operator's hand is allowed to move freely there.
        self.is_holding: bool = False

    def __call__(self, action: dict[str, Any]) -> dict[str, float]:
        enabled = bool(action["enabled"])
        gripper_pos = float(action["gripper_pos"])
        # The gripper is not IK-tracked: the teleop emits an absolute
        # motor-space target and it works whether or not the clutch is
        # engaged. Keep it in q_last so the seed stays a faithful command.
        self._q_last[self._gripper_idx] = gripper_pos

        # Operator-commanded reset: hold-to-ramp toward the attach-time
        # seed at _RESET_RAMP_RATE_DEG_PER_SEC, measured in wall-clock
        # time so the ramp feels the same regardless of how often the
        # transform is being polled (plain follower's main-loop
        # get_action() vs predictive follower's 90 Hz adapter thread).
        # Each call advances all arm joints by a clamped delta; release
        # the button at any time and the arm stops where it is. This is
        # the safety contract — snapping to the seed in a single call
        # would be ~600°/s on a plain follower with unbounded
        # max_relative_target. Intentionally bypasses the implausible-
        # jump backstop because each clamped step is itself well below
        # the backstop threshold (≤3° vs 20°). Gripper passthrough above
        # stays.
        if float(action.get("reset", 0.0)) > 0.5:
            now = time.perf_counter()
            # Rising edge of the hold: dt=0 warms up with zero step so a
            # fast first call doesn't snap; ramp starts from the next
            # call onward.
            dt = 0.0 if self._reset_prev_t is None else min(now - self._reset_prev_t, _RESET_RAMP_DT_CAP_S)
            self._reset_prev_t = now
            step_deg = _RESET_RAMP_RATE_DEG_PER_SEC * dt
            arm_diff = self._q_init[self._arm_idx] - self._q_last[self._arm_idx]
            clamped = np.clip(arm_diff, -step_deg, step_deg)
            self._q_last[self._arm_idx] += clamped
            self._ref = None
            self._last_pos = None
            self._prev_enabled = False
            self.is_holding = False
            return {f"{name}.pos": float(self._q_last[i]) for i, name in enumerate(self._motor_names)}
        # Reset button not held: drop the timestamp so a future press
        # re-arms cleanly from the rising-edge zero-warm-up.
        self._reset_prev_t = None

        if enabled:
            if not self._prev_enabled or self._ref is None:
                self._ref = self._kin.forward_kinematics(self._q_last)
                self._last_pos = self._ref[:3, 3].copy()

            delta_p = np.array(
                [float(action["target_x"]), float(action["target_y"]), float(action["target_z"])],
                dtype=float,
            )
            r_delta = Rotation.from_rotvec(
                [float(action["target_wx"]), float(action["target_wy"]), float(action["target_wz"])]
            ).as_matrix()

            desired = np.eye(4, dtype=float)
            # World-frame rotation composition. The teleop is expected to
            # emit ``target_wx/wy/wz`` as a rotation in the robot WORLD
            # frame (Quest VR's source computes
            # ``r_delta = quest_rot_now * quest_rot_at_engage.inv()``,
            # matching the May-17 ``sim_receiver`` reference: same
            # ordering). The world-frame delta must therefore be LEFT-
            # multiplied onto the engage rotation; right-multiplication
            # (``ref @ r_delta``) silently treats it as a local-frame
            # delta and yawing the wrist sends the gripper around the
            # engage pose's tilted Z instead of vertical, forcing the IK
            # to corkscrew the elbow / shoulder to reach a screwed-up
            # target. The two compositions are equal only when the
            # engage pose is identity, which is essentially never on a
            # real arm.
            desired[:3, :3] = r_delta @ self._ref[:3, :3]
            pos = self._ref[:3, 3] + delta_p
            pos = np.clip(pos, self._ws_min, self._ws_max)

            assert self._last_pos is not None
            step = pos - self._last_pos
            n = float(np.linalg.norm(step))
            if n > self._max_step > 0.0:
                pos = self._last_pos + step * (self._max_step / n)
            desired[:3, 3] = pos

            # Hold this tick if the QP can't solve at all (e.g. the user
            # stretched past reach). Symmetric with the implausible-jump
            # guard below — internal state stays put so the next tick
            # re-evaluates from the held pose, and the user just has to
            # move back into the workspace for tracking to resume.
            tag = f"[{self._label}] " if self._label else ""
            try:
                q_new = np.asarray(self._kin.inverse_kinematics(self._q_last, desired), dtype=float)
            except _NoSolutionFound:
                logger.warning("CartesianIKController %sIK has no solution — holding this tick", tag)
                self.is_holding = True
                self._prev_enabled = enabled
                return {f"{name}.pos": float(self._q_last[i]) for i, name in enumerate(self._motor_names)}
            # Backstop: hold this tick if IK swings an arm joint implausibly
            # far. _q_last / _ref / _last_pos are left untouched, so the next
            # tick re-evaluates from the held state.
            joint_step = float(np.max(np.abs(q_new[self._arm_idx] - self._q_last[self._arm_idx])))
            if joint_step > _MAX_JOINT_STEP_DEG:
                logger.warning(
                    "CartesianIKController %simplausible IK jump (%.0f deg) — holding this tick",
                    tag,
                    joint_step,
                )
                self.is_holding = True
            else:
                q_new[self._gripper_idx] = gripper_pos
                self._q_last = q_new
                self._last_pos = pos.copy()
                self.is_holding = False
        else:
            # Disengaged is not "holding" — the operator's hand is free to
            # move without any IK-target feedback.
            self.is_holding = False

        self._prev_enabled = enabled
        return {f"{name}.pos": float(self._q_last[i]) for i, name in enumerate(self._motor_names)}


def make_bimanual_ik_transform(
    left: Callable[[dict], dict],
    right: Callable[[dict], dict],
) -> Callable[[dict], dict]:
    """Build a ``left_``/``right_`` split-merge transform over two arm controllers.

    Returns a ``dict -> dict`` callable: it takes a bimanual EE-delta action
    (every key prefixed ``left_`` or ``right_``), routes each arm's
    deprefixed slice to that arm's controller, and merges the two
    ``{motor}.pos`` results back under the prefixes — the joint dict a
    bimanual follower's ``send_action`` consumes. A robot installs the
    result into a Cartesian teleop via ``set_action_transform``.
    """

    def _transform(ee_action: dict) -> dict:
        left_in = {k.removeprefix("left_"): v for k, v in ee_action.items() if k.startswith("left_")}
        right_in = {k.removeprefix("right_"): v for k, v in ee_action.items() if k.startswith("right_")}
        return {f"left_{k}": v for k, v in left(left_in).items()} | {
            f"right_{k}": v for k, v in right(right_in).items()
        }

    return _transform


def make_so107_arm_kinematics(
    alignment: dict[str, JointAlignment],
    *,
    posture_cost: float = 0.05,
    max_iters: int = 50,
) -> JointMappedKinematics:
    """Build the motor-space kinematics for one SO-107 arm.

    This is the slow half of the Cartesian-IK setup: ``PinkKinematics``
    parses the SO-107 URDF + meshes into a pinocchio model (~1-2 s, CPU-
    bound, holds the GIL). Call it once, eagerly — e.g. in a robot's
    ``__init__``, before ``connect()`` starts any camera read thread.
    Building it later, while a RealSense thread is warming up, starves
    that thread and trips its frame-age watchdog.

    Args:
        alignment: This arm's motor->URDF alignment (``LEFT_ARM_ALIGNMENT``
            or ``RIGHT_ARM_ALIGNMENT`` from :mod:`joint_alignment`).
        posture_cost: PostureTask weight relative to the FrameTask (which is
            1.0). Default 0.05 makes posture a null-space tiebreaker so the
            QP almost always satisfies the EE target. Raise (e.g. 0.3) for
            stronger "stay near previous pose" — tighter joint continuity
            near singularities, at the cost of small EE tracking lag. This
            is the primary feel lever for "twisty / wrist-flipped configs"
            near reach limits.
        max_iters: QP iteration budget per ``inverse_kinematics`` call. Pink
            defaults to 10; the SO-107 default here is 50 because a moving
            teleop target benefits from extra iterations to close per-call
            lag (cm- to mm-scale at typical teleop speeds). Negligible CPU
            impact at 30 Hz. Lower to 10–20 for more "stick to seed" feel
            at the cost of moving-target tracking accuracy.

    Requires the optional ``pin-pink`` dependency (raises ``ImportError``
    otherwise).
    """
    from lerobot.model.pink_kinematics import PinkKinematics

    from . import get_urdf_path

    inner = PinkKinematics(
        urdf_path=str(get_urdf_path()),
        target_frame_name=URDF_ANCHOR_FRAME,
        joint_names=list(URDF_JOINT_NAMES),
        posture_cost=posture_cost,
        max_iters=max_iters,
    )
    # IK against the L6_1 anchor + a fixed offset to the virtual closed-
    # gripper tip — the IK is then S7-independent and the controller's
    # post-IK S7 overwrite no longer drives orientation drift.
    with_tip = TipOffsetKinematics(inner, TIP_OFFSET)
    return JointMappedKinematics(with_tip, list(MOTOR_NAMES), alignment)


# Effectively-unbounded workspace box for the IK controller: a meter cubed
# on each side, far past anything an SO-107 link tree can physically reach
# (~0.5 m max). Use this for a perfect-tracker / sim robot that has no
# physical-safety concern — the production ``SO107_WORKSPACE_*`` clip is
# there to keep a real arm + jittery hand-tracking input from being driven
# into the table or into self-collision, neither of which applies to a
# motor-less follower.
SO107_WORKSPACE_UNBOUNDED_MIN: tuple[float, float, float] = (-1.0, -1.0, -1.0)
SO107_WORKSPACE_UNBOUNDED_MAX: tuple[float, float, float] = (+1.0, +1.0, +1.0)


def make_so107_arm_ik_controller(
    kinematics: JointMappedKinematics,
    q_init: np.ndarray,
    workspace_min: tuple[float, float, float] = SO107_WORKSPACE_MIN,
    workspace_max: tuple[float, float, float] = SO107_WORKSPACE_MAX,
    *,
    label: str = "",
) -> CartesianIKController:
    """Build a Cartesian-IK controller for one SO-107 arm.

    Fast: wires a pre-built kinematics, the seed configuration, and the
    SO-107 workspace box into a controller. Build ``kinematics`` ahead of
    time with :func:`make_so107_arm_kinematics`.

    Args:
        kinematics: This arm's kinematics from :func:`make_so107_arm_kinematics`.
        q_init: The arm's current joint configuration, motor-space degrees,
            in :data:`joint_alignment.MOTOR_NAMES` order.
        workspace_min, workspace_max: EE-position clip box, URDF world frame.
            Defaults are the production safety bounds (conservative; chosen
            to keep a real arm + Quest hand-tracking input out of trouble).
            Override with ``SO107_WORKSPACE_UNBOUNDED_{MIN,MAX}`` for a
            motor-less / sim robot with no physical-safety concern.
    """
    return CartesianIKController(
        kinematics=kinematics,
        motor_names=list(MOTOR_NAMES),
        q_init=q_init,
        workspace_min=workspace_min,
        workspace_max=workspace_max,
        label=label,
    )


def is_so107_bimanual_cartesian_teleop(teleop: Any) -> bool:
    """True iff ``teleop`` looks like a bimanual SO-107 Cartesian source.

    Cheap structural check: ``action_features.names`` contains
    ``left_target_x`` and ``right_target_x``. Use to gate the Cartesian
    branch in a follower's ``attach_teleop`` before pulling in
    pin-pink-dependent IK setup.

    Does NOT verify the teleop has ``set_action_transform`` /
    ``get_action_raw``; callers should assert those separately if
    they require them.
    """
    try:
        names = teleop.action_features.get("names", {})
    except (AttributeError, TypeError):
        return False
    return "left_target_x" in names and "right_target_x" in names


class BimanualSO107IKTransform:
    """Callable Cartesian-action → joint-action transform for a bimanual
    SO-107 follower, with per-arm IK-hold state readable by the teleop.

    Wraps :func:`make_bimanual_ik_transform` so callers that only need the
    dict→dict behavior (``set_action_transform``) keep working unchanged,
    and adds :attr:`hold_per_arm` for callers (the Quest VR teleop) that
    want to surface "the IK is holding the last command" to the operator
    as a tactile signal.
    """

    def __init__(self, left: CartesianIKController, right: CartesianIKController) -> None:
        self.left = left
        self.right = right
        self._inner = make_bimanual_ik_transform(left, right)

    def __call__(self, action: dict) -> dict:
        return self._inner(action)

    @property
    def hold_per_arm(self) -> tuple[bool, bool]:
        """``(left_holding, right_holding)`` from the last ``__call__``.

        True iff that arm's IK ended up returning the held command
        (NoSolutionFound, implausible-jump backstop). The teleop signals
        each rising / falling edge to the operator via haptic pulse.
        """
        return (self.left.is_holding, self.right.is_holding)


def build_so107_bimanual_ik_transform(
    ik_kinematics: dict[str, JointMappedKinematics],
    left_arm: Any,
    right_arm: Any,
    workspace_min: tuple[float, float, float] = SO107_WORKSPACE_MIN,
    workspace_max: tuple[float, float, float] = SO107_WORKSPACE_MAX,
) -> BimanualSO107IKTransform:
    """Build a Cartesian-action → joint-action transform for a bimanual
    SO-107 follower.

    Seeds per-arm IK controllers from each arm's current observation
    (both arms must be connected — ``arm.get_observation()`` is called
    once at build time to read the latch reference), then composes them
    through the bimanual prefix split/merge.

    The returned object is callable (the shape ``teleop.set_action_transform``
    expects: takes a bimanual Cartesian action dict, returns a
    motor-joint dict prefixed with ``left_`` / ``right_``) AND exposes
    :attr:`BimanualSO107IKTransform.hold_per_arm` for callers that want
    per-arm hold state.

    Used by:
      * ``BiSO107Follower.attach_teleop`` — installs synchronously via
        ``set_action_transform`` (IK runs on-demand at ``get_action()``).
      * ``BiSO107FollowerPredictive._attach_cartesian_teleop`` — wraps in
        :class:`~lerobot.robots.predictive.cartesian_adapter.BimanualCartesianIKAdapter`
        so IK runs in a background thread at WebXR rate.
      * ``VirtualBiSO107Follower.attach_teleop`` — passes
        ``SO107_WORKSPACE_UNBOUNDED_*`` since a perfect-tracker robot has
        no physical-safety concern.
    """

    def _seed(arm: Any) -> np.ndarray:
        obs = arm.get_observation()
        return np.array([float(obs[f"{m}.pos"]) for m in MOTOR_NAMES], dtype=float)

    left_ik = make_so107_arm_ik_controller(
        ik_kinematics["left"], _seed(left_arm), workspace_min, workspace_max, label="left"
    )
    right_ik = make_so107_arm_ik_controller(
        ik_kinematics["right"], _seed(right_arm), workspace_min, workspace_max, label="right"
    )
    return BimanualSO107IKTransform(left_ik, right_ik)
