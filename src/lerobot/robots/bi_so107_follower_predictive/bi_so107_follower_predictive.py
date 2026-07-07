#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Bimanual SO-107 follower with predictive-lookahead controllers on both arms.

Composes two ``SO107FollowerPredictive`` instances — one per arm. Each
arm owns its own ``FeetechMotorsBus`` and its own 200 Hz controller
thread; the buses are independent, so the two controllers don't
contend on anything. The composer's job is just argument routing and
joint-name prefix translation (``left_*`` / ``right_*``), identical
to plain ``BiSO107Follower``.

Why per-arm threads instead of one shared loop driving both:

  * Bus blip isolation. A transient Feetech retry on one bus (~10 ms)
    would stall the shared loop and delay the other arm's writes.
    Two threads × two buses keeps the slow side from infecting the
    fast side.
  * Independent adaptive L. The two arms have measurably different
    motor τ in practice (one bus is over-cabled in white profile,
    the other isn't). Each arm's amplitude-gated xcorr converges to
    its own τ. Sharing one L would force the wrong tuning on at
    least one arm.
  * Phase drift between arms is ≤ 1 control period = 5 ms at 200 Hz —
    below motor response time and below the human-teleop floor, so
    losing per-tick sync of the goal issue costs nothing measurable.
  * ``SO107FollowerPredictive`` stays self-contained and re-usable as
    a single-arm robot; the bimanual is just two of them stacked.

GIL note: ``bus.sync_write`` releases the GIL during the serial I/O,
so the two writer threads progress in parallel even on CPython.

Distinct ``robot_type`` (``bi_so107_follower_predictive``) so the
training-side embodiment contract is unambiguous — see the module
docstring on ``SO107FollowerPredictive`` for the long-form rationale.
"""

from __future__ import annotations

import logging
from typing import Any

from lerobot.types import ActionChunk

from ..bi_so107_follower.bi_so107_follower import BiSO107Follower
from ..robot import Robot
from ..so107_follower_predictive import (
    SO107FollowerPredictive,
    SO107FollowerPredictiveRobotConfig,
)
from .config_bi_so107_follower_predictive import BiSO107FollowerPredictiveConfig

logger = logging.getLogger(__name__)


class BiSO107FollowerPredictive(BiSO107Follower):
    """Bimanual SO-107 with the predictive controller on each arm.

    Everything caller-facing (``connect`` / ``get_observation`` /
    ``send_action`` / ``disconnect``) is inherited from
    ``BiSO107Follower`` — it forwards to the per-arm objects, which
    are now ``SO107FollowerPredictive`` instances that handle the
    200 Hz writer thread + lookahead + adaptive update internally.
    """

    config_class = BiSO107FollowerPredictiveConfig
    name = "bi_so107_follower_predictive"

    def __init__(self, config: BiSO107FollowerPredictiveConfig):
        # Bypass BiSO107Follower.__init__ — its arms are plain SO107Follower.
        # We want predictive arms, so call Robot.__init__ directly and rebuild
        # the per-arm + camera setup with our own arm class.
        Robot.__init__(self, config)
        self.config = config

        # Shared controller settings projected onto each per-arm config.
        # Both arms get the same lookahead / alpha / control rate — see
        # the config docstring for why this is shared rather than per-arm.
        # Enumerated via ``PredictiveControllerConfig.__dataclass_fields__``
        # so adding a new controller knob automatically propagates here
        # — the previous hand-maintained list silently dropped
        # ``velocity_lowpass_hz`` / ``amp_gate_lo`` / ``amp_gate_hi``.
        import dataclasses

        from ..predictive.config import PredictiveControllerConfig

        per_arm_kwargs = {
            f.name: getattr(config, f.name) for f in dataclasses.fields(PredictiveControllerConfig)
        }

        left_arm_config = SO107FollowerPredictiveRobotConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_port,
            disable_torque_on_disconnect=config.left_arm_disable_torque_on_disconnect,
            max_relative_target=config.left_arm_max_relative_target,
            use_degrees=config.left_arm_use_degrees,
            cameras={},
            **per_arm_kwargs,
        )

        right_arm_config = SO107FollowerPredictiveRobotConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_port,
            disable_torque_on_disconnect=config.right_arm_disable_torque_on_disconnect,
            max_relative_target=config.right_arm_max_relative_target,
            use_degrees=config.right_arm_use_degrees,
            cameras={},
            **per_arm_kwargs,
        )

        self.left_arm = SO107FollowerPredictive(left_arm_config)
        self.right_arm = SO107FollowerPredictive(right_arm_config)

        # Cameras + RealSense depth-edge post-grab processor installation —
        # identical to BiSO107Follower.__init__. Kept inline rather than
        # extracted into a helper because changing it would require also
        # changing BiSO107Follower, and the two are independent embodiments
        # that just happen to share hardware.
        from lerobot.cameras.realsense import RealSenseCamera
        from lerobot.cameras.utils import make_cameras_from_configs
        from lerobot.processor import DepthEdgeOverlayProcessorStep

        self.cameras = make_cameras_from_configs(config.cameras)
        for cam_key, cam in self.cameras.items():
            if isinstance(cam, RealSenseCamera) and cam.use_depth:
                cam.post_grab_processor = DepthEdgeOverlayProcessorStep(
                    camera_key=cam_key,
                    threshold_percentile=90,
                    blur_kernel=3,
                    dilation_kernel=2,
                    alpha=0.7,
                    min_depth=0.2,
                    max_depth=0.6,
                )

        # Per-arm IK kinematics — same pre-build logic as plain BiSO107Follower:
        # parse the URDF here (~1-2 s/arm, CPU-bound) before connect() spins
        # up the RealSense read thread, so the parse can't starve it.
        # None if pin-pink is missing (Cartesian teleop then no-ops).
        self._ik_kinematics: dict[str, Any] | None = None
        try:
            from lerobot.robots.so107_description.cartesian_ik import make_so107_arm_kinematics
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

        # Lifetime: started by attach_teleop when wrapping a Cartesian VR
        # teleop, stopped on detach. None when no Cartesian adapter is
        # active (e.g. joint-space leader, or no teleop attached).
        self._cartesian_adapter: Any = None

    def attach_teleop(self, teleop) -> None:
        """Wire a teleop to the per-arm predictive controllers' pull path.

        Three teleop shapes are supported:

        * **Bimanual joint-space leader** (``BiSO107Leader``,
          ``BiSO107LeaderHighRate`` — has ``left_arm`` / ``right_arm``
          sub-teleop attributes). Each predictive arm polls its arm's
          sub-teleop directly.

        * **Bimanual Cartesian VR teleop** (``QuestVRTeleop`` —
          ``action_features.names`` contains ``left_target_x`` etc.). A
          :class:`BimanualCartesianIKAdapter` is built that wraps the
          teleop + the bimanual IK transform, runs IK at WebXR rate in a
          background thread, and exposes ``.left_arm`` / ``.right_arm``
          sub-teleops returning per-arm joint dicts. Same downstream
          contract as the leader case, so the per-arm controllers don't
          know which kind of teleop is upstream. The Cartesian teleop
          also gets an ``action_transform`` installed that returns the
          adapter's cached joint dict, so the script-side
          ``teleop.get_action()`` keeps returning a recordable joint
          dict — consistent with the plain-follower path.

        * **Anything else** (chunk-aware policy outputs, etc.): skip.
          The push path (``send_action``) is the route for chunks.

        ``None`` detaches both arms, stops the Cartesian adapter if any,
        and clears the Cartesian teleop's installed transform.
        """
        if teleop is None:
            self._teardown_cartesian_adapter()
            self.left_arm.attach_teleop(None)
            self.right_arm.attach_teleop(None)
            return

        # Bimanual joint-space leader path (unchanged).
        if hasattr(teleop, "left_arm") and hasattr(teleop, "right_arm"):
            self.left_arm.attach_teleop(teleop.left_arm)
            self.right_arm.attach_teleop(teleop.right_arm)
            return

        # Bimanual Cartesian VR teleop path (same detection as plain
        # BiSO107Follower.attach_teleop).
        from lerobot.robots.so107_description.cartesian_ik import (
            is_so107_bimanual_cartesian_teleop,
        )

        if is_so107_bimanual_cartesian_teleop(teleop):
            self._attach_cartesian_teleop(teleop)
            return

        logger.info(
            "%s: teleop %r is not a recognised bimanual leader or Cartesian teleop — "
            "skipping pull-path attach; send_action chunk path will be used",
            self,
            type(teleop).__name__,
        )

    def _attach_cartesian_teleop(self, teleop: Any) -> None:
        """Build and start the bimanual Cartesian IK adapter; route sub-teleops."""
        if self._ik_kinematics is None:
            logger.warning(
                "%s: Cartesian teleop attached but IK kinematics are unavailable "
                "(is pin-pink installed?) — the arms will not be driven.",
                self.name,
            )
            return
        assert hasattr(teleop, "set_action_transform"), (
            "a Cartesian teleop must expose set_action_transform()"
        )
        assert hasattr(teleop, "get_action_raw"), (
            "a Cartesian teleop bound to the predictive follower must expose "
            "get_action_raw() — see BimanualCartesianIKAdapter._tick"
        )
        assert self.is_connected, "attach_teleop requires the robot to be connected"

        # Tear down any previous Cartesian adapter — re-attach is supported.
        self._teardown_cartesian_adapter()

        from lerobot.robots.so107_description.cartesian_ik import (
            build_so107_bimanual_ik_transform,
        )

        from ..predictive.cartesian_adapter import BimanualCartesianIKAdapter

        transform = build_so107_bimanual_ik_transform(self._ik_kinematics, self.left_arm, self.right_arm)
        adapter = BimanualCartesianIKAdapter(teleop, transform, rate_hz=90.0)
        adapter.start()
        self._cartesian_adapter = adapter

        # Install a teleop-side transform that returns the adapter's
        # cached joint dict (single source of truth — the per-arm sub-
        # teleops below read the same cache; no parallel IK runs).
        # Exposes ``hold_per_arm`` because the teleop's haptic-rumble
        # path polls ``action_transform.hold_per_arm``; a plain function
        # has no attribute to look up so the rumble silently dies at the
        # adapter wrapper on the predictive path.
        class _AdapterTransform:
            def __call__(self, _action: dict) -> dict:
                return adapter.get_full_joint_action() or {}

            @property
            def hold_per_arm(self) -> tuple[bool, bool]:
                return adapter.hold_per_arm

        teleop.set_action_transform(_AdapterTransform())

        # Route the adapter's per-arm sub-teleops to each predictive arm's
        # controller. Same downstream contract as the leader path.
        self.left_arm.attach_teleop(adapter.left_arm)
        self.right_arm.attach_teleop(adapter.right_arm)
        logger.info(
            "%s: installed BimanualCartesianIKAdapter for %s",
            self.name,
            type(teleop).__name__,
        )

    def _teardown_cartesian_adapter(self) -> None:
        """Stop and forget the Cartesian adapter, if any. Idempotent."""
        if self._cartesian_adapter is None:
            return
        try:
            self._cartesian_adapter.stop()
        except Exception:
            logger.exception("%s: failed to stop Cartesian adapter cleanly", self.name)
        self._cartesian_adapter = None

    def send_action(
        self,
        action: dict[str, Any] | ActionChunk,
        *,
        period_s: float | None = None,
    ) -> dict[str, Any]:
        """Route bimanual intent to each predictive arm.

        For an ``ActionChunk``: split each frame's prefixed keys into two
        per-arm sub-chunks (same fps) and forward each to its arm. Both
        arms therefore get the full chunk horizon — exact-lookup runs
        independently per-arm. The recorded action is the prefixed
        frames[0] (= "current intent" for both arms combined), preserving
        the dataset-writer contract.

        For a plain dict: delegate to ``BiSO107Follower.send_action``,
        which handles dry-run + per-arm prefix translation.

        ``period_s`` is forwarded to each arm's predictive controller
        (see :meth:`SO107FollowerPredictive.send_action`). For chunks it
        defaults to ``1/action.fps`` per arm.
        """
        if not isinstance(action, ActionChunk):
            # The plain-dict path goes through BiSO107Follower.send_action,
            # which doesn't accept period_s. The current dict-mode callers
            # (intervention reset, idle hold) don't need starvation
            # detection. If a future caller needs it, they can wire it
            # explicitly through left_arm / right_arm send_action calls.
            return super().send_action(action)

        if self.config.dry_run:
            # Don't even route — return the current frame so callers
            # recording the action keep working.
            if not getattr(self, "_dry_run_logged", False):
                logger.warning(
                    "%s: dry_run=True - send_action is a no-op. Motors will NOT move. "
                    "Disable dry_run in the robot config to drive the arms.",
                    self,
                )
                self._dry_run_logged = True
            return dict(action.frames[0])

        # Split: each frame is {left_*: ..., right_*: ...} -> ({*: ...}, {*: ...})
        left_frames = tuple(
            {k.removeprefix("left_"): v for k, v in f.items() if k.startswith("left_")} for f in action.frames
        )
        right_frames = tuple(
            {k.removeprefix("right_"): v for k, v in f.items() if k.startswith("right_")}
            for f in action.frames
        )
        # Fail fast on mixed frames (frame missing one arm's keys). Without
        # this guard, the empty side would pass ActionChunk's non-empty-tuple
        # check, hit left_arm first, succeed, then raise on right_arm's
        # strict-key check — leaving the arms out of sync. The fix is to
        # detect the mismatch before any side-effect.
        empty_left = [i for i, f in enumerate(left_frames) if not f]
        empty_right = [i for i, f in enumerate(right_frames) if not f]
        if empty_left or empty_right:
            raise ValueError(
                f"{self}: ActionChunk frames are missing per-arm keys. "
                f"Frames with no left_* keys: {empty_left}. "
                f"Frames with no right_* keys: {empty_right}. "
                f"Every frame must carry both left_* and right_* keys to keep "
                f"the arms in sync."
            )
        left_chunk = ActionChunk(fps=action.fps, frames=left_frames)
        right_chunk = ActionChunk(fps=action.fps, frames=right_frames)

        left_sent = self.left_arm.send_action(left_chunk, period_s=period_s)
        right_sent = self.right_arm.send_action(right_chunk, period_s=period_s)

        # Re-prefix and return — the dataset writer records this as the
        # action for the current tick.
        return {
            **{f"left_{k}": v for k, v in left_sent.items()},
            **{f"right_{k}": v for k, v in right_sent.items()},
        }
