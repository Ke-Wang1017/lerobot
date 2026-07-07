# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Scripted bimanual Cartesian-EE-delta trajectory teleop.

Same action surface as :class:`QuestVRTeleop` — the bimanual
``left_target_x/y/z, left_target_wx/wy/wz, left_gripper_pos`` + right
keys, ``set_action_transform`` / ``get_action_raw`` for the predictive
adapter, ``action_features`` exposing the keys so a follower's
``attach_teleop`` recognises the Cartesian-VR-shape and installs the IK
path. Per-tick output advances by wall-clock so any consumer (the loop
driver at 30 Hz, the predictive Cartesian adapter at 90 Hz) sees a
consistent unfolding shape.

The trajectory plays once: a ``ramp_ticks`` linear ramp from seed to the
shape anchor, ``n_waypoints`` shape ticks, then a linear ramp back. After
that, ``is_exhausted`` flips True so ``lerobot-teleoperate`` /
``lerobot-record`` end the session cleanly. The delta keys are zero
during the post-trajectory window (held at seed) until the loop driver
exits.

Both arms get the **same** delta in robot base frame — each arm's
``CartesianIKController`` latches its own reference pose on the rising
edge of ``enabled``, so the per-arm trajectories trace the same shape in
each arm's reachable workspace. The user stages the arms accordingly.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from ..teleoperator import Teleoperator
from .configuration_scripted_ee import ScriptedBimanualEETeleopConfig

logger = logging.getLogger(__name__)

# Output action keys: every per-arm Cartesian-VR key, doubled with
# left_ / right_ prefixes. Matches :class:`QuestVRTeleop`.
_PER_ARM_KEYS = (
    "enabled",
    "target_x",
    "target_y",
    "target_z",
    "target_wx",
    "target_wy",
    "target_wz",
    "gripper_pos",
)
_ACTION_KEYS: tuple[str, ...] = tuple(f"{p}{k}" for p in ("left_", "right_") for k in _PER_ARM_KEYS)


def _heart_unit(n: int) -> np.ndarray:
    """Parametric heart, fit to unit bbox, starting at (0, 0)."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = 16.0 * np.sin(t) ** 3
    y = 13.0 * np.cos(t) - 5.0 * np.cos(2 * t) - 2.0 * np.cos(3 * t) - np.cos(4 * t)
    pts = np.stack([x, y], axis=1)
    pts -= pts[0]
    return pts / float((pts.max(axis=0) - pts.min(axis=0)).max())


def _shape_deltas_local(shape: str, size_m: float, n: int) -> np.ndarray:
    """Return ``(n, 2)`` shape deltas in local (forward, lateral) space."""
    if shape == "circle":
        r = size_m
        # Center at (+r, 0). Start at (0, 0); orbits forward then back.
        t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        return np.stack([r * (1 - np.cos(t)), r * np.sin(t)], axis=1)
    if shape == "heart":
        return _heart_unit(n) * size_m
    if shape == "square":
        per_edge = n // 4
        out: list[np.ndarray] = []
        # Corners in (forward, lateral): (0,0) → (s,0) → (s,s) → (0,s) → ...
        corners = np.array([[0.0, 0.0], [size_m, 0.0], [size_m, size_m], [0.0, size_m]])
        for e in range(4):
            a, b = corners[e], corners[(e + 1) % 4]
            for k in range(per_edge):
                out.append(a + (b - a) * (k / per_edge))
        return np.asarray(out)
    if shape == "static_hold":
        # All n waypoints sit at (forward = +size_m, lateral = 0). Diagnostic
        # for "does state catch up to action when action is held constant?".
        return np.tile(np.array([[size_m, 0.0]]), (n, 1))
    raise ValueError(f"unknown shape: {shape!r}")


class ScriptedBimanualEETeleop(Teleoperator):
    """Scripted bimanual Cartesian-EE-delta teleop.

    Implements the same surface a robot's ``attach_teleop`` matches for
    Quest VR: ``action_features`` has ``left_target_x`` / ``right_target_x``,
    plus ``set_action_transform`` / ``get_action_raw`` so the predictive
    Cartesian adapter can pre-transform and poll the raw EE deltas at
    its own rate.

    Lifecycle:
        * ``connect()`` resets the wall clock and pre-computes the
          ``ramp_in + shape + ramp_out`` delta sequence in (forward,
          lateral, up) world-frame coordinates.
        * ``get_action_raw()`` returns the delta dict for the current
          wall-clock tick. After the trajectory completes, returns a
          held-at-anchor / idle dict (deltas == 0 in the local frame,
          relative to the seed reference latched by the controller on
          ``enabled`` rising edge).
        * ``is_exhausted`` flips True one tick after the trajectory ends,
          so loop drivers that check this property (``lerobot-record``,
          ``lerobot-teleoperate``) exit cleanly.
    """

    config_class = ScriptedBimanualEETeleopConfig
    name = "scripted_bimanual_ee"

    def __init__(self, config: ScriptedBimanualEETeleopConfig):
        super().__init__(config)
        self.config: ScriptedBimanualEETeleopConfig = config
        self._lock = threading.Lock()
        self._connected = False
        self._t0: float | None = None
        self._deltas: np.ndarray | None = None  # (n_total, 3) in robot base frame, meters
        self._shape_end: int = 0  # last tick index of the trajectory (inclusive of ramp_out)
        self._action_transform: Callable[[dict], dict] | None = None

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (len(_ACTION_KEYS),),
            "names": {k: i for i, k in enumerate(_ACTION_KEYS)},
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def is_exhausted(self) -> bool:
        """True once the trajectory has played out one full cycle.

        Loop drivers (``lerobot-record`` / ``lerobot-teleoperate``) end
        the session when this flips True; the scripted teleop is single-
        shot.
        """
        if self._t0 is None or self._deltas is None:
            return False
        tick = int((time.monotonic() - self._t0) * self.config.loop_hz)
        return tick >= self._shape_end

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        """Pre-compute the delta sequence and start the wall clock."""
        if self._connected:
            return

        cfg = self.config
        forward = np.asarray(cfg.forward_axis, dtype=float)
        forward /= float(np.linalg.norm(forward)) or 1.0
        lateral = np.asarray(cfg.lateral_axis, dtype=float)
        lateral /= float(np.linalg.norm(lateral)) or 1.0
        up = np.array([0.0, 0.0, 1.0], dtype=float)

        offset = cfg.offset_forward_m * forward + cfg.offset_up_m * up

        # Shape deltas in local (forward, lateral). Convert to robot-base
        # 3D by combining with the configured forward / lateral axes.
        shape_2d = _shape_deltas_local(cfg.shape, cfg.size_m, cfg.n_waypoints)
        shape_3d = shape_2d[:, 0:1] * forward[None, :] + shape_2d[:, 1:2] * lateral[None, :]
        shape_3d = shape_3d + offset[None, :]

        # ramp_in goes 0 → offset, shape_3d already includes +offset, ramp_out
        # goes offset → 0.
        ramp_in = np.linspace(0.0, 1.0, cfg.ramp_ticks, endpoint=False)[:, None] * offset[None, :]
        ramp_out = np.linspace(1.0, 0.0, cfg.ramp_ticks, endpoint=False)[:, None] * offset[None, :]

        self._deltas = np.concatenate([ramp_in, shape_3d, ramp_out], axis=0)
        self._shape_end = int(len(self._deltas))
        self._t0 = time.monotonic()
        self._connected = True
        logger.info(
            "ScriptedBimanualEETeleop connected: %s shape, size=%.3f m, %d total ticks at %.0f Hz",
            cfg.shape,
            cfg.size_m,
            self._shape_end,
            cfg.loop_hz,
        )

    def disconnect(self) -> None:
        with self._lock:
            self._connected = False
            self._t0 = None
            self._deltas = None
            self._shape_end = 0
            self._action_transform = None

    def send_feedback(self, feedback: dict[str, Any]) -> None:  # noqa: ARG002
        pass

    # ── Actions ───────────────────────────────────────────────────────────

    def set_action_transform(self, transform: Callable[[dict], dict] | None) -> None:
        """Install the per-tick action transform a follower's attach_teleop sets.

        For plain ``BiSO107Follower``: this transform is the bimanual IK
        callable — ``get_action()`` returns motor-joint dicts. For
        ``BiSO107FollowerPredictive``: the follower installs a transform
        that returns the Cartesian adapter's cached joint dict, so
        recording stays consistent across the two paths.
        """
        self._action_transform = transform

    def _current_delta(self) -> np.ndarray:
        """Return the current 3D delta for both arms, or zeros if not yet running.

        Linearly interpolates between adjacent ``self._deltas`` waypoints
        based on a *continuous* (float) tick index — so a consumer polling
        at the predictive controller's 200 Hz rate gets a fresh
        interpolated sample every poll, instead of the same 30 Hz-quantized
        waypoint value repeated for 6-7 polls. The pull-path velocity
        estimator's design assumes one new sample per controller tick;
        feeding it staircase-quantized intent collapses the velocity
        estimate (see comments in predictive/controller.py).
        """
        if self._t0 is None or self._deltas is None:
            return np.zeros(3, dtype=float)
        tick_f = (time.monotonic() - self._t0) * self.config.loop_hz
        if tick_f < 0:
            return np.zeros(3, dtype=float)
        if tick_f >= self._shape_end - 1:
            # Trajectory done — hold at seed (zero delta). is_exhausted is True.
            return np.zeros(3, dtype=float)
        i = int(tick_f)
        frac = tick_f - i
        # Linear interpolation between waypoint i and i+1.
        return (1.0 - frac) * self._deltas[i] + frac * self._deltas[i + 1]

    def get_action_raw(self) -> dict[str, float]:
        """Return the raw EE-delta action; same surface as Quest VR's raw."""
        delta = self._current_delta()
        grip = float(self.config.gripper_value)
        return {
            "left_enabled": 1.0,
            "left_target_x": float(delta[0]),
            "left_target_y": float(delta[1]),
            "left_target_z": float(delta[2]),
            "left_target_wx": 0.0,
            "left_target_wy": 0.0,
            "left_target_wz": 0.0,
            "left_gripper_pos": grip,
            "right_enabled": 1.0,
            "right_target_x": float(delta[0]),
            "right_target_y": float(delta[1]),
            "right_target_z": float(delta[2]),
            "right_target_wx": 0.0,
            "right_target_wy": 0.0,
            "right_target_wz": 0.0,
            "right_gripper_pos": grip,
        }

    def get_action(self):
        action = self.get_action_raw()
        if self._action_transform is not None:
            action = self._action_transform(action)
        return action
