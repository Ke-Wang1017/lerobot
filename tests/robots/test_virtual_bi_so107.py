#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for ``VirtualBiSO107Follower`` — the motor-less perfect-tracker.

Pure Python, no hardware. Two layers:

* Robot-contract unit tests — lifecycle (connect / disconnect /
  is_connected), send_action stores joints, get_observation reflects
  them, partial actions preserve untouched motors, attach_teleop
  branching matches the plain BiSO107Follower (None / non-Cartesian /
  Cartesian).
* End-to-end smoke through the scripted teleop — verifies the production
  ``attach_teleop`` → IK transform install → ``teleop.get_action()`` →
  ``robot.send_action()`` codepath runs without error and produces
  non-trivial joint motion. The numerical correctness of the IK math is
  covered by ``test_pink_ik_trajectory.py`` (on main) and
  ``test_bimanual_tracks_shape`` in ``test_cartesian_ik.py``; this test
  is about wiring.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES
from lerobot.robots.virtual_bi_so107 import (
    VirtualBiSO107Follower,
    VirtualBiSO107FollowerConfig,
)
from lerobot.utils.import_utils import _pin_pink_available


@pytest.fixture
def robot() -> VirtualBiSO107Follower:
    """Connected ``VirtualBiSO107Follower`` ready for action / observation."""
    r = VirtualBiSO107Follower(VirtualBiSO107FollowerConfig())
    r.connect()
    yield r
    r.disconnect()


# ── Robot contract ────────────────────────────────────────────────────────


def test_lifecycle_connect_disconnect():
    r = VirtualBiSO107Follower(VirtualBiSO107FollowerConfig())
    assert not r.is_connected
    r.connect()
    assert r.is_connected
    r.disconnect()
    assert not r.is_connected


def test_no_cameras_but_attr_present():
    """Empty ``cameras`` dict keeps ``hasattr(robot, "cameras")`` truthy and
    ``len(cameras) == 0`` so the record/teleop scripts cleanly skip image
    writers."""
    r = VirtualBiSO107Follower(VirtualBiSO107FollowerConfig())
    assert hasattr(r, "cameras")
    assert len(r.cameras) == 0


def test_action_observation_features_match(robot):
    """Perfect tracker: observation features == action features."""
    assert robot.action_features == robot.observation_features
    # Sanity: 7 motors per arm × 2 arms = 14 keys, all float.
    assert len(robot.action_features) == 2 * len(MOTOR_NAMES)
    assert all(t is float for t in robot.action_features.values())


def test_send_action_stores_and_observation_returns(robot):
    """``send_action`` writes the joint dict; ``get_observation`` returns it."""
    action = {f"left_{m}.pos": 1.0 + i for i, m in enumerate(MOTOR_NAMES)}
    action |= {f"right_{m}.pos": 10.0 + i for i, m in enumerate(MOTOR_NAMES)}

    sent = robot.send_action(action)
    obs = robot.get_observation()

    assert sent == action  # perfect tracker: applied == requested
    assert obs == action


def test_send_action_partial_keeps_other_motors(robot):
    """Sending only a gripper update keeps the other six joints unchanged."""
    # Prime the arm at a known pose.
    full = {f"left_{m}.pos": 5.0 for m in MOTOR_NAMES} | {f"right_{m}.pos": -5.0 for m in MOTOR_NAMES}
    robot.send_action(full)

    # Now send only the grippers.
    robot.send_action({"left_gripper.pos": 70.0, "right_gripper.pos": 30.0})
    obs = robot.get_observation()

    assert obs["left_gripper.pos"] == 70.0
    assert obs["right_gripper.pos"] == 30.0
    # Other motors stayed at 5.0 / -5.0.
    for m in MOTOR_NAMES:
        if m == "gripper":
            continue
        assert obs[f"left_{m}.pos"] == 5.0
        assert obs[f"right_{m}.pos"] == -5.0


# ── attach_teleop branching ───────────────────────────────────────────────


def test_attach_teleop_none_is_safe(robot):
    """``attach_teleop(None)`` is a no-op."""
    robot.attach_teleop(None)


def test_attach_teleop_skips_non_cartesian(robot):
    """A joint-space-leader-shaped teleop (no ``left_target_x`` action keys)
    is a no-op — no transform installed."""

    class _JointLeaderStub:
        action_features = {"names": {f"left_{m}.pos": i for i, m in enumerate(MOTOR_NAMES)}}
        installed = None  # noqa: RUF012

        def set_action_transform(self, t):
            self.installed = t

    stub = _JointLeaderStub()
    robot.attach_teleop(stub)
    assert stub.installed is None


@pytest.mark.skipif(not _pin_pink_available, reason="pin-pink (optional) not installed")
def test_attach_teleop_installs_ik_for_cartesian_teleop(robot):
    """A Cartesian-shaped teleop gets the IK transform installed —
    ``teleop.get_action()`` then returns motor-joint dicts."""
    from lerobot.teleoperators.scripted_ee import (
        ScriptedBimanualEETeleop,
        ScriptedBimanualEETeleopConfig,
    )

    cfg = ScriptedBimanualEETeleopConfig(
        shape="circle", size_m=0.030, n_waypoints=8, ramp_ticks=2, loop_hz=100.0
    )
    teleop = ScriptedBimanualEETeleop(cfg)
    teleop.connect()
    try:
        robot.attach_teleop(teleop)

        # The transform installed by attach_teleop converts the Cartesian
        # delta dict to a motor-joint dict prefixed left_*/right_*.
        joints = teleop.get_action()
        for arm in ("left_", "right_"):
            for m in MOTOR_NAMES:
                assert f"{arm}{m}.pos" in joints, f"missing {arm}{m}.pos in IK output"
    finally:
        teleop.disconnect()


# ── End-to-end smoke through the production codepath ─────────────────────


@pytest.mark.skipif(not _pin_pink_available, reason="pin-pink (optional) not installed")
def test_scripted_teleop_drives_virtual_robot_through_send_action(robot):
    """The load-bearing test for this PR.

    Wires the virtual robot to a scripted Cartesian teleop the same way
    the GUI run loop will: ``attach_teleop`` → IK transform installed →
    loop calls ``teleop.get_action()`` → result feeds ``robot.send_action``.

    Verifies that:

    1. After one trajectory cycle, the recorded joint observations differ
       from the initial pose (proves the IK actually ran and the virtual
       arms tracked the commanded joints).
    2. Both arms moved (not just one — bimanual transform is wired).
    3. The robot is still healthy (``is_connected``, observations
       well-formed) at end of trajectory.

    Numerical IK correctness is covered by ``test_pink_ik_trajectory.py``
    and ``test_bimanual_tracks_shape``; this is a wiring + GUI-path
    smoke test.
    """
    from lerobot.teleoperators.scripted_ee import (
        ScriptedBimanualEETeleop,
        ScriptedBimanualEETeleopConfig,
    )

    cfg = ScriptedBimanualEETeleopConfig(
        shape="circle",
        size_m=0.030,
        n_waypoints=24,
        ramp_ticks=4,
        loop_hz=200.0,  # fast — total trajectory is ~0.16 s wall clock
    )
    teleop = ScriptedBimanualEETeleop(cfg)
    teleop.connect()
    try:
        robot.attach_teleop(teleop)

        initial_obs = robot.get_observation()
        last_obs = initial_obs

        # Drive the full trajectory. At 200 Hz with 4 + 24 + 4 = 32 ticks,
        # wall-clock duration is 0.16 s; the loop sees a moving target
        # whatever its sampling rate is, so we don't need every tick.
        for _ in range(60):
            joints = teleop.get_action()
            last_obs = robot.send_action(joints)
            if teleop.is_exhausted:
                break
            time.sleep(0.005)

        assert robot.is_connected
        # Some joint moved by at least 1 mDeg — proves IK and tracking ran.
        # (CartesianIKController gates motion on ``enabled``; for this
        # scripted source the ramp_in starts at enabled=0, then enabled=1
        # for the shape phase. The ramp gives the test ~24 enabled ticks.)
        any_motion = any(abs(last_obs[k] - initial_obs[k]) > 1e-3 for k in initial_obs if k.endswith(".pos"))
        assert any_motion, (
            f"no joint motion observed after driving scripted teleop; initial={initial_obs}, final={last_obs}"
        )
        # Both arms must have moved (bimanual transform routing).
        left_moved = any(
            abs(last_obs[k] - initial_obs[k]) > 1e-3 for k in initial_obs if k.startswith("left_")
        )
        right_moved = any(
            abs(last_obs[k] - initial_obs[k]) > 1e-3 for k in initial_obs if k.startswith("right_")
        )
        assert left_moved and right_moved, f"only one arm moved (left={left_moved}, right={right_moved})"
    finally:
        robot.attach_teleop(None)
        teleop.disconnect()


def test_send_action_with_unknown_keys_silently_ignores(robot):
    """Extra keys (e.g. camera-image keys from a chunk-aware policy) don't
    crash; only the known motor keys are applied."""
    action = {f"left_{m}.pos": float(i) for i, m in enumerate(MOTOR_NAMES)}
    action |= {f"right_{m}.pos": float(i + 10) for i, m in enumerate(MOTOR_NAMES)}
    action |= {"some_unknown.key": 99.0, "left_image.front": np.zeros((1,), dtype=np.uint8)}

    sent = robot.send_action(action)

    for m in MOTOR_NAMES:
        assert sent[f"left_{m}.pos"] == float(MOTOR_NAMES.index(m))
        assert sent[f"right_{m}.pos"] == float(MOTOR_NAMES.index(m) + 10)
