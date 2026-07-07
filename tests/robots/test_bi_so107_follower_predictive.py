#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Mock-bus integration smoke test for BiSO107FollowerPredictive.

The single-arm tests in test_so107_follower_predictive.py exercise the
algorithmic and threading surface. This file's job is just to verify
the bimanual composer wires up correctly: two independent buses, two
independent controller threads, shared controller settings, joint-name
prefixing, and clean shutdown.

If you find yourself wanting to add per-tick algorithmic assertions
here, prefer adding them to the single-arm file — the bimanual just
inherits via composition.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.bi_so107_follower_predictive import (
    BiSO107FollowerPredictive,
    BiSO107FollowerPredictiveConfig,
)

_MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "forearm_roll",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def _make_bus_mock(name: str) -> MagicMock:
    bus = MagicMock(name=name)
    bus.is_connected = False
    bus.is_calibrated = True
    bus.sync_write_log: list[tuple[str, dict]] = []

    def _connect():
        bus.is_connected = True

    def _disconnect(_disable=True):
        bus.is_connected = False

    def _sync_write(data_name, goal_dict, **_kwargs):
        bus.sync_write_log.append((data_name, dict(goal_dict)))

    @contextmanager
    def _dummy_cm():
        yield

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect
    bus.sync_write.side_effect = _sync_write
    bus.torque_disabled.side_effect = _dummy_cm
    return bus


@pytest.fixture
def bi_follower():
    """Yield a connected BiSO107FollowerPredictive with two mock buses.

    Each call to FeetechMotorsBus(...) returns a distinct bus mock so
    we can verify that left and right writes go to *different* buses.
    """
    left_bus = _make_bus_mock("LeftBusMock")
    right_bus = _make_bus_mock("RightBusMock")
    buses_in_order = [left_bus, right_bus]
    bus_iter = iter(buses_in_order)

    def _bus_side_effect(*_args, **kwargs):
        bus = next(bus_iter)
        bus.motors = kwargs["motors"]
        bus.sync_read.return_value = dict.fromkeys(bus.motors, 0.0)
        bus.write.return_value = None
        return bus

    # Patch the import site used by SO107FollowerPredictive (each arm
    # is a predictive instance, so this is the only patch needed —
    # BiSO107FollowerPredictive doesn't construct buses directly).
    from lerobot.robots.so107_follower_predictive import (
        SO107FollowerPredictive,
    )

    with (
        patch(
            "lerobot.robots.so_follower.so_follower.FeetechMotorsBus",
            side_effect=_bus_side_effect,
        ),
        patch.object(SO107FollowerPredictive, "configure", lambda self: None),
    ):
        cfg = BiSO107FollowerPredictiveConfig(
            left_arm_port="/dev/null_left",
            right_arm_port="/dev/null_right",
            left_arm_use_degrees=False,
            right_arm_use_degrees=False,
            id="white_test",
            control_rate_hz=500.0,
            adaptive=False,
        )
        robot = BiSO107FollowerPredictive(cfg)
        robot.connect(calibrate=False)
        try:
            yield robot, left_bus, right_bus
        finally:
            if robot.is_connected:
                robot.disconnect()


def test_two_independent_controller_threads(bi_follower):
    """Each arm must have its own running controller thread."""
    robot, _l, _r = bi_follower
    left_thread = robot.left_arm._controller._thread
    right_thread = robot.right_arm._controller._thread
    assert left_thread is not None and right_thread is not None
    assert left_thread is not right_thread
    assert left_thread.is_alive()
    assert right_thread.is_alive()


def test_controller_settings_shared_across_arms(bi_follower):
    """The bimanual config's controller knobs propagate to both arms."""
    robot, _l, _r = bi_follower
    for arm in (robot.left_arm, robot.right_arm):
        assert arm.config.lookahead_ms == 80.0
        assert arm.config.max_lookahead_ms == 150.0  # PredictiveControllerConfig default
        assert arm.config.corrector_alpha == 1.0
        assert arm.config.velocity_window_ms == 70.0
        assert arm.config.control_rate_hz == 500.0
        assert arm.config.adaptive is False
        assert arm.config.max_step_deg == 3.0


def test_send_action_routes_prefixed_keys_to_correct_arm(bi_follower):
    """left_*.pos must land on left bus, right_*.pos on right bus."""
    robot, left_bus, right_bus = bi_follower
    left_bus.sync_write_log.clear()
    right_bus.sync_write_log.clear()

    action = {}
    for m in _MOTOR_NAMES:
        action[f"left_{m}.pos"] = 1.0
        action[f"right_{m}.pos"] = 2.0
    robot.send_action(action)
    # Wait for both control threads to land at least one tick.
    time.sleep(0.05)

    left_goals = [gd for name, gd in left_bus.sync_write_log if name == "Goal_Position"]
    right_goals = [gd for name, gd in right_bus.sync_write_log if name == "Goal_Position"]
    assert left_goals, "left arm controller never wrote Goal_Position"
    assert right_goals, "right arm controller never wrote Goal_Position"

    # Each bus saw values that match its arm's intent (1.0 vs 2.0), and
    # no cross-talk — left bus should never see a 2.0 goal in this test.
    # Tolerance is loose because the controller's velocity LSQ over a
    # near-constant intent stream produces a tiny non-zero slope, so
    # `intent + v * L` drifts ~1e-5 from the raw intent value. The
    # test's purpose is routing (left ≠ right), not bit-equality.
    for gd in left_goals:
        for v in gd.values():
            assert abs(v - 1.0) < 0.01, f"left bus got {v}, expected ~1.0"
    for gd in right_goals:
        for v in gd.values():
            assert abs(v - 2.0) < 0.01, f"right bus got {v}, expected ~2.0"


def test_disconnect_stops_both_threads(bi_follower):
    robot, _l, _r = bi_follower
    left_thread = robot.left_arm._controller._thread
    right_thread = robot.right_arm._controller._thread
    robot.disconnect()
    left_thread.join(timeout=1.0)
    right_thread.join(timeout=1.0)
    assert not left_thread.is_alive()
    assert not right_thread.is_alive()


def test_get_observation_returns_prefixed_keys(bi_follower):
    robot, _l, _r = bi_follower
    obs = robot.get_observation()
    expected = {f"left_{m}.pos" for m in _MOTOR_NAMES} | {f"right_{m}.pos" for m in _MOTOR_NAMES}
    # observation_features may include cameras; the motor subset must be
    # exactly the expected prefixed names.
    motor_keys = {k for k in obs if k.endswith(".pos")}
    assert motor_keys == expected


def test_attach_teleop_routes_per_arm(bi_follower):
    """A bimanual leader teleop with left_arm / right_arm sub-attributes
    must be split per-arm — each follower controller polls its own
    arm's teleop, not the bimanual wrapper."""
    robot, _l, _r = bi_follower

    bi_teleop = MagicMock()
    bi_teleop.left_arm = MagicMock()
    bi_teleop.right_arm = MagicMock()

    robot.attach_teleop(bi_teleop)

    assert robot.left_arm._controller._teleop is bi_teleop.left_arm
    assert robot.right_arm._controller._teleop is bi_teleop.right_arm

    robot.attach_teleop(None)
    assert robot.left_arm._controller._teleop is None
    assert robot.right_arm._controller._teleop is None


def test_attach_teleop_skips_non_bimanual_teleop(bi_follower):
    """A non-bimanual teleop (e.g. TrajectoryReplayTeleop has no
    left_arm/right_arm attrs but DOES support the chunk path via
    send_action) shouldn't trigger an error from attach_teleop —
    the bimanual robot just no-ops the attach and lets send_action
    route the chunk. Binding the same single-arm teleop to both
    arms would have mixed up left/right poses on the buses, so we
    skip rather than do that. Test asserts: no exception, and
    neither arm's controller has _teleop bound."""
    robot, _l, _r = bi_follower
    single_arm_teleop = MagicMock(spec=["get_action"])  # no left_arm / right_arm
    # Should NOT raise.
    robot.attach_teleop(single_arm_teleop)
    assert robot.left_arm._controller._teleop is None
    assert robot.right_arm._controller._teleop is None


def test_attach_teleop_wires_cartesian_adapter(bi_follower):
    """A bimanual Cartesian VR teleop (Quest-like — has ``left_target_x`` in
    ``action_features.names``) must trigger the Cartesian IK adapter path:
    the adapter is built and started, its per-arm sub-teleops are routed
    to each predictive arm's controller, and an ``action_transform`` is
    installed on the teleop so script-side reads return joint dicts.
    Detaching stops the adapter and clears the routing.
    """
    robot, _l, _r = bi_follower

    # Build a fake Cartesian teleop with the Quest-shaped action_features
    # and a settable action_transform. The adapter only reads
    # ``get_action()`` and ``set_action_transform()`` — that's enough.
    installed_transform: list = []  # captures whatever the follower installs

    cart_teleop = MagicMock()
    cart_teleop.action_features = {
        "names": {
            "left_enabled": 0,
            "left_target_x": 1,
            "right_enabled": 8,
            "right_target_x": 9,
        }
    }
    cart_teleop.get_action.return_value = {
        "left_enabled": 0.0,
        "left_target_x": 0.0,
        "left_target_y": 0.0,
        "left_target_z": 0.0,
        "left_target_wx": 0.0,
        "left_target_wy": 0.0,
        "left_target_wz": 0.0,
        "left_gripper_pos": 0.0,
        "right_enabled": 0.0,
        "right_target_x": 0.0,
        "right_target_y": 0.0,
        "right_target_z": 0.0,
        "right_target_wx": 0.0,
        "right_target_wy": 0.0,
        "right_target_wz": 0.0,
        "right_gripper_pos": 0.0,
    }
    cart_teleop.set_action_transform = lambda fn: installed_transform.append(fn)
    # The "left_arm"/"right_arm" attribute MUST NOT be there — otherwise the
    # leader-path branch fires first. MagicMock auto-creates attrs on access,
    # so explicitly hasattr-block.
    del cart_teleop.left_arm
    del cart_teleop.right_arm

    # Pretend IK kinematics are available even if pin-pink isn't installed
    # in the test env — we don't need real IK to test the wiring.
    fake_kin = MagicMock()
    fake_kin.forward_kinematics.return_value = MagicMock()
    robot._ik_kinematics = {"left": fake_kin, "right": fake_kin}

    # Patch the IK-controller and bimanual-transform factories so the
    # adapter gets a deterministic transform that doesn't need the real IK.
    with (
        patch(
            "lerobot.robots.so107_description.cartesian_ik.make_so107_arm_ik_controller",
            side_effect=lambda kin, q_init, *_args, **_kw: MagicMock(),
        ),
        patch(
            "lerobot.robots.so107_description.cartesian_ik.make_bimanual_ik_transform",
            return_value=lambda _action: {
                **{f"left_{m}.pos": 0.0 for m in _MOTOR_NAMES},
                **{f"right_{m}.pos": 0.0 for m in _MOTOR_NAMES},
            },
        ),
    ):
        robot.attach_teleop(cart_teleop)

    # Adapter installed and running, transform installed on teleop.
    assert robot._cartesian_adapter is not None
    assert robot._cartesian_adapter.is_running
    assert len(installed_transform) == 1, "an action_transform must be installed on the Cartesian teleop"

    # Per-arm controllers see the adapter's sub-teleops (not the bimanual
    # wrapper) — same contract as the leader path.
    assert robot.left_arm._controller._teleop is robot._cartesian_adapter.left_arm
    assert robot.right_arm._controller._teleop is robot._cartesian_adapter.right_arm

    # Let the adapter's background loop run at least once so the cache
    # has a sample, then verify the per-arm sub-teleops return one
    # arm's unprefixed motor keys.
    time.sleep(0.05)
    left_dict = robot._cartesian_adapter.left_arm.get_action()
    assert left_dict, "adapter's left sub-teleop must produce a non-empty dict after one tick"
    assert all(k.endswith(".pos") and "left_" not in k for k in left_dict), (
        f"unexpected keys in left arm dict: {list(left_dict)}"
    )

    # Detach: adapter stops, controllers' teleop refs cleared.
    robot.attach_teleop(None)
    assert robot._cartesian_adapter is None
    assert robot.left_arm._controller._teleop is None
    assert robot.right_arm._controller._teleop is None


def test_cartesian_adapter_install_surfaces_hold_per_arm_to_teleop(bi_follower):
    """The IK-hold rumble path on the Quest VR teleop polls its installed
    ``action_transform.hold_per_arm``. On the plain follower the installed
    transform IS the BimanualSO107IKTransform which exposes that property.
    On the predictive path the installed transform is a thin adapter shim
    (the per-arm controllers read the adapter cache directly), so the shim
    must ALSO expose hold_per_arm — otherwise the rumble dies at this seam.

    Regression for: on-hardware Quest VR session reported no haptic at all
    (predictive path is the on-hardware path).
    """
    from unittest.mock import patch

    robot, _l, _r = bi_follower
    installed = []

    teleop = MagicMock()
    teleop.action_features = {
        "names": {"left_target_x": 0, "right_target_x": 1, "left_enabled": 2, "right_enabled": 3}
    }
    teleop.get_action.return_value = {}
    teleop.set_action_transform = installed.append
    del teleop.left_arm
    del teleop.right_arm

    fake_kin = MagicMock()
    fake_kin.forward_kinematics.return_value = MagicMock()
    robot._ik_kinematics = {"left": fake_kin, "right": fake_kin}

    # Fake the bimanual transform with a settable hold_per_arm so we can
    # see whether the predictive-path shim propagates it.
    class _FakeTransform:
        def __init__(self):
            self.hold_per_arm = (False, False)

        def __call__(self, _action):
            return {}

    fake_transform = _FakeTransform()
    with patch(
        "lerobot.robots.so107_description.cartesian_ik.build_so107_bimanual_ik_transform",
        return_value=fake_transform,
    ):
        robot.attach_teleop(teleop)

    try:
        assert len(installed) == 1, "an action_transform must be installed"
        shim = installed[0]
        # Initial state propagates.
        assert shim.hold_per_arm == (False, False)
        # Flipping the underlying transform's hold state shows up live —
        # not snapshotted at install time, because the rumble path reads
        # this every tick.
        fake_transform.hold_per_arm = (True, False)
        assert shim.hold_per_arm == (True, False)
        fake_transform.hold_per_arm = (False, True)
        assert shim.hold_per_arm == (False, True)
        # Callable contract still works (used by script-side get_action).
        assert isinstance(shim({}), dict)
    finally:
        robot.attach_teleop(None)


def test_send_action_rejects_chunk_missing_one_arms_keys(bi_follower):
    """A frame missing all left_* keys (or right_*) would have produced an
    empty per-arm sub-chunk, which then races: left_arm.send_action
    succeeds, right_arm raises on the strict-key check — leaving the arms
    out of sync. The composer must fail fast before either side-effect."""
    from lerobot.types import ActionChunk

    robot, _left_bus, _right_bus = bi_follower

    # Frame 0 is well-formed; frame 1 omits all right_* keys.
    frame_full = {f"left_{m}.pos": 1.0 for m in _MOTOR_NAMES}
    frame_full.update({f"right_{m}.pos": 2.0 for m in _MOTOR_NAMES})
    frame_left_only = {f"left_{m}.pos": 1.0 for m in _MOTOR_NAMES}
    chunk = ActionChunk(fps=30.0, frames=(frame_full, frame_left_only))

    with pytest.raises(ValueError, match="missing per-arm keys"):
        robot.send_action(chunk)

    # Error happened before either arm's _latest_intent was updated to a
    # stale value — verify by re-sending a valid chunk and confirming
    # both arms accept it cleanly.
    valid_chunk = ActionChunk(fps=30.0, frames=(frame_full,))
    robot.send_action(valid_chunk)  # must not raise


def test_send_action_rejects_chunk_missing_first_arms_keys(bi_follower):
    """Mirror of the above: frame missing all left_* keys."""
    from lerobot.types import ActionChunk

    robot, _l, _r = bi_follower
    frame_right_only = {f"right_{m}.pos": 2.0 for m in _MOTOR_NAMES}
    chunk = ActionChunk(fps=30.0, frames=(frame_right_only,))

    with pytest.raises(ValueError, match="missing per-arm keys"):
        robot.send_action(chunk)
