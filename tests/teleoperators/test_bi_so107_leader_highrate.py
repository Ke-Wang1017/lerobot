"""Smoke test for BiSO107LeaderHighRate.

The single-arm tests in test_so107_leader_highrate.py exercise the
thread + cache surface; this file just verifies the bimanual composer
correctly stacks two high-rate arms with independent threads.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from lerobot.teleoperators.bi_so107_leader_highrate import (
    BiSO107LeaderHighRate,
    BiSO107LeaderHighRateConfig,
)
from lerobot.teleoperators.so107_leader_highrate import SO107LeaderHighRate


def _make_bus_mock(name: str):
    bus = MagicMock(name=name)
    bus.is_connected = False
    bus.is_calibrated = True

    def _connect():
        bus.is_connected = True

    def _disconnect(*_args, **_kwargs):
        bus.is_connected = False

    def _sync_read(*_args, **_kwargs):
        return dict.fromkeys(bus.motors, 0.0)

    @contextmanager
    def _dummy_cm():
        yield

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect
    bus.sync_read.side_effect = _sync_read
    bus.torque_disabled.side_effect = _dummy_cm
    return bus


@pytest.fixture
def bi_leader():
    left_bus = _make_bus_mock("LeftBus")
    right_bus = _make_bus_mock("RightBus")
    bus_iter = iter([left_bus, right_bus])

    def _bus_side_effect(*_args, **kwargs):
        bus = next(bus_iter)
        bus.motors = kwargs["motors"]
        return bus

    with (
        patch(
            "lerobot.teleoperators.so_leader.so_leader.FeetechMotorsBus",
            side_effect=_bus_side_effect,
        ),
        patch.object(SO107LeaderHighRate, "configure", lambda self: None),
    ):
        cfg = BiSO107LeaderHighRateConfig(
            id="test",
            left_arm_port="/dev/null_l",
            right_arm_port="/dev/null_r",
            read_rate_hz=500.0,
        )
        teleop = BiSO107LeaderHighRate(cfg)
        teleop.connect(calibrate=False)
        try:
            yield teleop, left_bus, right_bus
        finally:
            if teleop.is_connected:
                teleop.disconnect()


def test_both_arms_are_high_rate_instances(bi_leader):
    teleop, _l, _r = bi_leader
    assert isinstance(teleop.left_arm, SO107LeaderHighRate)
    assert isinstance(teleop.right_arm, SO107LeaderHighRate)


def test_per_arm_read_threads_are_independent(bi_leader):
    teleop, _l, _r = bi_leader
    lt = teleop.left_arm._read_thread
    rt = teleop.right_arm._read_thread
    assert lt is not None and rt is not None
    assert lt is not rt
    assert lt.is_alive() and rt.is_alive()


def test_read_rate_propagates_to_both_arms(bi_leader):
    teleop, _l, _r = bi_leader
    assert teleop.left_arm.config.read_rate_hz == 500.0
    assert teleop.right_arm.config.read_rate_hz == 500.0


def test_get_action_returns_prefixed_keys_from_both_arms(bi_leader):
    teleop, _l, _r = bi_leader
    # Wait for both caches to warm.
    import time

    for _ in range(50):
        if teleop.left_arm._cached_pose is not None and teleop.right_arm._cached_pose is not None:
            break
        time.sleep(0.005)
    action = teleop.get_action()
    motor_keys = {k for k in action if k.endswith(".pos")}
    expected_left = {"left_shoulder_pan.pos", "left_gripper.pos"}
    expected_right = {"right_shoulder_pan.pos", "right_gripper.pos"}
    assert expected_left.issubset(motor_keys)
    assert expected_right.issubset(motor_keys)


def test_disconnect_stops_both_read_threads(bi_leader):
    teleop, _l, _r = bi_leader
    lt = teleop.left_arm._read_thread
    rt = teleop.right_arm._read_thread
    teleop.disconnect()
    lt.join(timeout=1.0)
    rt.join(timeout=1.0)
    assert not lt.is_alive()
    assert not rt.is_alive()
