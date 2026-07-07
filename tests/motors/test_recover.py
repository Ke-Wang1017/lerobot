# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for ``lerobot.motors.recovery``.

The recovery helpers need to exercise the bus's ``_connect`` / ``ping`` /
``disable_torque`` / ``_write`` machinery. Rather than wire up the
byte-level mock-serial harness used elsewhere, we patch these methods on a
real ``FeetechMotorsBus`` instance to keep the tests focused on the
recovery logic.

**Scope:** these tests cover control flow and report shape only —
branching, retry ordering, accumulation of the ``RecoveryReport`` fields,
robot-level introspection, JSON serialisation. They do **not** verify
wire-protocol or motor-firmware behaviour (whether the SDK emits the right
bytes, whether ``Torque_Enable=0`` actually clears an STS3215 overload
latch, whether retry timing is sufficient for the firmware's protection
circuit to release). That contract is validated separately against real
hardware — see the "Validation" section in
``src/lerobot/motors/recovery.py`` for the procedure.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.recovery import RecoveryReport, recover_bus, recover_robot

try:
    import scservo_sdk  # noqa: F401  -- module skipped if Feetech SDK absent

    from lerobot.motors.feetech import FeetechMotorsBus
except (ImportError, ModuleNotFoundError):
    pytest.skip("scservo_sdk not available", allow_module_level=True)


def _three_sts_motors() -> dict[str, Motor]:
    return {
        "shoulder": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
        "elbow": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
        "wrist": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
    }


def _build_bus(motors: dict[str, Motor] | None = None) -> FeetechMotorsBus:
    motors = motors if motors is not None else _three_sts_motors()
    return FeetechMotorsBus(port="/dev/null-test", motors=motors)


def test_recover_bus_all_responsive():
    """Happy path: every motor pings, every torque write succeeds."""
    bus = _build_bus()
    bus.port_handler = MagicMock()

    with (
        patch.object(bus, "_connect"),
        patch.object(bus, "ping", return_value=777),
        patch.object(bus, "disable_torque") as mock_disable,
        patch.object(bus, "_write") as mock_write,
    ):
        report = recover_bus(bus, num_retry=2)

    assert report.responsive_ids == [1, 2, 3]
    assert report.torque_disabled_ids == [1, 2, 3]
    assert report.recovered_ids == []
    assert report.still_unreachable_ids == []
    assert report.errors == {}
    assert mock_disable.call_count == 3
    mock_write.assert_not_called()  # no Feetech latch-clear path needed


def test_recover_bus_clears_feetech_overload_latch():
    """Initial ping misses motor 2; direct ``Torque_Enable=0`` + retry ping
    recovers it. This is the exact path that fixed the user's wrist_roll."""
    bus = _build_bus()
    bus.port_handler = MagicMock()

    initial = {1: 777, 2: None, 3: 777}
    after_clear = {2: 777}

    def ping_side_effect(motor_id, num_retry=0):
        seen = ping_side_effect.calls.get(motor_id, 0)
        ping_side_effect.calls[motor_id] = seen + 1
        return initial.get(motor_id) if seen == 0 else after_clear.get(motor_id)

    ping_side_effect.calls = {}

    with (
        patch.object(bus, "_connect"),
        patch.object(bus, "ping", side_effect=ping_side_effect),
        patch.object(bus, "disable_torque") as mock_disable,
        patch.object(bus, "_write") as mock_write,
    ):
        report = recover_bus(bus, num_retry=2)

    assert sorted(report.responsive_ids) == [1, 3]
    assert report.recovered_ids == [2]
    assert sorted(report.torque_disabled_ids) == [1, 2, 3]
    assert report.still_unreachable_ids == []

    # Direct _write was used to clear the latch on the silent motor only
    assert [c.args[2] for c in mock_write.call_args_list] == [2]
    # disable_torque called for every recovered/responsive motor by name
    disabled_names = sorted(c.kwargs["motors"][0] for c in mock_disable.call_args_list)
    assert disabled_names == ["elbow", "shoulder", "wrist"]


def test_recover_bus_truly_dead_motor_marked_unreachable():
    """A motor that doesn't respond to either pass ends up in
    ``still_unreachable_ids``."""
    bus = _build_bus()
    bus.port_handler = MagicMock()

    with (
        patch.object(bus, "_connect"),
        patch.object(bus, "ping", side_effect=lambda mid, num_retry=0: 777 if mid != 2 else None),
        patch.object(bus, "disable_torque"),
        patch.object(bus, "_write"),
    ):
        report = recover_bus(bus, num_retry=1)

    assert sorted(report.responsive_ids) == [1, 3]
    assert report.recovered_ids == []
    assert report.still_unreachable_ids == [2]


def test_recover_bus_port_open_failure_returns_note():
    """Failure to open the port turns into a note on the report rather than
    an exception. Endpoints can show this verbatim to the user."""
    bus = _build_bus()
    bus.port_handler = MagicMock()

    with patch.object(bus, "_connect", side_effect=OSError("permission denied")):
        report = recover_bus(bus)

    assert report.responsive_ids == []
    assert report.torque_disabled_ids == []
    assert any("could not open port" in n for n in report.notes)


def test_recover_bus_disable_torque_failure_recorded_in_errors():
    """A motor pings but ``disable_torque`` fails — error captured per-id,
    other motors still processed."""
    bus = _build_bus()
    bus.port_handler = MagicMock()

    def disable_side_effect(motors, num_retry=0):
        if "elbow" in motors:
            raise ConnectionError("simulated write failure")

    with (
        patch.object(bus, "_connect"),
        patch.object(bus, "ping", return_value=777),
        patch.object(bus, "disable_torque", side_effect=disable_side_effect),
        patch.object(bus, "_write"),
    ):
        report = recover_bus(bus)

    assert sorted(report.responsive_ids) == [1, 2, 3]
    assert sorted(report.torque_disabled_ids) == [1, 3]
    assert 2 in report.errors
    assert "simulated write failure" in report.errors[2]


def test_recovery_report_to_dict_is_json_friendly():
    """Endpoint serialisation sanity check."""
    report = RecoveryReport(port="/dev/ttyACM2")
    report.responsive_ids = [1, 3]
    report.recovered_ids = [2]
    report.torque_disabled_ids = [1, 2, 3]
    report.errors = {4: "something"}
    report.notes = ["test note"]

    d = report.to_dict()
    assert d["port"] == "/dev/ttyACM2"
    assert d["responsive_ids"] == [1, 3]
    assert d["recovered_ids"] == [2]
    assert d["torque_disabled_ids"] == [1, 2, 3]
    assert d["errors"] == {"4": "something"}  # JSON-safe string keys
    assert d["notes"] == ["test note"]


def test_recover_robot_single_arm_invokes_recover_bus_once():
    """Robot with a single ``self.bus`` produces one report."""
    robot = MagicMock(spec=[])  # no spec -> only the attrs we set exist
    robot.bus = _build_bus()
    robot.bus.port_handler = MagicMock()

    fake_report = RecoveryReport(port="/dev/null-test", responsive_ids=[1, 2, 3])
    with patch("lerobot.motors.recovery.recover_bus", return_value=fake_report) as mock_rb:
        # robot is a MagicMock spec=[] so it isn't a Robot instance — call the
        # _walk path indirectly via recover_robot.
        from lerobot.robots.robot import Robot

        # Build a minimal Robot subclass instance for the isinstance check
        class _StubRobot(Robot):
            config_class = type("C", (), {})
            name = "stub"

            def __init__(self):  # bypass parent __init__ (needs config dirs etc.)
                self.bus = robot.bus

            @property
            def observation_features(self):
                return {}

            @property
            def action_features(self):
                return {}

            @property
            def is_connected(self):
                return False

            def connect(self, calibrate=True):
                pass

            @property
            def is_calibrated(self):
                return True

            def calibrate(self):
                pass

            def configure(self):
                pass

            def get_observation(self):
                return {}

            def send_action(self, action):
                return action

            def disconnect(self):
                pass

        reports = recover_robot(_StubRobot())

    assert len(reports) == 1
    assert reports[0] is fake_report
    assert mock_rb.call_count == 1


def test_recover_robot_bi_arm_recurses_into_nested_robots():
    """Bi-arm composite with ``self.left_arm`` / ``self.right_arm`` produces
    one report per arm, by recursing through nested Robot instances."""
    from lerobot.robots.robot import Robot

    class _Leaf(Robot):
        config_class = type("C", (), {})
        name = "leaf"

        def __init__(self, port):
            self.bus = _build_bus()
            self.bus.port = port
            self.bus.port_handler = MagicMock()

        @property
        def observation_features(self):
            return {}

        @property
        def action_features(self):
            return {}

        @property
        def is_connected(self):
            return False

        def connect(self, calibrate=True):
            pass

        @property
        def is_calibrated(self):
            return True

        def calibrate(self):
            pass

        def configure(self):
            pass

        def get_observation(self):
            return {}

        def send_action(self, action):
            return action

        def disconnect(self):
            pass

    class _Composite(_Leaf):
        name = "composite"

        def __init__(self):
            self.left_arm = _Leaf("/dev/ttyACM_LEFT")
            self.right_arm = _Leaf("/dev/ttyACM_RIGHT")

    seen_ports = []

    def fake_recover_bus(bus, num_retry=3):
        seen_ports.append(bus.port)
        return RecoveryReport(port=bus.port)

    with patch("lerobot.motors.recovery.recover_bus", side_effect=fake_recover_bus):
        reports = recover_robot(_Composite())

    assert sorted(seen_ports) == ["/dev/ttyACM_LEFT", "/dev/ttyACM_RIGHT"]
    assert len(reports) == 2


def test_recover_robot_with_no_buses_returns_placeholder():
    """A robot with no ``SerialMotorsBus`` attribute (e.g. gRPC-backed)
    produces a single placeholder report — never an empty list."""
    from lerobot.robots.robot import Robot

    class _NoBus(Robot):
        config_class = type("C", (), {})
        name = "nobus"

        def __init__(self):
            self.client = "grpc-client-stub"  # not a bus, not a robot

        @property
        def observation_features(self):
            return {}

        @property
        def action_features(self):
            return {}

        @property
        def is_connected(self):
            return False

        def connect(self, calibrate=True):
            pass

        @property
        def is_calibrated(self):
            return True

        def calibrate(self):
            pass

        def configure(self):
            pass

        def get_observation(self):
            return {}

        def send_action(self, action):
            return action

        def disconnect(self):
            pass

    reports = recover_robot(_NoBus())

    assert len(reports) == 1
    assert reports[0].port is None
    # Frontend (robot.js::recoverRobot) substring-matches the literal "no MotorsBus"
    # to surface "Recovery not supported" instead of a misleading success toast.
    # If you change this wording, also update the frontend detector.
    assert any("no MotorsBus" in n for n in reports[0].notes)


def test_recover_robot_unwraps_locked_bus():
    """Robot.bus wrapped in ``LockedBus`` (predictive controllers do this
    for thread-safety) should still be discoverable by recovery.

    Regression: the walker used to skip ``LockedBus`` because
    ``isinstance(LockedBus, SerialMotorsBus)`` is False — predictive
    robots would falsely report "Recovery not supported" even though
    they have a real Feetech bus underneath.
    """
    from lerobot.motors.locked_bus import LockedBus
    from lerobot.robots.robot import Robot

    inner_bus = _build_bus()
    inner_bus.port_handler = MagicMock()
    wrapped = LockedBus(inner_bus)

    class _PredictiveStubRobot(Robot):
        config_class = type("C", (), {})
        name = "predictive_stub"

        def __init__(self):
            self.bus = wrapped  # the LockedBus proxy, as predictive robots do

        @property
        def observation_features(self):
            return {}

        @property
        def action_features(self):
            return {}

        @property
        def is_connected(self):
            return False

        def connect(self, calibrate=True):
            pass

        @property
        def is_calibrated(self):
            return True

        def calibrate(self):
            pass

        def configure(self):
            pass

        def get_observation(self):
            return {}

        def send_action(self, action):
            return action

        def disconnect(self):
            pass

    fake_report = RecoveryReport(port="/dev/null-test", responsive_ids=[1, 2, 3])
    with patch("lerobot.motors.recovery.recover_bus", return_value=fake_report) as mock_rb:
        reports = recover_robot(_PredictiveStubRobot())

    # The walker MUST unwrap LockedBus and invoke recover_bus on the inner
    # SerialMotorsBus — not bail with a "no MotorsBus" placeholder.
    assert mock_rb.call_count == 1
    # The bus passed to recover_bus must be the unwrapped inner bus.
    passed_bus = mock_rb.call_args[0][0]
    assert passed_bus is inner_bus
    assert len(reports) == 1
    assert reports[0] is fake_report
    assert not any("no MotorsBus" in n for n in reports[0].notes)
