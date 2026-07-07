"""Behavioural-parity tests parameterized across predictive variants.

These tests pin the invariants that any class composed of
:class:`PredictiveLookaheadMixin` + a base follower must satisfy. If we
add a new variant (e.g. an Openarm predictive follower) and inherit from
the mixin, these tests run against it automatically — catching the
"forgot to forward field X" / "MRO is wrong" / "controller doesn't start"
class of regression that would otherwise only surface on hardware.

What's NOT covered here:
  * Algorithmic correctness of the controller itself (tests in
    ``test_predictive_controller_rate_agnostic.py`` cover that, and
    they're variant-independent because they instantiate the controller
    directly).
  * Per-motor-count behaviour (the controller's
    ``[f"{m}.pos" for m in bus.motors]`` mapping is motor-count-agnostic
    by construction — verified by the smoke tests below across 6 and 7
    motor counts).
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.so107_follower_predictive import (
    SO107FollowerPredictive,
    SO107FollowerPredictiveRobotConfig,
)
from lerobot.robots.so_follower_predictive import (
    SOFollowerPredictive,
    SOFollowerPredictiveRobotConfig,
)

# (variant_class, config_class, expected_motor_count)
VARIANTS = [
    pytest.param(SO107FollowerPredictive, SO107FollowerPredictiveRobotConfig, 7, id="so107"),
    pytest.param(SOFollowerPredictive, SOFollowerPredictiveRobotConfig, 6, id="so100_101"),
]


def _make_bus_mock():
    bus = MagicMock(name="FeetechBusMock")
    bus.is_connected = False
    bus.is_calibrated = True

    def _connect():
        bus.is_connected = True

    def _disconnect(*_a, **_kw):
        bus.is_connected = False

    @contextmanager
    def _dummy_cm():
        yield

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect
    bus.torque_disabled.side_effect = _dummy_cm
    bus.sync_write = MagicMock()
    bus.sync_read = MagicMock()
    return bus


@contextmanager
def _connected(variant_cls, config_cls):
    """Yield a connected variant instance backed by a mocked Feetech bus."""
    bus_mock = _make_bus_mock()

    def _bus_side_effect(*_a, **kw):
        bus_mock.motors = kw["motors"]
        bus_mock.sync_read.return_value = dict.fromkeys(bus_mock.motors, 0.0)
        return bus_mock

    with (
        patch(
            "lerobot.robots.so_follower.so_follower.FeetechMotorsBus",
            side_effect=_bus_side_effect,
        ),
        # Skip configure() — irrelevant to mixin tests, would require
        # more bus stubbing.
        patch.object(variant_cls, "configure", lambda self: None),
    ):
        cfg = config_cls(port="/dev/null", id=f"{variant_cls.name}_test", control_rate_hz=500.0)
        robot = variant_cls(cfg)
        robot.connect(calibrate=False)
        try:
            yield robot, bus_mock
        finally:
            if robot.is_connected:
                robot.disconnect()


# =============================================================================
# Per-variant invariants
# =============================================================================


@pytest.mark.parametrize("variant_cls,config_cls,n_motors", VARIANTS)
def test_bus_is_locked_after_construction(variant_cls, config_cls, n_motors):
    """The mixin's contract: ``self.bus`` must be a LockedBus proxy after
    construction. Recovery, GUI dropdowns, and downstream code all depend
    on this — the recovery walker specifically unwraps LockedBus."""
    from lerobot.motors.locked_bus import LockedBus

    with _connected(variant_cls, config_cls) as (robot, _bus_mock):
        assert isinstance(robot.bus, LockedBus), (
            f"{variant_cls.__name__}: self.bus is {type(robot.bus).__name__}, expected LockedBus"
        )


@pytest.mark.parametrize("variant_cls,config_cls,n_motors", VARIANTS)
def test_motor_count_passes_through_to_controller(variant_cls, config_cls, n_motors):
    """The controller's motor_keys is derived from the bus, NOT from the
    config or the variant class — adding a 6-motor or 7-motor variant
    should work with zero controller changes."""
    with _connected(variant_cls, config_cls) as (robot, _bus_mock):
        assert len(robot._controller._motor_keys) == n_motors, (
            f"{variant_cls.__name__}: controller saw {len(robot._controller._motor_keys)} "
            f"motor keys, expected {n_motors}"
        )


@pytest.mark.parametrize("variant_cls,config_cls,n_motors", VARIANTS)
def test_controller_thread_runs_after_connect(variant_cls, config_cls, n_motors):
    """``configure()`` is the mixin's slot for starting the worker thread.
    The thread MUST be alive after connect() (which the mixin chains
    through to base.connect → base.configure → controller.start)."""
    with _connected(variant_cls, config_cls) as (robot, _bus_mock):
        assert robot._controller._thread is not None, (
            f"{variant_cls.__name__}: controller thread not started after connect()"
        )
        assert robot._controller._thread.is_alive(), (
            f"{variant_cls.__name__}: controller thread is not alive after connect()"
        )


@pytest.mark.parametrize("variant_cls,config_cls,n_motors", VARIANTS)
def test_send_action_forwards_to_controller(variant_cls, config_cls, n_motors):
    """The mixin's send_action override must call controller.set_intent
    and return the operator's raw intent (dict path)."""
    with _connected(variant_cls, config_cls) as (robot, _bus_mock):
        intent = {f"{m}.pos": float(i) for i, m in enumerate(robot.bus.motors)}
        # The controller's set_intent is the load-bearing call.
        with patch.object(robot._controller, "set_intent") as mock_si:
            returned = robot.send_action(intent)
        assert mock_si.call_count == 1
        # Return value is the operator's raw intent (a dict for the dict path).
        assert returned == intent


@pytest.mark.parametrize("variant_cls,config_cls,n_motors", VARIANTS)
def test_attach_teleop_routes_to_controller(variant_cls, config_cls, n_motors):
    """attach_teleop is just a wrapper over controller.set_teleop with
    logging — confirming the wrapper exists across variants."""
    with _connected(variant_cls, config_cls) as (robot, _bus_mock):
        sentinel_teleop = MagicMock()
        robot.attach_teleop(sentinel_teleop)
        assert robot._controller._teleop is sentinel_teleop
        # Detach.
        robot.attach_teleop(None)
        assert robot._controller._teleop is None


@pytest.mark.parametrize("variant_cls,config_cls,n_motors", VARIANTS)
def test_disconnect_stops_controller_before_base(variant_cls, config_cls, n_motors):
    """Controller MUST stop before the base disconnects the bus —
    otherwise the writer thread's next sync_write races with port
    closure. The mixin enforces this by calling stop() before super()."""
    with _connected(variant_cls, config_cls) as (robot, _bus_mock):
        ctrl = robot._controller
        assert ctrl._thread is not None and ctrl._thread.is_alive()
        robot.disconnect()
        # After disconnect, controller stopped + bus closed.
        assert ctrl._thread is None
        # `is_connected` reads from bus.is_connected (set by the mock's
        # _disconnect side_effect).
        assert not robot.is_connected


# =============================================================================
# Cross-variant invariants — assert the two variants are CONFIGURED the same
# =============================================================================


def test_predictive_config_defaults_are_shared_across_variants():
    """Both variants inherit from PredictiveControllerConfig — defaults
    must be identical. If a knob's default ever diverges between the
    two, it's almost certainly a mistake (the variants share the same
    Feetech hardware family)."""
    so107 = SO107FollowerPredictiveRobotConfig(port="/dev/null", id="x")
    so100 = SOFollowerPredictiveRobotConfig(port="/dev/null", id="x")
    knobs = [
        "lookahead_ms",
        "max_lookahead_ms",
        "corrector_alpha",
        "velocity_window_ms",
        "velocity_estimator",
        "velocity_lowpass_hz",
        "amp_gate_lo",
        "amp_gate_hi",
        "control_rate_hz",
        "adaptive",
        "max_step_deg",
    ]
    for k in knobs:
        v107 = getattr(so107, k)
        v100 = getattr(so100, k)
        assert v107 == v100, (
            f"PredictiveControllerConfig knob {k!r} default diverges: SO107={v107!r} vs SO100/101={v100!r}"
        )
