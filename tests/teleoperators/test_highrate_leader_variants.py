"""Behavioural-parity tests parameterized across high-rate leader variants.

Mirrors ``tests/robots/test_predictive_mixin_variants.py`` for the
:class:`HighRateLeaderMixin` family. Any leader composed of the mixin
+ a base leader class must pass these tests — guards the "forgot to
forward config field" / "MRO is wrong" / "thread doesn't start" class
of regression that would only surface on hardware otherwise.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from lerobot.teleoperators.so107_leader_highrate import (
    SO107LeaderHighRate,
    SO107LeaderHighRateConfig,
)
from lerobot.teleoperators.so_leader_highrate import (
    SOLeaderHighRate,
    SOLeaderHighRateConfig,
)

# (variant_class, config_class, expected_motor_count)
VARIANTS = [
    pytest.param(SO107LeaderHighRate, SO107LeaderHighRateConfig, 7, id="so107"),
    pytest.param(SOLeaderHighRate, SOLeaderHighRateConfig, 6, id="so100_101"),
]


def _make_bus_mock():
    bus = MagicMock(name="FeetechBusMock")
    bus.is_connected = False
    bus.is_calibrated = True
    bus.sync_read_call_count = 0

    def _connect():
        bus.is_connected = True

    def _disconnect(*_a, **_kw):
        bus.is_connected = False

    def _sync_read(_data_name, **_kw):
        bus.sync_read_call_count += 1
        return dict.fromkeys(bus.motors, 0.0)

    @contextmanager
    def _dummy_cm():
        yield

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect
    bus.sync_read.side_effect = _sync_read
    bus.torque_disabled.side_effect = _dummy_cm
    return bus


@contextmanager
def _connected(variant_cls, config_cls):
    bus_mock = _make_bus_mock()

    def _bus_side_effect(*_a, **kw):
        bus_mock.motors = kw["motors"]
        return bus_mock

    with (
        patch(
            "lerobot.teleoperators.so_leader.so_leader.FeetechMotorsBus",
            side_effect=_bus_side_effect,
        ),
        # Skip configure() — bus.write calls aren't modelled by the mock.
        patch.object(variant_cls, "configure", lambda self: None),
    ):
        cfg = config_cls(port="/dev/null", id=f"{variant_cls.name}_test", read_rate_hz=500.0)
        teleop = variant_cls(cfg)
        teleop.connect(calibrate=False)
        try:
            yield teleop, bus_mock
        finally:
            if teleop.is_connected:
                teleop.disconnect()


@pytest.mark.parametrize("variant_cls,config_cls,n_motors", VARIANTS)
def test_bus_is_locked_after_construction(variant_cls, config_cls, n_motors):
    from lerobot.motors.locked_bus import LockedBus

    with _connected(variant_cls, config_cls) as (teleop, _bus):
        assert isinstance(teleop.bus, LockedBus), (
            f"{variant_cls.__name__}: self.bus is {type(teleop.bus).__name__}, expected LockedBus"
        )


@pytest.mark.parametrize("variant_cls,config_cls,n_motors", VARIANTS)
def test_read_thread_runs_after_connect(variant_cls, config_cls, n_motors):
    with _connected(variant_cls, config_cls) as (teleop, _bus):
        assert teleop._read_thread is not None
        assert teleop._read_thread.is_alive()


@pytest.mark.parametrize("variant_cls,config_cls,n_motors", VARIANTS)
def test_get_action_returns_cached_pose(variant_cls, config_cls, n_motors):
    """After the background thread warms up, get_action() returns the
    cache without re-reading the bus."""
    with _connected(variant_cls, config_cls) as (teleop, bus):
        # Wait for at least one read cycle.
        deadline = time.time() + 0.5
        while bus.sync_read_call_count < 2 and time.time() < deadline:
            time.sleep(0.005)
        assert bus.sync_read_call_count >= 1, "background thread never read the bus"
        # Now measure: get_action() must NOT bump the read count.
        baseline = bus.sync_read_call_count
        pose = teleop.get_action()
        # Tiny race: a tick may fire between get_action and the read of
        # sync_read_call_count. Allow at most 1 tick.
        assert bus.sync_read_call_count - baseline <= 1, (
            f"{variant_cls.__name__}: get_action triggered bus reads (delta={bus.sync_read_call_count - baseline})"
        )
        assert len(pose) == n_motors


@pytest.mark.parametrize("variant_cls,config_cls,n_motors", VARIANTS)
def test_disconnect_stops_thread_and_clears_cache(variant_cls, config_cls, n_motors):
    with _connected(variant_cls, config_cls) as (teleop, _bus):
        assert teleop._read_thread is not None
        teleop.disconnect()
        assert teleop._read_thread is None
        assert teleop._cached_pose is None, (
            "disconnect must clear the cache to prevent stale-data leak across reconnects"
        )


def test_highrate_config_defaults_are_shared_across_variants():
    """Both variants inherit HighRateLeaderConfig — read_rate_hz default
    must match."""
    so107 = SO107LeaderHighRateConfig(port="/dev/null", id="x")
    so100 = SOLeaderHighRateConfig(port="/dev/null", id="x")
    assert so107.read_rate_hz == so100.read_rate_hz
