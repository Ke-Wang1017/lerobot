"""Tests for SO107LeaderHighRate.

Drop-in replacement for SO107Leader that runs a background bus-read
thread. These tests verify the thread lifecycle, cache freshness, and
the bus-read-elimination contract — get_action() must not block on
the serial bus once the cache is warm.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from lerobot.teleoperators.so107_leader_highrate import (
    SO107LeaderHighRate,
    SO107LeaderHighRateConfig,
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


def _make_bus_mock(read_value: float = 0.0):
    """FeetechMotorsBus mock that records every sync_read call."""
    bus = MagicMock(name="FeetechBusMock")
    bus.is_connected = False
    bus.is_calibrated = True
    bus.sync_read_call_count = 0

    def _connect():
        bus.is_connected = True

    def _disconnect(*_args, **_kwargs):
        bus.is_connected = False

    def _sync_read(data_name, **_kwargs):
        bus.sync_read_call_count += 1
        return dict.fromkeys(bus.motors, read_value)

    @contextmanager
    def _dummy_cm():
        yield

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect
    bus.sync_read.side_effect = _sync_read
    bus.torque_disabled.side_effect = _dummy_cm
    return bus


@pytest.fixture
def leader():
    """Connected SO107LeaderHighRate with a mock bus."""
    bus_mock = _make_bus_mock()

    def _bus_side_effect(*_args, **kwargs):
        bus_mock.motors = kwargs["motors"]
        return bus_mock

    with (
        patch(
            "lerobot.teleoperators.so_leader.so_leader.FeetechMotorsBus",
            side_effect=_bus_side_effect,
        ),
        # Skip configure() — it issues bus.write calls that the mock
        # doesn't model and they aren't relevant to the read-thread tests.
        patch.object(SO107LeaderHighRate, "configure", lambda self: None),
    ):
        cfg = SO107LeaderHighRateConfig(
            port="/dev/null",
            id="test_leader",
            read_rate_hz=500.0,  # high so tests don't need to sleep long
        )
        teleop = SO107LeaderHighRate(cfg)
        teleop.connect(calibrate=False)
        try:
            yield teleop, bus_mock
        finally:
            if teleop.is_connected:
                teleop.disconnect()


class TestLifecycle:
    def test_connect_starts_read_thread(self, leader):
        teleop, _bus = leader
        assert teleop._read_thread is not None
        assert teleop._read_thread.is_alive()

    def test_disconnect_stops_read_thread(self, leader):
        teleop, _bus = leader
        thread = teleop._read_thread
        teleop.disconnect()
        thread.join(timeout=1.0)
        assert not thread.is_alive()

    def test_disconnect_stops_thread_before_closing_bus(self, leader):
        """Read-loop must stop before bus.disconnect() — otherwise the
        next sync_read in the loop races with the close and raises."""
        teleop, bus = leader

        # Track ordering: thread must have stopped before disconnect runs.
        stop_set_at = [None]
        original_set = teleop._read_stop.set

        def tracking_set():
            stop_set_at[0] = time.perf_counter()
            original_set()

        teleop._read_stop.set = tracking_set
        disconnect_at = [None]
        original_disconnect = bus.disconnect.side_effect

        def tracking_disconnect(*args, **kwargs):
            disconnect_at[0] = time.perf_counter()
            return original_disconnect(*args, **kwargs)

        bus.disconnect.side_effect = tracking_disconnect
        teleop.disconnect()
        assert stop_set_at[0] is not None
        assert disconnect_at[0] is not None
        assert stop_set_at[0] < disconnect_at[0]

    def test_no_thread_leak_across_connect_cycles(self, leader):
        teleop, _bus = leader
        baseline = threading.active_count()
        teleop.disconnect()
        teleop.connect(calibrate=False)
        teleop.disconnect()
        time.sleep(0.05)
        assert threading.active_count() <= baseline


class TestCacheBehavior:
    def test_get_action_returns_cached_value_after_thread_warms(self, leader):
        teleop, bus = leader
        # Wait for the read thread to have produced at least one sample.
        for _ in range(50):
            with teleop._cache_lock:
                if teleop._cached_pose is not None:
                    break
            time.sleep(0.005)
        assert teleop._cached_pose is not None, "read thread never produced a sample"

        # Record the bus call count, then call get_action many times.
        # The bus should NOT be hit again — cache reads only.
        count_before = bus.sync_read_call_count
        for _ in range(10):
            teleop.get_action()
        # All 10 calls hit the cache, not the bus.
        assert bus.sync_read_call_count == count_before

    def test_get_action_returns_motor_dict(self, leader):
        teleop, _bus = leader
        # Wait for first cache write.
        for _ in range(50):
            with teleop._cache_lock:
                if teleop._cached_pose is not None:
                    break
            time.sleep(0.005)
        action = teleop.get_action()
        expected_keys = {f"{m}.pos" for m in _MOTOR_NAMES}
        assert set(action.keys()) == expected_keys

    def test_get_action_returns_independent_copies(self, leader):
        """Caller mutations must not poison the cache."""
        teleop, _bus = leader
        for _ in range(50):
            with teleop._cache_lock:
                if teleop._cached_pose is not None:
                    break
            time.sleep(0.005)
        a = teleop.get_action()
        a["shoulder_pan.pos"] = 999.0
        b = teleop.get_action()
        assert b["shoulder_pan.pos"] != 999.0, "caller mutation poisoned the cache"

    def test_get_action_falls_back_to_blocking_read_when_cache_cold(self):
        """First call before the background thread populates the cache
        must fall through to a direct blocking sync_read — caller never
        sees None / empty dict."""
        bus_mock = _make_bus_mock(read_value=0.0)

        def _bus_side_effect(*_args, **kwargs):
            bus_mock.motors = kwargs["motors"]
            return bus_mock

        with (
            patch(
                "lerobot.teleoperators.so_leader.so_leader.FeetechMotorsBus",
                side_effect=_bus_side_effect,
            ),
            patch.object(SO107LeaderHighRate, "configure", lambda self: None),
        ):
            cfg = SO107LeaderHighRateConfig(
                port="/dev/null",
                id="cold",
                # Very slow read rate so the cache definitely isn't warm
                # before our get_action() call.
                read_rate_hz=1.0,
            )
            teleop = SO107LeaderHighRate(cfg)
            teleop.connect(calibrate=False)
            try:
                # Patch the cache so get_action sees it empty even if a
                # tick managed to slip in between connect and this call.
                with teleop._cache_lock:
                    teleop._cached_pose = None
                action = teleop.get_action()
                expected_keys = {f"{m}.pos" for m in _MOTOR_NAMES}
                assert set(action.keys()) == expected_keys
            finally:
                teleop.disconnect()


class TestStaleCache:
    """Regression: disconnect() must clear the cache so a reconnect can't
    return a stale pose from the previous session before the new read
    thread has warmed up."""

    def test_disconnect_clears_cache(self, leader):
        teleop, _bus = leader
        # Wait for the read thread to populate the cache.
        for _ in range(50):
            with teleop._cache_lock:
                if teleop._cached_pose is not None:
                    break
            time.sleep(0.005)
        assert teleop._cached_pose is not None

        teleop.disconnect()
        # After disconnect, the cache must be empty so a subsequent
        # get_action() can't return data from the previous session.
        with teleop._cache_lock:
            assert teleop._cached_pose is None


class TestReadRate:
    def test_thread_polls_at_configured_rate(self, leader):
        """Over a window, the cache should have been updated at
        roughly the configured rate. Loose tolerance because the
        thread shares the GIL and the mock's sync_read is essentially
        instant. Catches a regression where the rate-limit logic
        breaks and the thread spins."""
        teleop, bus = leader
        # Wait for thread warm-up.
        time.sleep(0.05)
        count_at_start = bus.sync_read_call_count
        start = time.perf_counter()
        time.sleep(0.2)
        elapsed = time.perf_counter() - start
        delta = bus.sync_read_call_count - count_at_start
        # Fixture's read_rate_hz=500 → expect ~100 calls in 200 ms.
        # Tolerate ±50% (50-150) since the mock is fast and GIL contention
        # is variable on CI.
        expected = elapsed * teleop.config.read_rate_hz
        assert 0.5 * expected < delta < 1.5 * expected, (
            f"read thread did {delta} reads in {elapsed * 1000:.0f}ms, "
            f"expected ~{int(expected)} at {teleop.config.read_rate_hz} Hz"
        )
