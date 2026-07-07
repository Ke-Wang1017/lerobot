#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Mock-bus integration tests for SO107FollowerPredictive.

The point of these tests is the *lifecycle and threading* of the
predictive controller as wired into the Robot interface — not the
algorithm itself. Algorithm correctness is covered by:

  * hardware-in-the-loop validation (separate manual scripts)
  * offline ground-truth comparison
  * closed-loop convergence simulation

Here we just make sure that, with a mocked FeetechMotorsBus:

  * connect() starts the controller thread and configures the bus
  * send_action(intent) is non-blocking and publishes intent
  * the controller thread eventually writes Goal_Position to the bus
  * get_observation() reads from the bus and feeds the controller's
    state log
  * the per-step max_step_deg clamp applies
  * disconnect() stops the controller cleanly with no leaked threads
  * missing-key actions raise (regression: the strict-key fix on
    _action_to_array)
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lerobot.robots.so107_follower_predictive import (
    SO107FollowerPredictive,
    SO107FollowerPredictiveRobotConfig,
)

# Motor map: SO-107 has 7 motors, same names as the parent SO107Follower.
_MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "forearm_roll",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def _make_bus_mock() -> MagicMock:
    """FeetechMotorsBus stand-in that records every sync_write call."""
    bus = MagicMock(name="FeetechBusMock")
    bus.is_connected = False
    bus.is_calibrated = True

    def _connect():
        bus.is_connected = True

    def _disconnect(_disable=True):
        bus.is_connected = False

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect

    @contextmanager
    def _dummy_cm():
        yield

    bus.torque_disabled.side_effect = _dummy_cm
    return bus


@pytest.fixture
def follower():
    """Yield a connected SO107FollowerPredictive with a mocked bus.

    The mock bus's sync_read returns a deterministic state vector
    (zeros, matching the expected motor count) so the controller's
    state-log path runs without raising.
    """
    bus_mock = _make_bus_mock()
    # Capture every (data_name, goal_dict) the controller writes — the
    # tests assert against this in lieu of probing the controller's
    # private state.
    sync_write_log: list[tuple[str, dict]] = []
    bus_mock.sync_write_log = sync_write_log

    def _sync_write(data_name, goal_dict, **_kwargs):
        sync_write_log.append((data_name, dict(goal_dict)))

    bus_mock.sync_write.side_effect = _sync_write

    def _bus_side_effect(*_args, **kwargs):
        bus_mock.motors = kwargs["motors"]
        # sync_read returns positions at the bus's natural zero.
        bus_mock.sync_read.return_value = dict.fromkeys(bus_mock.motors, 0.0)
        bus_mock.write.return_value = None
        bus_mock.disable_torque.return_value = None
        bus_mock.enable_torque.return_value = None
        return bus_mock

    with (
        patch(
            "lerobot.robots.so_follower.so_follower.FeetechMotorsBus",
            side_effect=_bus_side_effect,
        ),
        # Skip configure() — its bus.write side effects aren't relevant
        # to controller lifecycle and would require more bus stubbing.
        patch.object(SO107FollowerPredictive, "configure", lambda self: None),
    ):
        cfg = SO107FollowerPredictiveRobotConfig(
            port="/dev/null",
            id="test_arm",
            # Crank the rate so each test only has to sleep a few ms to
            # observe controller activity. Algorithmic behaviour is
            # rate-independent.
            control_rate_hz=500.0,
            # Default is True; turn off so tests don't race against the
            # 2 s adaptive update cadence (separate test exercises it).
            adaptive=False,
        )
        robot = SO107FollowerPredictive(cfg)
        robot.connect(calibrate=False)
        try:
            yield robot, bus_mock
        finally:
            if robot.is_connected:
                robot.disconnect()


# ============================================================================
# Lifecycle
# ============================================================================


class TestLifecycle:
    def test_connect_starts_controller_thread(self, follower):
        robot, _bus = follower
        # The controller's _thread should be alive after connect().
        assert robot._controller._thread is not None
        assert robot._controller._thread.is_alive()

    def test_disconnect_stops_controller_thread(self, follower):
        robot, bus = follower
        ctrl_thread = robot._controller._thread
        assert ctrl_thread is not None
        robot.disconnect()
        # Give it a moment beyond the join timeout.
        ctrl_thread.join(timeout=1.0)
        assert not ctrl_thread.is_alive()
        # Bus should be closed too.
        bus.disconnect.assert_called()

    def test_disconnect_runs_every_phase_on_error(self, follower):
        """If bus.disconnect raises, cameras still get cleaned up.

        Regression for f6ecbdb6b: previously a bus failure stranded any
        later cleanup. The predictive follower has no cameras here, but
        the controller stop must run before the bus error and be
        observable as 'thread is no longer alive'.
        """
        robot, bus = follower
        bus.disconnect.side_effect = RuntimeError("simulated torque-disable failure")
        ctrl_thread = robot._controller._thread
        with pytest.raises(RuntimeError, match="simulated"):
            robot.disconnect()
        ctrl_thread.join(timeout=1.0)
        assert not ctrl_thread.is_alive()  # controller stopped before bus error
        # Reset for fixture teardown.
        bus.disconnect.side_effect = None

    def test_no_thread_leak_across_connect_cycles(self, follower):
        """Disconnect + reconnect should not leave a zombie thread."""
        robot, _bus = follower
        baseline = threading.active_count()
        robot.disconnect()
        robot.connect(calibrate=False)
        robot.disconnect()
        # Give the OS a moment to reap threads.
        time.sleep(0.05)
        assert threading.active_count() <= baseline


# ============================================================================
# send_action contract
# ============================================================================


class TestSendAction:
    def _intent(self, value: float = 0.0) -> dict[str, float]:
        return {f"{m}.pos": value for m in _MOTOR_NAMES}

    def test_send_action_returns_intent_unchanged(self, follower):
        """send_action must echo back the *intent*, not a clamped value.

        Callers (dataset writer, in particular) record this as the
        action; the predictive robot's contract is that the recorded
        action IS the raw operator intent.
        """
        robot, _bus = follower
        intent = self._intent(3.14)
        returned = robot.send_action(intent)
        assert returned == intent

    def test_send_action_is_non_blocking(self, follower):
        """send_action must return well under one control period."""
        robot, _bus = follower
        intent = self._intent(0.0)
        start = time.perf_counter()
        robot.send_action(intent)
        elapsed_ms = (time.perf_counter() - start) * 1000
        # control_rate_hz=500 → period 2 ms. send_action must be far
        # cheaper than that since it just records to a deque.
        assert elapsed_ms < 1.0, f"send_action blocked for {elapsed_ms:.2f} ms"

    def test_controller_writes_goal_position(self, follower):
        """After send_action, the 200 Hz thread must publish to the bus."""
        robot, bus = follower
        bus.sync_write_log.clear()
        robot.send_action(self._intent(1.0))
        # Wait a few control ticks (rate is 500 Hz in tests = 2 ms / tick).
        time.sleep(0.05)
        # At least one Goal_Position write should have landed.
        goal_writes = [(name, gd) for name, gd in bus.sync_write_log if name == "Goal_Position"]
        assert goal_writes, "controller never wrote Goal_Position"
        # And it should have written every motor, not a partial set
        # (regression for the strict-key fix on _action_to_array).
        first_goal = goal_writes[0][1]
        assert set(first_goal.keys()) == set(_MOTOR_NAMES)

    def test_missing_motor_key_raises(self, follower):
        """Partial action dict must fail loud, not produce a partial write.

        Regression for commit 42bd67f3b — previously the filter
        ``[float(action[k]) for k in self._motor_keys if k in action]``
        silently shrank the array, the controller would issue a partial
        sync_write, and the missing motors would hold their last goal.
        """
        robot, _bus = follower
        partial = {f"{m}.pos": 0.0 for m in _MOTOR_NAMES[:-1]}  # drop gripper
        with pytest.raises(ValueError, match="action missing keys"):
            robot.send_action(partial)


# ============================================================================
# attach_teleop — pull-path intent source
# ============================================================================


class TestAttachTeleop:
    """When a teleop is bound via robot.attach_teleop, the controller's
    tick polls teleop.get_action() at control rate instead of waiting
    for send_action pushes. The new path is the one that makes a
    high-rate leader (SO107LeaderHighRate) actually useful — without
    it, intent samples only flow at the loop driver's cadence.
    """

    def test_attach_routes_through_controller(self, follower):
        robot, _bus = follower
        teleop = MagicMock()
        teleop.get_action.return_value = {f"{m}.pos": 0.0 for m in _MOTOR_NAMES}
        assert robot._controller._teleop is None
        robot.attach_teleop(teleop)
        assert robot._controller._teleop is teleop
        robot.attach_teleop(None)
        assert robot._controller._teleop is None

    def test_controller_polls_teleop_at_control_rate(self, follower):
        """A bound teleop's get_action MUST be called many times per
        second from the controller thread. Catches a regression where
        attach_teleop is wired but _tick still reads _latest_intent.
        """
        robot, _bus = follower
        teleop = MagicMock()
        teleop.get_action.return_value = {f"{m}.pos": 0.0 for m in _MOTOR_NAMES}
        robot.attach_teleop(teleop)
        # Fixture's control_rate_hz=500 → expect ~50 calls in 100 ms.
        # Tolerate a loose lower bound; the upper bound doesn't matter
        # for catching the bug.
        time.sleep(0.1)
        assert teleop.get_action.call_count >= 20, (
            f"controller called teleop.get_action() only "
            f"{teleop.get_action.call_count} times over ~100 ms at 500 Hz"
        )

    def test_attached_teleop_drives_goal_position(self, follower):
        """A bound teleop's poses must end up as Goal_Position writes,
        bypassing the send_action push path entirely.
        """
        robot, bus = follower
        teleop = MagicMock()
        # Distinct value so we can tell it apart from the fixture's default 0.0.
        teleop.get_action.return_value = {f"{m}.pos": 7.5 for m in _MOTOR_NAMES}
        robot.attach_teleop(teleop)
        bus.sync_write_log.clear()
        time.sleep(0.05)
        goal_writes = [gd for name, gd in bus.sync_write_log if name == "Goal_Position"]
        assert goal_writes
        # First goal write should reflect the teleop's pose (modulo the
        # max_step clamp; with fixture's max_step_deg=3 starting from 0,
        # first goal is capped at ~3, not 7.5 — but it must be > 0 to
        # confirm it came from the teleop, not the cached 0).
        first = goal_writes[0]
        for motor, val in first.items():
            assert val > 0.0, f"{motor} write {val}; expected teleop's pose to drive it"

    def test_teleop_get_action_error_is_caught_per_tick(self, follower):
        """A failing teleop.get_action must not kill the controller
        thread — the per-tick try/except should absorb it and the
        thread keeps running for the next tick."""
        robot, _bus = follower
        teleop = MagicMock()
        teleop.get_action.side_effect = RuntimeError("simulated bus error")
        robot.attach_teleop(teleop)
        time.sleep(0.05)
        # Thread should still be alive after the error stream.
        assert robot._controller._thread.is_alive()
        # And get_action was actually being called (proves it WASN'T
        # the falling-through-and-stalling bug).
        assert teleop.get_action.call_count > 0

    def test_detach_returns_to_send_action_path(self, follower):
        """Attaching then detaching restores the push-based path.
        send_action's intent should drive motors again."""
        robot, bus = follower
        teleop = MagicMock()
        teleop.get_action.return_value = {f"{m}.pos": 5.0 for m in _MOTOR_NAMES}
        robot.attach_teleop(teleop)
        time.sleep(0.02)
        robot.attach_teleop(None)
        bus.sync_write_log.clear()
        # Now drive via send_action — controller should use that.
        robot.send_action({f"{m}.pos": 0.0 for m in _MOTOR_NAMES})
        time.sleep(0.05)
        goal_writes = [gd for name, gd in bus.sync_write_log if name == "Goal_Position"]
        assert goal_writes


# ============================================================================
# Chunk path
# ============================================================================


class TestActionChunk:
    """Exact-lookup path: send_action(ActionChunk) feeds the controller a
    horizon, controller picks the target at now + L by interpolation.
    """

    def _frame(self, value: float) -> dict[str, float]:
        return {f"{m}.pos": value for m in _MOTOR_NAMES}

    def test_send_action_returns_frame_zero_for_chunk(self, follower):
        """The dataset writer records what send_action returns. For a
        chunk that must be frames[0] (= "current intent"), not the
        chunk itself."""
        from lerobot.types import ActionChunk

        robot, _bus = follower
        chunk = ActionChunk(
            fps=30.0,
            frames=(self._frame(1.0), self._frame(1.1), self._frame(1.2)),
        )
        returned = robot.send_action(chunk)
        assert returned == self._frame(1.0)

    def test_controller_writes_via_chunk_lookup(self, follower):
        """Chunk path → controller writes Goal_Position from the chunk's
        future values (not by velocity-extrapolating past intents)."""
        from lerobot.types import ActionChunk

        robot, bus = follower
        bus.sync_write_log.clear()
        # Linear ramp: frame[k] = 5 + 0.1 * k. At fps=30, frame index
        # for L=80ms is 80ms * 30fps = 2.4 → interp between frame 2 and 3.
        frames = tuple(self._frame(5.0 + 0.1 * k) for k in range(20))
        chunk = ActionChunk(fps=30.0, frames=frames)
        robot.send_action(chunk)
        # Wait a few ticks (rate is 500 Hz in tests = 2 ms / tick).
        time.sleep(0.05)
        goal_writes = [gd for name, gd in bus.sync_write_log if name == "Goal_Position"]
        assert goal_writes, "controller never wrote Goal_Position"
        # The very first goal write should reflect an L-shifted value
        # somewhere between frame 0 (5.0) and a few frames in. Exact
        # value depends on tick timing relative to send_action; just
        # verify the controller is reading from the future, not
        # holding at 5.0.
        first_goal = goal_writes[0]
        assert all(v > 5.0 for v in first_goal.values()), f"expected L-shifted future, got {first_goal}"

    def test_chunk_with_missing_key_raises(self, follower):
        """Strict-key check applies per-frame: a malformed frame in
        the chunk must fail loud at set_intent time, not later in the
        controller's tick."""
        from lerobot.types import ActionChunk

        robot, _bus = follower
        good = self._frame(0.0)
        bad = {k: v for k, v in good.items() if k != f"{_MOTOR_NAMES[-1]}.pos"}
        chunk = ActionChunk(fps=30.0, frames=(good, bad))
        with pytest.raises(ValueError, match="action missing keys"):
            robot.send_action(chunk)

    def test_chunk_at_arbitrary_fps_resolves_target_correctly(self, follower):
        """The controller's chunk-lookup math must work for any fps,
        not just 30. Regression guard against hardcoded-30 assumptions
        in the controller. With L=0 + a ramp chunk, the goal at tick 0
        should equal frame[0] regardless of fps."""
        import numpy as np

        from lerobot.types import ActionChunk

        robot, bus = follower
        # Try fps=15 (low) and fps=60 (high) — both have to land the
        # goal write near frame[0] when L=0. Use a non-zero base value
        # so we'd notice if the lookup landed elsewhere.
        for test_fps in (15.0, 60.0):
            bus.sync_write_log.clear()
            base = 7.0
            frames = tuple(self._frame(base + 0.1 * k) for k in range(20))
            robot.send_action(ActionChunk(fps=test_fps, frames=frames))
            time.sleep(0.02)  # a few control ticks
            goals = [gd for name, gd in bus.sync_write_log if name == "Goal_Position"]
            assert goals, f"no goal write at fps={test_fps}"
            # First goal should be very close to frame[0] (L=0 in this fixture).
            for v in goals[0].values():
                assert v == pytest.approx(base, abs=0.5), f"fps={test_fps}: goal {v}, expected ~{base}"
            # Math sanity: tail-velocity extrapolation past chunk end
            # uses fps in (frames[-1] - frames[-2]) * fps, so the *value*
            # past the end must depend on fps. Compute the predicted
            # value at "now past last frame" and assert the formula.
            tail_v = (np.array([base + 0.1 * 19]) - np.array([base + 0.1 * 18])) * test_fps
            excess_s = 0.001  # 1 ms past end
            expected_past_end = base + 0.1 * 19 + tail_v[0] * excess_s
            # Just verify our math; the controller will only reach this
            # in real runs where the chunk runs out, not in this test.
            assert expected_past_end == pytest.approx(base + 0.1 * 19 + 0.1 * test_fps * excess_s, abs=1e-9)

    def test_chunk_then_dict_clears_chunk_state(self, follower):
        """A single-dict send_action after a chunk drops the chunk
        record — subsequent ticks fall back to the velocity-extrapolation
        path. (Latest-send wins.)"""
        from lerobot.types import ActionChunk

        robot, _bus = follower
        chunk = ActionChunk(fps=30.0, frames=(self._frame(0.0), self._frame(0.1)))
        robot.send_action(chunk)
        time.sleep(0.005)
        robot.send_action(self._frame(0.5))
        time.sleep(0.005)
        # After the dict send, the controller's latest_chunk must be cleared.
        assert robot._controller._latest_chunk is None


# ============================================================================
# get_observation feeds the controller's state log
# ============================================================================


class TestGetObservation:
    def test_observation_includes_every_motor(self, follower):
        robot, _bus = follower
        obs = robot.get_observation()
        expected = {f"{m}.pos" for m in _MOTOR_NAMES}
        assert set(obs.keys()) == expected

    def test_observation_pushes_state_to_controller_log(self, follower):
        """The controller needs state samples for the adaptive update.

        get_observation() should append to _state_log; this verifies
        the feed path exists (the adaptive update itself is exercised
        in TestAdaptiveUpdate).
        """
        robot, _bus = follower
        n_before = len(robot._controller._state_log)
        for _ in range(3):
            robot.get_observation()
        assert len(robot._controller._state_log) >= n_before + 3


# ============================================================================
# max_step_deg safety clamp
# ============================================================================


class TestSafetyClamp:
    def test_per_step_delta_is_clamped(self, follower):
        """A large intent jump must be slewed by ≤ max_step_deg / tick.

        The controller's _last_action holds last tick's commanded
        position. When the new shifted target differs by more than
        max_step_deg per joint, the delta is clipped. This protects
        against extrapolation spikes during sharp leader reversals.
        """
        robot, bus = follower
        # First intent: settle at 0.0. Wait for a couple of ticks so
        # _last_action is populated.
        robot.send_action({f"{m}.pos": 0.0 for m in _MOTOR_NAMES})
        time.sleep(0.05)
        bus.sync_write_log.clear()
        # Now jump to a value far beyond max_step_deg (default = 3°).
        robot.send_action({f"{m}.pos": 100.0 for m in _MOTOR_NAMES})
        time.sleep(0.02)  # ~10 ticks at 500 Hz
        # Find the first Goal_Position write after the jump.
        goal_writes = [gd for name, gd in bus.sync_write_log if name == "Goal_Position"]
        assert goal_writes, "no Goal_Position writes after the jump"
        first = goal_writes[0]
        # Every motor's first commanded value should be within
        # max_step_deg of the previous (which was 0).
        max_step = robot.config.max_step_deg
        for motor, val in first.items():
            assert abs(val) <= max_step + 1e-6, f"{motor} jumped to {val}, exceeds max_step_deg={max_step}"


# ============================================================================
# Adaptive update path
# ============================================================================


class TestAdaptiveUpdate:
    def test_adaptive_update_runs_without_raising(self):
        """Build enough state + intent samples to drive _maybe_update_lookahead.

        The deterministic correctness of the cross-correlation is
        already covered by separate offline backtests. Here we just
        verify the threaded path doesn't raise on a real history shape.
        """
        bus_mock = _make_bus_mock()

        def _bus_side_effect(*_args, **kwargs):
            bus_mock.motors = kwargs["motors"]
            bus_mock.sync_read.return_value = dict.fromkeys(bus_mock.motors, 0.0)
            return bus_mock

        with (
            patch(
                "lerobot.robots.so_follower.so_follower.FeetechMotorsBus",
                side_effect=_bus_side_effect,
            ),
            patch.object(SO107FollowerPredictive, "configure", lambda self: None),
        ):
            cfg = SO107FollowerPredictiveRobotConfig(
                port="/dev/null",
                control_rate_hz=500.0,
                adaptive=True,
            )
            robot = SO107FollowerPredictive(cfg)
            # Shorten the update cadence so the test doesn't have to
            # wait the production 2 s.
            robot._controller._UPDATE_PERIOD_S = 0.1  # type: ignore[misc]
            robot._controller._WINDOW_S = 0.3  # type: ignore[misc]
            robot.connect(calibrate=False)
            try:
                # Drive a sinusoidal intent so v_leader is non-zero and
                # the amplitude-floor gate passes.
                import math

                t0 = time.perf_counter()
                while time.perf_counter() - t0 < 0.5:  # 0.5 s of activity
                    t = time.perf_counter() - t0
                    val = 5.0 * math.sin(2 * math.pi * 2 * t)
                    robot.send_action({f"{m}.pos": val for m in _MOTOR_NAMES})
                    robot.get_observation()
                    time.sleep(0.005)
                # If we got here without the controller thread raising,
                # the adaptive update path is alive. The thread catches
                # exceptions per-tick but logs them; if we want strict
                # detection we'd need to capture log output.
                assert robot._controller._thread.is_alive()
            finally:
                robot.disconnect()


# ============================================================================
# Adaptive xcorr correctness — algorithm-level tests
# ============================================================================
#
# These tests build a PredictiveLookaheadController directly without
# starting its thread or touching a bus. They pre-populate _intent_log
# and _state_log with synthetic signals that have a *known* shift, then
# call _maybe_update_lookahead() directly and assert the controller
# converges toward the synthetic ground truth.
#
# The threaded TestAdaptiveUpdate above only checks "the path doesn't
# raise". This class checks "the path computes the right number." A
# units bug in the time conversion (e.g. caller_rate vs control_rate
# mismatch — bug fixed in c6537b40a) would surface here as the recovered
# lag being scaled by a constant factor and these assertions would fail.


def _make_isolated_controller(
    *,
    control_rate_hz: float = 200.0,
    max_lookahead_ms: float = 110.0,
    lookahead_ms: float = 0.0,
    adaptive: bool = True,
):
    """Build a PredictiveLookaheadController without starting its thread
    or any bus traffic. The controller is fully usable for direct
    set_intent / _maybe_update_lookahead calls — tests bypass start()
    so the thread doesn't race with synthetic log injection.
    """
    from lerobot.robots.predictive.controller import PredictiveLookaheadController

    cfg = SO107FollowerPredictiveRobotConfig(
        port="/dev/null",
        adaptive=adaptive,
        lookahead_ms=lookahead_ms,
        max_lookahead_ms=max_lookahead_ms,
        control_rate_hz=control_rate_hz,
    )

    robot = MagicMock()
    robot.config = cfg
    robot.bus = MagicMock()
    robot.bus.motors = dict.fromkeys(_MOTOR_NAMES, None)
    robot._bus_lock = threading.Lock()
    # PredictiveLookaheadController formats the robot into log messages
    # via "%s". Give it a stable repr so failures are diagnosable.
    robot.__str__ = lambda self: "test_isolated_controller"

    return PredictiveLookaheadController(robot)


def _inject_sinusoid_pair(
    controller,
    tau_s: float,
    *,
    freq_hz: float = 1.0,
    amp: float = 5.0,
    state_rate_hz: float = 30.0,
):
    """Pre-populate _intent_log and _state_log with sinusoidal signals
    where state(t) = intent(t - tau_s).

    ``_intent_log`` is populated at the controller's control rate (that's
    how _tick fills it in production). ``_state_log`` is populated at
    ``state_rate_hz`` (default 30 Hz — the observation rate that
    observe_state is called at in production), so the 200-element
    deque's maxlen comfortably spans the 3-second xcorr window without
    overflowing. Sampling state at the control rate would overflow the
    deque to the last 1 s of data and break the xcorr.

    Returns: the wall-clock ``now`` to pass to _maybe_update_lookahead.
    """
    import numpy as np

    dt_ctrl = controller._control_dt
    dt_state = 1.0 / state_rate_hz
    duration = controller._WINDOW_S * 1.5 + 0.25  # window + headroom
    t0 = 1000.0  # arbitrary anchor; perf_counter-comparable

    # Intent: control-rate cadence (200 Hz default).
    n_intent = int(duration / dt_ctrl)
    for i in range(n_intent):
        t = t0 + i * dt_ctrl
        intent_val = amp * np.sin(2 * np.pi * freq_hz * t)
        # Same value on every motor — the xcorr is per-joint but with
        # identical signals all 7 joints agree on the recovered lag.
        controller._intent_log.append((t, np.array([intent_val] * len(_MOTOR_NAMES))))

    # State: observation-rate cadence (30 Hz default). state(t) =
    # intent(t - tau_s) by construction.
    n_state = int(duration / dt_state)
    for i in range(n_state):
        t = t0 + i * dt_state
        state_val = amp * np.sin(2 * np.pi * freq_hz * (t - tau_s))
        controller._state_log.append((t, np.array([state_val] * len(_MOTOR_NAMES))))

    return t0 + duration


class TestAdaptiveCorrectness:
    """Direct tests on PredictiveLookaheadController._maybe_update_lookahead.

    These are the tests that would have caught the caller-rate-vs-
    control-rate units bug (fixed in c6537b40a). Each test feeds the
    controller a synthetic signal pair with a known time shift and
    asserts the controller's L update lands at the right place.
    """

    def test_xcorr_recovers_positive_lag_when_state_lags_intent(self):
        """state(t) = intent(t - 80 ms) → reported lag ≈ +80 ms.

        Starting from L=0 with ADAPTIVE_ALPHA=0.5, the first update
        moves L to 0.5 * 0.080 = 40 ms.
        """
        ctrl = _make_isolated_controller(lookahead_ms=0.0)
        tau_s = 0.080
        now = _inject_sinusoid_pair(ctrl, tau_s)
        ctrl._maybe_update_lookahead(now)
        # alpha = 0.5; new_L = 0.5 * (current_L + lag) when current_L = 0
        # That's 0.5 * 0.080 = 0.040 s.
        assert ctrl._lookahead_s == pytest.approx(0.5 * tau_s, abs=0.005)

    def test_xcorr_units_invariant_to_control_rate(self):
        """The recovered lag MUST be in seconds, not "control_dt units".

        Regression for the bug fixed in c6537b40a: that bug scaled the
        reported lag by (caller_rate / control_rate). The bug was
        invisible at any single control_rate but blows up when the rate
        changes. Asserting the same recovered lag at 100/200/500 Hz
        pins the time-unit invariant.
        """
        tau_s = 0.080
        for control_rate in (100.0, 200.0, 500.0):
            ctrl = _make_isolated_controller(control_rate_hz=control_rate)
            now = _inject_sinusoid_pair(ctrl, tau_s)
            ctrl._maybe_update_lookahead(now)
            assert ctrl._lookahead_s == pytest.approx(0.5 * tau_s, abs=0.005), (
                f"control_rate={control_rate} Hz: expected L ≈ {0.5 * tau_s} s, got L = {ctrl._lookahead_s} s"
            )

    def test_xcorr_recovers_negative_lag_when_state_leads_intent(self):
        """state(t) = intent(t + 50 ms) → reported lag ≈ −50 ms.

        Negative lag must be reachable so adaptive can DECREASE L
        when the operator overshoots; otherwise the loop is a
        one-way ratchet.
        """
        ctrl = _make_isolated_controller(lookahead_ms=50.0)  # start L = 0.050 s
        # state leading intent by 50ms — implies controller's current
        # lookahead is already too large by 50ms.
        now = _inject_sinusoid_pair(ctrl, -0.050)
        ctrl._maybe_update_lookahead(now)
        # new_L = 0.5 * (0.050 + (-0.050)) + 0.5 * 0.050
        #       = 0.5 * 0 + 0.025 = 0.025 s
        assert ctrl._lookahead_s == pytest.approx(0.025, abs=0.005)

    def test_l_cap_is_enforced(self):
        """Huge lag must clamp L to max_lookahead_s, not run away."""
        ctrl = _make_isolated_controller(max_lookahead_ms=50.0, lookahead_ms=0.0)
        # tau = 0.2s, max scan is _MAX_LAG_S = 0.3s so the peak is
        # representable. With L=0 the update wants L→target=0.5*0.2=0.1s,
        # which exceeds the 50 ms cap → clamps.
        tau_s = 0.200
        now = _inject_sinusoid_pair(ctrl, tau_s)
        ctrl._maybe_update_lookahead(now)
        assert ctrl._lookahead_s == pytest.approx(0.050, abs=1e-6)
        assert ctrl._lookahead_s <= ctrl._max_lookahead_s + 1e-9

    def test_l_floor_at_zero(self):
        """Adaptive must not drive L negative even with strongly leading state."""
        ctrl = _make_isolated_controller(lookahead_ms=5.0)  # start L = 0.005 s
        now = _inject_sinusoid_pair(ctrl, -0.200)  # state leads intent by 200ms
        ctrl._maybe_update_lookahead(now)
        assert ctrl._lookahead_s >= 0.0

    def test_amplitude_floor_skips_stationary_joints(self):
        """A joint whose state is below _AMP_FLOOR (encoder noise) must
        be ignored in the aggregation, so a single moving joint can
        dominate the lag estimate without static joints' spurious
        0.95 correlations dragging it toward 0."""
        import numpy as np

        ctrl = _make_isolated_controller()
        tau_s = 0.060
        dt_ctrl = ctrl._control_dt
        dt_state = 1.0 / 30.0  # observe rate; see _inject_sinusoid_pair
        duration = ctrl._WINDOW_S * 1.5 + 0.25
        t0 = 1000.0
        rng = np.random.default_rng(seed=42)

        # Intent at control rate (200 Hz). Only joint 0 moves; rest have
        # tiny noise. std < _AMP_FLOOR=1.0 ensures the amplitude gate
        # excludes the noisy joints from the aggregation.
        n_intent = int(duration / dt_ctrl)
        for i in range(n_intent):
            t = t0 + i * dt_ctrl
            moving_intent = 5.0 * np.sin(2 * np.pi * 1.0 * t)
            noise = rng.normal(0, 0.05, size=len(_MOTOR_NAMES) - 1)
            ctrl._intent_log.append((t, np.concatenate([[moving_intent], noise])))

        # State at observation rate (30 Hz) so the deque (maxlen=200)
        # spans the 3s xcorr window. state[0] = moving_intent(t - tau).
        n_state = int(duration / dt_state)
        for i in range(n_state):
            t = t0 + i * dt_state
            moving_state = 5.0 * np.sin(2 * np.pi * 1.0 * (t - tau_s))
            noise = rng.normal(0, 0.05, size=len(_MOTOR_NAMES) - 1)
            ctrl._state_log.append((t, np.concatenate([[moving_state], noise])))

        ctrl._maybe_update_lookahead(t0 + duration)
        # The moving joint contributes; the noisy ones are filtered.
        # Recovered L should track the moving joint's tau.
        assert ctrl._lookahead_s == pytest.approx(0.5 * tau_s, abs=0.005)

    def test_below_min_samples_skips_update(self):
        """Too-little data must skip — don't update L from noise."""
        ctrl = _make_isolated_controller()
        starting_l = ctrl._lookahead_s
        # Only inject a tiny fraction of the expected sample count.
        import numpy as np

        dt = ctrl._control_dt
        for i in range(5):  # well below the threshold
            t = 1000.0 + i * dt
            ctrl._intent_log.append((t, np.zeros(len(_MOTOR_NAMES))))
            ctrl._state_log.append((t, np.zeros(len(_MOTOR_NAMES))))
        ctrl._maybe_update_lookahead(1000.0 + 5 * dt)
        assert ctrl._lookahead_s == starting_l

    def test_intent_log_populated_at_control_rate_not_caller_rate(self, follower):
        """Direct regression for commit c6537b40a.

        Before the fix, ``_intent_log`` was populated from ``set_intent``
        (caller thread, ~30 Hz). After the fix it's populated from
        ``_tick`` (control thread, 200+ Hz). The xcorr's
        ``best_k * control_dt`` index-to-time conversion is only correct
        under the latter — under the former, reported lag is scaled by
        ``caller_rate / control_rate`` (= 30/200 = 0.15 in the field).

        Asserting that the log grows at roughly control rate even when
        send_action isn't being called catches a regression that reverts
        the population path back into set_intent.
        """
        robot, _bus = follower
        # Prime: one send_action so the controller has an intent to log.
        robot.send_action({f"{m}.pos": 0.0 for m in _MOTOR_NAMES})

        start = time.perf_counter()
        n_before = len(robot._controller._intent_log)
        sleep_s = 0.2
        time.sleep(sleep_s)
        elapsed = time.perf_counter() - start
        log_delta = len(robot._controller._intent_log) - n_before

        # At the fixture's 500 Hz control rate over ~200 ms we expect
        # ~100 new entries. Pre-fix code would yield zero (only set_intent
        # populates, no new send_action calls in this window).
        expected_min = int(elapsed * robot.config.control_rate_hz * 0.5)
        assert log_delta >= expected_min, (
            f"_intent_log gained {log_delta} entries over {elapsed * 1000:.0f} ms "
            f"at {robot.config.control_rate_hz} Hz. Expected ≥ {expected_min}. "
            "Suggests intent log is populated at caller rate instead of "
            "control rate (caller_rate/control_rate units bug, c6537b40a)."
        )


# ============================================================================
# Chunk-lookup correctness — direct tests on the math
# ============================================================================


class TestChunkLookupMath:
    """Direct tests on PredictiveLookaheadController._lookup_in_chunk.

    The exact-lookup branch is the hot path under chunked sources. These
    tests verify the index math at multiple fps so a hardcoded-30 bug
    would never silently pass.
    """

    def test_lookup_at_t_zero_returns_frame_zero(self):
        """now = received_at, L = 0 → exact frame[0], no interpolation."""
        import numpy as np

        ctrl = _make_isolated_controller(lookahead_ms=0.0)
        frames_arr = np.array([[1.0] * 7, [2.0] * 7, [3.0] * 7])
        chunk = (100.0, 30.0, frames_arr)
        result = ctrl._lookup_in_chunk(chunk, now=100.0)
        np.testing.assert_allclose(result, frames_arr[0])

    @pytest.mark.parametrize("fps", [15.0, 30.0, 60.0, 120.0])
    def test_lookup_index_math_matches_fps(self, fps):
        """target_idx = (elapsed + L) * fps — same math at any fps."""
        import numpy as np

        ctrl = _make_isolated_controller(lookahead_ms=0.0)
        # 10 frames at the given fps. Each frame is k as its value.
        frames_arr = np.array([[float(k)] * 7 for k in range(10)])
        chunk = (100.0, fps, frames_arr)
        # Query at elapsed = 2.5 / fps → target_idx = 2.5 → interp
        # between frame[2]=2 and frame[3]=3 at alpha=0.5 → 2.5.
        result = ctrl._lookup_in_chunk(chunk, now=100.0 + 2.5 / fps)
        np.testing.assert_allclose(result, np.array([2.5] * 7))

    def test_lookup_past_end_uses_tail_velocity(self):
        """Past the last frame, extrapolate from forward diff of last two."""
        import numpy as np

        ctrl = _make_isolated_controller(lookahead_ms=0.0)
        # 3 frames at 30 fps. Values 1.0, 1.5, 2.0 → tail v = (2.0-1.5)*30 = 15.0 units/s.
        frames_arr = np.array([[1.0] * 7, [1.5] * 7, [2.0] * 7])
        fps = 30.0
        chunk = (100.0, fps, frames_arr)
        # Query 0.1 s past last frame (last frame at elapsed = 2/30 ≈ 0.0667).
        excess_s = 0.1
        elapsed_past_end = 2.0 / fps + excess_s
        result = ctrl._lookup_in_chunk(chunk, now=100.0 + elapsed_past_end)
        # Predicted: 2.0 + 15.0 * 0.1 = 3.5
        expected = 2.0 + 15.0 * excess_s
        np.testing.assert_allclose(result, np.array([expected] * 7), atol=1e-9)

    def test_lookahead_s_override_path(self):
        """lookahead_s_override=0.0 must use L=0 even when controller's
        _lookahead_s is set to something else. Regression for the
        adaptive-log path that computes "intent at now" with L=0."""
        import numpy as np

        ctrl = _make_isolated_controller(lookahead_ms=50.0)
        frames_arr = np.array([[float(k)] * 7 for k in range(10)])
        fps = 30.0
        chunk = (100.0, fps, frames_arr)
        # With override=0, we get frame[0]. Without override, we'd see
        # the L-shifted lookup (50 ms × 30 fps = 1.5 frames ahead).
        result_override = ctrl._lookup_in_chunk(chunk, now=100.0, lookahead_s_override=0.0)
        np.testing.assert_allclose(result_override, frames_arr[0])
        result_default = ctrl._lookup_in_chunk(chunk, now=100.0)
        np.testing.assert_allclose(result_default, np.array([1.5] * 7))


# ============================================================================
# Velocity estimator dispatch — algorithm correctness
# ============================================================================
#
# Verifies each of the three velocity estimators ("quad", "linear",
# "forward_diff") returns the expected slope on synthetic motion. The
# "quad" estimator is the new default per the backtest
# (offline backtests, not in this PR); these tests pin its
# unbiased-under-acceleration property and the dispatch wiring.


class TestVelocityEstimators:
    """Algorithm-level tests on the three estimators.

    Each estimator is a static method, so we test them directly without
    spinning up a controller instance.
    """

    def _ts(self, n: int, dt: float = 0.005, t0: float = 0.0):
        import numpy as np

        return t0 + dt * np.arange(n, dtype=np.float64)

    def test_linear_on_constant_velocity_returns_v(self):
        """All three estimators must recover the true velocity exactly
        when the signal is a perfect line — the easy case."""
        import numpy as np

        from lerobot.robots.predictive.controller import PredictiveLookaheadController

        ts = self._ts(20)
        v_true = 5.0  # units / s
        ps = np.stack([v_true * ts] * 7, axis=1)  # (n, 7), same on every joint
        for name, fn in (
            ("linear", PredictiveLookaheadController._velocity_lsq_linear),
            ("quad", PredictiveLookaheadController._velocity_lsq_quad_end),
            ("forward_diff", PredictiveLookaheadController._velocity_forward_diff),
        ):
            v = fn(ts, ps)
            assert v is not None, f"{name} returned None for clean linear signal"
            np.testing.assert_allclose(v, [v_true] * 7, atol=1e-9, err_msg=f"{name} biased on constant v")

    def test_quad_unbiased_under_acceleration_but_linear_is_biased(self):
        """Synthetic accelerating signal p(t) = 0.5·a·t². True v(t_end) = a·t_end.

        ``linear`` reports slope ≈ a·t_midpoint = a·t_end/2 (HALF the right
        answer in the limit). ``quad`` recovers a·t_end exactly. This is
        the head-to-head that justifies the "quad" default — the same
        artifact responsible for ~75 ms of apparent τ inflation on the
        leader-teleop path.
        """
        import numpy as np

        from lerobot.robots.predictive.controller import PredictiveLookaheadController

        ts = self._ts(20)  # 0..0.095 s
        a = 100.0  # accel units/s²
        # Centered around 0 so v at midpoint is a·t_mid = a·0.0475 ≈ 4.75
        # and v at end is a·t_end = a·0.095 ≈ 9.5.
        ps = np.stack([0.5 * a * ts * ts] * 7, axis=1)
        v_end_true = a * ts[-1]
        v_mid_expected = a * ts.mean()

        v_lin = PredictiveLookaheadController._velocity_lsq_linear(ts, ps)
        v_quad = PredictiveLookaheadController._velocity_lsq_quad_end(ts, ps)
        v_fd = PredictiveLookaheadController._velocity_forward_diff(ts, ps)

        # Linear: returns midpoint slope, ≈ HALF the true end-slope.
        np.testing.assert_allclose(v_lin, [v_mid_expected] * 7, atol=1e-6)
        # Quad: returns end slope exactly (parabola is degree-2 → fit is exact).
        np.testing.assert_allclose(v_quad, [v_end_true] * 7, atol=1e-6)
        # Forward-diff: ≈ a · (t_end - dt/2) (slope of the last segment).
        # This is between the linear and quad answers, closer to quad.
        v_fd_expected = a * (ts[-1] - (ts[-1] - ts[-2]) / 2)
        np.testing.assert_allclose(v_fd, [v_fd_expected] * 7, atol=1e-6)

        # And the headline assertion: quad and linear DISAGREE on this signal.
        assert abs(v_quad[0] - v_lin[0]) > 1.0  # ~4.75 apart by construction.

    def test_quad_falls_back_to_linear_with_2_samples(self):
        """Quadratic needs ≥ 3 samples; with fewer we transparently fall
        back to linear so callers never see a None when they could have
        gotten a slope estimate."""
        import numpy as np

        from lerobot.robots.predictive.controller import PredictiveLookaheadController

        ts = np.array([0.0, 0.005], dtype=np.float64)
        ps = np.array([[1.0] * 7, [2.0] * 7], dtype=np.float64)
        v = PredictiveLookaheadController._velocity_lsq_quad_end(ts, ps)
        assert v is not None
        # 2-sample slope: (2-1)/0.005 = 200 on each joint
        np.testing.assert_allclose(v, [200.0] * 7, atol=1e-6)

    def test_estimators_return_none_on_degenerate_input(self):
        """Single sample / all-same-timestamps → no slope possible."""
        import numpy as np

        from lerobot.robots.predictive.controller import PredictiveLookaheadController

        ts = np.array([1.0])
        ps = np.array([[0.0] * 7])
        for fn in (
            PredictiveLookaheadController._velocity_lsq_linear,
            PredictiveLookaheadController._velocity_forward_diff,
        ):
            assert fn(ts, ps) is None

        # All-same timestamps → linear and forward_diff degenerate.
        ts_flat = np.full(5, 1.0)
        ps5 = np.zeros((5, 7))
        assert PredictiveLookaheadController._velocity_lsq_linear(ts_flat, ps5) is None
        assert PredictiveLookaheadController._velocity_forward_diff(ts_flat, ps5) is None

    def test_controller_dispatches_to_configured_estimator(self):
        """Setting ``velocity_estimator`` in the config selects the right
        method at runtime. Regression for the dispatch wiring."""
        for name in ("quad", "linear", "forward_diff"):
            ctrl = _make_isolated_controller()
            ctrl._velocity_estimator = name  # type: ignore[assignment]
            # Construct an accelerating signal and check which value the
            # dispatch returns. Each estimator gives a different answer.
            import numpy as np

            ts = self._ts(20)
            ps = np.stack([0.5 * 100.0 * ts * ts] * 7, axis=1)
            v = ctrl._estimate_velocity(ts, ps)
            assert v is not None
            if name == "quad":
                # End slope exactly.
                np.testing.assert_allclose(v, [100.0 * ts[-1]] * 7, atol=1e-6)
            elif name == "linear":
                # Midpoint slope.
                np.testing.assert_allclose(v, [100.0 * ts.mean()] * 7, atol=1e-6)
            elif name == "forward_diff":
                dt = ts[-1] - ts[-2]
                np.testing.assert_allclose(v, [100.0 * (ts[-1] - dt / 2)] * 7, atol=1e-6)


# ============================================================================
# Integration: rate-agnostic intent ring (end-to-end via the mocked bus)
# ============================================================================


class TestRateAgnosticIntent:
    """End-to-end integration: same trajectory pushed at different rates
    should produce comparable motor_cmd output.

    These tests exercise the FULL controller pipeline (set_intent →
    intent_ring → _tick → velocity estimator → sync_write) against a
    mocked bus that records every written goal_position. After running
    for a fixed wall-clock duration at different push rates with the
    same underlying ramp signal, the recorded motor_cmd traces should
    converge to similar values within tolerance.

    Pre-fix behaviour (intent ring filled per controller tick with
    stair-stepped 30 Hz value): velocity estimator drastically
    under-estimated source-rate change → motor_cmd lagged by far more
    than the L=80ms target.

    Post-fix: motor_cmd ≈ intent + L · v across rates, within
    estimation residual.
    """

    # Ramp rate chosen so the joint moves > amp_gate_hi (default 3 deg)
    # within velocity_window_s (default 70 ms) — i.e. gate = 1 in steady
    # state. 60 deg/s × 70 ms = 4.2 deg p2p > 3.
    _RAMP_DEG_PER_S = 60.0

    @staticmethod
    def _push_ramp_and_collect(
        robot, bus, rate_hz: float, duration_s: float, ramp_deg_per_s: float
    ) -> tuple[list[tuple[float, float]], float]:
        """Push a deterministic ramp ``intent = ramp_deg_per_s * t`` at
        ``rate_hz`` for ``duration_s``.

        Returns ``(trace, t0)`` where trace is
        ``[(write_timestamp_relative_to_t0, shoulder_pan_motor_cmd), ...]``.
        ``t0`` is the start time of this run (controller's clock).
        """
        bus.sync_write_log.clear()
        period = 1.0 / rate_hz
        n_pushes = max(1, int(rate_hz * duration_s))
        # Tag each sync_write with its perf_counter timestamp so we can
        # slice the trace by elapsed time later.
        tagged: list[tuple[float, dict]] = []
        original_side_effect = bus.sync_write.side_effect

        def _tagged_side_effect(data_name, goal_dict, **kwargs):
            tagged.append((time.perf_counter(), dict(goal_dict)))
            # Also keep the existing fixture log behaviour.
            original_side_effect(data_name, goal_dict, **kwargs)

        bus.sync_write.side_effect = _tagged_side_effect
        try:
            t0 = time.perf_counter()
            for i in range(n_pushes):
                t_target = t0 + i * period
                sleep_for = t_target - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                t_caller = time.perf_counter()
                intent_val = ramp_deg_per_s * (t_caller - t0)
                robot.send_action(
                    {f"{m}.pos": intent_val for m in _MOTOR_NAMES},
                )
            # Brief drain so the last few writes settle.
            time.sleep(0.02)
        finally:
            bus.sync_write.side_effect = original_side_effect
        trace = [(t - t0, g["shoulder_pan"]) for t, g in tagged if "shoulder_pan" in g]
        return trace, t0

    def test_motor_cmd_consistent_across_push_rates(self, follower):
        """30 Hz push and 200 Hz push of the same ramp should produce
        motor_cmd traces that match in the steady-state mid-run window.
        The sampling is taken DURING the run, not after, so the
        velocity-estimator window is always fresh.

        Under stateful_lp (the default), motor_cmd at controller-tick
        time ``now`` is ``intent + v · (L + elapsed_since_publish)``,
        where ``v`` is the per-publish EMA velocity. That formula collapses
        to ``v · (now + L)`` for a clean ramp, so the two push rates
        should match within EMA convergence residual.

        Pre-fix (window-based estimator with no elapsed extension): the
        30 Hz path under-extrapolated by ~50% — both because n_samples
        oscillated 2↔3 in the window AND because the controller treated
        the latest-published intent as if it had just arrived.
        """
        robot, bus = follower
        duration_s = 0.6  # long enough for warmup + steady-state sampling

        # Run at 30 Hz, capture trace.
        trace_30, _ = self._push_ramp_and_collect(
            robot,
            bus,
            rate_hz=30,
            duration_s=duration_s,
            ramp_deg_per_s=self._RAMP_DEG_PER_S,
        )
        # Brief pause + reset the intent ring so the second run is isolated.
        time.sleep(0.05)
        robot._controller._intent_ring.clear()
        robot._controller._last_action = None  # reset per-step clamp baseline
        robot._controller._v_lp_state = None
        robot._controller._prev_publish_intent = None
        robot._controller._prev_publish_t = None

        # Run at 200 Hz, capture trace.
        trace_200, _ = self._push_ramp_and_collect(
            robot,
            bus,
            rate_hz=200,
            duration_s=duration_s,
            ramp_deg_per_s=self._RAMP_DEG_PER_S,
        )

        assert len(trace_30) > 50, f"30 Hz trace too short: {len(trace_30)}"
        assert len(trace_200) > 50, f"200 Hz trace too short: {len(trace_200)}"

        def _steady_state_motor_cmd(trace: list[tuple[float, float]]) -> float:
            """Median of motor_cmd in the steady-state window [0.2s, 0.45s].
            Skips warmup (max_step_deg clamp + EMA convergence) and the
            after-last-push tail (where the velocity-estimator window can
            flush out)."""
            mid = [v for t, v in trace if 0.2 <= t <= 0.45]
            assert mid, "no writes in steady-state window"
            return float(np.median(mid))

        # motor_cmd at tick time t ≈ v · (t + L). Median t in [0.2, 0.45]
        # is 0.325, so expected ≈ 60 · 0.405 = 24.3.
        ss_30 = _steady_state_motor_cmd(trace_30)
        ss_200 = _steady_state_motor_cmd(trace_200)
        lookahead_s = robot._controller._lookahead_s
        expected_median = self._RAMP_DEG_PER_S * (0.325 + lookahead_s)

        # Headline assertion: the two push rates yield matching motor_cmds.
        assert abs(ss_30 - ss_200) < 0.5, (
            f"motor_cmd steady-state: 30Hz={ss_30:.2f}, 200Hz={ss_200:.2f} — rate-agnostic invariant violated"
        )

        # Sanity: both are in the right ballpark. Wider tolerance because
        # EMA velocity is biased low during convergence (especially at
        # 30 Hz where α ≈ 0.45 per publish).
        for rate_label, ss in [("30Hz", ss_30), ("200Hz", ss_200)]:
            assert abs(ss - expected_median) < 2.0, (
                f"{rate_label}: motor_cmd steady-state={ss:.2f}, expected≈{expected_median:.2f}"
            )


# ============================================================================
# Starvation warning
# ============================================================================


class TestStarvationWarning:
    def test_warns_when_declared_period_elapses_without_sample(self, follower, caplog):
        """When the caller declares period_s but stops publishing for
        more than 3·period, a one-time WARNING fires explaining the gap.
        """
        import logging

        robot, _bus = follower
        # Push one sample with a declared 33 ms period.
        robot.send_action(
            {f"{m}.pos": 0.0 for m in _MOTOR_NAMES},
            period_s=1.0 / 30.0,
        )
        # Now wait 4× the declared period without pushing again.
        caplog.set_level(logging.WARNING)
        # Need ≥1 s for the throttled starvation check to fire (it
        # only runs at most once per second in _tick). Plus a bit of
        # margin for the controller to actually reach that check.
        time.sleep(1.3)
        # The warning should have fired exactly once.
        starvation_msgs = [r for r in caplog.records if "no intent samples received" in r.getMessage()]
        assert len(starvation_msgs) == 1, f"expected 1 starvation warning, got {len(starvation_msgs)}"

    def test_no_warning_when_period_not_declared(self, follower, caplog):
        """Period-less callers (the historical default) get no starvation
        spam — the controller has no expectation to violate."""
        import logging

        robot, _bus = follower
        robot.send_action(
            {f"{m}.pos": 0.0 for m in _MOTOR_NAMES},
            # period_s NOT declared
        )
        caplog.set_level(logging.WARNING)
        time.sleep(1.3)
        starvation_msgs = [r for r in caplog.records if "no intent samples received" in r.getMessage()]
        assert starvation_msgs == []

    def test_warning_resets_on_fresh_sample(self, follower, caplog):
        """If a fresh sample arrives after a starvation warning fired,
        the latch resets so a future starvation re-warns."""
        import logging

        robot, _bus = follower
        # Sample 1, with declared period → wait to starve → wait for warn.
        robot.send_action(
            {f"{m}.pos": 0.0 for m in _MOTOR_NAMES},
            period_s=1.0 / 30.0,
        )
        caplog.set_level(logging.WARNING)
        time.sleep(1.3)
        assert robot._controller._warned_starvation is True
        # Fresh sample resets the latch.
        robot.send_action(
            {f"{m}.pos": 1.0 for m in _MOTOR_NAMES},
            period_s=1.0 / 30.0,
        )
        assert robot._controller._warned_starvation is False
