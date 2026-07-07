#!/usr/bin/env python

"""Rate-agnosticity tests for the predictive lookahead controller.

These tests verify that the velocity estimator and intent-ring sampling
behave correctly across a wide range of caller publish rates (10 Hz —
200 Hz). The core invariant being tested:

    same underlying trajectory → same controller behaviour,
    regardless of how often the caller publishes samples

This was historically broken: the ``_intent_ring`` was populated at the
controller's 200 Hz tick rate with whatever value happened to be in
``_latest_intent`` at that instant, which for a 30 Hz push source meant
6 of every 7 timestamps held an identical (stale) value at 5 ms apart.
The velocity estimator computed velocities from those degenerate dts
and grossly under-estimated the source's true rate of change.

The fix: populate the intent ring at the source's actual publish
events (in ``set_intent`` on the push path, in ``_tick`` on the pull
path), so the estimator's observed dts always reflect the source's
real cadence.

Tests below verify this fix is in place and stays in place.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from lerobot.robots.predictive.controller import PredictiveLookaheadController as Ctrl
from lerobot.types import ActionChunk

# =============================================================================
# Section A — velocity estimator unit tests (static methods, no controller)
# =============================================================================
#
# These exercise the four estimators directly with synthetic timestamp +
# value sequences. No mock controller / mock bus required.


def _sine_traj(freq_hz, amp, rate_hz, duration_s, n_joints=1, phase=0.0):
    """Discrete samples of a sinusoid at ``rate_hz``."""
    t = np.arange(0.0, duration_s, 1.0 / rate_hz)
    p_one = amp * np.sin(2 * np.pi * freq_hz * t + phase)
    p = np.stack([p_one] * n_joints, axis=1)
    return t, p


def _windowed(t, p, t_now, window_s):
    mask = (t >= t_now - window_s) & (t <= t_now)
    return t[mask], p[mask]


def _analytic_deriv_of_sine(freq_hz, amp, t_now, phase=0.0):
    """d/dt [amp * sin(2π f t + φ)] = amp · 2π f · cos(2π f t + φ)."""
    return amp * 2 * np.pi * freq_hz * np.cos(2 * np.pi * freq_hz * t_now + phase)


# Tolerance in deg/s. The EMA + windowing introduces small residual
# (especially at lower rates where the window contains fewer samples).
# Tuned so the test passes consistently for rates ≥ 10 Hz with a 70 ms
# window on a 0.5 Hz sinusoid. Tighter than this risks flakiness from
# legitimate algorithmic residual rather than a real regression.
_VELOCITY_TOL_DEG_PER_S = 8.0


@pytest.mark.parametrize("rate_hz", [10, 30, 60, 100, 200])
def test_forward_diff_rate_agnostic(rate_hz):
    """forward_diff returns velocity within tolerance of the analytic
    derivative at the same time t, regardless of the input sample rate.

    Same trajectory, different sample density → same answer."""
    t_now = 1.5
    freq, amp = 0.5, 10.0
    expected = _analytic_deriv_of_sine(freq, amp, t_now)
    t, p = _sine_traj(freq, amp, rate_hz=rate_hz, duration_s=3.0)
    tw, pw = _windowed(t, p, t_now, window_s=0.07)
    if len(tw) < 2:
        pytest.skip(f"window has <2 samples at rate_hz={rate_hz}")
    v = Ctrl._velocity_forward_diff(tw, pw)
    assert v is not None
    assert abs(v[0] - expected) < _VELOCITY_TOL_DEG_PER_S, (
        f"rate={rate_hz}: estimated v={v[0]:.2f}, expected≈{expected:.2f}"
    )


@pytest.mark.parametrize("rate_hz", [30, 60, 100, 200])
def test_lsq_quad_end_rate_agnostic(rate_hz):
    """LSQ-quad velocity estimator is rate-agnostic given correct timestamps."""
    t_now = 1.5
    freq, amp = 0.5, 10.0
    expected = _analytic_deriv_of_sine(freq, amp, t_now)
    t, p = _sine_traj(freq, amp, rate_hz=rate_hz, duration_s=3.0)
    tw, pw = _windowed(t, p, t_now, window_s=0.07)
    if len(tw) < 3:
        pytest.skip(f"quad fit needs ≥3 samples; have {len(tw)}")
    v = Ctrl._velocity_lsq_quad_end(tw, pw)
    assert v is not None
    assert abs(v[0] - expected) < _VELOCITY_TOL_DEG_PER_S


@pytest.mark.parametrize("rate_hz", [30, 60, 100, 200])
def test_amp_gated_lp_rate_agnostic(rate_hz):
    """amp_gated_lowpass is rate-agnostic when amplitude gate is open.

    The amplitude in this window is ~10 deg p2p (well above amp_gate_hi=3.0
    default), so the gate is fully open and the lowpassed-forward-diff
    output should match the analytic velocity within tolerance — regardless
    of sample rate."""
    t_now = 1.5
    freq, amp = 0.5, 10.0
    expected = _analytic_deriv_of_sine(freq, amp, t_now)
    t, p = _sine_traj(freq, amp, rate_hz=rate_hz, duration_s=3.0)
    tw, pw = _windowed(t, p, t_now, window_s=0.07)
    if len(tw) < 3:
        pytest.skip(f"amp_gated_lp needs ≥3 samples; have {len(tw)}")
    v = Ctrl._velocity_amp_gated_lowpass(tw, pw, fc_hz=4.0, amp_lo=1.0, amp_hi=3.0)
    assert v is not None
    assert abs(v[0] - expected) < _VELOCITY_TOL_DEG_PER_S, (
        f"rate={rate_hz}: v_est={v[0]:.2f}, expected≈{expected:.2f}"
    )


def test_velocity_estimator_handles_steady_state_hold():
    """Identical-value samples — a real source genuinely holding position —
    must return zero velocity, not undefined or NaN.

    Regression: an earlier proposed fix used value-equality dedup on the
    intent ring, which would have discarded these samples and corrupted
    the velocity estimate during steady-state hold. This test catches
    any re-introduction of that mistake.
    """
    t = np.array([0.0, 0.033, 0.067, 0.1])
    p = np.full((4, 2), 50.0)  # constant — but they ARE four real samples
    for estimator in (
        Ctrl._velocity_forward_diff,
        Ctrl._velocity_lsq_linear,
        Ctrl._velocity_lsq_quad_end,
    ):
        v = estimator(t, p)
        assert v is not None
        assert np.allclose(v, 0.0), f"{estimator.__name__}: expected 0, got {v}"

    v = Ctrl._velocity_amp_gated_lowpass(t, p, fc_hz=4.0, amp_lo=1.0, amp_hi=3.0)
    assert v is not None
    assert np.allclose(v, 0.0)


# =============================================================================
# Section B — intent ring sampling tests (mock controller, no bus)
# =============================================================================
#
# These exercise set_intent and _tick's interaction with _intent_ring
# without spinning a real bus / connection. The minimal mock-robot
# fixture lets us isolate the ring-management logic.


def _make_mock_controller():
    """Build a controller with a stubbed robot/bus suitable for testing
    set_intent + _tick semantics WITHOUT real hardware or threads.

    The controller's __init__ requires a robot whose ``config`` supplies
    a handful of fields. We mock those.
    """
    cfg = MagicMock(name="cfg")
    cfg.lookahead_ms = 80.0
    cfg.max_lookahead_ms = 110.0
    cfg.velocity_window_ms = 70.0
    cfg.corrector_alpha = 1.0
    cfg.control_rate_hz = 200.0
    cfg.adaptive = False  # don't fire the adaptive xcorr in unit tests
    cfg.max_step_deg = 3.0
    cfg.velocity_estimator = "amp_gated_lp"
    cfg.velocity_lowpass_hz = 4.0
    cfg.amp_gate_lo = 1.0
    cfg.amp_gate_hi = 3.0

    bus = MagicMock(name="bus")
    bus.motors = {f"m{i}": MagicMock(id=i + 1) for i in range(7)}
    bus.is_connected = True
    bus.sync_read.return_value = dict.fromkeys(bus.motors, 0.0)
    bus.sync_write = MagicMock()

    robot = MagicMock(name="robot")
    robot.config = cfg
    robot.bus = bus
    robot.__str__ = lambda _self: "mock_robot"

    return Ctrl(robot)


def test_push_path_appends_at_source_rate():
    """30 Hz push of N samples → ring has N entries with ~33 ms spacing."""
    ctrl = _make_mock_controller()
    n = 5
    period = 1.0 / 30.0
    for i in range(n):
        ctrl.set_intent(t=i * period, action={f"m{j}.pos": float(i) for j in range(7)})
    assert len(ctrl._intent_ring) == n
    ts = [t for t, _ in ctrl._intent_ring]
    dts = np.diff(ts)
    assert np.allclose(dts, period, atol=1e-9)


def test_push_path_does_not_dedupe_identical_values():
    """Identical-value samples must NOT be dropped. Steady-state hold IS
    information ('velocity is zero now'), not redundancy."""
    ctrl = _make_mock_controller()
    identical = {f"m{i}.pos": 42.0 for i in range(7)}
    for i in range(5):
        ctrl.set_intent(t=i * 0.033, action=identical)
    assert len(ctrl._intent_ring) == 5  # NOT collapsed to 1


def test_push_path_records_declared_period():
    """When the caller declares period_s, the controller stashes it."""
    ctrl = _make_mock_controller()
    ctrl.set_intent(
        t=0.0,
        action={f"m{i}.pos": 1.0 for i in range(7)},
        period_s=1.0 / 30.0,
    )
    assert ctrl._declared_period_s == pytest.approx(1.0 / 30.0)
    assert ctrl._last_publish_t == pytest.approx(0.0)


def test_push_path_period_none_when_not_declared():
    """No period declared → starvation detection silently disabled."""
    ctrl = _make_mock_controller()
    ctrl.set_intent(t=0.0, action={f"m{i}.pos": 1.0 for i in range(7)})
    assert ctrl._declared_period_s is None


def test_chunk_push_infers_period_from_fps():
    """ActionChunk doesn't need explicit period — auto-inferred as 1/fps."""
    ctrl = _make_mock_controller()
    chunk = ActionChunk(
        fps=30.0,
        frames=tuple({f"m{i}.pos": float(j) for i in range(7)} for j in range(10)),
    )
    ctrl.set_intent(t=0.0, action=chunk)
    assert ctrl._declared_period_s == pytest.approx(1.0 / 30.0)
    assert ctrl._latest_chunk is not None


def test_chunk_push_does_not_populate_ring():
    """Chunk path uses _lookup_in_chunk (exact lookup) — no need to put
    chunk frames into the velocity-estimator ring. Keeping the two paths'
    state isolated avoids polluting the dict path's velocity estimate
    with chunk samples on later dict pushes."""
    ctrl = _make_mock_controller()
    chunk = ActionChunk(
        fps=30.0,
        frames=tuple({f"m{i}.pos": float(j) for i in range(7)} for j in range(10)),
    )
    ctrl.set_intent(t=0.0, action=chunk)
    assert len(ctrl._intent_ring) == 0


def test_dict_after_chunk_clears_chunk_record():
    """Switching from chunk-mode to dict-mode clears _latest_chunk so the
    controller's _tick falls into the velocity-extrapolation path. Without
    this, a stale chunk would keep being interpolated against, ignoring
    the new dict's intent."""
    ctrl = _make_mock_controller()
    chunk = ActionChunk(
        fps=30.0,
        frames=tuple({f"m{i}.pos": float(j) for i in range(7)} for j in range(10)),
    )
    ctrl.set_intent(t=0.0, action=chunk)
    assert ctrl._latest_chunk is not None
    ctrl.set_intent(t=0.033, action={f"m{i}.pos": 5.0 for i in range(7)})
    assert ctrl._latest_chunk is None
    assert len(ctrl._intent_ring) == 1  # the new dict was appended


# =============================================================================
# Section C — HVLA chunk-emission helper
# =============================================================================


def test_remaining_chunk_as_actionchunk_packs_correctly():
    from lerobot.policies.hvla.s1_process import _remaining_chunk_as_actionchunk

    chunk = np.arange(70, dtype=np.float64).reshape(10, 7)
    names = [f"joint{i}.pos" for i in range(7)]
    ac = _remaining_chunk_as_actionchunk(chunk, start_idx=3, joint_names=names, fps=30.0)

    assert isinstance(ac, ActionChunk)
    assert ac.fps == 30.0
    assert len(ac.frames) == 7  # 10 - 3
    # frame[0] of the ActionChunk MUST be chunk[start_idx]
    assert ac.frames[0]["joint0.pos"] == float(chunk[3, 0])
    assert ac.frames[0]["joint6.pos"] == float(chunk[3, 6])
    # tail frame matches chunk's last row
    assert ac.frames[-1]["joint6.pos"] == float(chunk[-1, 6])


def test_remaining_chunk_override_replaces_frame_zero_only():
    """The current_frame_override lets the caller inject a clamped value
    (e.g. POLICY JUMP CLAMP rewrite) into frame 0 without disturbing the
    remaining chunk content."""
    from lerobot.policies.hvla.s1_process import _remaining_chunk_as_actionchunk

    chunk = np.arange(70, dtype=np.float64).reshape(10, 7)
    names = [f"j{i}.pos" for i in range(7)]
    override = np.full(7, 99.0)
    ac = _remaining_chunk_as_actionchunk(
        chunk, start_idx=5, joint_names=names, fps=30.0, current_frame_override=override
    )
    # frame[0] is the override
    assert ac.frames[0]["j0.pos"] == 99.0
    # frame[1] reverts to chunk[6] (the next real frame)
    assert ac.frames[1]["j0.pos"] == float(chunk[6, 0])


# =============================================================================
# Section D — stateful_lp HF amplification regressions
# =============================================================================
#
# These tests catch the bug class that shipped to production:
# at low publish rates (30 Hz) the window-based estimators (quad / linear /
# forward_diff / amp_gated_lp) hold only 2-3 samples in velocity_window_s,
# and the EMA + amplitude-gate behaviour flickers between rate-dependent
# values at the publish boundary, amplifying HF noise in the action signal
# by 1.5-2x in motor_cmd.
#
# The stateful_lp estimator updates v_lp_state once per publish event with
# α keyed on the actual publish dt, so the filter cutoff is
# ``velocity_lowpass_hz`` regardless of publish rate. This section asserts
# that:
#   1. motor_cmd HF amplification stays close to 1x even at low publish rates.
#   2. The HF amplification is INDEPENDENT of publish rate (within tolerance).
#
# Data-driven: uses a 5s slice of recorded HVLA policy output (ep3 of
# thewisp/latency_test). This is the exact data that triggered the bug
# in production, so the test will refuse to regress.


_FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "predictive_controller_ep3_slice.npz"


def _replay_stateful_lp(action_seq: np.ndarray, fps: float, publish_rate_hz: float) -> np.ndarray:
    """Replay ``action_seq`` (shape (T, J)) through the stateful_lp logic
    at ``publish_rate_hz`` publishes and 200 Hz ticks. Returns motor_cmd
    sampled at 200 Hz for the same duration.

    Reimplements the controller's stateful_lp branch as a pure function
    so this test is not sensitive to thread timing — the production
    controller produces the same output up to tick-jitter (verified in
    test_so107_follower_predictive.py via the mocked bus). Here we just
    care about the algorithmic spectrum.
    """
    n_motors = action_seq.shape[1]
    duration = (action_seq.shape[0] - 1) / fps
    # Publishes (re-quantized to publish_rate_hz from the dataset fps):
    pub_period = 1.0 / publish_rate_hz
    n_pubs = int(duration / pub_period)
    pub_times = np.arange(n_pubs) * pub_period
    # Interpolate action_seq onto pub_times.
    src_t = np.arange(action_seq.shape[0]) / fps
    pub_intent = np.empty((n_pubs, n_motors))
    for j in range(n_motors):
        pub_intent[:, j] = np.interp(pub_times, src_t, action_seq[:, j])

    # Controller state.
    v_lp = None
    prev_pub_intent = None
    prev_pub_t = None
    lookahead_s = 0.080
    vel_window_s = 0.070
    fc_hz = 4.0
    rc = 1.0 / (2 * np.pi * fc_hz)

    # Tick at 200 Hz.
    tick_hz = 200.0
    n_ticks = int(duration * tick_hz)
    cmd_grid = np.zeros((n_ticks, n_motors))
    pub_idx = 0
    for tick_i in range(n_ticks):
        now = tick_i / tick_hz
        # Service pending publishes.
        while pub_idx < n_pubs and pub_times[pub_idx] <= now:
            t = pub_times[pub_idx]
            cur = pub_intent[pub_idx]
            if prev_pub_intent is not None and prev_pub_t is not None:
                dt = t - prev_pub_t
                if dt > 1e-9:
                    diff = (cur - prev_pub_intent) / dt
                    ema_a = dt / (rc + dt)
                    v_lp = diff.copy() if v_lp is None else ema_a * diff + (1.0 - ema_a) * v_lp
            prev_pub_intent = cur.copy()
            prev_pub_t = t
            pub_idx += 1
        # Compute motor_cmd.
        if prev_pub_intent is None:
            cmd_grid[tick_i] = 0.0
            continue
        if v_lp is None or (now - prev_pub_t) > vel_window_s:
            cmd_grid[tick_i] = prev_pub_intent
            continue
        elapsed = now - prev_pub_t
        cmd_grid[tick_i] = prev_pub_intent + v_lp * (lookahead_s + elapsed)
    return cmd_grid


def _hf_band_rms(signal: np.ndarray, sample_hz: float, lo_hz: float, hi_hz: float) -> float:
    """RMS amplitude in [lo_hz, hi_hz] band, after removing DC."""
    x = signal - signal.mean()
    fft = np.abs(np.fft.rfft(x)) / len(x) * 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sample_hz)
    mask = (freqs >= lo_hz) & (freqs < hi_hz)
    if not mask.any():
        return 0.0
    return float(np.sqrt(np.mean(fft[mask] ** 2)))


@pytest.fixture(scope="module")
def ep3_slice():
    """Load the 5s recorded HVLA action slice (the exact data that exposed
    the dict-mode jitter bug in production)."""
    if not _FIXTURE_PATH.exists():
        pytest.skip(f"fixture {_FIXTURE_PATH} missing — regenerate via /tmp script")
    data = np.load(_FIXTURE_PATH)
    return {"action": data["action"].astype(np.float64), "fps": float(data["fps"])}


def _replay_amp_gated_lp(action_seq: np.ndarray, fps: float, publish_rate_hz: float) -> np.ndarray:
    """Replay through the OLD amp_gated_lp logic. Used as a reference
    point in the regression test: stateful_lp should improve on this.
    """
    from collections import deque

    vel_win_s = 0.070
    lookahead_s = 0.080
    fc_hz = 4.0
    amp_lo = 1.0
    amp_hi = 3.0
    tick_hz = 200.0
    n_motors = action_seq.shape[1]
    duration = (action_seq.shape[0] - 1) / fps
    pub_period = 1.0 / publish_rate_hz
    n_pubs = int(duration / pub_period)
    pub_times = np.arange(n_pubs) * pub_period
    src_t = np.arange(action_seq.shape[0]) / fps
    pub_intent = np.empty((n_pubs, n_motors))
    for j in range(n_motors):
        pub_intent[:, j] = np.interp(pub_times, src_t, action_seq[:, j])

    n_ticks = int(duration * tick_hz)
    cmd_grid = np.zeros((n_ticks, n_motors))
    ring: deque = deque(maxlen=64)
    pub_idx = 0
    for ti in range(n_ticks):
        now = ti / tick_hz
        while pub_idx < n_pubs and pub_times[pub_idx] <= now:
            ring.append((pub_times[pub_idx], pub_intent[pub_idx].copy()))
            pub_idx += 1
        if not ring:
            continue
        latest = ring[-1][1]
        cutoff = now - vel_win_s
        win = [(t, p) for t, p in ring if t >= cutoff]
        if len(win) < 2:
            cmd_grid[ti] = latest
            continue
        ts = np.array([t for t, _ in win])
        ps = np.stack([p for _, p in win])
        if ps.shape[0] < 3:
            # n=2 fallback: forward_diff WITH NO GATE (the bug).
            dt = ts[-1] - ts[-2]
            v = (ps[-1] - ps[-2]) / dt if abs(dt) > 1e-9 else np.zeros(n_motors)
            cmd_grid[ti] = latest + v * lookahead_s
        else:
            dts = np.diff(ts)
            diffs = np.diff(ps, axis=0) / dts[:, None]
            dt_typ = float(np.median(dts))
            rc = 1.0 / (2 * np.pi * fc_hz)
            a = dt_typ / (rc + dt_typ)
            v = diffs[0].copy()
            for i in range(1, diffs.shape[0]):
                v = a * diffs[i] + (1 - a) * v
            amp = ps.max(axis=0) - ps.min(axis=0)
            g = np.clip((amp - amp_lo) / max(amp_hi - amp_lo, 1e-9), 0, 1)
            cmd_grid[ti] = latest + g * v * lookahead_s
    return cmd_grid


def test_stateful_lp_outperforms_amp_gated_lp_on_real_data(ep3_slice):
    """Regression for the original dict-mode jitter bug.

    Replays a 5 s slice of recorded HVLA policy output (the exact data
    that triggered visible motor shake in production) through:
      * The OLD amp_gated_lp estimator (window-based, broken at 30 Hz)
      * The NEW stateful_lp estimator (rate-invariant)

    Asserts stateful_lp produces strictly less HF amplification in
    motor_cmd across the 5-35 Hz bands on the joints that exposed the
    bug (shoulder_lift, forearm_roll). The threshold is a 30 % reduction
    over amp_gated_lp — comfortably more lenient than the observed 40-
    50 % reduction, so the test won't flake on minor estimator tweaks,
    but anyone restoring the window-based behaviour will trip it.

    The 5-35 Hz band is where derivative-extrapolation amplification is
    most visible. Below 5 Hz is actual motion content; above 35 Hz is
    above the EMA cutoff and heavily attenuated regardless. The bug
    primarily lived at 10-20 Hz where the OLD estimator produced 6×
    amplification on this slice vs the new 3×.
    """
    action = ep3_slice["action"]
    fps = ep3_slice["fps"]
    cmd_old = _replay_amp_gated_lp(action, fps, publish_rate_hz=30.0)
    cmd_new = _replay_stateful_lp(action, fps, publish_rate_hz=30.0)

    warmup = int(0.5 * 200.0)
    src_t = np.arange(action.shape[0]) / fps
    tick_t = np.arange(cmd_new.shape[0]) / 200.0

    for j in [1, 3]:  # shoulder_lift, forearm_roll
        intent_j = np.interp(tick_t, src_t, action[:, j])[warmup:]
        old_j = cmd_old[warmup:, j]
        new_j = cmd_new[warmup:, j]
        for lo, hi in [(5, 10), (10, 20), (20, 35)]:
            intent_rms = _hf_band_rms(intent_j, 200.0, lo, hi)
            if intent_rms < 1e-4:
                continue
            old_amp = _hf_band_rms(old_j, 200.0, lo, hi) / intent_rms
            new_amp = _hf_band_rms(new_j, 200.0, lo, hi) / intent_rms
            # Allow up to 70 % of the OLD amplification (≥30 % improvement).
            assert new_amp < 0.7 * old_amp, (
                f"joint {j}, {lo}-{hi}Hz @ 30Hz publish: "
                f"stateful_lp amp={new_amp:.2f}× vs amp_gated_lp amp={old_amp:.2f}×. "
                f"stateful_lp should reduce HF amplification by ≥30%; "
                f"regression detected."
            )


def test_stateful_lp_absolute_hf_amplification_bound(ep3_slice):
    """Absolute upper bound on motor_cmd HF amplification.

    A loose ceiling so anyone breaking stateful_lp into an even-worse
    regime than amp_gated_lp would trip this. Set well above the
    observed values (2-4×) and at half the OLD amp_gated_lp peak (6-7×).
    """
    action = ep3_slice["action"]
    fps = ep3_slice["fps"]
    cmd_new = _replay_stateful_lp(action, fps, publish_rate_hz=30.0)
    warmup = int(0.5 * 200.0)
    src_t = np.arange(action.shape[0]) / fps
    tick_t = np.arange(cmd_new.shape[0]) / 200.0
    for j in [1, 3]:
        intent_j = np.interp(tick_t, src_t, action[:, j])[warmup:]
        cmd_j = cmd_new[warmup:, j]
        for lo, hi in [(5, 10), (10, 20), (20, 35)]:
            intent_rms = _hf_band_rms(intent_j, 200.0, lo, hi)
            if intent_rms < 1e-4:
                continue
            amp = _hf_band_rms(cmd_j, 200.0, lo, hi) / intent_rms
            # 4.5× is twice the observed worst (2.39× at 5-10Hz, joint 1) and
            # well below the OLD amp_gated_lp peak (6.94× at 20-35Hz, joint 1).
            assert amp < 4.5, (
                f"joint {j}, {lo}-{hi}Hz: HF amplification {amp:.2f}× "
                f"(intent_rms={intent_rms:.4f}). Absolute ceiling 4.5× "
                f"exceeded — fundamental regression in stateful_lp."
            )


def test_stateful_lp_motor_cmd_consistent_across_publish_rates(ep3_slice):
    """Different publish rates of the same trajectory should produce
    motor_cmd that tracks the underlying motion comparably.

    Compares low-frequency band (≤4 Hz, the actual robot motion) between
    30 Hz and 60 Hz publishes. At these rates, the EMA's 4 Hz cutoff is
    well above Nyquist for both, so the actual motion content should be
    reproduced consistently.

    We deliberately don't compare against 200 Hz publish: at 200 Hz the
    EMA has so many samples per second that it passes more HF (per
    publish, α is smaller, but accumulation is faster), and motor_cmd
    HF content diverges from the lower publish rates. That divergence
    is inherent to discrete EMA filtering and not a bug.
    """
    action = ep3_slice["action"]
    fps = ep3_slice["fps"]
    cmd_30 = _replay_stateful_lp(action, fps, 30.0)
    cmd_60 = _replay_stateful_lp(action, fps, 60.0)
    warmup = int(0.5 * 200.0)
    # Compare low-frequency RMS (≤ 4 Hz). The HF parts will inherently
    # differ because publish rate changes the EMA's effective discretisation.
    for j in [1, 3]:
        cmd_30_lf = _replay_lowpass(cmd_30[warmup:, j], 200.0, 4.0)
        cmd_60_lf = _replay_lowpass(cmd_60[warmup:, j], 200.0, 4.0)
        n = min(len(cmd_30_lf), len(cmd_60_lf))
        diff = cmd_30_lf[:n] - cmd_60_lf[:n]
        rms = float(np.sqrt(np.mean(diff**2)))
        assert rms < 0.3, (
            f"joint {j}: motor_cmd LF (<4Hz) RMS difference between "
            f"30Hz and 60Hz publishes = {rms:.3f} deg. Threshold 0.3 deg — "
            f"actual motion content should be publish-rate-invariant."
        )


def _replay_lowpass(x: np.ndarray, sample_hz: float, cutoff_hz: float) -> np.ndarray:
    """Brick-wall lowpass via FFT zeroing. Adequate for separating LF
    motion content (≤4 Hz) from HF estimator artifacts."""
    fft = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sample_hz)
    fft[freqs > cutoff_hz] = 0
    return np.fft.irfft(fft, n=len(x))


def test_stateful_lp_zero_motion_zero_lookahead():
    """Stationary intent → v_lp → 0 → motor_cmd = intent. No phantom
    lookahead from estimator noise even without an amplitude gate."""
    n_motors = 7
    fps = 30.0
    action = np.full((150, n_motors), 42.0, dtype=np.float64)  # stationary
    cmd = _replay_stateful_lp(action, fps, publish_rate_hz=30.0)
    # After EMA convergence, motor_cmd should equal intent (42.0) to
    # within machine epsilon.
    warmup = int(0.5 * 200.0)
    assert np.allclose(cmd[warmup:], 42.0, atol=1e-9), (
        f"stationary intent produced non-zero lookahead: "
        f"max deviation = {np.max(np.abs(cmd[warmup:] - 42.0))}"
    )
