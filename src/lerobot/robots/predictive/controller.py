#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Background controller that performs predictive lookahead on a robot bus.

Builds on a prototype lookahead + amplitude-gated cross-correlation
design, with the stateful-lp velocity estimator added in the
dict-mode-shake fix. Extracted from
``so107_follower_predictive.py`` so it can serve any
``PredictiveLookaheadMixin``-backed robot (SO-100/101/107, future
variants). The controller assumes only that ``robot.bus`` exposes
``sync_write("Goal_Position", dict)`` and that ``robot.bus.motors`` is
iterable over motor keys — Feetech-flavoured names happen to match the
SO-series, but the controller doesn't hard-code them.
"""

from __future__ import annotations

import csv
import logging
import os
import threading
import time
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from lerobot.types import ActionChunk, RobotAction, RobotObservation
from lerobot.utils.robot_utils import precise_sleep

if TYPE_CHECKING:
    from lerobot.robots.robot import Robot

logger = logging.getLogger(__name__)


class PredictiveLookaheadController:
    """Background thread + state for the predictive-lookahead controller.

    Lifecycle:
      ``__init__(robot)`` → ``start()`` (called from robot.connect()) →
      caller drives main loop (sending intents via ``robot.send_action`` →
      ``set_intent``, reading state via ``robot.get_observation`` →
      ``observe_state``) → ``stop()`` (called from robot.disconnect()).

    The control thread runs at ``config.control_rate_hz``, polls the
    latest intent, extrapolates by ``L · v_leader``, applies the
    predictor-corrector smoothing, clamps the per-step delta, and writes
    ``Goal_Position`` to the bus. Bus thread safety is provided by the
    :class:`LockedBus` proxy that wraps ``robot.bus`` — every public I/O
    method on the bus is internally serialized, so this controller and
    the main thread coexist without caller-side coordination. The
    adaptive cross-correlation update runs from the same thread every 2 s.

    The controller is motor-count-agnostic: it reads
    ``[f"{m}.pos" for m in robot.bus.motors]`` once at construction and
    uses that ordering for every per-tick array conversion. Adding a new
    robot with a different motor count (e.g. SO-100 with 6 motors vs
    SO-107 with 7) requires no changes here.
    """

    # Cross-corr scan radius (seconds).
    _MAX_LAG_S: float = 0.3
    # Confidence floor for cross-corr peak.
    _CORR_FLOOR: float = 0.95
    # State-amplitude floor (motor units). Filters out stationary joints
    # whose encoder noise can produce spurious 0.95 correlations.
    _AMP_FLOOR: float = 1.0
    # Adaptive update cadence and rolling window.
    _UPDATE_PERIOD_S: float = 2.0
    _WINDOW_S: float = 3.0
    _ADAPTIVE_ALPHA: float = 0.5

    def __init__(self, robot: Robot):
        cfg = robot.config
        # Required-attribute guard: surface a clear error if a robot
        # tries to instantiate the controller before setting up its bus.
        # The mixin enforces order in __init__, but defensive against
        # subclasses that bypass it.
        assert hasattr(robot, "bus") and robot.bus is not None, (
            f"PredictiveLookaheadController requires {type(robot).__name__}.bus to be set"
        )
        assert hasattr(robot.bus, "motors") and robot.bus.motors, (
            f"PredictiveLookaheadController requires {type(robot).__name__}.bus.motors to be a non-empty mapping"
        )
        self._robot = robot
        self._bus = robot.bus
        # Lock-down: motor key ordering is captured once so we don't
        # rebuild the dict-to-array mapping every tick.
        self._motor_keys: list[str] = [f"{m}.pos" for m in self._bus.motors]

        # Parameters (immutable after __init__)
        self._max_lookahead_s = cfg.max_lookahead_ms / 1000.0
        self._velocity_window_s = cfg.velocity_window_ms / 1000.0
        self._corrector_alpha = cfg.corrector_alpha
        self._control_dt = 1.0 / cfg.control_rate_hz
        self._max_step = cfg.max_step_deg
        self._adaptive = cfg.adaptive
        self._velocity_estimator: str = cfg.velocity_estimator
        # Knobs for the "amp_gated_lp" / "stateful_lp" estimators. Inert
        # for quad / linear / forward_diff.
        self._velocity_lowpass_hz: float = getattr(cfg, "velocity_lowpass_hz", 4.0)
        self._amp_gate_lo: float = getattr(cfg, "amp_gate_lo", 1.0)
        self._amp_gate_hi: float = getattr(cfg, "amp_gate_hi", 3.0)

        # Mutable state (control thread)
        self._lookahead_s = cfg.lookahead_ms / 1000.0
        self._intent_ring: deque[tuple[float, np.ndarray]] = deque(maxlen=64)
        self._action_ring: deque[tuple[float, np.ndarray]] = deque(maxlen=64)
        self._last_action: np.ndarray | None = None
        self._last_adaptive_t: float = 0.0

        # State log fed from get_observation() — read by control thread.
        # deque append + iteration is atomic on CPython under the GIL.
        self._state_log: deque[tuple[float, np.ndarray]] = deque(maxlen=200)
        # Intent log fed from set_intent() — also read by control thread.
        self._intent_log: deque[tuple[float, np.ndarray]] = deque(maxlen=2000)

        # Cross-thread variables for the latest intent (control thread reads
        # this each tick to know where to aim). The same lock protects
        # _intent_ring on the push path — see set_intent + _tick.
        self._target_lock = threading.Lock()
        self._latest_intent: np.ndarray | None = None
        # Chunk record (one per send_action call that passed an ActionChunk).
        # Layout: (received_at, fps, frames_arr) where frames_arr is
        # shape (N, n_motors). ``None`` means the latest send_action was a
        # single dict → controller takes the velocity-extrapolation path.
        # Latest-chunk-wins: a fresh chunk fully replaces any previous one;
        # a single-dict send replaces with None. Tracked under _target_lock
        # alongside _latest_intent so a tick sees a consistent snapshot.
        self._latest_chunk: tuple[float, float, np.ndarray] | None = None

        # Caller-declared cadence (optional). When the caller passes
        # ``period_s`` to set_intent, the controller uses it for:
        #   * starvation detection (warn if no sample arrives within
        #     several declared periods)
        #   * future replay / observability (the declared cadence is
        #     ground truth for reconstructing the sample schedule).
        # When a chunk is passed, ``_declared_period_s`` is auto-inferred
        # as 1/chunk.fps. ``None`` means the source didn't declare a
        # cadence — velocity estimator still works from observed dts in
        # _intent_ring, but starvation detection is disabled.
        self._last_publish_t: float | None = None
        self._declared_period_s: float | None = None
        # Starvation-warning latch (fires at most once per session per
        # connection — re-issued only after disconnect/reconnect).
        self._warned_starvation: bool = False
        self._last_starvation_check_t: float = 0.0

        # Stateful per-publish lowpass velocity (rate-invariant by design).
        # Updated in ``set_intent`` on each new publish via a 1st-order EMA
        # with α keyed on the actual publish-to-publish dt — the filter's
        # cutoff is ``velocity_lowpass_hz`` regardless of publish rate or
        # controller tick rate. Read each tick in ``_tick`` as constant
        # state (no per-tick recomputation, no window-membership sensitivity).
        # Used only when ``velocity_estimator == "stateful_lp"``.
        # All three fields are guarded by ``_target_lock``.
        self._v_lp_state: np.ndarray | None = None
        self._prev_publish_intent: np.ndarray | None = None
        self._prev_publish_t: float | None = None

        # Optional direct teleop binding. When set, ``_tick`` polls
        # ``teleop.get_action()`` each iteration instead of reading
        # ``_latest_intent`` (which is updated at the caller's
        # send_action cadence — typically 30 Hz). Polling a high-rate
        # teleop at the controller's 200 Hz tick gives the velocity
        # estimator one fresh sample per tick. Setting this to None
        # reverts to the send_action-driven path.
        self._teleop = None

        # Thread control
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # ── Debug instrumentation ────────────────────────────────────────
        # Set LEROBOT_PREDICTIVE_TRACE=<path.csv> to capture per-tick raw
        # data (now, n_win, intent, v_smooth, gate, motor_cmd, ...). Dumped
        # to disk at controller stop. Off by default — zero overhead when
        # the env var is unset because the append is the only work and is
        # guarded by ``self._trace_rows is not None``.
        # The robot ``id`` is inserted before the extension so bi-arm
        # setups (two controllers) produce distinct files.
        trace_env = os.environ.get("LEROBOT_PREDICTIVE_TRACE")
        if trace_env:
            base, ext = os.path.splitext(trace_env)
            self._trace_path: str | None = f"{base}_{robot.id}{ext or '.csv'}"
            self._trace_rows: list[dict] | None = []
            logger.info("%s predictive trace ENABLED → %s", self._robot, self._trace_path)
        else:
            self._trace_path = None
            self._trace_rows = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the controller's 200 Hz writer thread. Idempotent."""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._control_loop,
            name=f"{self._robot}_predictive_lookahead",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the controller thread + flush trace. Idempotent.

        MUST be called before ``self._bus.disconnect()`` — the writer
        thread would otherwise crash trying to ``sync_write`` into a
        closed port.
        """
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning("%s controller thread did not stop within 2s", self._robot)
        self._thread = None
        self._flush_trace()

    def _flush_trace(self) -> None:
        """Write the per-tick trace buffer to ``self._trace_path`` as CSV.

        Called from ``stop()``. One row per tick. Each motor's per-joint
        scalar gets its own column (``intent_j0`` ... ``intent_jN``) so
        the CSV is grep-able without unpacking arrays.
        """
        if not self._trace_rows or not self._trace_path:
            return
        n_motors = len(self._motor_keys)
        scalar_keys = [
            "t",
            "path",
            "n_win",
            "ring_size",
            "t_oldest_in_win",
            "t_newest_in_win",
            "n_samples",
            "dt_typ",
            "ema_alpha",
            "fallback",
        ]
        per_joint_keys = ["intent", "v_smooth", "v_leader", "amplitude", "gate", "raw_shifted", "motor_cmd"]
        fieldnames = list(scalar_keys)
        for k in per_joint_keys:
            for j in range(n_motors):
                fieldnames.append(f"{k}_j{j}")
        try:
            with open(self._trace_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for row in self._trace_rows:
                    flat = {k: row.get(k, "") for k in scalar_keys}
                    for k in per_joint_keys:
                        v = row.get(k)
                        if v is None:
                            for j in range(n_motors):
                                flat[f"{k}_j{j}"] = ""
                        else:
                            arr = np.asarray(v)
                            for j in range(n_motors):
                                flat[f"{k}_j{j}"] = float(arr[j]) if j < arr.size else ""
                    w.writerow(flat)
            logger.info(
                "%s predictive trace flushed: %d rows → %s",
                self._robot,
                len(self._trace_rows),
                self._trace_path,
            )
        except OSError:
            logger.exception("%s failed to write predictive trace", self._robot)

    # ── Caller-facing hooks (invoked from main thread) ────────────────────

    def set_intent(
        self,
        t: float,
        action: RobotAction | ActionChunk,
        *,
        period_s: float | None = None,
    ) -> None:
        """Receive the latest intent from the caller. Non-blocking.

        Preconditions:
          * ``t`` is a monotonic perf_counter timestamp from the caller's
            thread; the controller stamps the publish event with this.
          * ``action`` contains every motor key in ``self._motor_keys``;
            missing keys raise ``ValueError`` so the caller fails fast
            rather than producing a partial sync_write that silently
            leaves some joints at their previous goal.

        Postconditions:
          * ``self._latest_intent`` reflects the new intent.
          * For dict path: ``self._intent_ring`` has the new (t, intent)
            entry; the stateful EMA velocity is updated; the chunk record
            is cleared.
          * For chunk path: ``self._latest_chunk`` is the new chunk; the
            ring/EMA are NOT touched (chunk lookup uses chunk frames
            directly).

        Args:
            t: Caller-side ``time.perf_counter()`` at the moment of publish.
                Drives velocity estimation in the dict path — by stamping
                with the caller's timestamp (not the controller's tick
                time), the velocity estimator's observed dts reflect the
                source's actual sample period, making it rate-agnostic.
            action: Single-frame ``RobotAction`` dict OR an ``ActionChunk``.
                Dict → velocity-extrapolation path. Chunk → exact-lookup
                path (``_lookup_in_chunk``); velocity estimator bypassed.
            period_s: Optional caller-declared publish period (e.g. 1/30
                for a 30 Hz policy). Enables starvation detection. When
                omitted on the chunk path, auto-inferred as ``1/action.fps``.

        For an ``ActionChunk``, ``frames[0]`` is the "current intent" used
        as the fallback when the chunk is consumed. The rest of the chunk
        is held under ``_latest_chunk`` for the control thread's
        exact-lookup path.

        For a single ``RobotAction`` dict, ``_latest_chunk`` is cleared so
        subsequent ticks fall back to velocity extrapolation. The push
        path also appends ``(t, current_intent)`` to ``_intent_ring`` —
        this is the source's publish event timestamped at the source's
        clock, so the velocity estimator sees dts that reflect the
        caller's actual sample rate. Identical-value samples are NOT
        deduped: a steady-state hold from the caller IS information
        ("velocity is zero now"), not redundant data.

        NB: ``_intent_log`` is NOT populated here. It's populated from
        ``_tick`` instead, at the control rate, so the adaptive xcorr can
        use ``control_dt`` as its time unit cleanly (matches the
        prototype design where ``leader_log`` was appended inside the
        control loop). Logging here would have
        sample spacing = caller rate (typically 30 Hz), and the xcorr
        would mis-convert index shifts to time by a factor of
        ``caller_rate / control_rate``.
        """
        if isinstance(action, ActionChunk):
            frames_arr = self._frames_to_array(action.frames)
            current_intent = frames_arr[0]
            # Chunk path: cadence is implicit in chunk.fps. Caller can
            # still override with an explicit period_s if e.g. they
            # publish chunks at a different cadence than chunk.fps.
            declared = period_s if period_s is not None else (1.0 / action.fps)
            with self._target_lock:
                self._latest_intent = current_intent
                self._latest_chunk = (t, action.fps, frames_arr)
                self._last_publish_t = t
                self._declared_period_s = declared
                self._warned_starvation = False  # fresh sample resets latch
        else:
            current_intent = self._action_to_array(action)
            with self._target_lock:
                self._latest_intent = current_intent
                self._latest_chunk = None
                self._last_publish_t = t
                self._declared_period_s = period_s
                self._warned_starvation = False
                # Push path: this call IS the source's publish event.
                # Stamp with the caller's t — NOT the controller's tick
                # time — so velocity estimator sees source-rate dts.
                # Skip when a teleop is bound: that path fills the ring
                # in _tick from polled samples instead.
                if self._teleop is None:
                    self._intent_ring.append((t, current_intent.copy()))
                    self._update_v_lp_locked(t, current_intent)

    def observe_state(self, t: float, obs: RobotObservation) -> None:
        """Receive a state sample from get_observation(). Non-blocking."""
        state_arr = self._observation_to_array(obs)
        if state_arr is not None:
            self._state_log.append((t, state_arr))

    def set_teleop(self, teleop) -> None:
        """Bind an intent source the control thread polls each tick.

        When set, ``_tick`` calls ``teleop.get_action()`` to get the
        latest intent dict. This bypasses the ``send_action`` →
        ``set_intent`` → ``_latest_intent`` push path. Use with a
        high-rate teleop (e.g. ``SOLeaderHighRate``) whose
        ``get_action()`` is a fast cached read; the controller's
        velocity estimator then sees one fresh sample per tick.

        Pass ``None`` to detach. The teleop is expected to be
        thread-safe with respect to concurrent ``get_action`` calls
        (``SOLeaderHighRate`` guarantees this via its background-thread
        cache).
        """
        self._teleop = teleop

    # ── Background control loop ───────────────────────────────────────────

    def _control_loop(self) -> None:
        next_tick = time.perf_counter()
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception:
                # Catch-all keeps the bus alive even if one tick blows up
                # (e.g. a transient TxRxResult). Log + continue.
                logger.exception("%s predictive-lookahead tick failed", self._robot)
            next_tick += self._control_dt
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                precise_sleep(sleep_for)
            else:
                # Fell behind — reset target so we don't spin trying to
                # catch up over many ticks.
                next_tick = time.perf_counter()

    def _tick(self) -> None:
        # Pull-path: a bound teleop is the authoritative intent source.
        # Polling here at control rate gives the velocity estimator one
        # fresh sample per tick (vs. the push-path's stair-stepped 30 Hz).
        # The polled intent is treated as a dict (no chunk semantics
        # for the leader case — chunks come through send_action from
        # sources that have a real future like trajectory_replay).
        teleop = self._teleop
        is_pull_path = teleop is not None
        if is_pull_path:
            try:
                action = teleop.get_action()
            except Exception:
                logger.exception("%s teleop.get_action() failed", self._robot)
                return
            if not action:
                return  # teleop returned empty — wait for next tick
            intent = self._action_to_array(action)
            chunk = None  # leader teleop never publishes a chunk
        else:
            with self._target_lock:
                intent = self._latest_intent
                chunk = self._latest_chunk
            if intent is None:
                return  # no intent received yet; nothing to send

        now = time.perf_counter()

        # Pull path: every tick is a fresh poll of the teleop's cache
        # (e.g. SOLeaderHighRate's 200 Hz background reader). Append per
        # tick — the controller-tick rate IS the source's rate.
        # Push path: the intent ring is already populated by set_intent
        # at the source's actual rate. DO NOT append here — that would
        # re-introduce the stair-step bug where 200 Hz timestamps with
        # values that update at the caller's rate fool the velocity
        # estimator into seeing 5 ms-spaced "samples" that mostly contain
        # repeats of the latest pushed value, collapsing the velocity
        # estimate.
        if is_pull_path:
            with self._target_lock:
                self._intent_ring.append((now, intent.copy()))
                # Pull-path publish event: feed the stateful EMA so the
                # stateful_lp estimator works regardless of intent source.
                # Also update _latest_intent + _prev_publish_t so the
                # stateful_lp helper reads consistent state.
                self._latest_intent = intent
                self._update_v_lp_locked(now, intent)

        # Log "intent at now" (no lookahead) for the adaptive xcorr.
        # Done from the control thread so the sample cadence matches
        # ``control_dt`` — the xcorr then converts index shifts to time
        # cleanly. Logging from set_intent() instead would sample at the
        # caller's rate (typically 30 Hz) while the xcorr's ``dt`` would
        # still be the control rate (200 Hz), under-reporting lag by a
        # factor of caller_rate / control_rate.
        # Chunk path: interpolate the chunk at "now" so the log captures
        # the source's smooth intent, not the stair-stepped latest_intent.
        # Dict path: use the latest_intent as-is — that's the truth (the
        # caller didn't publish a smoother signal).
        if chunk is not None:
            intent_at_now = self._lookup_in_chunk(chunk, now, lookahead_s_override=0.0)
        else:
            intent_at_now = intent
        self._intent_log.append((now, intent_at_now))

        # 1. Compute the lookahead target.
        # Trace dict — populated only when LEROBOT_PREDICTIVE_TRACE is set.
        # Captures the dict-path velocity-LSQ internals so the per-tick
        # cause of any motor_cmd wobble can be inspected post-hoc.
        trace_on = self._trace_rows is not None
        trace: dict = {"t": now, "path": "chunk" if chunk is not None else "dict"} if trace_on else None

        if chunk is not None:
            raw_shifted = self._lookup_in_chunk(chunk, now)
            if trace_on:
                trace["n_win"] = 0
                trace["v_leader"] = np.zeros_like(intent)
        elif self._velocity_estimator == "stateful_lp":
            # Rate-invariant path. Everything happens in one helper that
            # reads ``_latest_intent`` and ``_prev_publish_t`` atomically
            # (single lock) so a publish arriving mid-tick can't desync
            # the intent value from the elapsed-since-publish term.
            raw_shifted, lp_trace = self._stateful_lp_raw_shifted(now)
            if trace_on:
                trace["v_smooth"] = lp_trace["v_lp_state"]
                trace["fallback"] = lp_trace["fallback"]
                trace["elapsed_since_publish"] = lp_trace["elapsed_since_publish"]
                trace["v_leader"] = lp_trace["v_lp_state"]  # gate=1 always
        else:
            # Window-based estimators (quad/linear/forward_diff/
            # amp_gated_lp). Snapshot the ring under the lock — set_intent
            # may be appending from the caller thread concurrently.
            with self._target_lock:
                ring_snapshot = list(self._intent_ring)
            cutoff = now - self._velocity_window_s
            win = [(t, p) for t, p in ring_snapshot if t >= cutoff]
            n_win = len(win)
            if trace_on:
                trace["n_win"] = n_win
                trace["ring_size"] = len(ring_snapshot)
                trace["t_oldest_in_win"] = float(win[0][0]) if win else 0.0
                trace["t_newest_in_win"] = float(win[-1][0]) if win else 0.0
            if n_win < 2:
                raw_shifted = intent.copy()
                if trace_on:
                    trace["v_leader"] = np.zeros_like(intent)
            else:
                ts = np.array([t for t, _ in win], dtype=np.float64)
                ps = np.stack([p for _, p in win])
                v_leader = self._estimate_velocity(ts, ps, trace=trace)
                raw_shifted = intent + v_leader * self._lookahead_s if v_leader is not None else intent.copy()
                if trace_on:
                    trace["v_leader"] = v_leader.copy() if v_leader is not None else np.zeros_like(intent)

        # 2. Predictor-corrector smoothing
        alpha = self._corrector_alpha
        if alpha < 1.0 and self._last_action is not None and len(self._action_ring) >= 2:
            a_cutoff = now - self._velocity_window_s
            a_win = [(t, a) for t, a in self._action_ring if t >= a_cutoff]
            if len(a_win) >= 2:
                ts_a = np.array([t for t, _ in a_win], dtype=np.float64)
                ps_a = np.stack([a for _, a in a_win])
                ts_a_c = ts_a - ts_a.mean()
                denom_a = float((ts_a_c * ts_a_c).sum())
                v_action = (ts_a_c @ ps_a) / denom_a if denom_a > 1e-12 else np.zeros_like(intent)
            else:
                v_action = np.zeros_like(intent)
            predictor = self._last_action + v_action * self._control_dt
            shifted = alpha * raw_shifted + (1.0 - alpha) * predictor
        else:
            shifted = raw_shifted

        # 3. Per-step safety clamp
        if self._last_action is not None:
            delta = shifted - self._last_action
            if np.any(np.abs(delta) > self._max_step):
                shifted = self._last_action + np.clip(delta, -self._max_step, self._max_step)

        # 4. Write to motors. Thread safety provided by LockedBus.
        goal_dict = {self._motor_keys[i].removesuffix(".pos"): float(shifted[i]) for i in range(len(shifted))}
        self._bus.sync_write("Goal_Position", goal_dict)
        self._last_action = shifted
        self._action_ring.append((now, shifted.copy()))

        # Finalise the trace row (after the bus write so motor_cmd is
        # the actual value sent). Append last to avoid partial rows on
        # mid-tick exceptions.
        if trace_on:
            trace["intent"] = intent.copy()
            trace["raw_shifted"] = raw_shifted.copy()
            trace["motor_cmd"] = shifted.copy()
            self._trace_rows.append(trace)

        # 5. Periodic adaptive update
        if self._adaptive and (now - self._last_adaptive_t) >= self._UPDATE_PERIOD_S:
            self._last_adaptive_t = now
            self._maybe_update_lookahead(now)

        # 6. Starvation check (cheap; throttled to once per second).
        if (now - self._last_starvation_check_t) > 1.0:
            self._last_starvation_check_t = now
            self._check_starvation(now)

    def _check_starvation(self, now: float) -> None:
        """Warn once if the source declared a publish period but is silent.

        Fires at most once per session per (re)connection. Reset when a
        fresh sample arrives (see set_intent — clears ``_warned_starvation``).
        Only meaningful for the push path: the pull path's ``get_action``
        always returns the leader's latest cached pose, so "starvation" is
        a missing cache (separate failure mode, logged elsewhere).
        """
        if self._teleop is not None or self._warned_starvation:
            return
        with self._target_lock:
            declared = self._declared_period_s
            last_t = self._last_publish_t
        if declared is None or last_t is None:
            return
        elapsed = now - last_t
        if elapsed > 3.0 * declared:
            logger.warning(
                "%s: no intent samples received for %.0f ms (caller declared "
                "period %.0f ms). Controller is operating on stale velocity "
                "estimate — motor will drift toward extrapolation tail.",
                self._robot,
                elapsed * 1000,
                declared * 1000,
            )
            self._warned_starvation = True

    # ── Adaptive update ──────────────────────────────────────────────────

    def _maybe_update_lookahead(self, now: float) -> None:
        """Amplitude-gated cross-correlation update of self._lookahead_s.

        Amplitude-gated cross-correlation lag estimate: symmetric scan,
        corr ≥ 0.95 AND amplitude ≥ _AMP_FLOOR, amplitude-weighted
        aggregate, α=0.5
        low-pass, hard cap.
        """
        win_start = now - self._WINDOW_S
        # Snapshot the cross-thread deques into lists before filtering.
        # set_intent / observe_state can mutate _intent_log / _state_log
        # concurrently from the caller thread; a comprehension iterating
        # the deque directly can raise "deque mutated during iteration".
        # list(deque) is atomic under the GIL and gives a stable view.
        intent_snapshot = list(self._intent_log)
        state_snapshot = list(self._state_log)
        intent_samples = [(t, p) for t, p in intent_snapshot if t >= win_start]
        state_samples = [(t, s) for t, s in state_snapshot if t >= win_start - 0.5]
        # Minimum intent-sample floor scales with the expected count
        # (window_s × control_rate). 30 % of full is the same fraction the
        # prototype uses (100 of 600 at 3 s / 200 Hz) and stays sensible
        # across control_rate_hz / WINDOW_S changes — hardcoding 30
        # samples would be a different fraction at every control rate.
        expected_intent = self._WINDOW_S / self._control_dt
        min_intent_samples = max(10, int(expected_intent * 0.3))
        if len(intent_samples) < min_intent_samples or len(state_samples) < 5:
            return

        ts_intent = np.array([t for t, _ in intent_samples], dtype=np.float64)
        gt = np.stack([p for _, p in intent_samples])
        ts_state = np.array([t for t, _ in state_samples], dtype=np.float64)
        s_reads = np.stack([s for _, s in state_samples])
        state_at_intent = np.empty_like(gt)
        for j in range(gt.shape[1]):
            state_at_intent[:, j] = np.interp(ts_intent, ts_state, s_reads[:, j])

        dt = self._control_dt
        max_lag = int(self._MAX_LAG_S / dt)
        confident_lags: list[float] = []
        confident_amps: list[float] = []
        for j in range(gt.shape[1]):
            a = gt[:, j]
            s = state_at_intent[:, j]
            state_amp = float(s.std())
            if state_amp < self._AMP_FLOOR:
                continue
            best_k, best_c = 0, -np.inf
            for k in range(-max_lag, max_lag + 1):
                if k >= 0:
                    aa, ss = (a[: len(a) - k], s[k:]) if k > 0 else (a, s)
                else:
                    m = -k
                    aa, ss = a[m:], s[: len(s) - m]
                aa_c = aa - aa.mean()
                ss_c = ss - ss.mean()
                na, ns = float(np.linalg.norm(aa_c)), float(np.linalg.norm(ss_c))
                if na < 1e-12 or ns < 1e-12:
                    continue
                c = float(np.dot(aa_c, ss_c) / (na * ns))
                if c > best_c:
                    best_c, best_k = c, k
            if best_c >= self._CORR_FLOOR:
                confident_lags.append(best_k * dt)
                confident_amps.append(state_amp)
        if not confident_lags:
            return

        weights = np.asarray(confident_amps, dtype=np.float64)
        lag = float(np.sum(np.asarray(confident_lags) * weights) / weights.sum())
        target = self._lookahead_s + lag
        new_l = self._ADAPTIVE_ALPHA * target + (1.0 - self._ADAPTIVE_ALPHA) * self._lookahead_s
        self._lookahead_s = max(0.0, min(new_l, self._max_lookahead_s))
        logger.debug(
            "%s adaptive L update: lag=%+.1fms → L=%.1fms (cap %.0fms)",
            self._robot,
            lag * 1000,
            self._lookahead_s * 1000,
            self._max_lookahead_s * 1000,
        )

    # ── Velocity estimators ──────────────────────────────────────────────
    #
    # Five implementations selectable by ``config.velocity_estimator``.
    # See ``config.py`` for the field's docstring with the rationale for
    # each. The window-based four are static (no controller state); the
    # fifth (``stateful_lp``) is stateful per-publish — see
    # ``_update_v_lp_locked`` / ``_stateful_lp_raw_shifted`` below.

    def _estimate_velocity(
        self, ts: np.ndarray, ps: np.ndarray, trace: dict | None = None
    ) -> np.ndarray | None:
        """Dispatch to the configured estimator. ``trace`` forwarded only to
        amp_gated_lp (others have no internal state worth capturing)."""
        if self._velocity_estimator == "quad":
            return self._velocity_lsq_quad_end(ts, ps)
        if self._velocity_estimator == "linear":
            return self._velocity_lsq_linear(ts, ps)
        if self._velocity_estimator == "forward_diff":
            return self._velocity_forward_diff(ts, ps)
        if self._velocity_estimator == "amp_gated_lp":
            return self._velocity_amp_gated_lowpass(
                ts, ps, self._velocity_lowpass_hz, self._amp_gate_lo, self._amp_gate_hi, trace=trace
            )
        # Unknown — fall back to linear with a one-time warning. Should not
        # be reachable because the field is Literal-typed at construction,
        # but defensive in case someone instantiates the controller
        # bypassing the dataclass init.
        logger.warning(
            "Unknown velocity_estimator=%r; falling back to 'linear'",
            self._velocity_estimator,
        )
        return self._velocity_lsq_linear(ts, ps)

    @staticmethod
    def _velocity_lsq_linear(ts: np.ndarray, ps: np.ndarray) -> np.ndarray | None:
        """Original prototype behaviour. Slope = centered velocity over the
        window. Returns slope for each joint. Biased under acceleration."""
        ts_c = ts - ts.mean()
        denom = float((ts_c * ts_c).sum())
        if denom < 1e-12:
            return None
        return (ts_c @ ps) / denom

    @staticmethod
    def _velocity_lsq_quad_end(ts: np.ndarray, ps: np.ndarray) -> np.ndarray | None:
        """Fit p(t) = a + b·t + c·t² with t expressed relative to the
        most-recent sample. The slope at the most-recent sample is the
        coefficient b. Same column-rank check as linear; falls back to
        linear on rank deficiency (e.g. only 2 samples available)."""
        if ts.shape[0] < 3:
            # Quadratic fit needs at least 3 points to be defined.
            return PredictiveLookaheadController._velocity_lsq_linear(ts, ps)
        t_rel = ts - ts[-1]  # ≤ 0; slope at t_rel=0 is the "now" slope
        # Design matrix for [a, b, c] in p = a + b·t + c·t²
        design = np.stack([np.ones_like(t_rel), t_rel, t_rel * t_rel], axis=1)
        # lstsq returns coefficients shape (3,) or (3, n_motors) depending
        # on ps shape; ps is (n, n_motors), result is (3, n_motors). The
        # slope at t_rel=0 is coef[1].
        try:
            coef, *_ = np.linalg.lstsq(design, ps, rcond=None)
        except np.linalg.LinAlgError:
            return PredictiveLookaheadController._velocity_lsq_linear(ts, ps)
        return coef[1]

    @staticmethod
    def _velocity_forward_diff(ts: np.ndarray, ps: np.ndarray) -> np.ndarray | None:
        """v(now) ≈ (p[-1] − p[-2]) / dt. Two-sample causal difference at
        the tail of the window. Unbiased, but noise-sensitive at small
        dt — fine when the source's intent log is smooth, marginal when
        samples carry measurement noise."""
        if ts.shape[0] < 2:
            return None
        dt = float(ts[-1] - ts[-2])
        if abs(dt) < 1e-9:
            return None
        return (ps[-1] - ps[-2]) / dt

    @staticmethod
    def _velocity_amp_gated_lowpass(
        ts: np.ndarray,
        ps: np.ndarray,
        fc_hz: float,
        amp_lo: float,
        amp_hi: float,
        trace: dict | None = None,
    ) -> np.ndarray | None:
        """Amplitude-gated lowpass forward-difference, per joint.

        See ``config.py`` for the design rationale. WARNING: degrades to
        ungated raw forward_diff when the window holds <3 samples (e.g.
        30 Hz publish × 70 ms window → 90% of ticks). Use ``stateful_lp``
        for low-rate dict publishes instead.

        Returns shape (n_motors,) or None when the window is too short.
        """
        n_samples, n_motors = ps.shape
        if n_samples < 3:
            v = PredictiveLookaheadController._velocity_forward_diff(ts, ps)
            if trace is not None:
                trace["fallback"] = "forward_diff"
                trace["n_samples"] = n_samples
                trace["dt_typ"] = float(ts[-1] - ts[-2]) if n_samples >= 2 else 0.0
                trace["ema_alpha"] = 0.0
                trace["v_smooth"] = v.copy() if v is not None else np.zeros(n_motors)
                trace["amplitude"] = (
                    (ps.max(axis=0) - ps.min(axis=0)) if n_samples >= 1 else np.zeros(n_motors)
                )
                trace["gate"] = np.ones(n_motors)  # forward_diff has no gate
            return v
        # Per-sample forward differences (n_samples-1, n_motors).
        dts = np.diff(ts)
        dts_safe = np.where(np.abs(dts) > 1e-9, dts, 1.0)
        diffs = np.diff(ps, axis=0) / dts_safe[:, None]
        # First-order EMA with cutoff fc_hz, integrated across the window's
        # samples. Equivalent to running ``v_smooth = α·diff + (1−α)·v_prev``
        # from window start to end. Uses median dt for α to match the
        # filter's design cutoff under (usually negligible) tick jitter.
        dt_typ = float(np.median(dts))
        rc = 1.0 / (2 * np.pi * fc_hz)
        ema_alpha = dt_typ / (rc + dt_typ)
        v_smooth = diffs[0].copy()
        for i in range(1, diffs.shape[0]):
            v_smooth = ema_alpha * diffs[i] + (1.0 - ema_alpha) * v_smooth
        # Per-joint amplitude (peak-to-peak over the window).
        amplitude = ps.max(axis=0) - ps.min(axis=0)
        # Gate ∈ [0, 1] per joint. Hard zero below amp_lo, hard one above
        # amp_hi, linear in between. clip avoids division pathology when
        # amp_lo == amp_hi (configurable edge case).
        denom = max(amp_hi - amp_lo, 1e-9)
        gate = np.clip((amplitude - amp_lo) / denom, 0.0, 1.0)
        if trace is not None:
            trace["fallback"] = ""
            trace["n_samples"] = n_samples
            trace["dt_typ"] = dt_typ
            trace["ema_alpha"] = ema_alpha
            trace["v_smooth"] = v_smooth.copy()
            trace["amplitude"] = amplitude.copy()
            trace["gate"] = gate.copy()
        return gate * v_smooth

    # ── Stateful lowpass velocity ────────────────────────────────────────

    def _update_v_lp_locked(self, t: float, current_intent: np.ndarray) -> None:
        """Update the stateful per-publish EMA velocity on a new publish.

        MUST be called with ``self._target_lock`` already held.

        α is keyed on the actual publish-to-publish dt so the effective
        filter cutoff is ``velocity_lowpass_hz`` regardless of how often
        publishes arrive. Window-based estimators (quad / linear /
        forward_diff / amp_gated_lp) all degrade at low publish rates
        because they only see 2-3 ring entries inside the velocity_window_s;
        this stateful update is rate-invariant by construction.

        Both intent-ingress paths feed this:
          * Push path (caller's ``send_action`` → ``set_intent`` dict
            branch): the new publish is the caller's event.
          * Pull path (controller polls ``teleop.get_action()`` in
            ``_tick``): each tick is treated as a publish event from the
            teleop source.
        """
        if self._prev_publish_intent is not None and self._prev_publish_t is not None:
            dt_pub = t - self._prev_publish_t
            if dt_pub > 1e-9:
                diff = (current_intent - self._prev_publish_intent) / dt_pub
                rc = 1.0 / (2 * np.pi * self._velocity_lowpass_hz)
                ema_a = dt_pub / (rc + dt_pub)
                if self._v_lp_state is None:
                    self._v_lp_state = diff.copy()
                else:
                    self._v_lp_state = ema_a * diff + (1.0 - ema_a) * self._v_lp_state
        self._prev_publish_intent = current_intent.copy()
        self._prev_publish_t = t

    def _stateful_lp_raw_shifted(self, now: float) -> tuple[np.ndarray, dict]:
        """Return ``raw_shifted`` for the stateful_lp path.

        Computes ``intent + v_lp * (L + elapsed_since_publish)`` atomically
        — single ``_target_lock`` acquisition so the intent value and the
        publish timestamp that the elapsed term is derived from come from
        the same publish event. Without that atomicity, a publish that
        arrives between the tick's prelude (which captures
        ``_latest_intent``) and the elapsed-term computation makes
        ``elapsed`` momentarily near-zero while ``intent`` is still the
        previous value — net effect: motor_cmd lags by one publish period.
        ~50 ms at 30 Hz.

        Rate-invariant: the EMA α (set in ``set_intent``) is keyed on the
        publish-to-publish dt so the filter cutoff is ``velocity_lowpass_hz``
        whether publishes arrive at 30 Hz or 200 Hz. The control tick rate
        only affects how often this method is sampled; the value it
        produces depends only on time and the publish stream.

        NO amplitude gate. The original ``amp_gated_lp`` estimator added a
        window-based amplitude gate to suppress lookahead amplification of
        residual estimator noise. In the stateful path:
          * Tremor is already attenuated by the EMA's velocity_lowpass_hz
            cutoff (default 4 Hz → 10 Hz tremor band reduced ~3×).
          * Stationary intent → v_lp converges to 0 → no lookahead by
            construction.
          * The per-step clamp (max_step_deg) caps any residual jitter.
        Adding a window-p2p gate here would re-introduce the same
        sample-count flicker that the stateful design eliminates — at
        30 Hz publishes the window holds 2-3 entries and the gate value
        oscillates between rate-dependent values every publish period.

        Stale-publish safety: if no publish has arrived within
        ``velocity_window_s`` of ``now``, return the latest intent
        unchanged (no lookahead) and clear the stored state so the next
        publish starts the EMA fresh rather than blending against an old
        velocity estimate.
        """
        n_motors = len(self._motor_keys)
        trace = {
            "v_lp_state": np.zeros(n_motors),
            "elapsed_since_publish": 0.0,
            "fallback": "",
        }
        with self._target_lock:
            intent = self._latest_intent
            v_lp = self._v_lp_state
            last_pub_t = self._prev_publish_t
        if intent is None:
            trace["fallback"] = "no_intent"
            return np.zeros(n_motors), trace
        if v_lp is None or last_pub_t is None:
            trace["fallback"] = "no_state"
            return intent.copy(), trace
        elapsed = now - last_pub_t
        if elapsed > self._velocity_window_s:
            with self._target_lock:
                self._v_lp_state = None
                self._prev_publish_intent = None
                self._prev_publish_t = None
            trace["fallback"] = "stale"
            return intent.copy(), trace
        trace["v_lp_state"] = v_lp.copy()
        trace["elapsed_since_publish"] = elapsed
        return intent + v_lp * (self._lookahead_s + elapsed), trace

    # ── Chunk lookup ─────────────────────────────────────────────────────

    def _lookup_in_chunk(
        self,
        chunk: tuple[float, float, np.ndarray],
        now: float,
        lookahead_s_override: float | None = None,
    ) -> np.ndarray:
        """Exact-lookup target at ``now + L`` using a received chunk.

        Within the chunk: linear interpolation between adjacent frames.
        Past the chunk's last frame: linear extrapolation from the
        chunk's tail velocity. Sub-frame interpolation matches what
        TrajectoryReplayTeleop already does for its own playback head;
        keeping the math symmetric avoids a subtle bias between "what
        was recorded" and "what was replayed".

        ``lookahead_s_override``: when not None, use this in place of
        ``self._lookahead_s``. Used by the adaptive log path to compute
        "intent at now" (L=0) regardless of the controller's current
        lookahead.
        """
        lookahead_s = self._lookahead_s if lookahead_s_override is None else lookahead_s_override
        received_at, fps, frames_arr = chunk
        n_frames = frames_arr.shape[0]
        elapsed = now - received_at
        target_idx_f = (elapsed + lookahead_s) * fps
        if target_idx_f <= n_frames - 1:
            lo = max(0, int(target_idx_f))
            hi = min(n_frames - 1, lo + 1)
            alpha = target_idx_f - lo
            return frames_arr[lo] * (1.0 - alpha) + frames_arr[hi] * alpha
        # Past chunk end — extrapolate from the tail velocity. Using the
        # last two frames keeps the extrapolation responsive to the most
        # recent dynamics of the source rather than averaging the whole
        # chunk, which would over-smooth at sharp transitions (e.g. an
        # end-of-trajectory deceleration).
        tail_v = (frames_arr[-1] - frames_arr[-2]) * fps if n_frames >= 2 else np.zeros_like(frames_arr[-1])
        excess_s = (target_idx_f - (n_frames - 1)) / fps
        return frames_arr[-1] + tail_v * excess_s

    # ── Helpers ──────────────────────────────────────────────────────────

    def _frames_to_array(self, frames: tuple[dict[str, float], ...]) -> np.ndarray:
        """Stack ``ActionChunk.frames`` into shape ``(N, n_motors)``.

        Each frame goes through ``_action_to_array`` so the strict-key
        check applies — a missing motor in any frame raises immediately
        instead of producing a malformed chunk that the lookup path
        would silently propagate.
        """
        return np.stack([self._action_to_array(f) for f in frames])

    def _action_to_array(self, action: RobotAction) -> np.ndarray:
        # Strict: every motor key must be present. A short dict here would
        # produce a partial sync_write (the missing motors silently hold
        # their last goal), which is unsafe — fail fast instead.
        missing = [k for k in self._motor_keys if k not in action]
        if missing:
            raise ValueError(f"action missing keys for motors: {missing}")
        return np.array([float(action[k]) for k in self._motor_keys], dtype=np.float64)

    def _observation_to_array(self, obs: RobotObservation) -> np.ndarray | None:
        # State samples drive the adaptive update only. A missing motor key
        # here is unexpected (get_observation builds the dict from the same
        # bus) so log + drop the sample rather than fail the whole tick —
        # the controller can still extrapolate from the last good state.
        missing = [k for k in self._motor_keys if k not in obs]
        if missing:
            logger.warning("state observation missing keys %s; dropping sample", missing)
            return None
        return np.array([float(obs[k]) for k in self._motor_keys], dtype=np.float64)
