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

"""Mixin that adds the predictive-lookahead controller to any robot.

Use::

    class MyFollowerPredictive(PredictiveLookaheadMixin, MyFollower):
        config_class = MyFollowerPredictiveRobotConfig
        name = "my_follower_predictive"

The mixin order matters: ``PredictiveLookaheadMixin`` must come BEFORE
the base robot class in the MRO so its method overrides take precedence
and chain into the base via ``super()``.

The base robot is expected to:
  * set ``self.bus`` to a ``SerialMotorsBus`` (any flavour with
    ``sync_write("Goal_Position", dict)`` and a ``motors`` mapping) in
    its ``__init__``.
  * have a ``configure()`` method that does the per-motor PID / mode
    setup (so the mixin can start the controller AFTER configure but
    BEFORE the parent's "connected" log).
  * have a ``disconnect()`` method that closes the bus + cameras.

The config is expected to mix in :class:`PredictiveControllerConfig`
fields (lookahead_ms, velocity_estimator, etc.).
"""

from __future__ import annotations

import logging
import time

from lerobot.motors.locked_bus import LockedBus
from lerobot.types import ActionChunk, RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .controller import PredictiveLookaheadController

logger = logging.getLogger(__name__)


class PredictiveLookaheadMixin:
    """Inheritance mixin: layers the predictive controller onto a robot.

    Responsibilities:
      * Wrap ``self.bus`` in ``LockedBus`` so the controller's 200 Hz
        writer thread + the main loop's reader serialize through the
        proxy's RLock without caller-side coordination.
      * Instantiate :class:`PredictiveLookaheadController` (the worker).
      * Start the controller in ``configure()`` (called from the base
        robot's ``connect()`` flow at the right point: after motors are
        configured, before the "connected" log).
      * Stop the controller in ``disconnect()`` BEFORE the base
        disconnects the bus (writer would otherwise crash on closed port).
      * Forward ``send_action`` to ``controller.set_intent``.
      * Feed observed state to ``controller.observe_state`` for the
        adaptive xcorr lag estimator.
      * Expose ``attach_teleop`` for the controller's pull path.

    The mixin holds NO motor-map / motor-count assumptions — it inherits
    the base robot's ``self.bus`` whatever it is. Adding the mixin to a
    new robot is as simple as inheriting from it; no bus duplication.
    """

    # Declared for static-analysis: these are populated by the base
    # robot class's __init__ before the mixin's __init__ runs.
    bus: LockedBus
    _controller: PredictiveLookaheadController

    def __init__(self, config) -> None:  # noqa: D401 — mixin __init__
        # Base robot.__init__ creates self.bus as a SerialMotorsBus.
        super().__init__(config)
        # Invariant: the base must have set up a bus by now.
        assert hasattr(self, "bus") and self.bus is not None, (
            f"{type(self).__name__}: base class __init__ did not set self.bus before "
            f"PredictiveLookaheadMixin.__init__"
        )
        assert not isinstance(self.bus, LockedBus), (
            f"{type(self).__name__}: self.bus is already wrapped in LockedBus. "
            f"The mixin owns this wrapping — base classes must not pre-wrap."
        )
        # Wrap in LockedBus so the controller's 200 Hz writer thread +
        # the main-loop's 30 Hz reader + soft-land per-motor writes all
        # serialize through one proxy lock. Without this, the bus would
        # need caller-side coordination at every concurrent path.
        self.bus = LockedBus(self.bus)
        # Controller construction is cheap (no thread, no I/O). The
        # 200 Hz writer thread is launched in ``configure()``.
        self._controller = PredictiveLookaheadController(self)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """Run base connect() then start the controller thread.

        Controller start happens AFTER ``super().connect()`` so the
        motors have already been configured (PID + operating mode) and
        the bus is fully up. Starting it earlier would risk the 200 Hz
        writer scribbling into a half-configured motor. The cosmetic
        cost is that the "controller started" log appears after the
        base's "connected" log; functionally this is correct ordering.
        """
        super().connect(calibrate=calibrate)
        self._controller.start()
        cfg = self.config
        logger.info(
            "%s: predictive controller started (L=%.0fms, α=%.2f, %.0fHz, adaptive=%s, estimator=%s)",
            self,
            cfg.lookahead_ms,
            cfg.corrector_alpha,
            cfg.control_rate_hz,
            cfg.adaptive,
            cfg.velocity_estimator,
        )

    @check_if_not_connected
    def disconnect(self) -> None:
        """Stop the controller first, then chain to base disconnect.

        Best-effort: if the controller's stop() raises (rare; the join
        timeout is 2 s and is itself defensive), we still attempt base
        disconnect so the port + cameras don't leak. The controller's
        error is surfaced after.
        """
        ctrl_err: BaseException | None = None
        try:
            self._controller.stop()
        except BaseException as e:
            logger.exception("%s: predictive controller stop failed", self)
            ctrl_err = e
        super().disconnect()
        if ctrl_err is not None:
            # Re-raise so the operator hears about controller-stop failures
            # (e.g. trace flush errors that the controller wrapped + logged).
            raise ctrl_err

    # ── Caller-facing methods ─────────────────────────────────────────────

    @check_if_not_connected
    def send_action(
        self,
        action: RobotAction | ActionChunk,
        *,
        period_s: float | None = None,
    ) -> RobotAction:
        """Publish intent to the controller (non-blocking).

        Two payload shapes:
          * ``RobotAction`` (a dict) — single intent at "now". The
            controller extrapolates ``now + L`` via the configured
            velocity estimator over the recent intent stream.
          * :class:`ActionChunk` — fixed-cadence horizon of intent
            samples starting at "now". The controller picks the
            lookahead target at index ``L * fps`` by interpolation;
            falls back to chunk-tail velocity extrapolation past the
            last frame.

        Args:
            action: dict or ActionChunk as above.
            period_s: Optional caller-declared publish period (e.g.
                ``1/30`` for a 30 Hz policy). When provided, enables
                starvation detection inside the controller (warn once
                if no sample arrives within several periods). For
                ActionChunk inputs, ``period_s`` defaults to
                ``1/action.fps``; pass explicitly only to override.

        Returns the operator's raw intent — ``frames[0]`` when a chunk
        was passed, the dict itself otherwise. State observed via
        ``get_observation`` will track that intent with residual
        ≈ motor_τ − L (≈ 0 at adaptive convergence). The actual motor
        command (the L-shifted target) is computed by the controller's
        200 Hz thread and never visible to the caller.
        """
        t = time.perf_counter()
        self._controller.set_intent(t, action, period_s=period_s)
        return dict(action.frames[0]) if isinstance(action, ActionChunk) else action

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """Read state + cameras and feed state to the controller.

        Timestamp is captured BEFORE super().get_observation() because
        the base reads the bus first; using t_before keeps the
        controller's adaptive xcorr aligned to when state was actually
        sampled (within a few ms of bus latency) rather than after the
        20-60 ms of camera reads that come later. This residual error
        is well under one frame at 30 fps and doesn't bias the xcorr.
        """
        t = time.perf_counter()
        obs = super().get_observation()
        self._controller.observe_state(t, obs)
        return obs

    def attach_teleop(self, teleop) -> None:
        """Bind a teleop the controller polls at control rate.

        With a teleop attached, the controller's 200 Hz tick calls
        ``teleop.get_action()`` directly to get the latest intent
        sample, bypassing ``_latest_intent`` set by ``send_action``.
        This is the path that benefits from a high-rate leader (any
        ``HighRateLeaderMixin``-backed teleop) whose ``get_action`` is
        a fast cached read — the controller's velocity estimator then
        sees one distinct sample per tick.

        ``send_action`` continues to work as before; whichever path
        wrote most recently wins each tick. The loop driver typically
        keeps calling ``send_action`` for dataset recording. When a
        teleop is attached, the recorded action is what the loop
        driver passes in — usually ``teleop.get_action()`` — so the
        dataset still gets the operator's intent.

        Pass ``teleop=None`` to detach.
        """
        self._controller.set_teleop(teleop)
        if teleop is None:
            logger.info("%s: teleop detached; controller falls back to send_action push path", self)
        else:
            logger.info(
                "%s: bound to teleop %r — controller polls teleop.get_action() at %.0f Hz",
                self,
                type(teleop).__name__,
                self.config.control_rate_hz,
            )
