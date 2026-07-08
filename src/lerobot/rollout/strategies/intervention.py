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

"""Shared human-in-the-loop intervention mechanism for rollout strategies.

Both DAgger (interactive imitation learning) and RLT (online RL fine-tuning)
run the same core loop: a policy drives autonomously, a human takes over via
teleop to correct/rescue, and the takeover is recorded. This module holds the
reusable half of that mechanism so a strategy only has to supply what is
genuinely unique to it:

  * the **state machine** — :class:`InterventionPhase` (AUTONOMOUS / PAUSED /
    CORRECTING) and :class:`InterventionEvents` (thread-safe event container
    with a declarative legal-transition table),
  * the **input devices** — keyboard / foot-pedal listeners mapping raw keys
    onto abstract phase events,
  * the **smooth teleop handover** — :meth:`InterventionStrategy._apply_transition`,
    which drives the leader/follower so the operator takes over without a jerk,
  * a set of **no-op extension hooks** (:meth:`on_tick`, :meth:`on_transition`,
    :meth:`on_terminal`, :meth:`background_step`, …) that a training strategy
    like RLT overrides to store transitions / run gradient updates, and which
    a pure recording strategy like DAgger leaves untouched.

``DAggerStrategy`` and (later) ``RLTStrategy`` both subclass
:class:`InterventionStrategy`. The pre-existing ``DAggerPhase`` / ``DAggerEvents``
names are kept as aliases so existing imports and tests are unaffected.
"""

from __future__ import annotations

import enum
import logging
from threading import Event, Lock
from typing import TYPE_CHECKING

from lerobot.common.control_utils import (
    follower_smooth_move_to,
    teleop_smooth_move_to,
    teleop_supports_feedback,
)

from .core import RolloutStrategy

if TYPE_CHECKING:
    from ..configs import RolloutStrategyConfig
    from ..context import RolloutContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class InterventionPhase(enum.Enum):
    """Observable phases of a human-in-the-loop rollout episode."""

    AUTONOMOUS = "autonomous"  # Policy driving
    PAUSED = "paused"  # Engine paused, teleop aligned, awaiting input
    CORRECTING = "correcting"  # Human driving via teleop, recording interventions


# Valid (current_phase, event) -> next_phase
_INTERVENTION_TRANSITIONS: dict[tuple[InterventionPhase, str], InterventionPhase] = {
    (InterventionPhase.AUTONOMOUS, "pause_resume"): InterventionPhase.PAUSED,
    (InterventionPhase.PAUSED, "pause_resume"): InterventionPhase.AUTONOMOUS,
    (InterventionPhase.PAUSED, "correction"): InterventionPhase.CORRECTING,
    (InterventionPhase.CORRECTING, "correction"): InterventionPhase.PAUSED,
}


class InterventionEvents:
    """Thread-safe container for intervention input-device events.

    The keyboard/pedal threads write transition requests; the main loop
    consumes them.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._phase = InterventionPhase.AUTONOMOUS
        self._pending_transition: str | None = None

        # Session-level flags
        self.stop_recording = Event()
        self.upload_requested = Event()

    # -- Thread-safe phase access ------------------------------------------

    @property
    def phase(self) -> InterventionPhase:
        """Current phase of the intervention state machine."""
        with self._lock:
            return self._phase

    @phase.setter
    def phase(self, value: InterventionPhase) -> None:
        with self._lock:
            self._phase = value

    def request_transition(self, event: str) -> None:
        """Request a phase transition (called from keyboard/pedal threads).

        Only enqueues the request if it corresponds to a valid transition
        from the current phase, preventing impossible state changes.
        """
        with self._lock:
            if (self._phase, event) in _INTERVENTION_TRANSITIONS:
                self._pending_transition = event

    def consume_transition(self) -> tuple[InterventionPhase, InterventionPhase] | None:
        """Consume a pending transition (called from main loop)."""
        with self._lock:
            if self._pending_transition is None:
                return None
            key = (self._phase, self._pending_transition)
            self._pending_transition = None
            new_phase = _INTERVENTION_TRANSITIONS.get(key)
            if new_phase is None:
                return None
            old_phase = self._phase
            self._phase = new_phase
            return old_phase, new_phase

    def reset(self) -> None:
        """Reset all transient state for a fresh session."""
        with self._lock:
            self._phase = InterventionPhase.AUTONOMOUS
            self._pending_transition = None
        self.upload_requested.clear()


# ---------------------------------------------------------------------------
# Input device handlers
# ---------------------------------------------------------------------------


def init_intervention_keyboard(events: InterventionEvents, cfg):
    """Initialise a keyboard listener for the 3 intervention controls.

    Backend selection (pynput on X11 / trusted-macOS / Windows, a terminal reader on
    Wayland / headless TTY) is delegated to :func:`create_key_listener`. Returns the
    listener (exposing ``stop()``) or ``None`` when no keyboard backend is usable.
    """
    from lerobot.utils.keyboard_input import create_key_listener

    # Map config key names to abstract event names.
    key_to_event = {
        cfg.pause_resume: "pause_resume",
        cfg.correction: "correction",
    }

    def dispatch(name: str) -> None:
        """Apply a resolved key name to the intervention events."""
        if name == "esc":
            logger.info("Stop recording...")
            events.stop_recording.set()
            return
        if name in key_to_event:
            events.request_transition(key_to_event[name])
        if name == cfg.upload:
            events.upload_requested.set()

    return create_key_listener(
        dispatch,
        controls_help=(
            f"pause_resume='{cfg.pause_resume}', correction='{cfg.correction}', "
            f"upload='{cfg.upload}', ESC=stop"
        ),
    )


def init_intervention_pedal(events: InterventionEvents, cfg):
    """Initialise foot pedal listener with the 3-pedal intervention controls.

    Returns the pedal listener thread (or ``None`` if evdev is unavailable).
    """
    from lerobot.utils.pedal import start_pedal_listener

    code_to_event = {
        cfg.pause_resume: "pause_resume",
        cfg.correction: "correction",
    }

    def on_press(code: str) -> None:
        if code in code_to_event:
            events.request_transition(code_to_event[code])
        if code == cfg.upload:
            events.upload_requested.set()

    logger.info("Initializing intervention foot pedal listener (device=%s)", cfg.device_path)
    return start_pedal_listener(on_press, device_path=cfg.device_path)


# ---------------------------------------------------------------------------
# Intervention strategy base
# ---------------------------------------------------------------------------


class InterventionStrategy(RolloutStrategy):
    """Base for human-in-the-loop rollout strategies (DAgger, RLT).

    Provides the shared state machine, input-device wiring, and smooth teleop
    handover, plus a set of no-op extension hooks. Subclasses implement the
    abstract :meth:`setup` / :meth:`run` / :meth:`teardown` from
    :class:`RolloutStrategy` and override whichever hooks they need. A pure
    recording strategy (DAgger) uses none of the hooks; a training strategy
    (RLT) overrides :meth:`on_transition`, :meth:`on_terminal`,
    :meth:`background_step`, etc.
    """

    def __init__(self, config: RolloutStrategyConfig) -> None:
        super().__init__(config)
        self._listener = None
        self._pedal_thread = None
        self._events = InterventionEvents()

    # ------------------------------------------------------------------
    # Input device wiring (shared setup/teardown)
    # ------------------------------------------------------------------

    def _setup_input_device(self, input_device: str, keyboard_cfg, pedal_cfg) -> None:
        """Start the configured keyboard or pedal listener."""
        if input_device == "keyboard":
            self._listener = init_intervention_keyboard(self._events, keyboard_cfg)
        else:
            self._pedal_thread = init_intervention_pedal(self._events, pedal_cfg)

    def _teardown_input_device(self) -> None:
        """Stop the keyboard listener if one is running."""
        if self._listener is not None:
            logger.info("Stopping keyboard listener")
            self._listener.stop()

    # ------------------------------------------------------------------
    # Smooth teleop handover (shared)
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_transition(
        old_phase: InterventionPhase,
        new_phase: InterventionPhase,
        engine,
        interpolator,
        ctx: RolloutContext,
        prev_action: dict | None,
    ) -> None:
        """Execute side-effects for a validated phase transition, including smooth handovers.

        AUTONOMOUS -> PAUSED (actuated teleop):
            Pause the engine, then drive the leader arm to the follower's last
            commanded position so the operator takes over without a jerk.

        PAUSED -> CORRECTING (non-actuated teleop):
            Slide the follower to the teleop's current pose so the robot meets
            the operator's hand rather than jumping to it on the first frame.

        CORRECTING -> PAUSED (actuated teleop):
            Re-enable torque to hold position after correction.
            This will be potentially useful if cancelling the correction recording

        PAUSED -> AUTONOMOUS:
            Reset and resume the inference engine.
        """
        teleop = ctx.hardware.teleop
        robot = ctx.hardware.robot_wrapper

        logger.info("Phase transition: %s -> %s", old_phase.value, new_phase.value)
        if old_phase == InterventionPhase.AUTONOMOUS and new_phase == InterventionPhase.PAUSED:
            logger.info("Pausing engine - robot holds position")
            engine.pause()

            if teleop_supports_feedback(teleop) and prev_action is not None:
                # TODO(Maxime): prev_action is in robot action key space (output of robot_action_processor).
                # send_feedback expects teleop feedback key space. For homogeneous setups (e.g. SO-101
                # leader + SO-101 follower) the keys are identical so this works. If the processor pipeline
                # does non-trivial key renaming (e.g. a rename_map on action keys), the interpolation in
                # teleop_smooth_move_to silently no-ops and the arm doesn't move.
                logger.info("Smooth handover: moving leader arm to follower position")
                teleop_smooth_move_to(teleop, prev_action)

        elif old_phase == InterventionPhase.PAUSED and new_phase == InterventionPhase.CORRECTING:
            logger.info("Entering correction mode - human teleop control")
            if not teleop_supports_feedback(teleop) and prev_action is not None:
                logger.info("Smooth handover: sliding follower to teleop position")
                obs = robot.get_observation()
                teleop_action = teleop.get_action()
                processed = ctx.processors.teleop_action_processor((teleop_action, obs))
                target = ctx.processors.robot_action_processor((processed, obs))
                follower_smooth_move_to(robot, prev_action, target)

            # unlock the teleop for human control
            if teleop_supports_feedback(teleop):
                teleop.disable_torque()

        elif old_phase == InterventionPhase.CORRECTING and new_phase == InterventionPhase.PAUSED:
            if teleop_supports_feedback(teleop):
                teleop.enable_torque()

        elif new_phase == InterventionPhase.AUTONOMOUS:
            logger.info("Resuming autonomous mode - resetting engine and interpolator")
            interpolator.reset()
            engine.reset()
            engine.resume()

            # release teleop before resuming the policy
            if teleop_supports_feedback(teleop):
                teleop.disable_torque()

    # ------------------------------------------------------------------
    # Extension hooks — no-ops by default. A training strategy (RLT)
    # overrides these; DAgger leaves them untouched. None of them run any
    # work in the base, so subclassing costs nothing for pure recording.
    # ------------------------------------------------------------------

    def on_episode_begin(self, ctx: RolloutContext) -> None:  # noqa: B027
        """Called when a fresh episode starts (before autonomous control)."""

    def on_tick(self, phase: InterventionPhase, obs_processed: dict, action_dict: dict | None) -> None:  # noqa: B027
        """Called once per control-loop tick, after the action is dispatched."""

    def on_intervention_frame(self, human_action: dict, obs_processed: dict) -> None:  # noqa: B027
        """Called for each frame the human is teleoperating (CORRECTING phase)."""

    def on_transition(self, *args, **kwargs) -> None:  # noqa: B027
        """Called when a (s, a, r, s') transition is available (training strategies)."""

    def on_terminal(self, *args, **kwargs) -> None:  # noqa: B027
        """Called when the operator signals a terminal outcome (success/abort)."""

    def on_episode_commit(self, ctx: RolloutContext) -> None:  # noqa: B027
        """Called when an episode is finalized/committed."""

    def on_episode_discard(self, ctx: RolloutContext) -> None:  # noqa: B027
        """Called when an episode is discarded/rolled back (re-record)."""

    def background_step(self) -> None:  # noqa: B027
        """Called in idle time each tick — a slot for background gradient updates."""


# ---------------------------------------------------------------------------
# Backward-compatible aliases (DAgger was the original home of these names).
# ---------------------------------------------------------------------------

DAggerPhase = InterventionPhase
DAggerEvents = InterventionEvents
