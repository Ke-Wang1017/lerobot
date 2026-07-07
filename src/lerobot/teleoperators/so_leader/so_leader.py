# !/usr/bin/env python

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

import logging
import time
from typing import Any

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .config_so_leader import SO107LeaderConfig, SOLeaderTeleopConfig

logger = logging.getLogger(__name__)


class SOLeader(Teleoperator):
    """Generic SO leader base for SO-100/101/10X teleoperators."""

    config_class = SOLeaderTeleopConfig
    name = "so_leader"

    def __init__(self, config: SOLeaderTeleopConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return self.action_features

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            # Calibration file exists, ask user whether to use it or run new calibration
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        full_turn_motor = "wrist_roll"
        unknown_range_motors = [motor for motor in self.bus.motors if motor != full_turn_motor]
        print(
            f"Move all joints except '{full_turn_motor}' sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        range_mins[full_turn_motor] = 0
        range_maxes[full_turn_motor] = 4095

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def enable_torque(self) -> None:
        self.bus.enable_torque()

    def disable_torque(self) -> None:
        self.bus.disable_torque()

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position")
        action = {f"{motor}.pos": val for motor, val in action.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, float]) -> None:
        goals = {k.removesuffix(".pos"): v for k, v in feedback.items() if k.endswith(".pos")}
        if goals:
            self.bus.sync_write("Goal_Position", goals)

    @check_if_not_connected
    def disconnect(self) -> None:
        self.bus.disconnect()
        logger.info(f"{self} disconnected.")


SO100Leader = SOLeader
SO101Leader = SOLeader


class SO107Leader(SOLeader):
    """
    SO-107 Leader Arm designed by TheRobotStudio and Hugging Face.
    7-axis arm with forearm_roll joint.
    """

    config_class = SO107LeaderConfig
    name = "so107_leader"

    def __init__(self, config: SO107LeaderConfig):
        # Call Teleoperator.__init__ directly to set up config, then define our own motors
        Teleoperator.__init__(self, config)
        self.config = config

        # SO-107 specific: 7 motors including forearm_roll
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "forearm_roll": Motor(4, "sts3215", norm_mode_body),  # Extra motor for SO-107
                "wrist_flex": Motor(5, "sts3215", norm_mode_body),
                "wrist_roll": Motor(6, "sts3215", norm_mode_body),
                "gripper": Motor(7, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

        # Intervention state (toggle - press SPACE to switch between policy/teleop)
        self._intervention_active = False
        self._keyboard_listener = None
        # Guard rails on the SPACE toggle:
        # - ``_intervention_transition_lock``: set externally by s1_process
        #   while a transition (servo sync, etc.) is in progress. SPACE
        #   presses are silently ignored while True. Released after the
        #   transition completes, so the operator can normally toggle
        #   again. Prevents the failure mode where a 2nd SPACE press
        #   during a 1-3s servo sync wastes the sync round-trip.
        # - ``_last_intervention_toggle_ts``: timestamp of the most
        #   recent accepted SPACE press. A subsequent SPACE within
        #   ``_intervention_debounce_s`` is ignored — catches accidental
        #   double-tap even when no transition lock is active (e.g.
        #   intervention OFF, which is fast).
        self._intervention_transition_lock: bool = False
        # Init to -inf so the very first SPACE press is always far past
        # the debounce window. Works in production (where monotonic()
        # is in the millions of seconds anyway) and avoids surprising
        # rejection in tests that use small synthetic ``now`` values.
        self._last_intervention_toggle_ts: float = float("-inf")
        self._intervention_debounce_s: float = 0.5

    def connect(self, calibrate: bool = True) -> None:
        """Connect and optionally setup gripper bounce if configured."""
        super().connect(calibrate)

        # Setup gripper bounce if configured
        if self.config.gripper_bounce:
            logger.info("Setting up gripper bounce to neutral position (50% open)...")
            self.set_gripper_bounce()

        # Start keyboard listener if intervention enabled
        if self.config.intervention_enabled:
            self._start_keyboard_listener()

    def set_gripper_bounce(self) -> None:
        """Set gripper to bounce back to 50% open position with weak torque."""
        weak_torque = 100
        neutral_position = 50.0  # 50% open

        logger.info(f"Applying weak torque limit: {weak_torque}/1000 ({weak_torque / 10:.0f}%)")
        logger.info(f"Setting gripper neutral position to: {neutral_position}%")

        # Set weak torque limit for gentle return
        self.bus.write("Torque_Limit", "gripper", weak_torque, normalize=False)

        # Command to neutral position with weak torque
        self.bus.write("Goal_Position", "gripper", neutral_position, normalize=True)

    @property
    def feedback_features(self) -> dict[str, type]:
        """Features that can be sent as feedback (motor positions)."""
        return {f"{motor}.pos": float for motor in self.bus.motors}

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """Command leader motors to given positions (for inverse-follow).

        Args:
            feedback: Dict mapping keys like "shoulder_pan.pos" to target positions.
        """
        if not feedback:
            return

        goal_positions = {}
        for motor_name in self.bus.motors:
            pos_key = f"{motor_name}.pos"
            if pos_key in feedback:
                goal_positions[motor_name] = feedback[pos_key]

        if goal_positions:
            self.bus.sync_write("Goal_Position", goal_positions)

    def enable_torque(self, num_retry: int = 5) -> None:
        """Enable torque on leader motors (for inverse-follow mode)."""
        self.bus.enable_torque(num_retry=num_retry)

    def disable_torque(self) -> None:
        """Disable torque on leader motors (for human control)."""
        self.bus.disable_torque()

    def _try_toggle_intervention(self, now: float) -> bool:
        """Pure decision function — called by the SPACE handler. Extracted
        so the toggle logic (lock + debounce + flip) is unit-testable
        without spinning up a pynput listener.

        Returns True if the toggle was accepted (state flipped), False
        if it was rejected (lock active or debounce window). Logs the
        reason on rejection so train.log shows when the operator's key
        press was filtered.
        """
        if self._intervention_transition_lock:
            logger.info(
                "SPACE ignored — intervention transition in progress. Wait for the transition to complete."
            )
            return False
        elapsed = now - self._last_intervention_toggle_ts
        if elapsed < self._intervention_debounce_s:
            logger.info(
                "SPACE ignored — debounce (%.0fms since last toggle, min %.0fms).",
                elapsed * 1000.0,
                self._intervention_debounce_s * 1000.0,
            )
            return False
        self._last_intervention_toggle_ts = now
        self._intervention_active = not self._intervention_active
        if self._intervention_active:
            logger.info("INTERVENTION ON - Switched to teleop mode")
        else:
            logger.info("INTERVENTION OFF - Returning to policy mode")
        return True

    def _start_keyboard_listener(self) -> None:
        """Start keyboard listener for intervention detection (space key)."""
        import time as _time

        from pynput import keyboard

        def on_press(key):
            if key != keyboard.Key.space:
                return
            self._try_toggle_intervention(_time.monotonic())

        self._keyboard_listener = keyboard.Listener(on_press=on_press)
        self._keyboard_listener.start()
        logger.info("Intervention enabled: Press SPACE to toggle between policy and teleop")

    def set_intervention_transition_lock(self, locked: bool) -> None:
        """Lock or unlock SPACE toggling during a transition.

        s1_process calls this around the policy↔intervention transition
        blocks (servo sync, torque enable/disable). While locked,
        SPACE presses log "ignored — transition in progress" instead
        of toggling. The lock is independent of the time-debounce —
        an unlock followed immediately by SPACE still respects the
        debounce window since the last accepted toggle.
        """
        self._intervention_transition_lock = bool(locked)

    def get_teleop_events(self) -> dict[str, Any]:
        """Return intervention status.

        Returns:
            Dict with TeleopEvents keys indicating current intervention state.
        """
        return {
            TeleopEvents.IS_INTERVENTION: self._intervention_active
            if self.config.intervention_enabled
            else False,
            TeleopEvents.TERMINATE_EPISODE: False,
            TeleopEvents.SUCCESS: False,
            TeleopEvents.RERECORD_EPISODE: False,
        }

    def reset_intervention(self) -> None:
        """Reset intervention state for new episode."""
        self._intervention_active = False

    def disconnect(self) -> None:
        """Disconnect and clean up keyboard listener."""
        if self._keyboard_listener is not None:
            self._keyboard_listener.stop()
            self._keyboard_listener = None
        super().disconnect()
