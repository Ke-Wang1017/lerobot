#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.so_follower import SO107Follower, SO107FollowerConfig
from lerobot.types import ActionChunk, action_first_frame

from ..robot import Robot
from .config_bi_so107_follower import BiSO107FollowerConfig

logger = logging.getLogger(__name__)


class BiSO107Follower(Robot):
    """
    Bimanual SO-107 Follower Arms designed by TheRobotStudio and Hugging Face.
    7-axis arms with forearm_roll joint for advanced manipulation.
    """

    config_class = BiSO107FollowerConfig
    name = "bi_so107_follower"

    def __init__(self, config: BiSO107FollowerConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = SO107FollowerConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_port,
            disable_torque_on_disconnect=config.left_arm_disable_torque_on_disconnect,
            max_relative_target=config.left_arm_max_relative_target,
            use_degrees=config.left_arm_use_degrees,
            cameras={},
        )

        right_arm_config = SO107FollowerConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_port,
            disable_torque_on_disconnect=config.right_arm_disable_torque_on_disconnect,
            max_relative_target=config.right_arm_max_relative_target,
            use_degrees=config.right_arm_use_degrees,
            cameras={},
        )

        self.left_arm = SO107Follower(left_arm_config)
        self.right_arm = SO107Follower(right_arm_config)
        self.cameras = make_cameras_from_configs(config.cameras)

        # Install the depth-edge overlay as an in-camera post-grab processor
        # on every RealSense camera that has depth enabled. The grab thread
        # consumes the depth frame and caches the overlay-RGB directly, so
        # the control loop never sees the depth frame or pays the overlay
        # cost. Pre-refactor this ran as a downstream ``ObservationProcessor``
        # on the control thread (~9.8 ms p50 per iteration); pushing it to
        # the grab thread is the same idea as the streaming-encoder offload.
        from lerobot.cameras.realsense import RealSenseCamera
        from lerobot.processor import DepthEdgeOverlayProcessorStep

        for cam_key, cam in self.cameras.items():
            if isinstance(cam, RealSenseCamera) and cam.use_depth:
                cam.post_grab_processor = DepthEdgeOverlayProcessorStep(
                    camera_key=cam_key,
                    threshold_percentile=90,  # Edge sensitivity (85-95, higher = fewer edges)
                    blur_kernel=3,  # Noise reduction (1, 3, 5, 7)
                    dilation_kernel=2,  # Edge thickness (0-5)
                    alpha=0.7,  # Edge opacity (0.0-1.0)
                    min_depth=0.2,  # Min depth in meters (20cm)
                    max_depth=0.6,  # Max depth in meters (60cm)
                )

        # Build the per-arm IK kinematics now: PinkKinematics parses the
        # SO-107 URDF (~1-2 s/arm, CPU-bound). Doing it here — before
        # connect() starts the RealSense read thread — keeps that parse
        # from starving the camera and tripping its frame-age watchdog.
        # None if pin-pink is missing (Cartesian teleop then no-ops).
        self._ik_kinematics: dict[str, Any] | None = None
        try:
            from lerobot.robots.so107_description.cartesian_ik import make_so107_arm_kinematics
            from lerobot.robots.so107_description.joint_alignment import (
                LEFT_ARM_ALIGNMENT,
                RIGHT_ARM_ALIGNMENT,
            )

            self._ik_kinematics = {
                "left": make_so107_arm_kinematics(
                    LEFT_ARM_ALIGNMENT,
                    posture_cost=config.ik_posture_cost,
                    max_iters=config.ik_max_iters,
                ),
                "right": make_so107_arm_kinematics(
                    RIGHT_ARM_ALIGNMENT,
                    posture_cost=config.ik_posture_cost,
                    max_iters=config.ik_max_iters,
                ),
            }
        except Exception:
            logger.exception("%s: Cartesian-IK kinematics unavailable; Cartesian teleop disabled", self.name)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"left_{motor}.pos": float for motor in self.left_arm.bus.motors} | {
            f"right_{motor}.pos": float for motor in self.right_arm.bus.motors
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        features = {}
        for cam_key in self.cameras:
            # Only RGB frames are stored in dataset
            # Depth frames (if enabled) are used by processors for visualization but not stored
            features[cam_key] = (self.config.cameras[cam_key].height, self.config.cameras[cam_key].width, 3)

        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return (
            self.left_arm.bus.is_connected
            and self.right_arm.bus.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

        for cam in self.cameras.values():
            cam.connect()

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    def attach_teleop(self, teleop: Any) -> None:
        """Wire a teleoperator as this robot's intent source.

        For a bimanual Cartesian VR teleop (Quest), build a per-arm IK
        controller and install it into the teleop via
        ``set_action_transform`` so ``teleop.get_action()`` returns
        motor-space joint commands. The IK stays robot-owned and the
        upstream teleoperate / record / replay loops are untouched — they
        just call ``get_action()`` and receive joints.

        For a joint-space leader teleop this is a no-op: it already emits
        joint dicts, which ``send_action`` consumes directly.

        Precondition: the robot is connected — each arm's IK controller is
        seeded from that arm's current joint configuration.
        """
        # Detect a bimanual Cartesian teleop by its action features. Done
        # before importing the IK module so a joint-space leader never
        # triggers the optional pin-pink import.
        from lerobot.robots.so107_description.cartesian_ik import (
            build_so107_bimanual_ik_transform,
            is_so107_bimanual_cartesian_teleop,
        )

        if not is_so107_bimanual_cartesian_teleop(teleop):
            return

        assert self.is_connected, "attach_teleop requires the robot to be connected"
        assert hasattr(teleop, "set_action_transform"), (
            "a Cartesian teleop must expose set_action_transform()"
        )

        if self._ik_kinematics is None:
            logger.warning(
                "%s: Cartesian teleop attached but IK kinematics are unavailable "
                "(is pin-pink installed?) — the arms will not be driven.",
                self.name,
            )
            return

        transform = build_so107_bimanual_ik_transform(self._ik_kinematics, self.left_arm, self.right_arm)
        teleop.set_action_transform(transform)
        logger.info("%s: installed Cartesian-IK transform into %s", self.name, type(teleop).__name__)

    def _read_camera_frame(self, cam) -> Any:
        """Read one camera frame using the configured strategy.

        ``latest`` returns whatever is in the grab thread's buffer
        (``cam.read_latest``). Falls back to ``cam.async_read`` on the
        very first call before any frame has been captured (read_latest
        raises in that case; async_read blocks until one arrives).
        """
        if self.config.camera_read_strategy == "latest":
            try:
                return cam.read_latest()
            except RuntimeError:
                # No frame captured yet — block once to populate the buffer.
                return cam.async_read()
        return cam.async_read()

    def get_observation(self) -> dict[str, Any]:
        from lerobot.utils.latency import current_span

        obs_dict = {}

        # Per-arm motor reads. Each arm's get_observation runs sync_read on
        # its bus; ``motor_read_left`` / ``motor_read_right`` let the
        # dashboard show whether one arm is dominating (e.g. when one bus
        # is over-cabled or has retries) rather than just a combined number.
        with current_span("motor_read_left"):
            left_obs = self.left_arm.get_observation()
        obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})

        with current_span("motor_read_right"):
            right_obs = self.right_arm.get_observation()
        obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})

        # Per-camera frame reads. Strategy is ``latest`` by default — see
        # ``BiSO107FollowerConfig.camera_read_strategy`` for the trade-off.
        # Per-camera spans isolate which one (if any) is the hot stage when
        # using ``wait_for_new``.
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()

            # For RealSense cameras with depth enabled, read aligned color+depth together
            from lerobot.cameras.realsense import RealSenseCamera

            with current_span(f"camera_read_{cam_key}"):
                # If the camera has a post-grab processor installed, depth has
                # already been consumed in the grab thread and the cached frame
                # is the final processed RGB. Skip the aligned-depth path so we
                # don't drag depth back across the thread boundary.
                if isinstance(cam, RealSenseCamera) and cam.use_depth and cam.post_grab_processor is None:
                    try:
                        color_frame, depth_frame = cam.read_color_and_aligned_depth()
                        obs_dict[cam_key] = color_frame
                        obs_dict[f"{cam_key}_depth"] = depth_frame
                        dt_ms = (time.perf_counter() - start) * 1e3
                        logger.debug(f"{self} read {cam_key} + aligned depth: {dt_ms:.1f}ms")
                    except Exception as e:
                        logger.warning(f"{self} failed to read aligned frames for {cam_key}: {e}")
                        # Fallback: try reading color only via the configured strategy.
                        obs_dict[cam_key] = self._read_camera_frame(cam)
                else:
                    obs_dict[cam_key] = self._read_camera_frame(cam)
                    dt_ms = (time.perf_counter() - start) * 1e3
                    logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any] | ActionChunk) -> dict[str, Any]:
        # Plain bimanual doesn't consume the chunk horizon — falls back to
        # frames[0]. The predictive bimanual overrides this method to keep
        # the chunk intact and route per-arm sub-chunks downstream.
        action = action_first_frame(action)
        # Dry-run mode: drop the command but return the requested action
        # unchanged so callers expecting the "sent" dict back keep working.
        # First-call logging makes the mode obvious in the run log so
        # nobody mistakes a quiet motor bus for normal behaviour.
        if self.config.dry_run:
            if not getattr(self, "_dry_run_logged", False):
                logger.warning(
                    "%s: dry_run=True — send_action is a no-op. Motors will NOT move. "
                    "Disable dry_run in the robot config to drive the arms.",
                    self,
                )
                self._dry_run_logged = True
            return action

        # Remove "left_" prefix
        left_action = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }
        # Remove "right_" prefix
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }

        send_action_left = self.left_arm.send_action(left_action)
        send_action_right = self.right_arm.send_action(right_action)

        # Add prefixes back
        prefixed_send_action_left = {f"left_{key}": value for key, value in send_action_left.items()}
        prefixed_send_action_right = {f"right_{key}": value for key, value in send_action_right.items()}

        return {**prefixed_send_action_left, **prefixed_send_action_right}

    def disconnect(self):
        self.left_arm.disconnect()
        self.right_arm.disconnect()

        for cam in self.cameras.values():
            cam.disconnect()

    def get_observation_processor_steps(self) -> list:
        """Return custom observation processor steps for this robot.

        Empty by default: ``DepthEdgeOverlayProcessorStep`` used to live here
        as a downstream step on the control thread, but is now installed
        directly on the RealSense cameras as a ``post_grab_processor`` (see
        ``__init__``). The grab thread consumes depth and caches the overlay-
        RGB, so the control thread doesn't pay the ~9.8 ms per-iteration cost.
        """
        return []
