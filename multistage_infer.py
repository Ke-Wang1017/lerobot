#!/usr/bin/env python3
"""
Multi-stage inference: run a sequence of ACT policies back-to-back.

Controls:
  Right arrow  ->  advance to next stage (current stage succeeded)
  Left arrow   <-  retry current stage from the beginning
  Escape       ->  abort everything

Usage:
  python multistage_infer.py
"""

import contextlib
import json
import logging
import signal
import time
from pathlib import Path

import draccus

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts

# Register draccus ChoiceRegistry subclasses before any decode/from_pretrained calls
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: F401
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import make_default_processors
from lerobot.robots import (
    bi_so107_follower,  # noqa: F401
    make_robot_from_config,
)
from lerobot.robots.config import RobotConfig
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener, predict_action
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, init_logging, log_say

# --------------- Configuration ---------------

STAGES = [
    {
        "name": "Pick up the socket",
        "task": "Pick up the socket",
        "policy_path": "/home/feit/Documents/lerobot/outputs/act_training_pickup_socket_mar_01/checkpoints/last/pretrained_model",
        "time_limit_s": 120,  # max seconds before auto-advancing (set high, press -> when done)
    },
    {
        "name": "Pick up the cylinder",
        "task": "Pick up the cylinder",
        "policy_path": "/home/feit/Documents/lerobot/outputs/act_training_pickup_cylinder_mar_01/checkpoints/last/pretrained_model",
        "time_limit_s": 120,
    },
    {
        "name": "Insert the toy",
        "task": "Insert the toy",
        "policy_path": "/home/feit/Documents/lerobot/outputs/act_cylinder_socket_feb_18/checkpoints/last/pretrained_model",
        "time_limit_s": 120,
    },
]

FPS = 30

# Chop config paths
CONFIG_DIR = Path.home() / ".config" / "chop"
ROBOT_CONFIG_FILE = CONFIG_DIR / "robot_config.json"


def load_policy_and_processors(policy_path: str):
    """Load a pretrained policy + its pre/post processors from a checkpoint."""
    cfg = PreTrainedConfig.from_pretrained(policy_path)
    cfg.pretrained_path = policy_path

    # Get the training dataset repo_id from the saved train config
    with open(Path(policy_path) / "train_config.json") as f:
        train_cfg = json.load(f)
    repo_id = train_cfg["dataset"]["repo_id"]

    # Load dataset metadata (features + normalization stats)
    ds = LeRobotDataset(repo_id)

    policy = make_policy(cfg, ds_meta=ds.meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=policy_path,
        dataset_stats=ds.meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": cfg.device},
        },
    )

    return policy, preprocessor, postprocessor, cfg


def soft_land(robot, duration_s=4.0, steps=20):
    """Gradually reduce torque so the robot lowers gently instead of dropping."""
    if not robot.is_connected:
        return

    buses = []
    if hasattr(robot, "left_arm"):
        buses.append(robot.left_arm.bus)
    if hasattr(robot, "right_arm"):
        buses.append(robot.right_arm.bus)
    if not buses:
        return

    try:
        # Command current position as goal so servos hold still
        obs = robot.get_observation()
        hold_action = {k: v for k, v in obs.items() if k.endswith(".pos")}
        if hold_action:
            robot.send_action(hold_action)

        # Ramp torque limit from 1000 (100%) down to 0 over `steps` increments
        step_delay = duration_s / steps
        for i in range(steps):
            torque_value = int(1000 * (1.0 - (i + 1) / steps))
            for bus in buses:
                for motor_name in bus.motors:
                    with contextlib.suppress(Exception):
                        bus.write("Torque_Limit", motor_name, torque_value, normalize=False)
            time.sleep(step_delay)

        logging.info("Soft landing complete")
    except Exception as e:
        logging.warning(f"Error during soft landing: {e}")


def main():
    init_logging()

    # Load configs from chop
    with open(ROBOT_CONFIG_FILE) as f:
        robot_raw = json.load(f)
    robot_cfg = draccus.decode(RobotConfig, robot_raw)
    robot = make_robot_from_config(robot_cfg)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # Add custom observation processor steps from the robot
    custom_steps = robot.get_observation_processor_steps()
    if custom_steps:
        robot_observation_processor.steps = custom_steps + robot_observation_processor.steps

    # Build dataset features (needed for frame building / action mapping)
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )

    robot.connect()

    # Restore full torque limit (soft_land from a previous run may have left it at 0)
    for arm_name in ("left_arm", "right_arm"):
        arm = getattr(robot, arm_name, None)
        if arm is not None:
            for motor_name in arm.bus.motors:
                with contextlib.suppress(Exception):
                    arm.bus.write("Torque_Limit", motor_name, 1000, normalize=False)

    listener, events = init_keyboard_listener()

    # Preload all policies so stage transitions are instant
    logging.info("Preloading all policies...")
    preloaded = []
    for stage in STAGES:
        logging.info(f"  Loading: {stage['name']}")
        policy, preprocessor, postprocessor, policy_cfg = load_policy_and_processors(stage["policy_path"])
        preloaded.append((policy, preprocessor, postprocessor, policy_cfg))
    logging.info("All policies loaded.")

    # CUDA warmup: run one dummy inference per policy to avoid first-frame stalls
    logging.info("Warming up policies on GPU...")
    obs = robot.get_observation()
    obs_processed = robot_observation_processor(obs)
    warmup_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)
    for i, (policy, preprocessor, postprocessor, policy_cfg) in enumerate(preloaded):
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        predict_action(
            observation=warmup_frame,
            policy=policy,
            device=get_safe_torch_device(policy_cfg.device),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy_cfg.use_amp,
            task=STAGES[i]["task"],
            robot_type=robot.robot_type,
        )
    logging.info("Warmup complete.")

    try:
        stage_idx = 0
        while stage_idx < len(STAGES):
            stage = STAGES[stage_idx]
            logging.info(f"\n{'=' * 60}")
            logging.info(f"STAGE {stage_idx + 1}/{len(STAGES)}: {stage['name']}")
            logging.info(f"{'=' * 60}")
            print(f"\n>>> STAGE {stage_idx + 1}/{len(STAGES)}: {stage['name']}")
            print("    Press -> to advance, <- to retry, ESC to abort\n")

            result = run_stage_preloaded(
                stage=stage,
                preloaded=preloaded[stage_idx],
                robot=robot,
                events=events,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                dataset_features=dataset_features,
            )

            if result == "abort":
                logging.info("Aborted by user")
                break
            elif result == "retry":
                logging.info(f"Retrying stage: {stage['name']}")
                continue
            else:  # "next"
                stage_idx += 1

        if stage_idx >= len(STAGES):
            log_say("All stages complete", True)
            logging.info("All stages complete!")

    finally:
        # Block Ctrl+C during shutdown to prevent partial torque states
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        log_say("Stopping", True)
        soft_land(robot)

        try:
            if robot.is_connected:
                robot.disconnect()
        except Exception as e:
            logging.warning(f"Error disconnecting robot (non-fatal): {e}")

        if listener:
            listener.stop()
        log_say("Done", True)


def run_stage_preloaded(
    stage: dict,
    preloaded: tuple,
    robot,
    events: dict,
    robot_action_processor,
    robot_observation_processor,
    dataset_features: dict,
):
    """Run a single policy stage with a preloaded policy. Returns 'next', 'retry', or 'abort'."""
    policy, preprocessor, postprocessor, policy_cfg = preloaded
    task = stage["task"]
    time_limit = stage["time_limit_s"]
    device = get_safe_torch_device(policy_cfg.device)

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    log_say(f"Stage: {stage['name']}", True)

    # Clear any leftover key events
    events["exit_early"] = False
    events["rerecord_episode"] = False

    start_t = time.perf_counter()

    while True:
        loop_start = time.perf_counter()
        elapsed = loop_start - start_t

        if elapsed >= time_limit:
            logging.info(f"Time limit reached for stage: {stage['name']}")
            break

        if events["stop_recording"]:
            return "abort"

        if events["exit_early"]:
            if events["rerecord_episode"]:
                events["exit_early"] = False
                events["rerecord_episode"] = False
                return "retry"
            events["exit_early"] = False
            return "next"

        # Get observation
        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)
        observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)

        # Policy inference
        action_values = predict_action(
            observation=observation_frame,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy_cfg.use_amp,
            task=task,
            robot_type=robot.robot_type,
        )

        act_processed = make_robot_action(action_values, dataset_features)

        # Send to robot
        robot_action = robot_action_processor((act_processed, obs))
        robot.send_action(robot_action)

        dt = time.perf_counter() - loop_start
        precise_sleep(max(1 / FPS - dt, 0.0))

    return "next"


if __name__ == "__main__":
    main()
