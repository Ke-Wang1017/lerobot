#!/usr/bin/env python3
"""
OpenPI inference client for bi_so107_follower robot.

Connects to Ke's async_pi05_websocket_server via JSON websocket.
The server auto-generates subtasks and actions from images + state.

Controls:
  Escape  ->  abort (soft landing)

Usage:
  # Terminal 1: Start server
  cd ~/Documents/openpi_subtask
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/async_pi05/async_pi05_websocket_server.py \
    --config soarm_pi05_flow_lora \
    --checkpoint ~/.cache/openpi/checkpoints/soarm-pi05-flow-lora-8000 \
    --gpu-id 0 --port 8765

  # Terminal 2: Run robot
  python openpi_infer.py --task "assemble cylinder into ring"
"""

import argparse
import asyncio
import contextlib
import json
import logging
import signal
import time
from pathlib import Path

import draccus
import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401

# Register draccus ChoiceRegistry subclasses before any decode calls
from lerobot.robots import (
    bi_so107_follower,  # noqa: F401
    make_robot_from_config,
)
from lerobot.robots.config import RobotConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import init_logging, log_say

try:
    import websockets
except ImportError:
    raise ImportError("websockets is required: pip install websockets") from None


# --------------- Constants ---------------

FPS = 30
ACTION_HORIZON = 50
PREFETCH_AT = 47

# Joint names in the order matching training data (left arm first, then right).
# From bi_so107_follower._motors_ft and so_follower.py motor definitions.
JOINT_NAMES = [
    "left_shoulder_pan.pos",
    "left_shoulder_lift.pos",
    "left_elbow_flex.pos",
    "left_forearm_roll.pos",
    "left_wrist_flex.pos",
    "left_wrist_roll.pos",
    "left_gripper.pos",
    "right_shoulder_pan.pos",
    "right_shoulder_lift.pos",
    "right_elbow_flex.pos",
    "right_forearm_roll.pos",
    "right_wrist_flex.pos",
    "right_wrist_roll.pos",
    "right_gripper.pos",
]

IMAGE_KEYS = ["front", "top", "left_wrist", "right_wrist"]

CONFIG_DIR = Path.home() / ".config" / "chop"
ROBOT_CONFIG_FILE = CONFIG_DIR / "robot_config.json"


# --------------- Helpers ---------------


def build_request(robot_obs: dict, task: str) -> dict:
    """Build a JSON-serializable request for the async server."""
    # Extract 14-dim state vector in correct joint order
    state = [float(robot_obs[name]) for name in JOINT_NAMES]

    # Write images to /dev/shm (RAM-backed tmpfs) and send paths — zero serialization cost on localhost
    images = {}
    for key in IMAGE_KEYS:
        if key in robot_obs:
            img = np.asarray(robot_obs[key], dtype=np.uint8)
            shm_path = f"/dev/shm/openpi_{key}.npy"
            np.save(shm_path, img)
            images[key] = {"shm_path": shm_path}

    return {
        "images": images,
        "high_level_prompt": task,
        "state": state,
    }


def actions_to_dicts(actions_list: list) -> list[dict]:
    """Convert list of action arrays to list of joint-name dicts."""
    result = []
    for action in actions_list:
        d = {}
        for i, name in enumerate(JOINT_NAMES):
            if i < len(action):
                d[name] = float(action[i])
        result.append(d)
    return result


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
        obs = robot.get_observation()
        hold_action = {k: v for k, v in obs.items() if k.endswith(".pos")}
        if hold_action:
            robot.send_action(hold_action)

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


def restore_torque(robot):
    """Restore full torque (in case a previous soft_land left it at 0)."""
    for arm_name in ("left_arm", "right_arm"):
        arm = getattr(robot, arm_name, None)
        if arm is not None:
            for motor_name in arm.bus.motors:
                with contextlib.suppress(Exception):
                    arm.bus.write("Torque_Limit", motor_name, 1000, normalize=False)


# --------------- Main ---------------


async def control_loop(robot, events, args):
    """Async control loop: connect to server, query model, execute actions."""
    uri = f"ws://{args.host}:{args.port}"
    logging.info(f"Connecting to OpenPI server at {uri}")

    async with websockets.connect(
        uri,
        ping_interval=60,
        ping_timeout=60,
        max_size=50 * 1024 * 1024,
    ) as ws:
        # Receive server metadata
        metadata = json.loads(await ws.recv())
        logging.info(f"Server metadata: {metadata}")

        # Log initial state for verification
        obs = robot.get_observation()
        state = [float(obs[name]) for name in JOINT_NAMES]
        logging.info(f"Initial state ({len(state)} joints): {[f'{v:.1f}' for v in state]}")
        logging.info(f"Joint order: {JOINT_NAMES}")

        action_buffer = []
        step_count = 0
        query_count = 0
        chunk_step = 0  # steps executed within current chunk
        pending_task: asyncio.Task | None = None
        # How many steps into a chunk before firing the next prefetch query.
        # With prefetch_at=25 and horizon=40 at 30fps, the query fires at
        # ~833ms into execution, giving ~830ms of inference time that overlaps
        # with the remaining 15 actions (~500ms). Result should be ready
        # just as the buffer empties.
        prefetch_at = args.prefetch_at

        async def _send_query(ws_conn, task_str, robot_ref):
            """Capture observation, send request, return parsed response."""
            obs = robot_ref.get_observation()
            req = build_request(obs, task_str)
            t0 = time.perf_counter()
            await ws_conn.send(json.dumps(req))
            resp = json.loads(await ws_conn.recv())
            return resp, (time.perf_counter() - t0) * 1000

        while True:
            loop_start = time.perf_counter()

            # Check abort
            if events.get("stop_recording"):
                logging.info("Escape pressed — aborting")
                break

            # Buffer empty — need a new prediction
            if not action_buffer:
                if pending_task is not None:
                    # Prefetch in flight — await it
                    response, query_ms = await pending_task
                    pending_task = None
                else:
                    # No prefetch (first iteration or fallback) — blocking query
                    response, query_ms = await _send_query(ws, args.task, robot)

                if response.get("status") == "error":
                    logging.error(f"Server error: {response.get('error')}")
                    break

                query_count += 1
                subtask = response.get("subtask", "")
                actions = response.get("actions")
                timing = response.get("timing", {})

                t_parts = []
                for k in (
                    "decode_images_ms",
                    "normalize_state_ms",
                    "subtask_ms",
                    "action_ms",
                    "unnormalize_ms",
                ):
                    v = timing.get(k)
                    if v is not None and v > 0:
                        t_parts.append(f"{k.replace('_ms', '')}={v:.0f}ms")
                logging.info(
                    f'[Query {query_count}] subtask="{subtask}" | '
                    f"server: {timing.get('total_ms', 0):.0f}ms ({', '.join(t_parts)}) | "
                    f"roundtrip: {query_ms:.0f}ms"
                )

                if actions is None:
                    logging.warning("No actions returned, skipping")
                    continue

                action_buffer = actions_to_dicts(actions)[: args.action_horizon]
                chunk_step = 0

                if query_count <= 3:
                    first = actions[0] if actions else []
                    logging.info(f"  First action: {[f'{v:.1f}' for v in first[:14]]}")

            # Fire prefetch query partway through the chunk
            if chunk_step == prefetch_at and pending_task is None:
                pending_task = asyncio.create_task(_send_query(ws, args.task, robot))

            # Execute next action
            action = action_buffer.pop(0)
            robot.send_action(action)
            step_count += 1
            chunk_step += 1

            # FPS control — asyncio.sleep yields to event loop for prefetch
            dt = time.perf_counter() - loop_start
            await asyncio.sleep(max(1.0 / args.fps - dt, 0.0))

        logging.info(f"Done. {step_count} steps executed, {query_count} queries made.")


def main():
    parser = argparse.ArgumentParser(description="OpenPI inference for SOARM robot")
    parser.add_argument("--host", default="localhost", help="OpenPI server host")
    parser.add_argument("--port", type=int, default=8765, help="OpenPI server port")
    parser.add_argument("--task", required=True, help="High-level task prompt")
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=ACTION_HORIZON,
        help="Steps to execute per action chunk (default: 40)",
    )
    parser.add_argument("--fps", type=int, default=FPS, help="Control loop FPS (default: 30)")
    parser.add_argument(
        "--prefetch-at",
        type=int,
        default=PREFETCH_AT,
        help="Fire prefetch query at this step within a chunk (default: 25)",
    )
    parser.add_argument(
        "--robot-config", type=str, default=str(ROBOT_CONFIG_FILE), help="Path to robot config JSON"
    )
    args = parser.parse_args()

    init_logging()

    # Load robot
    logging.info(f"Loading robot config from {args.robot_config}")
    with open(args.robot_config) as f:
        robot_raw = json.load(f)
    robot_cfg = draccus.decode(RobotConfig, robot_raw)
    robot = make_robot_from_config(robot_cfg)
    robot.connect()
    restore_torque(robot)

    listener, events = init_keyboard_listener()

    try:
        asyncio.run(control_loop(robot, events, args))
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        log_say("Stopping", True)
        soft_land(robot)
        try:
            if robot.is_connected:
                robot.disconnect()
        except Exception as e:
            logging.warning(f"Error disconnecting robot: {e}")
        if listener:
            listener.stop()
        log_say("Done", True)


if __name__ == "__main__":
    main()
