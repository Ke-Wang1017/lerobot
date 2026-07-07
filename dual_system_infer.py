#!/usr/bin/env python
"""Dual-System VLA Inference: S2 (Pi0.5) + S1 (ACTWithVLM)

S2 coroutine: Async WebSocket query to Pi0.5 at ~1Hz for scene-understanding latent.
S1 loop: Synchronous ACTWithVLM at ~30Hz for reactive action chunking.

The S2 latent is shared via a thread-safe cache. If S2 disconnects or is slow,
S1 continues with the last cached latent (graceful degradation).

Controls:
  Escape  ->  abort (soft landing)

Usage:
  # Terminal 1: Start Pi0.5 server
  cd ~/Documents/openpi_subtask && XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run python -u \
    scripts/async_pi05/async_pi05_websocket_server.py \
    --config soarm_pi05_flow_lora \
    --checkpoint ~/.cache/openpi/checkpoints/soarm-pi05-state-11997 \
    --gpu-id 0 --port 8765

  # Terminal 2: Run dual-system inference
  python dual_system_infer.py \
    --s1-checkpoint outputs/act_vlm_cylinder_ring_v4/checkpoint-80000 \
    --task "assemble cylinder into ring" \
    --s2-host localhost --s2-port 8765 \
    --resize-images 224x224
"""

import argparse
import asyncio
import contextlib
import json
import logging
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import draccus
import numpy as np
import torch
import torchvision.transforms.functional as TF

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


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("dual_system")


# --------------- Constants ---------------

FPS = 30

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

# S2 uses all 4 cameras (full scene understanding)
S2_IMAGE_KEYS = ["front", "top", "left_wrist", "right_wrist"]

CONFIG_DIR = Path.home() / ".config" / "chop"
ROBOT_CONFIG_FILE = CONFIG_DIR / "robot_config.json"


# --------------- S2 Latent Cache ---------------


@dataclass
class S2LatentCache:
    """Thread-safe cache for the S2 (VLM) latent vector."""

    latent: np.ndarray | None = None
    timestamp: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)
    update_count: int = 0

    def update(self, latent: np.ndarray):
        with self.lock:
            self.latent = latent
            self.timestamp = time.time()
            self.update_count += 1

    def get(self) -> tuple[np.ndarray | None, float]:
        with self.lock:
            return self.latent, self.timestamp

    @property
    def age_ms(self) -> float:
        with self.lock:
            if self.timestamp == 0:
                return float("inf")
            return (time.time() - self.timestamp) * 1000


# --------------- Shared Observation Cache ---------------


class ObservationCache:
    """Thread-safe cache for robot observations.

    S1 captures observations synchronously (sole caller of robot methods)
    and publishes them here. S2 reads the latest cached observation.
    No separate capture thread = no lock contention with send_action.
    """

    def __init__(self):
        self._latest_obs: dict | None = None
        self._lock = threading.Lock()

    def update(self, obs: dict):
        with self._lock:
            self._latest_obs = obs

    def get_latest(self) -> dict | None:
        with self._lock:
            return self._latest_obs


# --------------- Helpers ---------------


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

        logger.info("Soft landing complete")
    except Exception as e:
        logger.warning(f"Error during soft landing: {e}")


def restore_torque(robot):
    """Restore full torque (in case a previous soft_land left it at 0)."""
    for arm_name in ("left_arm", "right_arm"):
        arm = getattr(robot, arm_name, None)
        if arm is not None:
            for motor_name in arm.bus.motors:
                with contextlib.suppress(Exception):
                    arm.bus.write("Torque_Limit", motor_name, 1000, normalize=False)


def build_s2_request(
    robot_obs: dict, task: str, decode_subtask: bool = False, subtask_temperature: float = 0.0
) -> dict:
    """Build extract_latent request for S2 (Pi0.5), using shared memory."""
    state = [float(robot_obs[name]) for name in JOINT_NAMES]

    images = {}
    for key in S2_IMAGE_KEYS:
        if key in robot_obs:
            img = np.asarray(robot_obs[key], dtype=np.uint8)
            shm_path = f"/dev/shm/dual_s2_{key}.npy"
            np.save(shm_path, img)
            images[key] = {"shm_path": shm_path}

    return {
        "mode": "extract_latent",
        "images": images,
        "high_level_prompt": task,
        "state": state,
        "with_subtask": decode_subtask,
        "subtask_temperature": subtask_temperature,
    }


def obs_to_s1_batch(
    robot_obs: dict,
    s1_image_keys: list[str],
    s2_cache: S2LatentCache,
    s2_latent_key: str,
    device: torch.device,
    resize_to: tuple[int, int] | None = None,
    _profile: bool = False,
) -> dict:
    """Convert robot observation to S1 input batch.

    Images are converted to [1, C, H, W] float tensors normalized to [0, 1].
    State is [1, 14] float tensor.
    S2 latent is pulled from cache (or omitted for zero fallback).
    """
    batch = {}
    profile_parts = {} if _profile else None

    # Images → [1, C, H, W] float tensors
    # Resize on CPU first to minimize CPU→GPU transfer (0.6MB vs 10.5MB per image)
    for key in s1_image_keys:
        # key is like "observation.images.front" → extract camera name
        cam_name = key.split(".")[-1]
        if cam_name in robot_obs:
            if _profile:
                t0 = time.perf_counter()
            img = np.asarray(robot_obs[cam_name], dtype=np.uint8)  # HWC
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # CHW, uint8
            if _profile:
                t1 = time.perf_counter()
            if resize_to is not None:
                img_tensor = TF.resize(
                    img_tensor, list(resize_to), interpolation=TF.InterpolationMode.BILINEAR, antialias=True
                )
            if _profile:
                t2 = time.perf_counter()
            img_tensor = img_tensor.unsqueeze(0).float().div_(255.0)
            if _profile:
                t3 = time.perf_counter()
            img_tensor = img_tensor.to(device)
            if _profile:
                torch.cuda.synchronize()
                t4 = time.perf_counter()
                profile_parts[cam_name] = {
                    "np_permute": (t1 - t0) * 1000,
                    "resize": (t2 - t1) * 1000,
                    "float_div": (t3 - t2) * 1000,
                    "to_gpu": (t4 - t3) * 1000,
                }
            batch[key] = img_tensor

    # State → [1, 14]
    if _profile:
        t0 = time.perf_counter()
    state = [float(robot_obs[name]) for name in JOINT_NAMES]
    batch["observation.state"] = torch.tensor([state], dtype=torch.float32, device=device)
    if _profile:
        t1 = time.perf_counter()
        profile_parts["state"] = (t1 - t0) * 1000

    # S2 latent from cache
    if _profile:
        t0 = time.perf_counter()
    latent, ts = s2_cache.get()
    if latent is not None:
        batch[s2_latent_key] = torch.from_numpy(latent).unsqueeze(0).to(device)
        age_seconds = (time.time() - ts) if ts > 0 else 0.0
        batch["observation.s2_latent_age"] = torch.tensor([[age_seconds]], dtype=torch.float32, device=device)
    if _profile:
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        profile_parts["s2_latent"] = (t1 - t0) * 1000

    if _profile:
        batch["_profile"] = profile_parts

    return batch


# --------------- S2 Async Worker ---------------


async def s2_worker(
    obs_cache: ObservationCache,
    cache: S2LatentCache,
    server_uri: str,
    task: str,
    running: threading.Event,
    decode_subtask: bool = False,
    subtask_temperature: float = 0.0,
):
    """Async S2 loop: queries Pi0.5 for latent extraction, reads from shared obs cache."""
    retry_delay = 1.0

    while running.is_set():
        try:
            async with websockets.connect(
                server_uri,
                ping_interval=60,
                ping_timeout=60,
                max_size=50 * 1024 * 1024,
            ) as ws:
                metadata = json.loads(await ws.recv())
                logger.info("S2 connected to %s (%s)", server_uri, metadata.get("model", "?"))
                retry_delay = 1.0

                while running.is_set():
                    start = time.perf_counter()

                    obs = obs_cache.get_latest()
                    if obs is None:
                        await asyncio.sleep(0.01)
                        continue
                    request = build_s2_request(
                        obs, task, decode_subtask=decode_subtask, subtask_temperature=subtask_temperature
                    )

                    await ws.send(json.dumps(request))
                    response = json.loads(await ws.recv())

                    if response.get("status") == "success":
                        latent = np.array(response["s2_latent"], dtype=np.float32)
                        prev_latent, _ = cache.get()
                        cache.update(latent)
                        elapsed_ms = (time.perf_counter() - start) * 1000

                        # Log S2 summary every 10 seconds (not every query)
                        now = time.monotonic()
                        if not hasattr(s2_worker, "_last_log_time") or now - s2_worker._last_log_time >= 10.0:
                            s2_worker._last_log_time = now
                            timing = response.get("timing", {})
                            latent_norm = float(np.linalg.norm(latent))
                            diff_str = ""
                            if prev_latent is not None:
                                diff_str = f" | Δlatent={float(np.linalg.norm(latent - prev_latent)):.3f}"
                            subtask_str = ""
                            if "subtask" in response:
                                subtask_str = f' | subtask="{response["subtask"]}"'
                            logger.info(
                                "S2 #%d: %.0fms (prefix: %.0fms) | norm=%.2f%s%s",
                                cache.update_count,
                                elapsed_ms,
                                timing.get("prefix_ms", 0),
                                latent_norm,
                                diff_str,
                                subtask_str,
                            )
                    else:
                        logger.warning("S2 error: %s", response.get("error"))

        except (websockets.exceptions.ConnectionClosed, OSError) as e:
            if running.is_set():
                logger.warning("S2 disconnected (%s), retrying in %.0fs...", e, retry_delay)
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30.0)


def run_s2_thread(obs_cache, cache, server_uri, task, running, decode_subtask=False, subtask_temperature=0.0):
    """Run S2 worker in a dedicated thread with its own event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            s2_worker(
                obs_cache,
                cache,
                server_uri,
                task,
                running,
                decode_subtask=decode_subtask,
                subtask_temperature=subtask_temperature,
            )
        )
    except Exception as e:
        logger.error("S2 thread crashed: %s", e)
    finally:
        loop.close()


# --------------- Main ---------------


def control_loop(robot, events, args):
    """S1 control loop running at target FPS with cached S2 latent."""
    from lerobot.policies.act_vlm.modeling_act_vlm import S2_LATENT_KEY, ACTWithVLMPolicy
    from lerobot.policies.factory import make_pre_post_processors

    device = torch.device(args.device)

    # Parse resize
    resize_to = None
    if args.resize_images:
        h, w = args.resize_images.lower().split("x")
        resize_to = (int(h), int(w))
        logger.info("Will resize S1 images to %s", resize_to)

    # Load S1 policy
    logger.info("Loading S1 policy from %s...", args.s1_checkpoint)
    policy = ACTWithVLMPolicy.from_pretrained(pretrained_name_or_path=args.s1_checkpoint)

    # Override chunking config for inference
    if args.temporal_ensemble_coeff is not None:
        policy.config.temporal_ensemble_coeff = args.temporal_ensemble_coeff
        policy.config.n_action_steps = 1  # required by temporal ensembling
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

        policy.temporal_ensembler = ACTTemporalEnsembler(
            args.temporal_ensemble_coeff, policy.config.chunk_size
        )
        logger.info(
            "Temporal ensembling enabled (coeff=%.3f), re-querying every step", args.temporal_ensemble_coeff
        )
    elif args.n_action_steps is not None:
        policy.config.n_action_steps = args.n_action_steps
        logger.info(
            "n_action_steps=%d (re-query every %.1fs at %d FPS)",
            args.n_action_steps,
            args.n_action_steps / args.fps,
            args.fps,
        )

    policy.to(device)
    policy.eval()
    policy.reset()

    s1_image_keys = list(policy.config.image_features.keys())
    logger.info("S1 loaded. Image keys: %s | Device: %s", s1_image_keys, device)

    # Load preprocessor (normalization) and postprocessor (unnormalization)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.s1_checkpoint,
    )

    # Shared obs cache: S1 publishes, S2 reads (no lock contention with robot)
    obs_cache = ObservationCache()

    # S2 setup: either zero-stub (ablation) or live server
    s2_cache = S2LatentCache()
    s2_running = threading.Event()
    s2_thread = None

    if args.zero_s2:
        logger.info("--zero-s2: S2 disabled, using zero latent (ablation mode)")
        zero_latent = np.zeros(policy.config.s2_latent_dim, dtype=np.float32)
        s2_cache.update(zero_latent)
    else:
        s2_running.set()
        s2_thread = threading.Thread(
            target=run_s2_thread,
            args=(obs_cache, s2_cache, f"ws://{args.s2_host}:{args.s2_port}", args.task, s2_running),
            kwargs={"decode_subtask": args.decode_subtask, "subtask_temperature": args.subtask_temperature},
            daemon=True,
        )
        s2_thread.start()

    # Publish one observation so S2 can make its first request
    if not args.zero_s2:
        logger.info("Capturing initial observation for S2...")
        init_obs = robot.get_observation()
        obs_cache.update(init_obs)

    # Wait for first S2 latent (skip if zero-stub already populated).
    # torch.compile warmup can take 30-60s on first run, so be patient.
    s2_wait_timeout = 120.0
    if not args.zero_s2:
        logger.info(
            "Waiting for first S2 latent (may take up to %.0fs for torch.compile warmup)...", s2_wait_timeout
        )
    wait_start = time.time()
    while s2_cache.latent is None and (time.time() - wait_start) < s2_wait_timeout:
        time.sleep(0.5)
    if s2_cache.latent is not None:
        logger.info(
            "Got first S2 latent (dim=%d) after %.1fs", s2_cache.latent.shape[0], time.time() - wait_start
        )
    else:
        logger.warning("No S2 latent after %.0fs, starting S1 with zero latent", s2_wait_timeout)

    # S1 control loop
    n_steps = policy.config.n_action_steps
    te_mode = policy.config.temporal_ensemble_coeff is not None
    logger.info(
        "Starting S1 loop at %d FPS (n_action_steps=%d, temporal_ensemble=%s)", args.fps, n_steps, te_mode
    )
    step_count = 0
    s1_infer_times = []

    try:
        while True:
            loop_start = time.perf_counter()

            # Check abort
            if events.get("stop_recording"):
                logger.info("Escape pressed — aborting")
                break

            # S1 captures observation synchronously (sole caller of robot methods)
            t_obs = time.perf_counter()
            obs = robot.get_observation()
            obs_ms = (time.perf_counter() - t_obs) * 1000

            # Publish for S2 to read asynchronously
            obs_cache.update(obs)

            t_prep = time.perf_counter()
            do_profile = step_count < 3 or step_count % 500 == 0
            batch = obs_to_s1_batch(
                obs, s1_image_keys, s2_cache, S2_LATENT_KEY, device, resize_to=resize_to, _profile=do_profile
            )
            prep_profile = batch.pop("_profile", None)
            t_preproc = time.perf_counter()
            batch = preprocessor(batch)
            preproc_ms = (time.perf_counter() - t_preproc) * 1000
            prep_ms = (time.perf_counter() - t_prep) * 1000
            if prep_profile is not None:
                parts = []
                for cam, timings in prep_profile.items():
                    if isinstance(timings, dict):
                        parts.append(
                            f"{cam}: np={timings['np_permute']:.1f} resize={timings['resize']:.1f} "
                            f"float={timings['float_div']:.1f} gpu={timings['to_gpu']:.1f}ms"
                        )
                    else:
                        parts.append(f"{cam}: {timings:.1f}ms")
                logger.info(
                    "Step %d prep profile (%.1fms total, preproc=%.1fms): %s",
                    step_count,
                    prep_ms,
                    preproc_ms,
                    " | ".join(parts),
                )

            t_infer = time.perf_counter()
            with torch.no_grad():
                action = policy.select_action(batch)  # [batch, action_dim]
                action = postprocessor(action)
            infer_ms = (time.perf_counter() - t_infer) * 1000
            s1_infer_times.append(infer_ms)

            # Convert to joint-name dict and send
            action_np = action.cpu().numpy()[0]
            action_dict = {}
            for i, name in enumerate(JOINT_NAMES):
                if i < len(action_np):
                    action_dict[name] = float(action_np[i])

            if logger.isEnabledFor(logging.DEBUG) and (step_count < 5 or step_count % 100 == 0):
                raw_state = np.array([float(obs[name]) for name in JOINT_NAMES])
                delta = action_np - raw_state
                logger.debug(
                    "Step %d LEFT  | state=%s | action=%s | delta=%s",
                    step_count,
                    np.array2string(raw_state[:7], precision=1, suppress_small=True),
                    np.array2string(action_np[:7], precision=1, suppress_small=True),
                    np.array2string(delta[:7], precision=1, suppress_small=True),
                )
                logger.debug(
                    "Step %d RIGHT | state=%s | action=%s | delta=%s",
                    step_count,
                    np.array2string(raw_state[7:], precision=1, suppress_small=True),
                    np.array2string(action_np[7:], precision=1, suppress_small=True),
                    np.array2string(delta[7:], precision=1, suppress_small=True),
                )

            t_send = time.perf_counter()
            robot.send_action(action_dict)
            send_ms = (time.perf_counter() - t_send) * 1000
            step_count += 1

            if step_count % 100 == 0 and s1_infer_times:
                loop_ms = (time.perf_counter() - loop_start) * 1000
                avg_infer = np.mean(s1_infer_times[-20:])
                logger.info(
                    "Step %d | loop: %.1fms (obs: %.1fms, prep: %.1fms, infer: %.1fms avg %.1fms, send: %.1fms) "
                    "| S2 age: %.0fms | S2 updates: %d",
                    step_count,
                    loop_ms,
                    obs_ms,
                    prep_ms,
                    infer_ms,
                    avg_infer,
                    send_ms,
                    s2_cache.age_ms,
                    s2_cache.update_count,
                )

            # FPS control
            dt = time.perf_counter() - loop_start
            sleep_s = max(1.0 / args.fps - dt, 0.0)
            if sleep_s > 0:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        s2_running.clear()
        if s2_thread is not None:
            s2_thread.join(timeout=5.0)

        if s1_infer_times:
            logger.info(
                "Done. %d steps | S1 avg infer: %.1fms | S2 updates: %d",
                step_count,
                np.mean(s1_infer_times),
                s2_cache.update_count,
            )


def main():
    parser = argparse.ArgumentParser(description="Dual-System VLA Inference (S2: Pi0.5 + S1: ACTWithVLM)")
    parser.add_argument("--s1-checkpoint", required=True, help="Path to trained ACTWithVLM checkpoint")
    parser.add_argument("--task", required=True, help="High-level task prompt (for S2)")
    parser.add_argument("--s2-host", default="localhost", help="Pi0.5 server host")
    parser.add_argument("--s2-port", type=int, default=8765, help="Pi0.5 server port")
    parser.add_argument("--fps", type=int, default=FPS, help="S1 control loop FPS (default: 30)")
    parser.add_argument("--device", default="cuda", help="S1 device")
    parser.add_argument(
        "--robot-config", type=str, default=str(ROBOT_CONFIG_FILE), help="Path to robot config JSON"
    )
    parser.add_argument(
        "--decode-subtask",
        action="store_true",
        help="Also AR-decode subtask text from S2 for debugging (adds ~AR latency per S2 query)",
    )
    parser.add_argument(
        "--subtask-temperature",
        type=float,
        default=0.0,
        help="Temperature for subtask AR decoding (0=greedy argmax, >0=multinomial sampling). "
        "Try 0.3-0.7 to avoid subtask getting stuck on one answer.",
    )
    parser.add_argument(
        "--zero-s2", action="store_true", help="Ablation: disable S2 server and feed zero latent to S1"
    )
    parser.add_argument(
        "--resize-images",
        type=str,
        default=None,
        help="Resize images to HxW (e.g. 224x224). Must match training config.",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="Execute N steps from each chunk before re-querying (default: chunk_size=100)",
    )
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=None,
        help="Enable temporal ensembling with this coefficient (e.g. 0.01). "
        "Re-queries every step and blends overlapping chunks. Overrides --n-action-steps.",
    )
    args = parser.parse_args()

    init_logging()

    # Load robot (same as openpi_infer.py)
    logger.info("Loading robot config from %s", args.robot_config)
    with open(args.robot_config) as f:
        robot_raw = json.load(f)
    robot_cfg = draccus.decode(RobotConfig, robot_raw)
    robot = make_robot_from_config(robot_cfg)
    robot.connect()
    restore_torque(robot)

    listener, events = init_keyboard_listener()

    try:
        control_loop(robot, events, args)
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        log_say("Stopping", True)
        soft_land(robot)
        try:
            if robot.is_connected:
                robot.disconnect()
        except Exception as e:
            logger.warning("Error disconnecting robot: %s", e)
        if listener:
            listener.stop()
        log_say("Done", True)


if __name__ == "__main__":
    main()
