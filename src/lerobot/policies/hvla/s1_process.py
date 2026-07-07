"""S1 process: loads action policy, reads shared latent, runs robot control loop.

Runs in the main process (needs direct robot/camera access). S1 is latency-critical (~30Hz).
Adapted from dual_system_infer.py.
"""

import contextlib
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

from lerobot.policies.hvla.ipc import SharedImageBuffer, SharedLatentCache
from lerobot.policies.hvla.rlt.episode import EpisodeLifecycle, TerminalKind
from lerobot.types import ActionChunk

logger = logging.getLogger(__name__)

# Default joint names for SO107 bimanual robot (backward compat / tests)
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

# Default S1→S2 camera key map for SO107 (override via --s2-camera-map)
S2_CAM_KEY_MAP = {
    "front": "base_0_rgb",
    "top": "base_1_rgb",
    "left_wrist": "left_wrist_0_rgb",
    "right_wrist": "right_wrist_0_rgb",
}


def _joint_names_from_robot(robot) -> list[str]:
    """Derive joint names from a connected LeRobot Robot instance."""
    return list(robot.action_features.keys())


def _camera_keys_from_robot(robot) -> list[str]:
    """Derive camera keys from a connected LeRobot Robot instance."""
    return [k for k, v in robot.observation_features.items() if isinstance(v, tuple)]


def _compute_chunk_index(t_now: float, t_origin: float, fps: int, chunk_len: int) -> int:
    """Time-based index into current action chunk."""
    elapsed = t_now - t_origin
    idx = round(elapsed * fps)
    return max(0, min(idx, chunk_len - 1))


def _remaining_chunk_as_actionchunk(
    chunk_np: np.ndarray,
    start_idx: int,
    joint_names: list[str],
    fps: float,
    *,
    current_frame_override: np.ndarray | None = None,
) -> ActionChunk:
    """Pack ``chunk_np[start_idx:]`` as an ``ActionChunk`` whose frame 0 is
    the value the controller should aim at *now*.

    Why this exists: HVLA emits a numpy chunk per inference and indexes
    into it frame-by-frame. Sending one frame at a time as a dict via
    ``robot.send_action(dict)`` forces the predictive controller to
    extrapolate velocity from stair-stepped 30 Hz input — which works
    correctly only after the rate-agnostic intent-ring fix, and is
    still strictly less accurate than the chunk-exact-lookup path
    (``_lookup_in_chunk``). Sending the *remaining* chunk activates
    the exact-lookup path: the controller reads the value at the time
    ``now + L`` directly from the chunk's frames, with zero
    estimation error.

    Args:
        chunk_np: shape ``(N_frames, n_motors)`` — the policy's chunk.
        start_idx: index of the frame being executed at "now". Frames
            before this have already been consumed.
        joint_names: column order for the output frame dicts.
        fps: the chunk's frame rate (used for the controller's index
            math via ``elapsed * fps``).
        current_frame_override: if set, replaces ``chunk_np[start_idx]``
            as the chunk's frame 0. Callers use this to inject a clamped
            value (e.g. POLICY JUMP CLAMP rewrite) without disturbing
            the rest of the chunk's content. Defaults to using the raw
            chunk value at start_idx.

    Returns:
        ActionChunk with ``fps=fps`` and frames covering ``[start_idx, N)``
        — i.e. the policy's prediction from "now" out to the chunk tail.
        For non-predictive robots, ``send_action(ActionChunk)`` consumes
        only ``frames[0]`` and ignores the rest, so this is backward-
        compatible with any robot type.
    """
    remaining = chunk_np[start_idx:]
    n_frames = len(remaining)
    if current_frame_override is not None:
        # Build frames[0] from the override, frames[1:] from the rest.
        frames = (
            {name: float(current_frame_override[j]) for j, name in enumerate(joint_names)},
            *(
                {name: float(remaining[i, j]) for j, name in enumerate(joint_names)}
                for i in range(1, n_frames)
            ),
        )
    else:
        frames = tuple(
            {name: float(remaining[i, j]) for j, name in enumerate(joint_names)} for i in range(n_frames)
        )
    return ActionChunk(fps=float(fps), frames=frames)


def _osc_skip(chunk: np.ndarray, idx: int, step_count: int) -> int:
    """Oscillation skip: if all joints have low displacement in the next 5
    frames, scan ahead to where movement starts. Robot-agnostic."""
    if idx >= len(chunk) - 10:
        return idx
    origin = chunk[idx]
    check_at = min(idx + 5, len(chunk) - 1)
    total_disp = np.linalg.norm(chunk[check_at] - origin)
    if total_disp >= 2.0:
        return idx
    for k in range(idx + 5, min(idx + 30, len(chunk))):
        disp = np.linalg.norm(chunk[k] - origin)
        if disp > 3.0:
            if step_count % 50 == 0:
                logger.info("S1 osc-skip: flat [%d]→[%d] (disp=%.1f)", idx, k, disp)
            return max(0, min(k, len(chunk) - 1))
    return idx


def _apply_delta_filter(
    action_np: np.ndarray, prev_action_np: np.ndarray, max_step_delta: float
) -> np.ndarray:
    """Per-joint delta filter: hold joints that jump > max_step_delta."""
    diff = np.abs(action_np - prev_action_np)
    bad_mask = diff > max_step_delta
    if bad_mask.any():
        action_np[bad_mask] = prev_action_np[bad_mask]
    return action_np


def _atomic_torch_save(obj, path) -> None:
    """``torch.save`` with crash-safe atomicity.

    Writes to ``<path>.tmp`` first, then ``os.replace``s it onto the
    final path. Either the new file is fully present or the previous
    one is untouched — never a half-written file. Use this for every
    .pt write to avoid torn checkpoints if the process crashes mid-save.

    On most filesystems ``os.replace`` is atomic for files in the same
    directory, which is the case here (tmp lives in same dir).
    """

    target = str(path)
    tmp = target + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, target)


def _rlt_flush_intervention_terminal(rlt_state, rlt_recorder, rlt_replay, infer_thread, obs) -> None:
    """Flush a terminal transition for an intervention-rescued episode.

    When the operator presses 'r' (success) or LEFT-ARROW (abort) while
    intervention is still active, the inference thread is paused and so
    cannot record the terminal transition. Without this flush the +1
    (or ``cfg.abort_reward``) is silently dropped — the critic sees
    every intervention-rescued episode as a generic r=0 timeout. Paper
    Alg 1 line 12 stores the transition regardless of who was driving.

    Precondition:
        ``rlt_recorder.frames_observed > 0`` (caller checks). At least
        one complete C-frame chunk inside ``flush_terminal`` is required
        to anchor the terminal; if intervention was shorter than C
        frames the recorder logs a warning and no-ops.

    Postcondition:
        On success, one transition is appended to ``rlt_replay`` with
        ``done=True`` and the operator-selected reward, and
        ``rlt_state['total_transitions']`` is incremented by 1.
        Buffer's most recent slot is asserted to reflect the terminal.
        The lifecycle's terminal flag is consumed so the next
        ``begin()`` call doesn't trip the unconsumed-terminal assert.
    """
    lifecycle = rlt_state["lifecycle"]
    # Use consume_terminal_for_storage rather than peek — calling it
    # here marks the terminal as consumed so the next begin() doesn't
    # raise. The intervention path is the analog of inference-thread
    # storage for intervention-rescued episodes; consuming here is
    # semantically correct (the helper is the storage path).
    terminal = lifecycle.consume_terminal_for_storage()
    if terminal is None:
        return
    terminal_reward = float(rlt_state["config"].abort_reward) if terminal == TerminalKind.ABORT else 1.0
    if not rlt_recorder.flush_terminal(
        reward=terminal_reward,
        current_z_rl=infer_thread._rlt_latest_z_rl,
        current_obs=obs,
    ):
        return
    rlt_state["total_transitions"] += 1
    # Cross-module invariant: flush_terminal claimed it wrote a
    # done=True transition with the requested reward. The most
    # recently STAGED entry must reflect that — catches regressions
    # where the add path drops the done flag silently. With the
    # two-stage buffer, the terminal isn't in the committed arrays
    # yet (it'll move there at episode end), so we inspect staging.
    last = rlt_replay.peek_last_pending()
    assert last is not None, "flush_terminal returned True but the buffer has no pending writes"
    assert bool(last["done"]) is True and abs(float(last["reward"]) - terminal_reward) < 1e-6, (
        f"flush_terminal returned True but the last pending transition has "
        f"done={last['done']} reward={last['reward']:.3f} "
        f"(expected done=True, reward={terminal_reward:+.3f})"
    )
    logger.info(
        "RLT: intervention-terminal r=%+.2f flushed (%s)",
        terminal_reward,
        terminal.name,
    )


def _save_infer_drop(
    chunk_np: np.ndarray, obs: dict, infer_count: int, save_dir: str, joint_names: list[str] | None = None
):
    """Detect large jumps in any joint of predicted chunk and save obs for analysis."""
    # Max per-joint jump in first 20 steps
    per_joint_max = np.max(np.abs(np.diff(chunk_np[:20], axis=0)), axis=0)
    max_jump = np.max(per_joint_max)
    if max_jump <= 10:
        return
    worst_joint = int(np.argmax(per_joint_max))
    names = joint_names or JOINT_NAMES
    joint_label = names[worst_joint] if worst_joint < len(names) else f"joint_{worst_joint}"

    drop_dir = os.path.join(save_dir, f"infer_drop_{infer_count}")
    os.makedirs(drop_dir, exist_ok=True)
    for k, v in obs.items():
        if isinstance(v, np.ndarray) and v.ndim == 3:
            safe = k.replace("/", "_").replace(".", "_")
            cv2.imwrite(os.path.join(drop_dir, f"{safe}.jpg"), v[:, :, ::-1])
    state_arr = np.array([float(obs.get(j, 0)) for j in names])
    np.save(os.path.join(drop_dir, "state.npy"), state_arr)
    np.save(os.path.join(drop_dir, "chunk.npy"), chunk_np)
    logger.info(
        "S1 INFER DROP infer#%d | max_jump=%.1f (%s) | saved to %s",
        infer_count,
        max_jump,
        joint_label,
        drop_dir,
    )


def _log_joint_jump(
    action_np: np.ndarray,
    prev_action_np: np.ndarray,
    step_count: int,
    idx: int,
    chunk_data: np.ndarray | None,
    robot_state: np.ndarray | None = None,
    save_dir: str | None = None,
    obs_images: dict | None = None,
    joint_names: list[str] | None = None,
):
    """Log large per-joint jumps with chunk trajectory and optional obs saving."""
    per_joint_delta = np.abs(action_np - prev_action_np)
    max_delta = np.max(per_joint_delta)
    if max_delta <= 10:
        return
    if chunk_data is None:
        return
    names = joint_names or JOINT_NAMES
    worst = int(np.argmax(per_joint_delta))
    joint_label = names[worst] if worst < len(names) else f"joint_{worst}"
    delta = np.linalg.norm(action_np - prev_action_np)

    # Show trajectory of worst joint
    traj = [f"{chunk_data[i, worst]:.0f}" for i in range(min(20, len(chunk_data)))]
    state_str = ""
    if robot_state is not None:
        state_str = "\n  state: " + " ".join(f"{v:6.1f}" for v in robot_state)
    logger.info(
        "S1 JUMP step %d idx=%d | %s: %.1f→%.1f (Δ%.1f) | Δaction=%.1f\n  chunk %s[0:20]: %s%s",
        step_count,
        idx,
        joint_label,
        prev_action_np[worst],
        action_np[worst],
        per_joint_delta[worst],
        delta,
        joint_label,
        " ".join(traj),
        state_str,
    )
    # Save observation snapshot for offline analysis
    if save_dir and obs_images:
        import cv2

        drop_dir = os.path.join(save_dir, f"joint_jump_{step_count}")
        os.makedirs(drop_dir, exist_ok=True)
        for cam_name, img_np in obs_images.items():
            safe_name = cam_name.replace("/", "_").replace(".", "_")
            cv2.imwrite(os.path.join(drop_dir, f"{safe_name}.jpg"), img_np[:, :, ::-1])
        if robot_state is not None:
            np.save(os.path.join(drop_dir, "state.npy"), robot_state)
        if chunk_data is not None:
            np.save(os.path.join(drop_dir, "chunk.npy"), chunk_data)


def obs_to_s1_batch(
    robot_obs: dict,
    s1_image_keys: list[str],
    shared_cache: SharedLatentCache,
    s2_latent_key: str,
    device: torch.device,
    resize_to: tuple[int, int] | None = None,
    joint_names: list[str] | None = None,
) -> dict:
    """Convert robot observation to S1 input batch.

    Images are resized on CPU before GPU transfer (0.6MB vs 10.5MB per image).
    """
    if joint_names is None:
        joint_names = JOINT_NAMES

    batch = {}

    for key in s1_image_keys:
        cam_name = key.split(".")[-1]
        if cam_name in robot_obs:
            # Resize path must match training (FlowMatchingDataset uses
            # torchvision bilinear + antialias=True) — see parity test
            # test_inference_image_resize_matches_training. cv2.resize without
            # antialiasing aliases high-frequency content, producing inputs
            # DINOv2 never saw at train time.
            img = np.asarray(robot_obs[cam_name], dtype=np.uint8)
            img_tensor = (
                torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)
            )  # CPU float [3,H,W] in [0,1]
            if resize_to is not None:
                img_tensor = TF.resize(
                    img_tensor,
                    list(resize_to),
                    interpolation=TF.InterpolationMode.BILINEAR,
                    antialias=True,
                )
            batch[key] = img_tensor.unsqueeze(0).to(device)

    state = [float(robot_obs[name]) for name in joint_names]
    batch["observation.state"] = torch.tensor([state], dtype=torch.float32, device=device)

    if shared_cache is not None:
        latent, age_seconds = shared_cache.read_with_age()
        # Clamp age to training range — model was trained with max_delay_seconds
        # (e.g., 0.15s). Ages beyond that are out-of-distribution.
        max_train_age = 0.15  # must match --max-delay used during training
        age_seconds = min(age_seconds, max_train_age)
        batch[s2_latent_key] = latent.unsqueeze(0).to(device)
        batch["observation.s2_latent_age"] = torch.tensor([[age_seconds]], dtype=torch.float32, device=device)

    return batch


def _warmup_s1(
    policy,
    preprocessor,
    s1_image_keys,
    device,
    resize_to,
    state_dim: int = 14,
    s2_latent_dim: int = 2048,
    use_s2: bool = True,
):
    """Run dummy forward passes to trigger torch.compile kernel compilation.

    Uses fake data — no robot needed. After this, the control loop runs
    at full compiled speed with no compilation stalls.
    """
    import time as _time

    h, w = resize_to if resize_to else (224, 224)
    dummy_batch = {}
    for key in s1_image_keys:
        dummy_batch[key] = torch.randn(1, 3, h, w, device=device)
    dummy_batch["observation.state"] = torch.zeros(1, state_dim, device=device)
    if use_s2:
        dummy_batch["observation.s2_latent"] = torch.zeros(1, s2_latent_dim, device=device)
        dummy_batch["observation.s2_latent_age"] = torch.zeros(1, 1, device=device)

    dummy_batch = preprocessor(dummy_batch)

    t0 = _time.perf_counter()
    with torch.no_grad():
        for i in range(3):
            if hasattr(policy, "select_action"):
                policy.select_action(dummy_batch)
            else:
                policy.predict_action_chunk(dummy_batch)
            if i == 0:
                logger.info("S1: First compiled forward done (%.1fs)", _time.perf_counter() - t0)
    policy.reset()  # clear any state from warmup
    logger.info("S1: Warmup complete (%.1fs total)", _time.perf_counter() - t0)


def _create_or_resume_dataset(repo_id: str, fps: int, features: dict, robot_type: str):
    """Wrapper over LeRobotDataset.create that resumes if dir exists.

    Safety: when the directory exists, resume must succeed. If resume fails
    (e.g. corrupted metadata), this raises rather than silently rmtree-ing the
    user's data. The caller can manually delete the directory after backing up.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME

    dataset_root = HF_LEROBOT_HOME / repo_id
    if dataset_root.exists():
        # Pass explicit root so we don't trigger a Hub probe (which can 404
        # for local-only datasets and was the root cause of the May 2026 incident).
        return LeRobotDataset.resume(repo_id, root=dataset_root)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        robot_type=robot_type,
        use_videos=True,
    )


def _create_recording_dataset(repo_id: str, fps: int, robot, task: str):
    """Create a LeRobotDataset for recording inference episodes.

    Features are derived from the robot's action/observation specs.
    """

    obs_ft = robot.observation_features
    action_ft = robot.action_features

    # Joint names = non-camera observation features
    joint_names = [k for k, v in action_ft.items() if not isinstance(v, tuple)]
    cam_features = {k: v for k, v in obs_ft.items() if isinstance(v, tuple)}

    features = {}
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (len(joint_names),),
        "names": list(joint_names),
    }
    for cam_name, shape in cam_features.items():
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "channels"],
        }
    features["action"] = {
        "dtype": "float32",
        "shape": (len(joint_names),),
        "names": list(joint_names),
    }

    dataset = _create_or_resume_dataset(
        repo_id=repo_id,
        fps=fps,
        features=features,
        robot_type=robot.robot_type,
    )
    logger.info(
        "S1: Recording dataset '%s' (%d joints, %d cameras, %d existing episodes)",
        repo_id,
        len(joint_names),
        len(cam_features),
        dataset.meta.total_episodes,
    )
    return dataset


def _add_frame_to_dataset(dataset, obs: dict, action_np: np.ndarray, joint_names: list[str], task: str):
    """Add a single frame (obs + action) to the recording dataset."""
    frame = {"task": task}
    frame["observation.state"] = np.array(
        [float(obs.get(j, 0)) for j in joint_names],
        dtype=np.float32,
    )
    frame["action"] = action_np.astype(np.float32)
    for k, v in obs.items():
        if isinstance(v, np.ndarray) and v.ndim == 3:
            frame[f"observation.images.{k}"] = v
    dataset.add_frame(frame)


def inference_target_fps_for(fps: int | float, query_interval_steps: int) -> float | None:
    """Effective inference-thread FPS, given the control loop's fps and
    the query interval.

    The inference thread runs once per ``query_interval_steps`` control
    frames (the main loop executes that many actions per inferred chunk
    before re-querying). So its rate is the control rate divided by the
    query interval:

        inference_fps = fps / query_interval_steps

    When ``query_interval_steps == 0`` the thread runs as fast as it can
    — there's no meaningful FPS target, so return ``None`` and let the
    latency-monitoring stack skip the overrun / budget rules.

    Sharing this helper with the test suite (rather than inlining the
    formula at the call site) makes the relationship explicit and
    catchable if a future refactor changes the cadence.
    """
    if query_interval_steps <= 0:
        return None
    return float(fps) / query_interval_steps


def run_s1(
    s1_checkpoint: str,
    shared_cache: SharedLatentCache,
    shared_images: SharedImageBuffer,
    task: str,
    robot_config_path: str | None = None,
    fps: int = 30,
    device: str = "cuda",
    resize_images: tuple[int, int] | None = (224, 224),
    temporal_ensemble_coeff: float | None = None,
    n_action_steps: int | None = None,
    compile_s1: bool = False,
    s1_type: str = "act",
    stop_event=None,
    osc_skip: bool = False,
    query_interval_steps: int = 0,
    num_denoise_steps: int | None = None,
    max_step_delta: float | None = None,
    grip_drop_save_dir: str | None = None,
    record_dataset: str | None = None,
    num_episodes: int = 1,
    episode_time_s: float = 0,
    reset_time_s: float = 20,
    teleop_config_path: str | None = None,
    intervention_dataset: str | None = None,
    # RLT parameters
    rlt_mode: bool = False,
    rl_token_checkpoint: str | None = None,
    rlt_checkpoint: str | None = None,
    rlt_deploy: bool = False,
    rl_chunk_length: int = 10,
    rlt_output_dir: str = "outputs/rlt_online",
    rlt_start_engaged: bool = True,
    rlt_shared_noise_per_chunk: bool = False,
    # Latency monitoring (mirrors --latency_monitor / --latency_output_dir
    # from teleop/record). Snapshot is published to <output_dir>/latency_snapshot.json
    # so the GUI dashboard can render HVLA inference timings alongside teleop/record.
    latency_monitor: bool = False,
    latency_output_dir: str | None = None,
    # Send shape on the policy path. "chunk" (default) packs the
    # remaining chunk frames into an ActionChunk so the predictive
    # robot uses its exact-lookup path. "dict" sends only the single
    # frame at idx — primarily for A/B comparison against the pre-fix
    # behaviour. Non-predictive robots get frames[0] either way, so
    # this toggle is a no-op on them.
    send_action_shape: str = "chunk",
):
    """S1 control loop with robot. Runs in main process."""
    # Main process logging should already be configured by launch.py,
    # but ensure it works even if run standalone.
    from lerobot.policies.hvla.logging_utils import setup_process_logging

    if not logging.getLogger().handlers:
        setup_process_logging()

    # Add a per-run FileHandler at the TOP of the function so the file log
    # captures the full startup (policy load, dataset preload, flash-DAgger
    # pre-encoding, robot/camera connect). Previously this handler was added
    # later, after all of those steps — so a crash during startup left no
    # file-log evidence and post-hoc debugging required the GUI SSE buffer.
    import datetime

    run_log_dir = Path("outputs/hvla_runs")
    run_log_dir.mkdir(parents=True, exist_ok=True)
    run_log_file = run_log_dir / f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    _run_fh = logging.FileHandler(str(run_log_file), mode="w")
    _run_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(_run_fh)
    logger.info("S1: Run log → %s", run_log_file)

    # keyboard listener removed — Ctrl+C via stop_event is sufficient
    # Register robot/camera types for draccus. Auto-discover every
    # submodule under lerobot.robots / lerobot.teleoperators so any new
    # robot type (e.g. bi_so107_follower_predictive) is registered
    # without an explicit import edit here. Each import is wrapped in
    # contextlib.suppress(Exception) so packages whose optional SDK
    # extras aren't installed stay silent. Same pattern as
    # gui/api/robot.py::_ensure_configs_loaded.
    import importlib
    import json
    import pkgutil

    import draccus

    # Side-effect import: registers the typing.Literal decoder against
    # draccus. Without this, any config field annotated Literal[...]
    # (e.g. camera_read_strategy on bi_so107) fails to decode here even
    # though the GUI path works (which imports parser transitively).
    import lerobot.configs.parser  # noqa: F401
    import lerobot.robots as _lr_robots
    import lerobot.teleoperators as _lr_teleops
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
    from lerobot.policies.hvla.s1.protocol import S2_LATENT_KEY
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.config import RobotConfig

    for _pkg in (_lr_robots, _lr_teleops):
        for _importer, _modname, _ispkg in pkgutil.walk_packages(_pkg.__path__, prefix=_pkg.__name__ + "."):
            with contextlib.suppress(Exception):
                importlib.import_module(_modname)

    device = torch.device(device)

    # Load S1 policy based on type
    logger.info("S1: Loading %s policy from %s", s1_type, s1_checkpoint)

    if s1_type == "flow":
        from lerobot.policies.hvla.s1.flow_matching import FlowMatchingS1Config, FlowMatchingS1Policy

        config = FlowMatchingS1Config()
        policy = FlowMatchingS1Policy.from_pretrained(s1_checkpoint, config=config)

        def preprocessor(batch):
            return batch  # flow matching handles its own normalization

        def postprocessor(actions):
            return actions

        s1_image_keys = list(config.image_features.keys())
    else:
        from lerobot.policies.act_vlm.modeling_act_vlm import ACTWithVLMPolicy
        from lerobot.policies.factory import make_pre_post_processors

        policy = ACTWithVLMPolicy.from_pretrained(pretrained_name_or_path=s1_checkpoint)

        if temporal_ensemble_coeff is not None:
            policy.config.temporal_ensemble_coeff = temporal_ensemble_coeff
            policy.config.n_action_steps = 1
            from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

            policy.temporal_ensembler = ACTTemporalEnsembler(
                temporal_ensemble_coeff, policy.config.chunk_size
            )
            logger.info("S1: Temporal ensembling (coeff=%.3f)", temporal_ensemble_coeff)
        elif n_action_steps is not None:
            policy.config.n_action_steps = n_action_steps
            logger.info("S1: n_action_steps=%d", n_action_steps)

        s1_image_keys = list(policy.config.image_features.keys())
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=s1_checkpoint,
        )

    policy.to(device)
    policy.eval()
    policy.reset()

    # Enable TF32 matmul for faster float32 ops on Ampere+ GPUs
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    logger.info("S1: Policy loaded (%s). Image keys: %s", s1_type, s1_image_keys)

    # torch.compile is opt-in via --compile-s1 flag.
    # DINOv2 attention layers are not compatible with torch.compile out of the box.
    # TODO: investigate torch.compile with fullgraph=False or compiling only the
    # ACT encoder/decoder (excluding DINOv2 backbone).
    if compile_s1:
        logger.info("S1: Compiling denoise_step with torch.compile...")
        inner = policy.model if hasattr(policy, "model") else policy
        inner.denoise_step = torch.compile(inner.denoise_step, mode="default")
        logger.info("S1: Warming up compiled model...")
        _warmup_s1(
            policy,
            preprocessor,
            s1_image_keys,
            device,
            resize_images,
            state_dim=config.action_dim if s1_type == "flow" else 14,
            s2_latent_dim=config.s2_latent_dim if s1_type == "flow" else 2048,
            use_s2=shared_cache is not None,
        )

    # Load robot
    config_path = robot_config_path or str(Path.home() / ".config" / "lerobot" / "robots" / "white.json")
    logger.info("S1: Loading robot from %s", config_path)
    with open(config_path) as f:
        profile = json.load(f)
    # Support both formats:
    # - LeRobot profile: {type, name, fields: {ports...}, cameras, ...}
    # - Flat config: {type, left_arm_port, ..., cameras, ...}
    if "fields" in profile:
        config_dict = {"type": profile["type"]}
        for k, v in profile["fields"].items():
            config_dict[k] = v
        if "cameras" in profile:
            config_dict["cameras"] = profile["cameras"]
    else:
        config_dict = profile
    robot_cfg = draccus.decode(RobotConfig, config_dict)
    robot = make_robot_from_config(robot_cfg)
    logger.info("S1: Connecting to robot...")
    robot.connect()
    logger.info("S1: Robot connected")

    # Derive joint names and camera keys from the robot (robot-agnostic)
    joint_names = _joint_names_from_robot(robot)
    camera_keys = _camera_keys_from_robot(robot)
    action_dim = len(joint_names)
    logger.info("S1: Robot joints (%d): %s", action_dim, joint_names)
    logger.info("S1: Robot cameras: %s", camera_keys)

    # Apply observation processor steps (e.g., depth edge overlay for RealSense)
    obs_processor_steps = (
        robot.get_observation_processor_steps() if hasattr(robot, "get_observation_processor_steps") else []
    )

    # Append obs stream writer as the last step (writes processed obs to shared memory for GUI)
    from lerobot.robots.obs_stream import make_obs_stream_writer_step

    obs_stream_step = make_obs_stream_writer_step()
    if obs_stream_step is not None:
        obs_processor_steps.append(obs_stream_step)

    if obs_processor_steps:
        logger.info("S1: Observation processors: %s", [type(s).__name__ for s in obs_processor_steps])

    # Episode recording
    dataset = None
    if record_dataset:
        dataset = _create_recording_dataset(record_dataset, fps, robot, task)

    # Intervention recording dataset
    int_dataset = None
    if intervention_dataset:
        int_dataset = _create_recording_dataset(intervention_dataset, fps, robot, task)
        logger.info("S1: Intervention dataset '%s' created", intervention_dataset)

    # Teleop (leader arm) for intervention / inverse follow
    teleop = None
    if teleop_config_path:
        logger.info("S1: Loading teleop from %s", teleop_config_path)
        with open(teleop_config_path) as f:
            teleop_profile = json.load(f)
        # Support LeRobot profile format: {type, fields: {...}}
        if "fields" in teleop_profile:
            teleop_config_dict = {"type": teleop_profile["type"]}
            for k, v in teleop_profile["fields"].items():
                teleop_config_dict[k] = v
        else:
            teleop_config_dict = teleop_profile
        # Force intervention_enabled on
        teleop_config_dict["intervention_enabled"] = True
        from lerobot.teleoperators import TeleoperatorConfig
        from lerobot.teleoperators.utils import make_teleoperator_from_config

        teleop_cfg = draccus.decode(TeleoperatorConfig, teleop_config_dict)
        teleop = make_teleoperator_from_config(teleop_cfg)
        teleop.connect()
        logger.info("S1: Teleop connected (intervention enabled, press SPACE to toggle)")

    # Note: S2 wait happens after inference thread starts (below),
    # because the inference thread publishes images that S2 needs.

    # (Per-run FileHandler installed at top of run_s1 — see line ~497.
    # Removed from this location so the file log captures full startup.)

    # Route uncaught exceptions through the logger so their tracebacks
    # land in the persistent run_*.log alongside everything else.
    # Python's default uncaught-exception handler writes to sys.stderr,
    # which is captured by the GUI subprocess pipe (visible live in the
    # output panel) but does NOT flow through the FileHandler installed
    # above. Without this, post-hoc debugging on a crash from yesterday
    # finds the log truncated right before the actual failure.
    import sys
    import threading

    def _log_uncaught(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logger.error("Uncaught exception in main thread", exc_info=(exc_type, exc_value, exc_tb))

    def _log_thread_uncaught(args):
        if issubclass(args.exc_type, SystemExit):
            return
        logger.error(
            "Uncaught exception in thread %s",
            args.thread.name if args.thread else "<unknown>",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = _log_uncaught
    threading.excepthook = _log_thread_uncaught

    logger.info("S1: Starting control loop at %d FPS", fps)

    # --- RLT setup ---
    rl_token_encoder = None
    rlt_agent = None
    rlt_replay = None
    rlt_state = None

    if rlt_mode:
        from lerobot.policies.hvla.rlt.actor_critic import TD3Agent
        from lerobot.policies.hvla.rlt.config import RLTConfig
        from lerobot.policies.hvla.rlt.replay_buffer import TransactionalReplayBuffer
        from lerobot.policies.hvla.rlt.token import RLTokenEncoder, load_rlt_token_config

        rlt_config = RLTConfig(
            rl_token_dim=policy.config.hidden_dim,
            rl_chunk_length=rl_chunk_length,
            shared_noise_per_chunk=rlt_shared_noise_per_chunk,
        )

        # Resolve RL token encoder checkpoint:
        # 1. Explicit --rl-token-checkpoint (required by the GUI/API)
        # 2. Auto-discovered from training_state.pt of the RLT checkpoint
        #    (legacy escape hatch — only fires when the user passes an RLT
        #    checkpoint via CLI without --rl-token-checkpoint)
        # 3. Hard error — refuse to silently build a random encoder. The
        #    actor's input dim depends on the encoder's rl_token_dim, so
        #    a missing token checkpoint would let the actor build at the
        #    default 768 dim and then crash with a state_dict size mismatch
        #    during actor.pt load. Fail fast with an actionable message.
        resolved_token_ckpt = rl_token_checkpoint
        if not resolved_token_ckpt and rlt_checkpoint:
            # Mirror the load_dir resolution below: if rlt_checkpoint is a
            # run dir (no actor.pt at the top), look inside latest/.
            ts_search_dir = Path(rlt_checkpoint)
            if (
                not (ts_search_dir / "training_state.pt").exists()
                and (ts_search_dir / "latest" / "training_state.pt").exists()
            ):
                ts_search_dir = ts_search_dir / "latest"
            ts_path = ts_search_dir / "training_state.pt"
            if ts_path.exists():
                try:
                    ts = torch.load(str(ts_path), weights_only=False, map_location=device)
                    resolved_token_ckpt = ts.get("rlt_token_checkpoint")
                    if resolved_token_ckpt:
                        logger.info(
                            "RLT: Auto-discovered token encoder from checkpoint: %s", resolved_token_ckpt
                        )
                except Exception as e:
                    logger.warning("RLT: Failed to read token path from training state: %s", e)
        if not resolved_token_ckpt:
            raise RuntimeError(
                "RLT mode requires an RL Token Encoder checkpoint. Pass "
                "--rl-token-checkpoint=<path-to-encoder-dir> on the CLI "
                "(or fill the 'RL Token Encoder' field in the GUI). "
                "Example: outputs/rlt_token_v4_4layer_d2048/checkpoint-10000. "
                "Without it the encoder would be random and the actor would "
                "be built at the default rl_token_dim=768, crashing on "
                "state_dict load whenever the saved actor uses a different "
                "(e.g. widened) dim."
            )

        # Apply the trained checkpoint's architecture manifest BEFORE
        # instantiating the encoder — state_dict load fails otherwise
        # when the checkpoint was trained with a non-default arch (e.g.
        # widened d=2048). ``load_rlt_token_config`` raises if no
        # config.json is present (the deprecated 2L d=768 family).
        ckpt_path = Path(resolved_token_ckpt)
        ckpt_dir = ckpt_path if ckpt_path.is_dir() else ckpt_path.parent
        rlt_config = load_rlt_token_config(ckpt_dir, base=rlt_config)

        rl_token_encoder = RLTokenEncoder(rlt_config).to(device)

        enc_file = ckpt_path / "encoder.pt" if ckpt_path.is_dir() else ckpt_path
        rl_token_encoder.load_state_dict(torch.load(str(enc_file), weights_only=True, map_location=device))
        logger.info(
            "S1 RLT: Loaded RL token encoder from %s (enc_layers=%d dec_layers=%d)",
            enc_file,
            rlt_config.token_encoder_layers,
            rlt_config.token_decoder_layers,
        )
        rl_token_encoder.eval()
        for p in rl_token_encoder.parameters():
            p.requires_grad = False

        action_dim = policy.config.action_dim
        state_dim = policy.config.state_dim
        rlt_agent = TD3Agent(rlt_config, state_dim, action_dim, device)

        # Deploy mode: actor only, no critic/replay/training
        if rlt_deploy:
            assert rlt_checkpoint, "--rlt-deploy requires --rlt-checkpoint (which actor to deploy?)"
            rlt_replay = None
        else:
            rlt_replay = TransactionalReplayBuffer(
                rlt_config.replay_capacity,
                rlt_config.rl_token_dim,
                state_dim,
                action_dim,
                rl_chunk_length,
                device,
            )

        rlt_state = {
            "config": rlt_config,
            # 0-indexed episode counter. -1 means "no episode has started yet";
            # incremented at the top of each episode so the first runs as ep0.
            # Stored value on disk = index of last completed episode.
            "episode": -1,
            "total_updates": 0,
            "total_transitions": 0,
            "successes": [],
            # Per-episode operator-event tracking. Replaces the prior
            # racy ``reward_triggered`` / ``abort_triggered`` /
            # ``ignore_triggered`` / ``_episode_had_intervention`` /
            # ``buffer_size_at_episode_start`` flags. The bug at ep124
            # (17 done=True transitions written instead of 1) was the
            # racy flags being re-read by the inference thread on every
            # cycle; EpisodeLifecycle's one-shot
            # ``consume_terminal_for_storage`` makes that
            # structurally impossible. See rlt/episode.py.
            "lifecycle": EpisodeLifecycle(),
            "output_dir": Path(rlt_output_dir),
            "deploy": rlt_deploy,
        }
        rlt_state["output_dir"].mkdir(parents=True, exist_ok=True)

        # Log file
        rlt_log_file = rlt_state["output_dir"] / ("deploy.log" if rlt_deploy else "train.log")
        _rlt_fh = logging.FileHandler(str(rlt_log_file), mode="a")
        _rlt_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logging.getLogger().addHandler(_rlt_fh)
        logger.info("RLT: Logging to %s", rlt_log_file)

        # Metrics file for GUI dashboard
        from lerobot.policies.hvla.rlt.metrics import set_metrics_path

        set_metrics_path(str(rlt_state["output_dir"] / "metrics.json"))

        # Load checkpoint: explicit path via --rlt-checkpoint only. We used to
        # auto-resume when the output_dir had a latest/ subdir, but that
        # silently clobbered prior metrics on a "fresh" launch and hid the
        # fact that it was really a resume. Now: if the user wants to resume
        # they must pass --rlt-checkpoint explicitly; otherwise a populated
        # output_dir is an error.
        load_dir = None
        if rlt_checkpoint:
            load_dir = Path(rlt_checkpoint)
            # If this is a run dir (not a checkpoint dir), look for latest/ inside
            if not (load_dir / "actor.pt").exists() and (load_dir / "latest" / "actor.pt").exists():
                load_dir = load_dir / "latest"
        else:
            latest_dir = rlt_state["output_dir"] / "latest"
            if (latest_dir / "actor.pt").exists():
                raise RuntimeError(
                    f"RLT output_dir already contains a trained checkpoint at "
                    f"{latest_dir}. Refusing to start a fresh session here — "
                    f"that would overwrite prior training. To resume, pass "
                    f"--rlt-checkpoint={rlt_state['output_dir']} (or select "
                    f"the existing checkpoint in the GUI). To start a new "
                    f"training run, pass a different --rlt-output-dir."
                )

        if load_dir and (load_dir / "actor.pt").exists():
            logger.info("RLT: Loading checkpoint from %s", load_dir)
            rlt_agent.actor.load_state_dict(
                torch.load(str(load_dir / "actor.pt"), weights_only=True, map_location=device)
            )
            if not rlt_deploy:
                # Training mode: load critic, optimizer, replay buffer
                if (load_dir / "critic.pt").exists():
                    rlt_agent.critic.load_state_dict(
                        torch.load(str(load_dir / "critic.pt"), weights_only=True, map_location=device)
                    )
                else:
                    logger.warning("RLT: critic.pt not found, using random init")
                if (load_dir / "critic_target.pt").exists():
                    rlt_agent.critic_target.load_state_dict(
                        torch.load(str(load_dir / "critic_target.pt"), weights_only=True, map_location=device)
                    )
                else:
                    logger.warning("RLT: critic_target.pt not found, using random init")
                if (load_dir / "training_state.pt").exists():
                    ts = torch.load(
                        str(load_dir / "training_state.pt"), weights_only=True, map_location=device
                    )
                    rlt_agent.actor_opt.load_state_dict(ts["actor_opt"])
                    rlt_agent.critic_opt.load_state_dict(ts["critic_opt"])
                    rlt_state["episode"] = ts["episode"]
                    rlt_state["total_transitions"] = ts["total_transitions"]
                    rlt_state["total_updates"] = ts["total_updates"]
                    rlt_state["successes"] = ts["successes"]
                    logger.info(
                        "RLT: Loaded training state (ep=%d, updates=%d)",
                        rlt_state["episode"],
                        rlt_state["total_updates"],
                    )
                else:
                    logger.warning("RLT: training_state.pt not found, starting from ep=0")
                if rlt_replay is not None and (load_dir / "replay_buffer.pt").exists():
                    rlt_replay.load(str(load_dir / "replay_buffer.pt"))
                    logger.info("RLT: Loaded replay buffer (%d transitions)", len(rlt_replay))
                else:
                    logger.warning(
                        "RLT: replay buffer not loaded (replay=%s, file exists=%s)",
                        rlt_replay is not None,
                        (load_dir / "replay_buffer.pt").exists(),
                    )

            # Restore metrics. The aggregator's ``restore`` deserializes
            # each group (episodes / inferences / grad_updates) atomically
            # and logs the loaded shapes. If the file is in the legacy
            # flat-series format (saved before the 3-group refactor),
            # convert it here so episode-level history isn't lost.
            import json

            metrics_path = str(rlt_state["output_dir"] / "metrics.json")
            if os.path.exists(metrics_path):
                try:
                    from lerobot.policies.hvla.rlt.metrics import get_metrics

                    with open(metrics_path) as f:
                        saved = json.load(f)
                    series = saved.get("series", {})
                    if "episodes" not in series and "episode_successes" in series:
                        # Legacy flat format. Promote per-episode keys into
                        # the new ``episodes`` group; per-step series are
                        # incompatible (length-divergent across siblings)
                        # and would corrupt the grad_updates invariant if
                        # round-tripped — drop them with a warning so the
                        # operator knows chart history starts fresh.
                        series = {
                            **series,
                            "episodes": {
                                "successes": series.get("episode_successes", []),
                                "autonomous": series.get("episode_autonomous", []),
                                "timestamps": series.get("episode_timestamps", []),
                                "lengths_s": series.get("episode_lengths_s", []),
                            },
                        }
                        logger.warning(
                            "RLT: legacy metrics format — restored episode "
                            "history; per-step training series (Q values, "
                            "critic loss, actor delta) start fresh.",
                        )
                        saved = {**saved, "series": series}
                    get_metrics().restore(saved)
                except Exception as e:
                    logger.warning("RLT: Failed to restore metrics: %s", e)

            mode_str = "DEPLOY" if rlt_deploy else "TRAIN"
            logger.info(
                "RLT: %s mode — loaded from %s (ep=%d, updates=%d)",
                mode_str,
                load_dir,
                rlt_state["episode"],
                rlt_state["total_updates"],
            )
        else:
            logger.info(
                "S1 RLT: Online RL enabled — C=%d, UTD=%d, beta=%.2f, expl_sigma=%.3f, "
                "target_sigma=%.3f, shared_noise_per_chunk=%s",
                rl_chunk_length,
                rlt_config.utd_ratio,
                rlt_config.beta,
                rlt_config.exploration_sigma,
                rlt_config.target_sigma,
                rlt_config.shared_noise_per_chunk,
            )

    # --- Pipelined inference thread ---
    from lerobot.policies.hvla.s1_inference import InferenceThread
    from lerobot.utils.latency import LatencySession

    # HVLA writes one snapshot per thread under a shared parent dir so the
    # GUI dashboard can render them as stacked tracks of the same process.
    # Subdirs (not different filenames) keep the snapshot writer simple
    # and match LATENCY_SOURCES["hvla_main"] / ["hvla_infer"] in run.py.
    if latency_monitor and latency_output_dir is not None:
        main_dir: str | None = os.path.join(latency_output_dir, "main")
        inference_dir: str | None = os.path.join(latency_output_dir, "inference")
    else:
        main_dir = inference_dir = None

    main_session = LatencySession.from_config(
        enabled=latency_monitor,
        loop_kind="hvla_main",
        process="hvla",
        track="main",
        target_fps=float(fps),
        output_dir=main_dir,
    )
    # The inference thread runs at its OWN cadence — once per
    # ``query_interval_steps`` control frames, NOT once per control frame.
    # See ``inference_target_fps_for`` for the derivation. Using the
    # control loop's 1/fps as the inference budget would falsely flag the
    # thread as "overrunning" whenever a single inference takes longer
    # than one control period, even though that's the expected design.
    inference_session = LatencySession.from_config(
        enabled=latency_monitor,
        loop_kind="hvla_infer",
        process="hvla",
        track="inference",
        target_fps=inference_target_fps_for(fps, query_interval_steps),
        output_dir=inference_dir,
    )

    infer_thread = InferenceThread(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        shared_cache=shared_cache,
        s2_latent_key=S2_LATENT_KEY,
        s1_image_keys=s1_image_keys,
        joint_names=joint_names,
        device=device,
        resize_to=resize_images,
        fps=fps,
        num_denoise_steps=num_denoise_steps,
        query_interval_steps=query_interval_steps,
        grip_drop_save_dir=grip_drop_save_dir,
        rl_token_encoder=rl_token_encoder,
        rlt_actor=rlt_agent.actor if rlt_agent else None,
        rlt_agent=rlt_agent,
        rlt_state=rlt_state,
        rlt_replay=rlt_replay,
        latency_session=inference_session,
    )

    # Instantiate the intervention recorder once per process. It owns the
    # human-action → replay-buffer pipeline and raises on the first frame
    # where z_rl can't be sourced from the inference thread.
    if rlt_mode and rlt_replay is not None:
        from lerobot.policies.hvla.rlt.intervention import InterventionRecorder

        rlt_recorder = InterventionRecorder(
            replay=rlt_replay,
            policy=policy,
            device=device,
            chunk_length=rl_chunk_length,
            joint_names=joint_names,
        )
    else:
        rlt_recorder = None

    _supports_rtc = getattr(policy, "supports_rtc", False)
    _needs_ensemble = getattr(policy, "needs_temporal_ensemble", True)
    if _supports_rtc:
        logger.info(
            "S1: RTC enabled (max_delay=%d, dynamic prefix from prev chunk)",
            getattr(policy, "rtc_prefix_length", 5),
        )
    elif _needs_ensemble:
        logger.info("S1: Using temporal ensembling (no RTC)")
    if query_interval_steps > 0:
        logger.info(
            "S1: Query interval = %d steps (%.0fms)", query_interval_steps, query_interval_steps / fps * 1000
        )

    infer_thread.start()

    # Smoothness tracking
    prev_action_np = None
    action_deltas = []
    loop_intervals = []
    last_send_time = None

    # Capture initial obs, publish to both S2 and inference thread
    logger.info("S1: Capturing initial observation...")
    init_obs = robot.get_observation()
    if shared_images is not None:
        shared_images.write_images(init_obs, S2_CAM_KEY_MAP, joint_names)
    infer_thread.publish_obs(init_obs, time.perf_counter())

    # Wait for S2 (skip if running without S2)
    if shared_cache is not None:
        logger.info("S1: Waiting for first S2 latent (up to 120s)...")
        if not shared_cache.wait_for_first(timeout=120.0):
            logger.warning("S1: No S2 latent after 120s, starting with zero latent")
        else:
            logger.info("S1: Got first S2 latent (count=%d)", shared_cache.count)
    else:
        logger.info("S1: Running without S2 conditioning")

    # Wait for first chunk
    logger.info("S1: Waiting for first action chunk...")
    if not infer_thread.wait_for_first_chunk(timeout=60.0):
        logger.error("S1: No action chunk in 60s, aborting")
        infer_thread.stop()
        return

    logger.info("S1: First chunk ready, starting at %d FPS", fps)

    # Multi-episode rollout support
    # RLT always needs multi-episode mode
    multi_episode = num_episodes > 1 or episode_time_s > 0 or rlt_mode
    if multi_episode:
        from lerobot.common.control_utils import init_keyboard_listener

        listener, events = init_keyboard_listener()
        logger.info(
            "S1: Rollout mode — %d episodes, %.0fs/episode, %.0fs reset",
            num_episodes,
            episode_time_s,
            reset_time_s,
        )
        logger.info("S1: Press RIGHT ARROW to advance, ESC to stop")
    else:
        events = {"exit_early": False, "stop_recording": False}
        listener = None

    # RLT: hook R key into existing keyboard listener for reward signal
    if rlt_mode and rlt_state is not None and listener is not None:
        _orig_on_press = listener.on_press

        def _rlt_on_press(key, *args):
            try:
                if hasattr(key, "char") and key.char == "r":
                    rlt_state["lifecycle"].signal_terminal(TerminalKind.SUCCESS)
                    events["exit_early"] = True
                    logger.info("RLT: SUCCESS — reward +1, ending episode")
                    # Play success sound
                    try:
                        import subprocess

                        import numpy as _np

                        sr = 16000
                        t = _np.linspace(0, 0.3, int(sr * 0.3), dtype=_np.float32)
                        tone = (_np.sin(2 * _np.pi * 800 * t) * 20000).astype(_np.int16)
                        tone2 = (_np.sin(2 * _np.pi * 1200 * t[: len(t) // 2]) * 20000).astype(_np.int16)
                        sound = _np.concatenate([tone, tone2])
                        proc = subprocess.Popen(
                            ["aplay", "-f", "S16_LE", "-r", str(sr), "-c", "1", "-q"],
                            stdin=subprocess.PIPE,
                            stderr=subprocess.DEVNULL,
                        )
                        proc.stdin.write(sound.tobytes())
                        proc.stdin.close()
                    except Exception:
                        pass
                    return
                # LEFT ARROW = abort/disaster in RLT mode. Mirrors LeRobot's
                # convention (right = next, left = retry/reset) but with
                # RL-specific semantics: episode ends, terminal reward is
                # cfg.abort_reward (default -1.0). Intentionally short-
                # circuits before the original handler so we DON'T also set
                # rerecord_episode (which would discard the trajectory from
                # the dataset; we want it stored as a negative-reward
                # transition, not dropped).
                from pynput import keyboard as _kb

                if key == _kb.Key.left:
                    rlt_state["lifecycle"].signal_terminal(TerminalKind.ABORT)
                    events["exit_early"] = True
                    logger.info(
                        "RLT: ABORT — terminal reward %.2f, ending episode",
                        rlt_state["config"].abort_reward,
                    )
                    # Distinct sound: descending tone (failure cue).
                    try:
                        import subprocess

                        import numpy as _np

                        sr = 16000
                        t = _np.linspace(0, 0.3, int(sr * 0.3), dtype=_np.float32)
                        tone = (_np.sin(2 * _np.pi * 600 * t) * 20000).astype(_np.int16)
                        tone2 = (_np.sin(2 * _np.pi * 300 * t[: len(t) // 2]) * 20000).astype(_np.int16)
                        sound = _np.concatenate([tone, tone2])
                        proc = subprocess.Popen(
                            ["aplay", "-f", "S16_LE", "-r", str(sr), "-c", "1", "-q"],
                            stdin=subprocess.PIPE,
                            stderr=subprocess.DEVNULL,
                        )
                        proc.stdin.write(sound.tobytes())
                        proc.stdin.close()
                    except Exception:
                        pass
                    return
                if hasattr(key, "char") and key.char == "e":
                    from lerobot.utils.utils import log_say

                    engaged = not infer_thread._rlt_user_engaged
                    infer_thread._rlt_user_engaged = engaged
                    label = "Policy + RL" if engaged else "Policy"
                    logger.info("RLT: RL actor %s (E key)", "ENGAGED" if engaged else "DISENGAGED")
                    log_say(label)
                    color = "#4fc3f7" if engaged else "#2ecc71"
                    print(f"##OVERLAY:{label}:{color}##", flush=True)
                    return
                # DOWN ARROW = "ignore current episode". For OOD scenes
                # (camera glitch, accidental table bump, anything where
                # neither success nor abort is a valid label) the
                # operator marks this episode as never-happened. All
                # transitions added during the episode are truncated
                # from the replay buffer at episode end; the episode
                # counter and metrics are not advanced. Has priority
                # over abort/success — pressing DOWN after R or LEFT
                # discards their effect too.
                if key == _kb.Key.down:
                    rlt_state["lifecycle"].signal_ignore()
                    events["exit_early"] = True
                    # Mark the dataset for rerecord so the LeRobot
                    # framework's existing dataset-rollback path
                    # discards the recorded frames too.
                    events["rerecord_episode"] = True
                    logger.info(
                        "RLT: IGNORE — episode ep%d will be discarded",
                        rlt_state["episode"],
                    )
                    # Neutral two-beep cue (distinguishable from success
                    # ascending and abort descending).
                    try:
                        import subprocess

                        import numpy as _np

                        sr = 16000
                        t = _np.linspace(0, 0.15, int(sr * 0.15), dtype=_np.float32)
                        tone = (_np.sin(2 * _np.pi * 500 * t) * 18000).astype(_np.int16)
                        gap = _np.zeros(int(sr * 0.05), dtype=_np.int16)
                        sound = _np.concatenate([tone, gap, tone])
                        proc = subprocess.Popen(
                            ["aplay", "-f", "S16_LE", "-r", str(sr), "-c", "1", "-q"],
                            stdin=subprocess.PIPE,
                            stderr=subprocess.DEVNULL,
                        )
                        proc.stdin.write(sound.tobytes())
                        proc.stdin.close()
                    except Exception:
                        pass
                    return
            except AttributeError:
                pass
            if _orig_on_press is not None:
                _orig_on_press(key)

        listener.on_press = _rlt_on_press
        logger.info("RLT: Press 'r' = success (+1 reward, ends episode)")
        logger.info(
            "RLT: Press LEFT ARROW = abort/disaster (%.2f reward, ends episode)",
            rlt_state["config"].abort_reward,
        )
        logger.info("RLT: Press DOWN ARROW = ignore episode (discard transitions, no count)")
        logger.info("RLT: Press 'e' = toggle RL actor on/off")

    recorded_episodes = 0

    # Tracks whether the main loop ended cleanly (normal exit or Ctrl+C).
    # An unhandled exception leaves this False and the finally block skips
    # the final RLT checkpoint save — partial / mid-update state shouldn't
    # land in latest/. The user can fall back to a per-10-ep snapshot.
    clean_exit = False

    try:
        while recorded_episodes < num_episodes and not events.get("stop_recording"):
            # --- Reset phase ---
            if multi_episode:
                from lerobot.utils.utils import log_say

                _soft_land(robot, duration_s=2.0, steps=10)
                _disable_torque(robot)
                # Also disable leader torque for manual reset
                if teleop is not None and hasattr(teleop, "disable_torque"):
                    teleop.disable_torque()

                print("##OVERLAY:Resetting:#888888##", flush=True)
                if recorded_episodes > 0:
                    log_say(f"Reset the environment. Episode {recorded_episodes} of {num_episodes} done.")
                else:
                    log_say("Reset the environment. Press right arrow to start.")

                events["exit_early"] = False
                reset_start = time.time()
                while not events["exit_early"] and not events.get("stop_recording"):
                    if stop_event is not None and stop_event.is_set():
                        break
                    # Only auto-advance on timeout for between-episode resets, not first
                    if (
                        recorded_episodes > 0
                        and reset_time_s > 0
                        and (time.time() - reset_start) >= reset_time_s
                    ):
                        break

                    loop_start = time.perf_counter()

                    # Teleop during reset: read leader → send to follower
                    if teleop is not None:
                        act = teleop.get_action()
                        action_dict = {name: float(act.get(name, 0)) for name in joint_names}
                        robot.send_action(action_dict)

                    # Keep obs stream alive for GUI camera preview
                    obs = robot.get_observation()
                    for step in obs_processor_steps:
                        obs = step.observation(obs)

                    dt = time.perf_counter() - loop_start
                    time.sleep(max(1.0 / fps - dt, 0.0))
                events["exit_early"] = False

                if events.get("stop_recording") or (stop_event is not None and stop_event.is_set()):
                    break

                _enable_torque(robot)

            # --- Recording phase ---
            from lerobot.utils.utils import log_say

            log_say(f"Recording episode {recorded_episodes + 1} of {num_episodes}")
            step_count = 0
            episode_start = time.time()
            policy.reset()
            prev_action_np = None  # reset per episode to avoid stale delta checks

            # RLT: activate collection for this episode
            if rlt_mode:
                # Advance to this episode's 0-indexed number BEFORE anything reads
                # `rlt_state["episode"]` (warmup check in inference thread,
                # chunk_compare dump, metrics). Fresh run: -1 -> 0 for ep0.
                rlt_state["episode"] += 1
                infer_thread._rlt_system_active = True
                infer_thread._rlt_user_engaged = rlt_start_engaged
                infer_thread._rlt_prev = None
                # Initialize per-episode lifecycle. ``begin`` resets all
                # operator-event flags AND captures buffer size for IGNORE
                # rollback. Asserts loud if the previous episode left a
                # signaled-but-unconsumed terminal — caught early instead
                # of silently corrupting future bookkeeping.
                rlt_state["lifecycle"].begin(
                    buffer_size=len(rlt_replay) if rlt_replay is not None else 0,
                )
                engaged_str = "engaged" if rlt_start_engaged else "disengaged (press E to engage)"
                # Warmup: during the first ``warmup_episodes`` the actor's output is
                # discarded and the S1 reference is executed instead (actor + critic
                # still train on the collected transitions). Overlay label should
                # reflect that — otherwise "Policy + RL" is misleading since the
                # actor isn't actually driving.
                _in_warmup = rlt_state["config"].is_warmup(rlt_state["episode"])
                if rlt_start_engaged:
                    if _in_warmup:
                        print("##OVERLAY:Policy + RL (warmup):#ffa726##", flush=True)
                    else:
                        print("##OVERLAY:Policy + RL:#4fc3f7##", flush=True)
                else:
                    print("##OVERLAY:Policy:#2ecc71##", flush=True)
                logger.info(
                    "RLT: collection RESUMED (episode start), actor %s%s",
                    engaged_str,
                    " [warmup]" if _in_warmup else "",
                )

            # Ensure inference thread is running and has fresh data.
            # May have been paused by intervention in previous episode.
            if infer_thread.is_paused:
                logger.info("S1: Resuming inference thread (was paused from previous episode)")
                if teleop is not None and hasattr(teleop, "enable_torque"):
                    teleop.enable_torque()
                infer_thread.resume()

            # Publish fresh observation and wait for a chunk produced AFTER it.
            # Clear the existing chunk first so we don't accept a stale one
            # that was computed from a previous episode's observation.
            fresh_obs = robot.get_observation()
            for step in obs_processor_steps:
                fresh_obs = step.observation(fresh_obs)
            _t_publish = time.perf_counter()
            # Invalidate current chunk so wait loop only accepts new ones
            with infer_thread._chunk_lock:
                infer_thread._chunk_data = None
                infer_thread._chunk_t_origin = 0.0
            infer_thread.publish_obs(fresh_obs, _t_publish)
            logger.info("S1: Waiting for fresh chunk (published obs at t=%.3f)", _t_publish)
            while True:
                chunk, t_origin, _ = infer_thread.get_chunk()
                # Only accept chunks produced AFTER we published the fresh obs
                if chunk is not None and t_origin >= _t_publish - 0.1:
                    logger.info("S1: Got fresh chunk (age=%.3fs)", time.perf_counter() - t_origin)
                    break
                if time.perf_counter() - _t_publish > 10.0:
                    logger.warning("S1: Timeout waiting for fresh chunk at episode start")
                    break
                time.sleep(0.01)

            # --- Episode start assertions ---
            if infer_thread.is_paused:
                raise RuntimeError("BUG: inference thread still paused at episode start")
            if chunk is None:
                raise RuntimeError("BUG: no chunk available at episode start")
            _chunk_age = time.perf_counter() - t_origin
            if _chunk_age > 5.0:
                raise RuntimeError(f"BUG: chunk is stale at episode start (age={_chunk_age:.1f}s)")
            if events.get("exit_early"):
                raise RuntimeError("BUG: exit_early flag not cleared before episode start")

            # Intervention tracking for this episode
            was_intervening = False
            last_follower_pos_sent: dict[str, float] = {}
            pending_int_episodes: list[dict] = []
            if teleop is not None and hasattr(teleop, "reset_intervention"):
                teleop.reset_intervention()
                logger.info("S1: Intervention flag reset for new episode")
                # Verify it's actually off
                if hasattr(teleop, "get_teleop_events"):
                    from lerobot.teleoperators.utils import TeleopEvents

                    _ev = teleop.get_teleop_events()
                    _is_int = _ev.get(TeleopEvents.IS_INTERVENTION, False)
                    if _is_int:
                        raise RuntimeError("BUG: intervention still active after reset_intervention()")
                    logger.info("S1: Intervention state confirmed OFF")
            if int_dataset is not None:
                int_dataset.writer.episode_buffer = int_dataset.writer._create_episode_buffer()
                # Restart video encoders for the new intervention episode.
                # Discard any active encoder from previous episode first.
                for enc in int_dataset.writer.video_encoders.values():
                    if enc._episode_active:
                        enc.discard()
                if int_dataset.writer.video_encoders:
                    int_dataset.writer._start_video_encoders()
                # Verify encoders are ready
                for key, enc in int_dataset.writer.video_encoders.items():
                    assert enc._episode_active, (
                        f"BUG: int_dataset video encoder '{key}' not active at episode start"
                    )

            _first_iter = True
            while stop_event is None or not stop_event.is_set():
                loop_start = time.perf_counter()
                # Latency: open the iteration record. Branchy loop body
                # below uses ``continue`` and ``break`` freely; the next
                # ``start_iter()`` resets cleanly so partial iterations
                # are simply discarded (record_loop has the same shape).
                main_session.start_iter()

                if _first_iter:
                    _first_iter = False
                    logger.info(
                        "S1: Episode loop start — was_intervening=%s, infer_paused=%s, "
                        "prev_action=%s, chunk_age=%.1fs",
                        was_intervening,
                        infer_thread.is_paused,
                        "None" if prev_action_np is None else "set",
                        time.perf_counter() - t_origin if chunk is not None else -1,
                    )

                if stop_event is not None and stop_event.is_set():
                    logger.info("S1: Stop signal received")
                    break

                # Check episode time limit
                if episode_time_s > 0 and (time.time() - episode_start) >= episode_time_s:
                    logger.info("S1: Episode time limit (%.0fs) reached", episode_time_s)
                    break

                # Check exit_early (right/left arrow or R/X key in RLT mode)
                if events.get("exit_early"):
                    _peek = rlt_state["lifecycle"].peek_terminal() if (rlt_mode and rlt_state) else None
                    if _peek == TerminalKind.SUCCESS:
                        logger.info("S1: R key — episode SUCCESS, ending early")
                    elif _peek == TerminalKind.ABORT:
                        logger.info("S1: X key — episode ABORTED, ending early")
                    elif events.get("rerecord_episode"):
                        logger.info("S1: Left arrow — ending episode (will rerecord)")
                    else:
                        logger.info("S1: Right arrow — ending episode early")
                    events["exit_early"] = False
                    break

                # 1. Capture observation (main loop owns robot)
                with main_session.span("get_observation"):
                    obs = robot.get_observation()
                    for step in obs_processor_steps:
                        obs = step.observation(obs)
                t_now = time.perf_counter()
                main_session.cam_consume_all(getattr(robot, "cameras", None))

                # Runtime check: inference thread must be alive
                if not infer_thread._thread.is_alive():
                    logger.critical(
                        "S1: Inference thread DIED — stopping episode. Check logs above for the exception."
                    )
                    events["exit_early"] = True
                    break

                # 2. Deep-copy image arrays before publishing. Camera background
                # threads continuously overwrite their frame buffers; without a
                # copy, the inference thread may read a partially-updated frame
                # (causing "Corrupt JPEG" artifacts and bad action predictions).
                obs_copy = {}
                for k, v in obs.items():
                    if isinstance(v, np.ndarray) and v.ndim == 3:  # image: HWC
                        obs_copy[k] = v.copy()
                    else:
                        obs_copy[k] = v  # scalars/strings are immutable, no copy needed

                # Publish to inference thread + S2 (keep publishing even during
                # intervention so S2 latent stays current for policy resume)
                with main_session.span("publish_obs"):
                    infer_thread.publish_obs(obs_copy, t_now)
                    if shared_images is not None:
                        shared_images.write_images(obs, S2_CAM_KEY_MAP, joint_names)

                # Check intervention state
                is_intervention = False
                if teleop is not None and hasattr(teleop, "get_teleop_events"):
                    from lerobot.teleoperators.utils import TeleopEvents

                    teleop_events = teleop.get_teleop_events()
                    is_intervention = teleop_events.get(TeleopEvents.IS_INTERVENTION, False)
                # Tag the iteration so the dashboard can color-code
                # intervention vs policy without inferring it from
                # action-source span presence.
                main_session.set_field("intervention", bool(is_intervention))

                if is_intervention and teleop is not None:
                    # --- INTERVENTION MODE: human controls via leader arm ---
                    if not was_intervening:
                        # Transition: policy → intervention. Lock SPACE so
                        # accidental double-taps during the 1-3s servo sync
                        # don't waste a sync round-trip toggling back to
                        # policy. Released in a finally below so the lock
                        # never sticks even if sync raises.
                        if hasattr(teleop, "set_intervention_transition_lock"):
                            teleop.set_intervention_transition_lock(True)
                        if rlt_mode:
                            # RLT: keep inference running (for z_rl) but stop actor + collection
                            infer_thread._rlt_system_active = False
                            infer_thread._rlt_prev = None
                            logger.info("S1: INTERVENTION ON — inference continues (RLT), actor paused")
                            rlt_state["lifecycle"].mark_intervention()
                            logger.info("RLT: collection PAUSED (intervention)")
                            # Recorder owns the human-chunk state; reset it here
                            # so previous-intervention state can't leak in.
                            if rlt_recorder is not None:
                                rlt_recorder.reset()
                        else:
                            logger.info("S1: INTERVENTION ON — pausing inference, human takes over")
                            infer_thread.pause()

                        # Measure initial leader-follower delta before any sync
                        leader_pos_before = teleop.get_action()
                        follower_pos_now = {k: v for k, v in obs.items() if k.endswith(".pos")}
                        initial_deltas = {
                            k: abs(leader_pos_before.get(k, 0) - follower_pos_now.get(k, 0))
                            for k in follower_pos_now
                        }
                        max_initial_delta = max(initial_deltas.values()) if initial_deltas else 0
                        worst_joint = max(initial_deltas, key=initial_deltas.get) if initial_deltas else "?"
                        logger.info(
                            "S1: Intervention start — leader-follower delta: max=%.1f° (%s), mean=%.1f°",
                            max_initial_delta,
                            worst_joint,
                            sum(initial_deltas.values()) / max(len(initial_deltas), 1),
                        )
                        if max_initial_delta > 90:
                            raise RuntimeError(
                                f"BUG: leader-follower delta {max_initial_delta:.1f}° on {worst_joint} "
                                f"is dangerously large — inverse follow may not be working"
                            )

                        # Compensate servo tracking error before releasing torque.
                        # Uses the same logic as lerobot_record: iterative correction
                        # with audio feedback (the beep sleep provides natural timing
                        # that prevents divergence).
                        if last_follower_pos_sent and hasattr(teleop, "send_feedback"):
                            import subprocess as _sp

                            target = last_follower_pos_sent
                            goal = dict(target)
                            settle_tolerance = 1.0
                            sync_timeout = 5.0
                            sync_start = time.perf_counter()
                            beep_proc = None
                            while True:
                                pos = teleop.get_action()
                                max_err = max(abs(pos.get(k, 0) - target[k]) for k in target)
                                if max_err < settle_tolerance:
                                    if beep_proc is not None:
                                        beep_proc.terminate()
                                    break
                                if time.perf_counter() - sync_start > sync_timeout:
                                    logger.warning("S1: Servo sync timed out (max_err=%.2f)", max_err)
                                    if beep_proc is not None:
                                        beep_proc.terminate()
                                    break
                                for k in target:
                                    error = pos.get(k, 0) - target[k]
                                    goal[k] = goal[k] - error
                                teleop.send_feedback(goal)
                                almost_ready = max_err < settle_tolerance * 2
                                if almost_ready:
                                    if beep_proc is None or beep_proc.poll() is not None:
                                        sr = 8000
                                        dur = 1.0
                                        t_arr = np.linspace(0, dur, int(sr * dur), dtype=np.float32)
                                        tone = (np.sin(2 * np.pi * 1000 * t_arr) * 16000).astype(np.int16)
                                        beep_proc = _sp.Popen(
                                            ["aplay", "-f", "S16_LE", "-r", str(sr), "-c", "1", "-q"],
                                            stdin=_sp.PIPE,
                                            stderr=_sp.DEVNULL,
                                        )
                                        beep_proc.stdin.write(tone.tobytes())
                                        beep_proc.stdin.close()
                                    time.sleep(0.05)
                                else:
                                    if beep_proc is not None and beep_proc.poll() is None:
                                        beep_proc.terminate()
                                        beep_proc = None
                                    interval = min(0.5, max(0.05, 0.05 + 0.45 * (1 - max_err / 50)))
                                    if beep_proc is None or beep_proc.poll() is not None:
                                        sr = 8000
                                        dur = min(interval * 0.6, 0.08)
                                        t_arr = np.linspace(0, dur, int(sr * dur), dtype=np.float32)
                                        freq = 600 + min(max_err, 50) * 8
                                        tone = (np.sin(2 * np.pi * freq * t_arr) * 16000).astype(np.int16)
                                        beep_proc = _sp.Popen(
                                            ["aplay", "-f", "S16_LE", "-r", str(sr), "-c", "1", "-q"],
                                            stdin=_sp.PIPE,
                                            stderr=_sp.DEVNULL,
                                        )
                                        beep_proc.stdin.write(tone.tobytes())
                                        beep_proc.stdin.close()
                                    time.sleep(interval)
                            logger.info(
                                "S1: Servo sync done in %.0fms (max_err=%.2f)",
                                (time.perf_counter() - sync_start) * 1e3,
                                max_err,
                            )
                        else:
                            logger.warning("S1: No last_follower_pos_sent — skipping servo sync")

                        if hasattr(teleop, "disable_torque"):
                            teleop.disable_torque()
                        # NOTE: do NOT disable follower torque — it must hold position
                        # while the human grabs the leader. The follower will be driven
                        # by send_action() once the human starts moving the leader.
                        log_say("Intervention")
                        print("##OVERLAY:Intervention:#e5c07b##", flush=True)
                        # Servo sync + torque transfer complete — operator may
                        # now toggle SPACE again to end intervention.
                        if hasattr(teleop, "set_intervention_transition_lock"):
                            teleop.set_intervention_transition_lock(False)

                    # Read leader arm positions and send to robot
                    act = teleop.get_action()
                    action_np = np.array([float(act.get(j, 0)) for j in joint_names], dtype=np.float32)

                    # Safety guard: abort if leader position jumps too far from
                    # current follower position (indicates servo released badly)
                    if prev_action_np is not None:
                        max_delta = np.abs(action_np - prev_action_np).max()
                        if max_delta > 30.0:
                            logger.warning(
                                "S1: INTERVENTION SAFETY — leader jump %.1f° > 30° threshold, "
                                "holding follower position. Leader pos: %s",
                                max_delta,
                                {
                                    joint_names[i]: f"{action_np[i]:.1f}"
                                    for i in range(min(4, len(action_np)))
                                },
                            )
                            # Don't send this action — hold previous position
                            action_np = prev_action_np.copy()

                    action_dict = {
                        name: float(action_np[i]) for i, name in enumerate(joint_names) if i < len(action_np)
                    }
                    with main_session.span("action_send"):
                        robot.send_action(action_dict)
                    t_after_send = time.perf_counter()

                    # Record to intervention dataset
                    if int_dataset is not None:
                        _add_frame_to_dataset(int_dataset, obs, action_np, joint_names, task)

                    # Record to main dataset too (so episode is continuous)
                    if dataset is not None:
                        _add_frame_to_dataset(dataset, obs, action_np, joint_names, task)

                    # RLT: route human actions into the replay buffer via the
                    # recorder. Paper Alg 1 lines 9, 11, 12 during intervention.
                    # z_rl is sourced from the inference thread; the recorder
                    # asserts it is non-None at every frame so any failure to
                    # expose a fresh value upstream surfaces immediately.
                    if rlt_mode and rlt_recorder is not None:
                        # Read through the dedicated accessor. The inference
                        # thread updates this every cycle regardless of
                        # rlt_active, so it stays fresh during intervention.
                        current_z_rl = infer_thread._rlt_latest_z_rl
                        rlt_recorder.on_frame(
                            human_action_np=action_np,
                            current_z_rl=current_z_rl,
                            current_obs=obs,
                        )
                        # ``total_transitions`` is reconciled in one batch at
                        # intervention end. No need to poll mid-intervention —
                        # the counter is only consumed at episode end.

                else:
                    # --- POLICY MODE: S1 inference controls robot ---
                    if was_intervening:
                        # Transition: intervention → policy. Lock SPACE
                        # while we re-enable torque and resume inference.
                        # Faster than the policy→intervention sync (no
                        # servo loop) but still non-instant.
                        if hasattr(teleop, "set_intervention_transition_lock"):
                            teleop.set_intervention_transition_lock(True)
                        logger.info("S1: INTERVENTION OFF — resuming inference")

                        # Save pending intervention episode buffer
                        if (
                            int_dataset is not None
                            and int_dataset.writer.episode_buffer is not None
                            and int_dataset.writer.episode_buffer.get("size", 0) > 0
                        ):
                            import copy

                            pending_int_episodes.append(copy.deepcopy(int_dataset.writer.episode_buffer))
                            int_dataset.writer.episode_buffer = int_dataset.writer._create_episode_buffer()
                            for enc in int_dataset.writer.video_encoders.values():
                                if enc._episode_active:
                                    enc.discard()
                            if int_dataset.writer.video_encoders:
                                int_dataset.writer._start_video_encoders()
                            for key, enc in int_dataset.writer.video_encoders.items():
                                assert enc._episode_active, (
                                    f"BUG: int_dataset encoder '{key}' not active after restart"
                                )

                        _enable_torque(robot)
                        if hasattr(teleop, "enable_torque"):
                            try:
                                teleop.enable_torque()
                            except ConnectionError as e:
                                logger.warning("S1: Failed to enable leader torque: %s", e)
                        if rlt_mode:
                            # Resume RLT collection — actor runs again
                            infer_thread._rlt_system_active = True
                            infer_thread._rlt_prev = None  # fresh start after intervention
                            if rlt_recorder is not None:
                                rlt_state["total_transitions"] += rlt_recorder.chunks_stored
                                rlt_recorder.log_summary()
                                rlt_recorder.reset()
                            logger.info("RLT: collection RESUMED (intervention ended)")
                        else:
                            infer_thread.resume()
                            assert not infer_thread.is_paused, (
                                "BUG: inference thread still paused after resume"
                            )
                        if rlt_mode and infer_thread._rlt_user_engaged:
                            _resumed_in_warmup = rlt_state["config"].is_warmup(rlt_state["episode"])
                            if _resumed_in_warmup:
                                log_say("Policy + RL warmup")
                                print("##OVERLAY:Policy + RL (warmup):#ffa726##", flush=True)
                            else:
                                log_say("Policy + RL")
                                print("##OVERLAY:Policy + RL:#4fc3f7##", flush=True)
                        else:
                            log_say("Policy")
                            print("##OVERLAY:Policy:#2ecc71##", flush=True)
                        # Resume complete — operator may now toggle SPACE
                        # again to re-enter intervention.
                        if hasattr(teleop, "set_intervention_transition_lock"):
                            teleop.set_intervention_transition_lock(False)

                    # 3. Read latest chunk
                    with main_session.span("get_chunk"):
                        chunk, t_origin, t_obs = infer_thread.get_chunk()

                    if chunk is None:
                        time.sleep(1.0 / fps)
                        was_intervening = is_intervention
                        continue

                    # Runtime check: chunk must not be stale (>5s = definitely a bug)
                    chunk_age_s = time.perf_counter() - t_origin
                    if chunk_age_s > 5.0:
                        logger.error(
                            "S1: STALE CHUNK — age %.1fs, inference paused=%s. "
                            "Skipping action to avoid executing outdated commands.",
                            chunk_age_s,
                            infer_thread.is_paused,
                        )
                        time.sleep(1.0 / fps)
                        was_intervening = is_intervention
                        continue

                    # 4. Index chunk and send action
                    t_before_send = time.perf_counter()
                    idx = _compute_chunk_index(t_before_send, t_origin, fps, len(chunk))
                    if osc_skip:
                        idx = _osc_skip(chunk, idx, step_count)

                    action_np = chunk[idx].copy()
                    if max_step_delta is not None and prev_action_np is not None:
                        action_np = _apply_delta_filter(action_np, prev_action_np, max_step_delta)

                    # Sanity: actions must be finite
                    if not np.isfinite(action_np).all():
                        logger.error("S1: NaN/Inf in action — holding previous position")
                        if prev_action_np is not None:
                            action_np = prev_action_np.copy()
                        else:
                            was_intervening = is_intervention
                            continue

                    # Safety: clamp large jumps (>30° any joint) to prevent damage
                    if prev_action_np is not None:
                        delta = action_np - prev_action_np
                        max_delta = np.abs(delta).max()
                        if max_delta > 30.0:
                            worst_idx = int(np.abs(delta).argmax())
                            logger.warning(
                                "S1: POLICY JUMP CLAMP — %.1f° on %s (step %d idx %d), clamping to ±30°",
                                max_delta,
                                joint_names[worst_idx],
                                step_count,
                                idx,
                            )
                            action_np = prev_action_np + np.clip(delta, -30.0, 30.0)

                    # Forward the remaining chunk (frame `idx` onward) to the
                    # robot. Two payload shapes selected by send_action_shape:
                    #   * "chunk" (default): pack chunk[idx:] as an ActionChunk
                    #     so a predictive robot activates the exact-lookup path
                    #     (_lookup_in_chunk) for zero-estimation-error
                    #     compensation at L. Non-predictive robots use
                    #     frames[0] only — same as the dict path effectively.
                    #   * "dict": send only the single frame at idx. Used to
                    #     A/B-compare against the pre-fix behaviour or as a
                    #     rollback. Predictive robots fall back to the
                    #     velocity-LSQ extrapolation path (which is now
                    #     rate-agnostic post-fix #1).
                    # The recorded "action" column in the dataset is
                    # action_np (a single frame) regardless of which shape
                    # is sent — see _add_frame_to_dataset below.
                    if send_action_shape == "chunk":
                        payload: dict | ActionChunk = _remaining_chunk_as_actionchunk(
                            chunk, idx, joint_names, fps, current_frame_override=action_np
                        )
                    else:
                        payload = {
                            name: float(action_np[i])
                            for i, name in enumerate(joint_names)
                            if i < len(action_np)
                        }
                    with main_session.span("action_send"):
                        robot.send_action(payload)
                    t_after_send = time.perf_counter()

                    # Inverse follow: send follower position to leader so it mirrors
                    if teleop is not None and hasattr(teleop, "send_feedback"):
                        follower_pos = {k: v for k, v in obs.items() if k.endswith(".pos")}
                        leader_actual = teleop.get_action()
                        compensated = {}
                        correction_gain = 0.3
                        for k, target in follower_pos.items():
                            error = leader_actual.get(k, target) - target
                            compensated[k] = target - correction_gain * error
                        teleop.send_feedback(compensated)
                        last_follower_pos_sent = follower_pos

                    # Record frame to dataset
                    if dataset is not None:
                        _add_frame_to_dataset(dataset, obs, action_np, joint_names, task)

                    # Track chunk execution index for RTC prefix extraction
                    if _supports_rtc:
                        infer_thread.update_exec_index(idx + 1)

                    # Smoothness tracking (after send, not on critical path)
                    if prev_action_np is not None:
                        action_deltas.append(np.linalg.norm(action_np - prev_action_np))
                        diag_chunk, _, _ = infer_thread.get_chunk()
                        _state = np.array([float(obs.get(j, 0)) for j in joint_names])
                        _imgs = {k: v for k, v in obs.items() if isinstance(v, np.ndarray) and v.ndim == 3}
                        _log_joint_jump(
                            action_np,
                            prev_action_np,
                            step_count,
                            idx,
                            diag_chunk,
                            robot_state=_state,
                            save_dir=grip_drop_save_dir,
                            obs_images=_imgs,
                            joint_names=joint_names,
                        )

                prev_action_np = action_np.copy()

                if last_send_time is not None:
                    loop_intervals.append((t_after_send - last_send_time) * 1000)
                last_send_time = t_after_send
                step_count += 1
                was_intervening = is_intervention

                # Periodic logging
                if step_count % 100 == 0:
                    mode_str = "INTERVENTION" if is_intervention else "POLICY"
                    smooth_str = ""
                    if action_deltas:
                        r = action_deltas[-20:]
                        smooth_str += f" | Δaction: {np.mean(r):.3f}/{np.max(r):.3f}"
                    smooth_str += f" | max_joint={np.max(np.abs(action_np)):.1f}"
                    if loop_intervals:
                        r = loop_intervals[-20:]
                        smooth_str += f" | interval: {np.mean(r):.1f}±{np.std(r):.1f}ms"
                    if infer_thread.infer_times:
                        smooth_str += f" | infer: {np.mean(infer_thread.infer_times[-10:]):.0f}ms"
                    if not is_intervention:
                        smooth_str += f" | chunk_age: {(t_before_send - t_origin) * 1000:.0f}ms"

                    if shared_cache is not None:
                        logger.info(
                            "S1 step %d [%s] | S2 age: %.0fms | S2 #%d%s",
                            step_count,
                            mode_str,
                            shared_cache.age_ms,
                            shared_cache.count,
                            smooth_str,
                        )
                    else:
                        logger.info(
                            "S1 step %d [%s]%s",
                            step_count,
                            mode_str,
                            smooth_str,
                        )

                # Latency: commit BEFORE sleeping so loop_dt_ms captures
                # work-only time (matches teleop / record convention; the
                # overrun check fires when work alone exceeds 1000/fps).
                main_session.end_iter()

                # Fixed-rate sleep
                dt = time.perf_counter() - loop_start
                sleep_s = max(1.0 / fps - dt, 0.0)
                if sleep_s > 0:
                    time.sleep(sleep_s)

            # --- End of recording phase for this episode ---
            # Collect any remaining intervention buffer
            if int_dataset is not None and (
                int_dataset.writer.episode_buffer is not None
                and int_dataset.writer.episode_buffer.get("size", 0) > 0
            ):
                import copy

                pending_int_episodes.append(copy.deepcopy(int_dataset.writer.episode_buffer))
                int_dataset.writer.episode_buffer = int_dataset.writer._create_episode_buffer()

            if dataset is not None and step_count > 0:
                try:
                    dataset.save_episode()
                    logger.info(
                        "S1: Episode %d saved (%d frames, %.1fs)",
                        recorded_episodes + 1,
                        step_count,
                        time.time() - episode_start,
                    )
                except Exception as e:
                    logger.warning("S1: Failed to save episode %d: %s", recorded_episodes + 1, e)

            # Save intervention episodes after main episode
            if int_dataset is not None and pending_int_episodes:
                for ep_buffer in pending_int_episodes:
                    try:
                        int_dataset.save_episode(episode_data=ep_buffer)
                        logger.info("S1: Saved intervention episode %d", int_dataset.num_episodes - 1)
                    except Exception as e:
                        logger.warning("S1: Failed to save intervention episode: %s", e)

            recorded_episodes += 1

            # RLT: episode bookkeeping
            if rlt_mode and rlt_state is not None:
                # Pause collection during reset + clear recorder state.
                infer_thread._rlt_system_active = False
                infer_thread._rlt_prev = None

                lifecycle = rlt_state["lifecycle"]
                if lifecycle.is_ignored():
                    # Operator pressed DOWN — discard this episode entirely.
                    # Drop staging (transitions accumulated this episode never
                    # entered the committed buffer in the first place — the
                    # TransactionalReplayBuffer's structural guarantee), drop
                    # pending intervention chunks, decrement the episode
                    # counter so the next iteration reuses it, and skip
                    # metrics + checkpoint save. ``latest/`` stays at the
                    # previous episode's clean save.
                    if rlt_recorder is not None:
                        rlt_recorder.reset()
                    dropped = rlt_replay.discard() if rlt_replay is not None else 0
                    rlt_state["total_transitions"] -= dropped
                    # Roll back the counter that was incremented at episode start.
                    ep_ignored = rlt_state["episode"]
                    rlt_state["episode"] -= 1
                    lifecycle.end_episode()
                    logger.info(
                        "RLT ep%d: IGNORED — discarded %d transitions; "
                        "episode counter rolled back to %d (next episode reuses ep%d)",
                        ep_ignored,
                        dropped,
                        rlt_state["episode"],
                        ep_ignored,
                    )
                    # Skip the rest of the bookkeeping branch (no metrics row,
                    # no checkpoint save). The ``else`` below handles the
                    # normal episode-end path.
                else:
                    if rlt_recorder is not None:
                        # Episode may have ended while intervention was still
                        # active (operator presses 'r' to mark success without
                        # releasing intervention first — the common workflow).
                        # In that case the intervention-OFF transition never ran,
                        # so reconcile the counter, emit the watchdog log, and
                        # flush a terminal transition if R/LEFT was the trigger.
                        if rlt_recorder.frames_observed > 0:
                            rlt_state["total_transitions"] += rlt_recorder.chunks_stored
                            rlt_recorder.log_summary()
                            _rlt_flush_intervention_terminal(
                                rlt_state,
                                rlt_recorder,
                                rlt_replay,
                                infer_thread,
                                obs,
                            )
                        rlt_recorder.reset()
                    logger.info("RLT: collection PAUSED (episode end)")

                    # Read the (consumed-or-still-set) terminal kind for logging.
                    # peek_terminal does NOT consume — it's purely for the bookkeeping
                    # path's classification of the episode outcome.
                    terminal = lifecycle.peek_terminal()
                    success = terminal == TerminalKind.SUCCESS
                    aborted = terminal == TerminalKind.ABORT
                    had_intervention = lifecycle.had_intervention
                    autonomous = success and not had_intervention
                    ep_duration = time.time() - episode_start

                    rlt_state["successes"].append(success)
                    # NOTE: episode counter is incremented at episode start, not here.
                    recent = rlt_state["successes"][-20:]

                    if aborted:
                        auto_label = "ABORTED"
                    elif autonomous:
                        auto_label = "AUTONOMOUS"
                    elif success:
                        auto_label = "ASSISTED"
                    else:
                        auto_label = "FAIL"
                    logger.info(
                        "RLT ep%d: %s (intervention=%s) | transitions=%d updates=%d "
                        "success_rate(20)=%.0f%% | %.1fs",
                        rlt_state["episode"],
                        auto_label,
                        "yes" if had_intervention else "no",
                        rlt_state["total_transitions"],
                        rlt_state["total_updates"],
                        np.mean(recent) * 100 if recent else 0,
                        ep_duration,
                    )
                    from lerobot.policies.hvla.rlt.metrics import get_metrics, save_metrics_to_file

                    get_metrics().record_episode(
                        rlt_state["episode"],
                        success,
                        autonomous=not had_intervention,
                        duration_s=ep_duration,
                    )
                    save_metrics_to_file()

                    # Commit staged transitions into the committed buffer.
                    # Done AFTER metrics + log so the printed
                    # ``total_transitions`` tally matches what was accumulated
                    # during the episode. From the next episode onward,
                    # grad-update sampling sees this episode's transitions.
                    if rlt_replay is not None:
                        rlt_replay.commit()

                    lifecycle.end_episode()

                # Save checkpoint (training mode only)
                if not rlt_state.get("deploy"):
                    try:
                        save_dir = rlt_state["output_dir"] / "latest"
                        save_dir.mkdir(parents=True, exist_ok=True)
                        _atomic_torch_save(rlt_agent.actor.state_dict(), save_dir / "actor.pt")
                        _atomic_torch_save(rlt_agent.critic.state_dict(), save_dir / "critic.pt")
                        _atomic_torch_save(
                            rlt_agent.critic_target.state_dict(), save_dir / "critic_target.pt"
                        )
                        _atomic_torch_save(
                            {
                                "actor_opt": rlt_agent.actor_opt.state_dict(),
                                "critic_opt": rlt_agent.critic_opt.state_dict(),
                                "episode": rlt_state["episode"],
                                "total_transitions": rlt_state["total_transitions"],
                                "total_updates": rlt_state["total_updates"],
                                "successes": rlt_state["successes"],
                                "rlt_token_checkpoint": rl_token_checkpoint,
                            },
                            save_dir / "training_state.pt",
                        )
                        rlt_replay.save(str(save_dir / "replay_buffer.pt"))
                        logger.info(
                            "RLT: Saved checkpoint → %s (ep=%d, buf=%d, updates=%d)",
                            save_dir,
                            rlt_state["episode"],
                            len(rlt_replay),
                            rlt_state["total_updates"],
                        )
                        # Every 10 episodes, mirror to a never-overwritten snapshot
                        # alongside latest/. Lets us roll back when Q explodes or the
                        # actor drifts, without losing the rolling latest/ state.
                        ep = rlt_state["episode"]
                        if ep > 0 and (ep + 1) % 10 == 0:
                            import shutil

                            snap_dir = rlt_state["output_dir"] / f"ep_{ep + 1}"
                            try:
                                if snap_dir.exists():
                                    # safe-destruct: RLT checkpoint snapshot we just wrote — refresh
                                    shutil.rmtree(snap_dir)
                                shutil.copytree(save_dir, snap_dir)
                                logger.info("RLT: Snapshot → %s (permanent, for rollback)", snap_dir)
                            # safe-destruct: RLT checkpoint snapshot we just wrote — refresh
                            except Exception as e:
                                logger.error("RLT: Snapshot failed: %s", e)
                    except Exception as e:
                        logger.error("RLT: Failed to save checkpoint: %s", e)

            # Reset tracking state for next episode
            prev_action_np = None
            action_deltas.clear()
            loop_intervals.clear()
            last_send_time = None

        # Reached the end of the episode loop normally (all episodes done
        # or stop_recording was set). Mark exit as clean.
        clean_exit = True

    except KeyboardInterrupt:
        # Ctrl+C is a clean stop — state at this point is valid.
        logger.info("S1: Interrupted by user")
        clean_exit = True
    finally:
        infer_thread.stop()
        if infer_thread.infer_times:
            s2_info = f" | S2 updates: {shared_cache.count}" if shared_cache is not None else ""
            logger.info(
                "S1: Done. %d episodes, avg infer: %.1fms%s",
                recorded_episodes,
                np.mean(infer_thread.infer_times),
                s2_info,
            )
        # Per-phase shutdown timings so we can see which phase is eating the
        # SIGTERM→SIGKILL grace window. Total is logged at the end.
        shutdown_totals: dict[str, float] = {}
        if dataset is not None:
            with _shutdown_phase("dataset_finalize", shutdown_totals):
                try:
                    if (
                        dataset.writer.episode_buffer is not None
                        and dataset.writer.episode_buffer.get("size", 0) > 0
                    ):
                        dataset.save_episode()
                        logger.info("S1: Final partial episode saved")
                    dataset.finalize()
                    logger.info("S1: Dataset '%s' finalized (%d episodes)", record_dataset, recorded_episodes)
                except Exception as e:
                    logger.warning("S1: Failed to finalize dataset: %s", e)
        if int_dataset is not None:
            with _shutdown_phase("int_dataset_finalize", shutdown_totals):
                try:
                    int_dataset.finalize()
                    logger.info("S1: Intervention dataset finalized (%d episodes)", int_dataset.num_episodes)
                except Exception as e:
                    logger.warning("S1: Failed to finalize intervention dataset: %s", e)
        if listener is not None:
            with _shutdown_phase("listener_stop", shutdown_totals):
                listener.stop()
        # RLT: final save (training mode only). Skipped on unhandled
        # exception — partial / mid-update state shouldn't land in
        # latest/. The user falls back to a per-10-ep snapshot
        # (output_dir/ep_N/) which only writes at clean episode-end
        # points.
        if rlt_mode and rlt_agent is not None and not rlt_state.get("deploy"):
            if not clean_exit:
                logger.warning(
                    "RLT: skipping final checkpoint save due to unhandled "
                    "exception. Resume from a per-episode snapshot "
                    "(<output_dir>/ep_N/) instead of latest/, which is "
                    "from the previous episode's clean save."
                )
            else:
                with _shutdown_phase("rlt_final_save", shutdown_totals):
                    try:
                        save_dir = rlt_state["output_dir"] / "latest"
                        save_dir.mkdir(parents=True, exist_ok=True)
                        _atomic_torch_save(rlt_agent.actor.state_dict(), save_dir / "actor.pt")
                        _atomic_torch_save(rlt_agent.critic.state_dict(), save_dir / "critic.pt")
                        _atomic_torch_save(
                            rlt_agent.critic_target.state_dict(), save_dir / "critic_target.pt"
                        )
                        _atomic_torch_save(
                            {
                                "actor_opt": rlt_agent.actor_opt.state_dict(),
                                "critic_opt": rlt_agent.critic_opt.state_dict(),
                                "episode": rlt_state["episode"],
                                "total_transitions": rlt_state["total_transitions"],
                                "total_updates": rlt_state["total_updates"],
                                "successes": rlt_state["successes"],
                            },
                            save_dir / "training_state.pt",
                        )
                        rlt_replay.save(str(save_dir / "replay_buffer.pt"))
                        from lerobot.policies.hvla.rlt.metrics import save_metrics_to_file

                        save_metrics_to_file()
                        logger.info("RLT: Final checkpoint → %s", save_dir)
                    except Exception as e:
                        logger.error("RLT: Failed to save final checkpoint: %s", e)
        with _shutdown_phase("soft_land", shutdown_totals):
            _soft_land(robot)
        if teleop is not None:
            with _shutdown_phase("teleop_disconnect", shutdown_totals):
                try:
                    teleop.disconnect()
                except Exception as e:
                    logger.warning("Teleop disconnect error (non-fatal): %s", e)
        with _shutdown_phase("robot_disconnect", shutdown_totals):
            try:
                robot.disconnect()
            except Exception as e:
                logger.warning("Robot disconnect error (non-fatal): %s", e)
        logger.info(
            "S1 shutdown total: %.3fs (%s)",
            sum(shutdown_totals.values()),
            ", ".join(f"{k}={v:.2f}s" for k, v in shutdown_totals.items()),
        )


@contextlib.contextmanager
def _shutdown_phase(name: str, totals: dict):
    """Time a shutdown phase and accumulate into ``totals`` for a final summary.

    Phase timings let us see whether the SIGTERM→SIGKILL grace window in
    `gui/api/run.stop_process` is too short by design (e.g. _soft_land alone
    is 4s) or whether some specific phase is randomly slow. Every phase is
    logged even on exception so a failing phase still records its cost.
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        totals[name] = elapsed
        logger.info("S1 shutdown[%s]: %.3fs", name, elapsed)


def _get_motor_buses(robot) -> list:
    """Extract motor bus objects from any LeRobot robot."""
    buses = []
    for attr in ("left_arm", "right_arm", "arm"):
        arm = getattr(robot, attr, None)
        if arm is not None and hasattr(arm, "bus"):
            buses.append(arm.bus)
    return buses


def _stop_robot_controllers(robot) -> None:
    """Stop any background bus-driving controllers on the robot.

    The predictive lookahead controller runs a 200 Hz writer thread
    that sync_writes Goal_Position. If we proceed straight to
    soft-landing (which writes Torque_Limit per motor on the same bus),
    the two threads race for the half-duplex serial port and the
    writer logs "Port is in use!" for the duration of the landing
    (~4 s of stack traces). Pre-empt by stopping any ``_controller``
    objects we can find. No-op for robots without a controller (plain
    so_follower etc.) — getattr returns None, the loop just skips.

    Always pair with ``_start_robot_controllers`` on the resume path
    (typically called from ``_enable_torque``). Otherwise the writer
    thread stays dead for the rest of the session and ``send_action``
    becomes a no-op (it updates ``_latest_intent`` / ``_latest_chunk``
    but nothing writes to the bus → robot frozen).
    """
    for attr in ("left_arm", "right_arm", "arm"):
        arm = getattr(robot, attr, None)
        if arm is None:
            continue
        controller = getattr(arm, "_controller", None)
        if controller is None or not hasattr(controller, "stop"):
            continue
        with contextlib.suppress(Exception):
            controller.stop()


def _start_robot_controllers(robot) -> None:
    """Restart any background controllers that ``_stop_robot_controllers``
    stopped. Idempotent — the predictive controller's ``start()`` returns
    early when its thread is already alive.
    """
    for attr in ("left_arm", "right_arm", "arm"):
        arm = getattr(robot, attr, None)
        if arm is None:
            continue
        controller = getattr(arm, "_controller", None)
        if controller is None or not hasattr(controller, "start"):
            continue
        with contextlib.suppress(Exception):
            controller.start()


def _disable_torque(robot):
    """Disable torque on all motors so the robot can be moved by hand."""
    for bus in _get_motor_buses(robot):
        with contextlib.suppress(Exception):
            bus.disable_torque()
    logger.info("S1: Torque disabled (robot can be moved by hand)")


def _enable_torque(robot):
    """Re-enable torque on all motors."""
    for bus in _get_motor_buses(robot):
        with contextlib.suppress(Exception):
            bus.enable_torque()
    # Symmetric with the soft-land path's stop: restart any background
    # writer threads that were stopped to avoid the bus race. Without
    # this restart, robots with a predictive lookahead controller stay
    # frozen after the pre-record reset — send_action becomes a silent
    # no-op because the writer thread is dead.
    _start_robot_controllers(robot)
    logger.info("S1: Torque enabled")


def _soft_land(robot, duration_s=4.0, steps=20):
    """Gradually reduce torque so the robot lowers gently instead of dropping.

    Uses Torque_Limit register for gradual reduction (Feetech/Dynamixel specific),
    then bus.disable_torque() for clean shutdown.
    """
    if not robot.is_connected:
        return

    # Stop any background controllers before touching the bus directly —
    # the predictive lookahead controller's 200 Hz writer would otherwise
    # cascade "Port is in use!" errors for the duration of the landing.
    _stop_robot_controllers(robot)

    buses = _get_motor_buses(robot)
    if not buses:
        return

    try:
        # Hold current position first
        obs = robot.get_observation()
        hold_action = {k: v for k, v in obs.items() if k.endswith(".pos")}
        if hold_action:
            robot.send_action(hold_action)

        # Gradually reduce torque limit (best-effort — not all motors support this)
        step_delay = duration_s / steps
        for i in range(steps):
            torque_value = int(1000 * (1.0 - (i + 1) / steps))
            for bus in buses:
                for motor_name in bus.motors:
                    with contextlib.suppress(Exception):
                        bus.write("Torque_Limit", motor_name, torque_value, normalize=False)
            time.sleep(step_delay)

        # Disable torque completely
        for bus in buses:
            with contextlib.suppress(Exception):
                bus.disable_torque()

        # Restore torque limit so next enable isn't stuck at 0
        for bus in buses:
            for motor_name in bus.motors:
                with contextlib.suppress(Exception):
                    bus.write("Torque_Limit", motor_name, 1000, normalize=False)

        logger.info("S1: Soft landing complete")
    except Exception as e:
        logger.warning("S1: Soft landing error: %s", e)
