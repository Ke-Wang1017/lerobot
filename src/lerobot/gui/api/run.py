"""Run tab API: launch teleoperate/record/replay subprocesses and stream output."""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

if TYPE_CHECKING:
    from lerobot.gui.state import AppState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/run", tags=["run"])

# Tuple of timeout-error classes to catch from `asyncio.wait_for(...)`.
#
# Why this exists: pyupgrade and ruff UP041 ("timeout-error-alias") rewrite
# any literal `asyncio.TimeoutError` token in the source to `TimeoutError`
# under py311+ targets, and pyproject.toml declares
# `requires-python = ">=3.12"`. But the project's actual conda env runs
# Python 3.10 (per CLAUDE.md / project memory), where
# `asyncio.TimeoutError` is still a *distinct* class from builtin
# `TimeoutError` — the alias only landed in 3.11. So a naive
# `except asyncio.TimeoutError:` (or even
# `except (TimeoutError, asyncio.TimeoutError):`) gets rewritten to
# `except TimeoutError:`, which then doesn't catch on 3.10 and the SSE
# keepalive crashes with an unhandled traceback every 2s.
#
# Resolution: the except clauses reference _TIMEOUT_EXCS by name so no
# rewriteable literal `asyncio.TimeoutError` token appears in an except
# context. The constant assignment itself isn't a UP041 target.
_TIMEOUT_EXCS: tuple[type[BaseException], ...] = (TimeoutError, asyncio.TimeoutError)

# Where the teleop subprocess writes its latency_snapshot.json. The frontend
# reads this via /api/run/latency-metrics (cross-process). Mirrors the RLT
# pattern (outputs/rlt_online/metrics.json).
LATENCY_OUTPUT_DIR_TELEOP = "outputs/teleop"
LATENCY_OUTPUT_DIR_RECORD = "outputs/record"
# Back-compat alias: existing call sites pass this for teleop. Don't rename
# without updating every caller.
LATENCY_OUTPUT_DIR = LATENCY_OUTPUT_DIR_TELEOP

# Known latency snapshot sources. Each subprocess (teleop / record / HVLA)
# writes ``latency_snapshot.json`` into its OWN directory so the dashboard
# doesn't render the same data twice when two source keys would otherwise
# share a file. /api/run/latency-metrics?source=<key> reads from the
# matching directory. Adding a new loop kind means appending to this map
# and (if needed) plumbing the output_dir through that script's CLI args.
# Keys must match the ``loop_kind`` field the writer publishes.
# HVLA writes one snapshot per thread under outputs/hvla_runs/<track>/, so
# the dashboard can render the control thread (track=main) and the
# inference thread (track=inference) as stacked tracks of the same
# process. Single-track loops (teleop, record) write directly into their
# loop directory.
LATENCY_SOURCES: dict[str, str] = {
    "teleop": LATENCY_OUTPUT_DIR_TELEOP,
    "record": LATENCY_OUTPUT_DIR_RECORD,
    "hvla_main": "outputs/hvla_runs/main",
    "hvla_infer": "outputs/hvla_runs/inference",
}

_app_state: AppState = None  # type: ignore


def set_app_state(state: AppState) -> None:
    global _app_state
    _app_state = state


# ============================================================================
# Profile → CLI args conversion
# ============================================================================


def _get_known_fields(profile_type: str, prefix: str) -> set[str] | None:
    """Return the set of valid field names for a config type, or None if unknown.

    Imports all robot/teleoperator modules to ensure config subclasses are registered.
    """
    import dataclasses
    import importlib
    import pkgutil

    import lerobot.robots
    import lerobot.teleoperators
    from lerobot.robots.config import RobotConfig
    from lerobot.teleoperators.config import TeleoperatorConfig

    # Trigger registration of all config subclasses
    pkg = lerobot.robots if prefix == "robot" else lerobot.teleoperators
    for _importer, modname, _ispkg in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        with contextlib.suppress(Exception):
            importlib.import_module(modname)

    base_cls = RobotConfig if prefix == "robot" else TeleoperatorConfig
    choices = base_cls.get_known_choices()
    config_cls = choices.get(profile_type)
    if config_cls is None:
        return None
    return {f.name for f in dataclasses.fields(config_cls)}


def _profile_to_cli_args(profile_data: dict, prefix: str, *, include_cameras: bool = True) -> list[str]:
    """Convert a profile data dict to draccus CLI arguments.

    Args:
        profile_data: {type, fields, cameras} from the frontend.
        prefix: "robot" or "teleop".
        include_cameras: Whether to include camera args (False for replay,
            which doesn't import camera config modules).

    Returns:
        List of CLI arg strings like ["--robot.type=bi_so107_follower", ...].
    """
    args = [f"--{prefix}.type={profile_data['type']}"]
    known_fields = _get_known_fields(profile_data["type"], prefix)

    for key, value in profile_data.get("fields", {}).items():
        if value is None:
            continue
        if known_fields is not None and key not in known_fields:
            logger.warning(
                f"Ignoring unknown {prefix} field '{key}' (not in {profile_data['type']} config). "
                f"It may belong to a different branch or have been removed."
            )
            continue
        if isinstance(value, bool):
            args.append(f"--{prefix}.{key}={str(value).lower()}")
        else:
            args.append(f"--{prefix}.{key}={value}")

    if include_cameras:
        cameras = profile_data.get("cameras", {})
        if cameras:
            args.append(f"--{prefix}.cameras={json.dumps(cameras)}")

    return args


# ============================================================================
# Request models
# ============================================================================


class DebugModelConfig(BaseModel):
    checkpoint: str
    policy_type: str
    task: str = ""
    decode_subtask: bool = True


class TeleoperateRequest(BaseModel):
    robot: dict[str, Any]
    teleop: dict[str, Any]
    fps: int = 60
    debug_model: DebugModelConfig | None = None


class RecordRequest(BaseModel):
    robot: dict[str, Any]
    teleop: dict[str, Any] | None = None
    repo_id: str
    root: str | None = None
    policy_path: str | None = None
    single_task: str
    fps: int = 30
    episode_time_s: float = 60
    reset_time_s: float = 60
    num_episodes: int = 50
    video: bool = True
    vcodec: str = "libsvtav1"
    play_sounds: bool = True
    resume: bool = False
    debug_model: DebugModelConfig | None = None
    intervention_repo_id: str | None = None


class ReplayRequest(BaseModel):
    robot: dict[str, Any]
    repo_id: str
    root: str | None = None
    episode: int
    # No fps: replay must always pace at the dataset's recording fps. Upstream
    # `lerobot-replay` declares `cfg.dataset.fps` but its loop paces by
    # `dataset.fps` directly, so the field is dead config.


class HVLARunRequest(BaseModel):
    robot: dict[str, Any]
    s1_checkpoint: str
    s2_checkpoint: str | None = None
    task: str
    fps: int = 30
    s1_type: str = "flow"
    decode_subtask: bool = False
    s1_query_interval: int | None = None
    denoise_steps: int | None = None
    record_dataset: str | None = None
    num_episodes: int = 1
    episode_time_s: float = 60
    reset_time_s: float = 20
    teleop: dict[str, Any] | None = None
    intervention_dataset: str | None = None
    # RLT (RL Token)
    rlt_mode: bool = False
    rlt_token_checkpoint: str | None = None  # Phase 1: RL token encoder
    rlt_checkpoint: str | None = None  # Phase 2: existing actor checkpoint dir
    rlt_deploy: bool = False  # True = inference only, no training
    rlt_chunk_length: int = 10
    rlt_output_dir: str = "outputs/rlt_online"
    rlt_start_engaged: bool = True
    rlt_shared_noise_per_chunk: bool = True  # default after v2_widened A/B
    # Action send shape on the policy path. "chunk" (default) routes the
    # remaining chunk frames as an ActionChunk → predictive robots use
    # the chunk-exact-lookup path (zero estimation residual). "dict"
    # sends only the single frame at idx → predictive robots fall back
    # to the velocity-LSQ extrapolation path. Non-predictive robots are
    # unaffected (they use frames[0] either way). Primarily for A/B
    # testing the chunk-vs-dict latency improvement; default matches
    # the recommended best-perf path.
    send_action_shape: str = "chunk"


# ============================================================================
# Subprocess state
# ============================================================================

_active_process: asyncio.subprocess.Process | None = None
_active_command: str | None = None
_active_config: dict | None = None
_debug_process: asyncio.subprocess.Process | None = None  # optional model debug alongside teleop
_debug_output_path: Path | None = None  # log file for debug model output
_debug_output_lines: list[str] = []
_debug_output_event: asyncio.Event = asyncio.Event()
_debug_read_task: asyncio.Task | None = None
_output_lines: list[str] = []
_output_event: asyncio.Event = asyncio.Event()
_OUTPUT_MAX_LINES = 2000


_overlay_state: dict | None = None  # {"text": "...", "color": "..."}


def _append_output(line: str) -> None:
    """Append a line to the output buffer and notify SSE waiters.

    Lines matching ##OVERLAY:text:color## are intercepted as overlay
    updates and not appended to the terminal. Any subprocess can set
    the camera-feed overlay by printing this format to stdout.
    """
    global _output_lines, _overlay_state
    if line.startswith("##OVERLAY:"):
        parts = line.strip().strip("#").split(":")
        # OVERLAY:text or OVERLAY:text:color
        text = parts[1] if len(parts) > 1 else ""
        color = parts[2] if len(parts) > 2 else "#ffffff"
        _overlay_state = {"text": text, "color": color}
        _output_event.set()  # wake SSE to send overlay update
        return
    _output_lines.append(line)
    if len(_output_lines) > _OUTPUT_MAX_LINES:
        _output_lines = _output_lines[-_OUTPUT_MAX_LINES:]
    _output_event.set()


def _ensure_no_active_process() -> None:
    """Raise 409 if a process is already running."""
    if _active_process is not None and _active_process.returncode is None:
        raise HTTPException(
            409, f"A '{_active_command}' process is already running (PID {_active_process.pid})"
        )


async def _read_stream(stream: asyncio.StreamReader, prefix: str = "") -> None:
    """Read lines from a subprocess stream and append to output buffer."""
    while True:
        line = await stream.readline()
        if not line:
            break
        text = line.decode("utf-8", errors="replace").rstrip("\n\r")
        _append_output(f"{prefix}{text}")


_stream_tasks: list[asyncio.Task] = []


async def _wait_for_exit() -> None:
    """Wait for the subprocess to finish, log the exit, and clean up."""
    global _active_process, _active_command, _active_config
    if _active_process is None:
        return
    proc = _active_process  # capture reference before clearing
    await proc.wait()
    # Wait for stream readers to finish draining pipes before clearing state,
    # otherwise the SSE sends "done" before error output is captured.
    if _stream_tasks:
        await asyncio.gather(*_stream_tasks, return_exceptions=True)
    rc = proc.returncode
    _append_output(f"\n--- Process exited with code {rc} ---")
    # Only clear if this is still the same process (not replaced by a new launch)
    if _active_process is proc:
        _active_process = None
        _active_command = None
        _active_config = None
        _close_obs_reader()
        # NOTE: debug model is NOT stopped here — it stays warm for reuse
        logger.info(f"Process exited (rc={rc}), state cleared")
    _output_event.set()


# ── Parent-death detector ────────────────────────────────────────────────
#
# Why: a SIGKILL on the GUI server (e.g. VS Code closing the terminal,
# OOM kill, segfault) leaves any in-flight teleop/record subprocess
# running indefinitely — observed in the wild as a `lerobot-teleoperate`
# subprocess alive almost 2 hours after the GUI exited, holding ttyACM
# ports and cameras and blocking the next launch.
#
# Phase 1 fix: use Linux's PR_SET_PDEATHSIG so the kernel sends SIGTERM
# to the child the moment its parent dies, independent of FastAPI's
# lifespan hooks. Set via preexec_fn (runs in the forked child before
# exec, async-signal-safe — only a single libc.prctl call). Linux-only;
# no-op elsewhere.
#
# Phase 2 (still TODO in gui/TODO.md): file-based heartbeat for the
# corner case where SIGTERM races a stuck child.

# PR_SET_PDEATHSIG value from <sys/prctl.h>. Kept as a module constant so
# the preexec function doesn't reach into ctypes at signal-handler time.
_PR_SET_PDEATHSIG = 1


def _set_pdeathsig_preexec() -> None:
    """preexec_fn for subprocess: ask the kernel to SIGTERM us if our parent dies.

    Runs in the forked child between fork and exec. Stays minimal: a
    single libc.prctl call. Failure is silent (we can't log from
    preexec_fn — the parent would never see it). Worst case we just
    don't get the auto-cleanup behaviour, which is the current state
    anyway.
    """
    if sys.platform != "linux":
        return
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        # prctl(PR_SET_PDEATHSIG, SIGTERM, 0, 0, 0). The signal is
        # cleared on exec of a setuid binary; not a concern here.
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except (OSError, AttributeError):
        pass


async def _launch_subprocess(
    args: list[str], command: str, config: dict, extra_env: dict[str, str] | None = None
) -> None:
    """Launch a subprocess and start reading its output."""
    global _active_process, _active_command, _active_config, _output_lines, _stream_tasks

    _output_lines = []
    _active_command = command
    _active_config = config

    # Close stale obs reader — the new process will create fresh shared memory
    # segments; any existing reader is mapped to old (possibly unlinked) segments.
    _close_obs_reader()

    env = {**__import__("os").environ, "LEROBOT_OBS_STREAM": "1"}
    if extra_env:
        env.update(extra_env)
    cmd_str = " ".join(args)
    logger.info(f"Launching: {cmd_str}")
    _append_output(f"--- Starting {command} ---")
    _append_output(f"$ {cmd_str}\n")

    _active_process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        preexec_fn=_set_pdeathsig_preexec,
    )

    _stream_tasks = [
        asyncio.create_task(_read_stream(_active_process.stdout)),
        asyncio.create_task(_read_stream(_active_process.stderr, prefix="[stderr] ")),
    ]
    asyncio.create_task(_wait_for_exit())


async def _launch_debug_s2(config: DebugModelConfig) -> None:
    """Launch HVLA S2 standalone as a debug model process alongside teleop."""
    global _debug_process, _debug_output_path, _debug_output_lines, _debug_read_task
    import tempfile

    await _stop_debug_process()

    args = [
        "python",
        "-u",
        "-m",
        "lerobot.policies.hvla.s2_standalone",
        f"--checkpoint={Path(config.checkpoint).expanduser() / 'model.safetensors'}"
        if not config.checkpoint.endswith(".safetensors")
        else f"--checkpoint={Path(config.checkpoint).expanduser()}",
        f"--task={config.task}",
    ]
    if config.decode_subtask:
        args.append("--decode-subtask")

    # Write output to a dedicated log file (not mixed with main process output).
    # mkstemp -> path-only: the fd is closed immediately, then the subprocess
    # opens the path with O_TRUNC. Avoids mktemp's TOCTOU race.
    _fd, _log_path = tempfile.mkstemp(prefix="lerobot_debug_model_", suffix=".log")
    os.close(_fd)
    _debug_output_path = Path(_log_path)
    _debug_output_lines = []

    env = {**__import__("os").environ}
    logger.info(f"Launching debug S2: {' '.join(args)}")
    logger.info(f"Debug model output: {_debug_output_path}")

    debug_log_fd = os.open(str(_debug_output_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    _debug_process = await asyncio.create_subprocess_exec(
        *args,
        stdout=debug_log_fd,
        stderr=debug_log_fd,
        env=env,
        preexec_fn=_set_pdeathsig_preexec,
    )
    os.close(debug_log_fd)  # subprocess inherited the fd, we can close our copy

    # Tail the log file in background
    _debug_read_task = asyncio.create_task(_tail_debug_log())


async def _tail_debug_log() -> None:
    """Tail the debug model log file, appending lines to _debug_output_lines."""
    global _debug_output_lines
    if _debug_output_path is None:
        return
    # Wait for file to exist
    for _ in range(50):
        if _debug_output_path.exists():
            break
        await asyncio.sleep(0.1)
    if not _debug_output_path.exists():
        return
    with open(_debug_output_path) as f:
        while True:
            line = f.readline()
            if line:
                line = line.rstrip("\n\r")
                if line:
                    _debug_output_lines.append(line)
                    if len(_debug_output_lines) > 1000:
                        _debug_output_lines = _debug_output_lines[-1000:]
                    _debug_output_event.set()
            else:
                # No new data — check if process exited
                if _debug_process is None or _debug_process.returncode is not None:
                    break
                await asyncio.sleep(0.1)


async def _stop_debug_process() -> None:
    """Stop the debug model process if running."""
    global _debug_process, _debug_read_task, _debug_output_path, _s2_subtask_cache
    _s2_subtask_cache = None
    if _debug_process is not None and _debug_process.returncode is None:
        _debug_process.terminate()
        try:
            await asyncio.wait_for(_debug_process.wait(), timeout=5.0)
        except _TIMEOUT_EXCS:
            _debug_process.kill()
            await _debug_process.wait()
    _debug_process = None
    if _debug_read_task is not None:
        _debug_read_task.cancel()
        _debug_read_task = None
    # Clean up log file
    if _debug_output_path is not None:
        with contextlib.suppress(Exception):
            # safe-destruct: our debug log temp file
            _debug_output_path.unlink(missing_ok=True)
        _debug_output_path = None


# ============================================================================
# Debug model management
# ============================================================================

_debug_lock = asyncio.Lock()  # prevent concurrent load/unload

# Serializes the check-then-launch sequence in start_teleoperate / start_record /
# start_replay / start_hvla. Without it, two requests arriving close together
# can both see `_active_process is None` (the synchronous check passes),
# both call `await asyncio.create_subprocess_exec(...)`, and the second
# overwrites `_active_process` before the first launch finishes — orphaning
# the first subprocess holding cameras and serial ports. Acquired around the
# whole `_ensure_no_active_process() + _launch_subprocess(...)` block so the
# check and the assign-to-`_active_process` happen atomically.
_launch_lock = asyncio.Lock()


def _is_debug_loaded() -> bool:
    return _debug_process is not None and _debug_process.returncode is None


@router.post("/debug/load")
async def load_debug_model(config: DebugModelConfig) -> dict:
    """Load a debug model process (keeps it warm for reuse across teleop sessions)."""
    async with _debug_lock:
        if _is_debug_loaded():
            raise HTTPException(409, "Debug model already loaded — unload first")

        if config.policy_type == "hvla_s2_vlm":
            await _launch_debug_s2(config)
        else:
            raise HTTPException(400, f"Unsupported debug model type: {config.policy_type}")

        return {"status": "loaded", "policy_type": config.policy_type, "pid": _debug_process.pid}


@router.post("/debug/unload")
async def unload_debug_model() -> dict:
    """Unload the debug model process (frees GPU memory)."""
    async with _debug_lock:
        if not _is_debug_loaded():
            return {"status": "not_loaded"}
        await _stop_debug_process()
        return {"status": "unloaded"}


@router.get("/debug/status")
async def debug_model_status() -> dict:
    """Check if a debug model is loaded."""
    return {"loaded": _is_debug_loaded(), "pid": _debug_process.pid if _is_debug_loaded() else None}


@router.get("/debug/output")
async def debug_output_sse():
    """SSE stream of debug model output lines."""
    sent = 0

    async def event_generator():
        nonlocal sent
        while True:
            _debug_output_event.clear()
            # Send any new lines
            while sent < len(_debug_output_lines):
                line = _debug_output_lines[sent]
                sent += 1
                yield f"data: {line}\n\n"
            # Wait for new data or timeout (keepalive)
            try:
                await asyncio.wait_for(_debug_output_event.wait(), timeout=5.0)
            except _TIMEOUT_EXCS:
                yield ": keepalive\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# TODO: generalize — currently hardcoded for HVLA S2 subtask.
# When multiple debug model types exist, this should read from a generic
# model output schema rather than importing HVLA-specific IPC.
_s2_subtask_cache = None  # SharedLatentCache | None (read-only attach)


@router.get("/debug/subtask")
async def debug_subtask() -> dict:
    """Read the latest S2 subtask text directly from shared memory.

    Returns {"subtask": str, "age_ms": float} or {"subtask": ""} if unavailable.
    """
    global _s2_subtask_cache
    if not _is_debug_loaded():
        _s2_subtask_cache = None
        return {"subtask": ""}
    # Lazily attach to S2's shared memory
    if _s2_subtask_cache is None:
        try:
            from lerobot.policies.hvla.ipc import SharedLatentCache

            _s2_subtask_cache = SharedLatentCache(create=False)
        except FileNotFoundError:
            return {"subtask": ""}
    try:
        text, ts, confidence = _s2_subtask_cache.read_subtask()
        age_ms = (time.time() - ts) * 1000 if ts > 0 else 0.0
        return {"subtask": text, "age_ms": age_ms, "confidence": confidence}
    except Exception:
        _s2_subtask_cache = None
        return {"subtask": ""}


# ============================================================================
# Endpoints
# ============================================================================


@router.post("/teleoperate")
async def start_teleoperate(req: TeleoperateRequest) -> dict:
    async with _launch_lock:
        _ensure_no_active_process()

        args = ["lerobot-teleoperate"]
        args.extend(_profile_to_cli_args(req.robot, "robot"))
        args.extend(_profile_to_cli_args(req.teleop, "teleop"))
        args.append(f"--fps={req.fps}")
        # Always-on latency monitoring for GUI sessions; the GUI polls
        # outputs/teleop/latency_snapshot.json for the live overlay.
        args.append("--latency_monitor=true")
        args.append(f"--latency_output_dir={LATENCY_OUTPUT_DIR}")

        # Ensure debug model is loaded if selected (lazy load, stays warm after teleop).
        # _debug_lock is nested under _launch_lock: this lock-ordering is safe
        # because _debug_lock holders (load_debug_model / unload_debug_model)
        # never reach for _launch_lock — no cycle is possible.
        extra_env = None
        if req.debug_model and req.debug_model.policy_type == "hvla_s2_vlm":
            async with _debug_lock:
                if _debug_process is None or _debug_process.returncode is not None:
                    await _launch_debug_s2(req.debug_model)
            extra_env = {"LEROBOT_S2_IMAGE_BUFFER": "1"}

        await _launch_subprocess(args, command="teleoperate", config=req.model_dump(), extra_env=extra_env)
        return {"status": "started", "command": "teleoperate", "pid": _active_process.pid}


@router.post("/record")
async def start_record(req: RecordRequest) -> dict:
    async with _launch_lock:
        _ensure_no_active_process()

        if req.teleop is None and req.policy_path is None:
            raise HTTPException(400, "Either teleop or policy_path must be provided")

        args = ["lerobot-record"]
        args.extend(_profile_to_cli_args(req.robot, "robot"))
        if req.teleop is not None:
            args.extend(_profile_to_cli_args(req.teleop, "teleop"))
        if req.policy_path:
            args.append(f"--policy.path={req.policy_path}")
        args.append(f"--dataset.repo_id={req.repo_id}")
        if req.root:
            args.append(f"--dataset.root={req.root}")
        args.append(f"--dataset.single_task={req.single_task}")
        args.append(f"--dataset.fps={req.fps}")
        args.append(f"--dataset.episode_time_s={req.episode_time_s}")
        args.append(f"--dataset.reset_time_s={req.reset_time_s}")
        args.append(f"--dataset.num_episodes={req.num_episodes}")
        args.append(f"--dataset.video={'true' if req.video else 'false'}")
        args.append("--dataset.push_to_hub=false")
        args.append(f"--dataset.vcodec={req.vcodec}")
        args.append(f"--play_sounds={'true' if req.play_sounds else 'false'}")
        if req.resume:
            args.append("--resume=true")
        if req.intervention_repo_id:
            args.append(f"--intervention_repo_id={req.intervention_repo_id}")
        # Always-on latency monitoring for GUI sessions. Record writes to its
        # own directory so the dashboard doesn't double-render when both
        # teleop and record source keys exist in LATENCY_SOURCES.
        # _ensure_no_active_process guarantees only one writer at a time.
        args.append("--latency_monitor=true")
        args.append(f"--latency_output_dir={LATENCY_OUTPUT_DIR_RECORD}")

        extra_env = None
        if req.debug_model and req.debug_model.policy_type == "hvla_s2_vlm":
            async with _debug_lock:
                if _debug_process is None or _debug_process.returncode is not None:
                    await _launch_debug_s2(req.debug_model)
            extra_env = {"LEROBOT_S2_IMAGE_BUFFER": "1"}

        await _launch_subprocess(args, command="record", config=req.model_dump(), extra_env=extra_env)
        return {"status": "started", "command": "record", "pid": _active_process.pid}


@router.post("/replay")
async def start_replay(req: ReplayRequest) -> dict:
    async with _launch_lock:
        _ensure_no_active_process()

        # Note: --dataset.fps is intentionally omitted — `lerobot-replay` declares
        # it as config but ignores it (the loop paces by `dataset.fps` directly),
        # so passing it from the GUI was dead wiring.
        args = ["lerobot-replay"]
        args.extend(_profile_to_cli_args(req.robot, "robot"))
        args.append(f"--dataset.repo_id={req.repo_id}")
        if req.root:
            args.append(f"--dataset.root={req.root}")
        args.append(f"--dataset.episode={req.episode}")

        await _launch_subprocess(args, command="replay", config=req.model_dump())
        return {"status": "started", "command": "replay", "pid": _active_process.pid}


@router.post("/hvla")
async def start_hvla(req: HVLARunRequest) -> dict:
    """Launch HVLA dual-system inference (S1 + S2)."""
    import tempfile

    async with _launch_lock:
        _ensure_no_active_process()

        # Write robot profile to temp file (HVLA launch reads robot config from file)
        robot_config = dict(req.robot)
        if "fields" in robot_config:
            flat = {"type": robot_config["type"]}
            flat.update(robot_config["fields"])
            if "cameras" in robot_config:
                flat["cameras"] = robot_config["cameras"]
            robot_config = flat

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, prefix="hvla_robot_") as tmp:
            tmp.write(json.dumps(robot_config, indent=2))
            tmp_name = tmp.name

        args = [
            "python",
            "-m",
            "lerobot.policies.hvla.launch",
            f"--s1-checkpoint={req.s1_checkpoint}",
            f"--task={req.task}",
            f"--robot-config={tmp_name}",
            f"--fps={req.fps}",
            f"--s1-type={req.s1_type}",
        ]
        if req.s1_query_interval is not None:
            args.append(f"--s1-query-interval={req.s1_query_interval}")
        if req.denoise_steps is not None:
            args.append(f"--denoise-steps={req.denoise_steps}")
        if req.s2_checkpoint:
            args.append(f"--s2-checkpoint={Path(req.s2_checkpoint).expanduser()}")
        else:
            args.append("--zero-s2")
        if req.decode_subtask:
            args.append("--decode-subtask")
        if req.record_dataset:
            args.append(f"--record-dataset={req.record_dataset}")
            args.append(f"--num-episodes={req.num_episodes}")
            args.append(f"--episode-time-s={req.episode_time_s}")
            args.append(f"--reset-time-s={req.reset_time_s}")

        # Teleop for intervention / inverse follow
        if req.teleop:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, prefix="hvla_teleop_"
            ) as teleop_tmp:
                teleop_tmp.write(json.dumps(req.teleop, indent=2))
                teleop_tmp_name = teleop_tmp.name
            args.append(f"--teleop-config={teleop_tmp_name}")
        if req.intervention_dataset:
            args.append(f"--intervention-dataset={req.intervention_dataset}")

        # RLT
        if req.rlt_mode:
            # Refuse to launch without an explicit RL Token Encoder. Every other
            # path tried — silent warning + actor.pt load → state_dict size
            # mismatch crash deep inside model construction. Fail fast at the
            # request boundary instead.
            if not req.rlt_token_checkpoint or not str(req.rlt_token_checkpoint).strip():
                raise HTTPException(
                    400,
                    "rlt_token_checkpoint is required when rlt_mode is True. "
                    "Set the 'RL Token Encoder' field to a trained Phase-1 "
                    "encoder dir (e.g. outputs/rlt_token_v4_4layer_d2048/"
                    "checkpoint-10000). The actor's input dim depends on it.",
                )
            args.append("--rlt-mode")
            args.append(f"--rl-token-checkpoint={Path(req.rlt_token_checkpoint).expanduser()}")
            if req.rlt_checkpoint:
                args.append(f"--rlt-checkpoint={Path(req.rlt_checkpoint).expanduser()}")
            if req.rlt_deploy:
                args.append("--rlt-deploy")
            args.append(f"--rl-chunk-length={req.rlt_chunk_length}")
            args.append(f"--rlt-output-dir={req.rlt_output_dir}")
            if not req.rlt_start_engaged:
                args.append("--rlt-start-disengaged")
            if req.rlt_shared_noise_per_chunk:
                args.append("--rlt-shared-noise-per-chunk")
            # RLT needs multi-episode mode
            if not req.record_dataset:
                args.append(f"--num-episodes={req.num_episodes}")
                args.append(f"--episode-time-s={req.episode_time_s}")
                args.append(f"--reset-time-s={req.reset_time_s}")

        # Always enable latency monitoring when launched from the GUI so the
        # dashboard's hvla tracks have data without requiring an extra
        # checkbox. s1_process appends /main and /inference to this parent
        # path; the resulting subdirs match LATENCY_SOURCES.
        args.append("--latency-monitor")
        args.append("--latency-output-dir=outputs/hvla_runs")

        # Action send shape (default "chunk"). Only emit the flag when the
        # value differs from the default to keep the CLI tidy in the GUI's
        # subprocess log.
        if req.send_action_shape != "chunk":
            args.append(f"--send-action-shape={req.send_action_shape}")

        await _launch_subprocess(args, command="hvla", config=req.model_dump())
        return {"status": "started", "command": "hvla", "pid": _active_process.pid}


@router.get("/rlt-metrics")
async def get_rlt_metrics() -> dict:
    """Return current RLT training metrics from file (cross-process)."""
    from lerobot.gui.api._run_core import get_rlt_metrics as _impl

    return _impl()


@router.get("/latency-metrics")
async def get_latency_metrics(source: str = "teleop") -> dict:
    """Return the latest latency snapshot for the requested source."""
    from lerobot.gui.api._run_core import get_latency_metrics as _impl

    return _impl(source)


@router.get("/latency-sources")
async def list_latency_sources() -> dict:
    """List which latency snapshot sources are available and recent.

    A source is considered "fresh" when its snapshot file was written in the
    last 5 seconds (the writer publishes at ~1 Hz; 5s tolerates jitter and
    pause/resume). Stale snapshots are reported with ``fresh=False`` so the
    dashboard can grey them out instead of dropping them entirely (useful
    for post-mortem viewing of the last run).

    Returns: ``{"sources": [{"key": str, "loop_kind": str, "fresh": bool,
    "age_s": float | None}, ...]}``.
    """
    out: list[dict] = []
    now = time.time()
    for key, dir_path in LATENCY_SOURCES.items():
        snap_path = Path(dir_path) / "latency_snapshot.json"
        if not snap_path.exists():
            out.append({"key": key, "loop_kind": None, "fresh": False, "age_s": None})
            continue
        try:
            mtime = snap_path.stat().st_mtime
            age_s = max(0.0, now - mtime)
            with open(snap_path, encoding="utf-8") as f:
                snap = json.load(f)
            out.append(
                {
                    "key": key,
                    "loop_kind": snap.get("loop_kind"),
                    "fresh": age_s <= 5.0,
                    "age_s": age_s,
                }
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("latency-sources read failed (key=%s): %s", key, e)
            out.append({"key": key, "loop_kind": None, "fresh": False, "age_s": None})
    return {"sources": out}


@router.get("/rlt-config")
async def get_rlt_config() -> dict:
    """Return current RLT override values so the GUI sliders can show the
    actually-applied settings instead of their hardcoded HTML defaults.

    Falls back to ``RLTConfig`` defaults for any key not present in the file,
    so the GUI always has a value to display. Returns empty dict when no RLT
    session is active.
    """
    import json as _json

    from lerobot.policies.hvla.rlt.config import RLTConfig

    defaults = RLTConfig()
    # ``active`` lets the GUI distinguish "no session — defaults" from
    # "session running — these are the live values". Without it the GUI
    # silently lets users toggle the diag button before launch / after
    # stop and the click goes nowhere (POST returns 409, frontend used
    # to swallow it). See ``set_rlt_config``.
    active = bool(_active_config and _active_config.get("rlt_output_dir"))
    result = {
        "active": active,
        "beta": defaults.beta,
        "exploration_sigma": defaults.exploration_sigma,
        "target_sigma": defaults.target_sigma,
        "dump_chunks": False,
    }
    if not active:
        return result
    override_path = Path(_active_config["rlt_output_dir"]) / "rlt_overrides.json"
    if not override_path.exists():
        return result
    try:
        with open(override_path) as f:
            overrides = _json.load(f)
        # Back-compat: legacy "actor_sigma" is a synonym for exploration_sigma.
        if "actor_sigma" in overrides and "exploration_sigma" not in overrides:
            overrides["exploration_sigma"] = overrides["actor_sigma"]
        for key in ("beta", "exploration_sigma", "target_sigma"):
            if key in overrides:
                result[key] = float(overrides[key])
        if "dump_chunks" in overrides:
            result["dump_chunks"] = bool(overrides["dump_chunks"])
    except Exception as e:
        logger.warning("RLT config read failed: %s", e)
    return result


@router.post("/rlt-config")
async def set_rlt_config(body: dict) -> dict:
    """Write RLT config overrides for the training subprocess to pick up."""
    import json as _json

    if not _active_config or not _active_config.get("rlt_output_dir"):
        raise HTTPException(409, "No active RLT session")
    # Validate: only known keys, numeric values in range. Legacy "actor_sigma"
    # is accepted as a synonym for exploration_sigma so older GUI builds don't
    # break mid-session.
    if "actor_sigma" in body and "exploration_sigma" not in body:
        body["exploration_sigma"] = body["actor_sigma"]
    allowed = {
        "beta": (0.0, 10.0),
        "exploration_sigma": (0.0, 1.0),
        "target_sigma": (0.0, 1.0),
    }
    filtered = {}
    for key, (lo, hi) in allowed.items():
        if key in body:
            val = float(body[key])
            filtered[key] = max(lo, min(hi, val))
    # Boolean flags (no range)
    if "dump_chunks" in body:
        filtered["dump_chunks"] = bool(body["dump_chunks"])
    if not filtered:
        raise HTTPException(
            400,
            "No valid keys. Allowed: beta, exploration_sigma, target_sigma, dump_chunks",
        )
    override_path = Path(_active_config["rlt_output_dir"]) / "rlt_overrides.json"
    # Ensure the output dir exists. The subprocess creates it inside ``run_s1``
    # but only after model imports finish — there's a multi-second window
    # right after launch where _active_config is set but the dir doesn't
    # exist yet, and a Diagnostic-toggle click in that window used to fail
    # with FileNotFoundError → 500 → red toast. Creating it here closes
    # that window without needing to rendezvous with the subprocess.
    override_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Merge with existing file so partial updates (e.g. only dump_chunks)
        # don't wipe other keys (beta, actor_sigma).
        existing = {}
        if override_path.exists():
            with contextlib.suppress(Exception), open(override_path) as f:
                existing = _json.load(f)
        merged = {**existing, **filtered}
        tmp = str(override_path) + ".tmp"
        with open(tmp, "w") as f:
            _json.dump(merged, f)
        os.replace(tmp, str(override_path))
        logger.info("RLT config override written: %s", filtered)
        return {"status": "ok", **filtered}
    except Exception as e:
        logger.warning("RLT config write failed: %s", e)
        raise HTTPException(500, str(e)) from e


@router.post("/stop")
async def stop_process() -> dict:
    global _active_process, _active_command, _active_config

    if _active_process is None:
        raise HTTPException(409, "No active process to stop")

    pid = _active_process.pid
    try:
        _active_process.send_signal(signal.SIGINT)
        try:
            await asyncio.wait_for(_active_process.wait(), timeout=5.0)
        except _TIMEOUT_EXCS:
            _active_process.kill()
            await _active_process.wait()
    except ProcessLookupError:
        pass

    _append_output(f"\n--- Process stopped (PID {pid}) ---\n")
    _active_process = None
    _active_command = None
    _active_config = None
    _close_obs_reader()
    # NOTE: debug model is NOT stopped here — it stays warm for reuse
    return {"status": "stopped", "pid": pid}


@router.get("/status")
async def get_status() -> dict:
    from lerobot.gui.api._run_core import get_run_status

    return get_run_status()


@router.get("/output")
async def stream_output() -> StreamingResponse:
    """Server-Sent Events stream of subprocess output."""

    async def event_generator():
        sent_lines = 0
        sent_overlay = None  # track last overlay sent to avoid duplicates
        proc_at_start = _active_process

        logger.info(
            f"SSE: connected, buffered={len(_output_lines)} lines, process={'running' if proc_at_start and proc_at_start.returncode is None else 'none/done'}"
        )

        # Send existing buffered lines first
        while sent_lines < len(_output_lines):
            line = _output_lines[sent_lines]
            yield f"data: {json.dumps({'line': line})}\n\n"
            sent_lines += 1

        # Then wait for new lines
        while True:
            _output_event.clear()

            # Send overlay update if changed
            if _overlay_state is not None and _overlay_state != sent_overlay:
                sent_overlay = _overlay_state.copy()
                yield f"data: {json.dumps({'overlay': sent_overlay})}\n\n"

            while sent_lines < len(_output_lines):
                line = _output_lines[sent_lines]
                yield f"data: {json.dumps({'line': line})}\n\n"
                sent_lines += 1

            # Check if process is done
            if _active_process is None or _active_process.returncode is not None:
                while sent_lines < len(_output_lines):
                    line = _output_lines[sent_lines]
                    yield f"data: {json.dumps({'line': line})}\n\n"
                    sent_lines += 1
                logger.info(
                    f"SSE: done (process={'exited rc=' + str(_active_process.returncode) if _active_process else 'None'}, sent={sent_lines} lines)"
                )
                yield f"data: {json.dumps({'done': True})}\n\n"
                return

            try:
                await asyncio.wait_for(_output_event.wait(), timeout=2.0)
            except _TIMEOUT_EXCS:
                yield ": keepalive\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ============================================================================
# Observation stream endpoints (shared-memory camera/state viewer)
# ============================================================================

_obs_reader = None  # ObservationStreamReader | None
_obs_reader_meta_ino: int | None = None  # inode of /dev/shm/lerobot_obs_meta at attach time
_jpeg_cache: dict[str, tuple[int, bytes]] = {}  # cam_key → (seq, jpeg_bytes)

_OBS_META_SHM_PATH = "/dev/shm/lerobot_obs_meta"  # nosec B108  # POSIX shared memory (well-known path)


def _get_obs_reader():
    """Lazily attach to the robot's observation stream.

    Detects stream recreation (e.g. new teleop session after a crash) by
    comparing the inode of ``/dev/shm/lerobot_obs_meta`` — if it changed,
    the old reader is stale (mapped to unlinked segments) and we re-attach.
    """
    global _obs_reader, _obs_reader_meta_ino

    if _obs_reader is not None:
        # Cheap staleness check: has the meta segment been recreated?
        try:
            current_ino = os.stat(_OBS_META_SHM_PATH).st_ino
        except FileNotFoundError:
            _close_obs_reader()
            return None
        if current_ino != _obs_reader_meta_ino:
            logger.info("ObservationStream recreated (inode changed), re-attaching reader")
            _close_obs_reader()
            # Fall through to create new reader
        else:
            return _obs_reader

    try:
        from lerobot.robots.obs_stream import ObservationStreamReader

        _obs_reader = ObservationStreamReader()
        try:
            _obs_reader_meta_ino = os.stat(_OBS_META_SHM_PATH).st_ino
        except FileNotFoundError:
            _obs_reader_meta_ino = None
        logger.info(
            "ObservationStreamReader attached: %d scalars, %d cameras",
            len(_obs_reader.obs_scalar_keys),
            len(_obs_reader.image_keys),
        )
        return _obs_reader
    except Exception:
        return None


def _close_obs_reader():
    global _obs_reader, _obs_reader_meta_ino
    if _obs_reader is not None:
        _obs_reader.close()
        _obs_reader = None
    _obs_reader_meta_ino = None
    _jpeg_cache.clear()


@router.get("/obs-stream/meta")
async def obs_stream_meta() -> dict:
    """Return observation stream layout (feature names, image dims)."""
    reader = _get_obs_reader()
    if reader is None:
        logger.debug("obs-stream/meta: reader not available")
        return {"available": False}
    logger.info("obs-stream/meta: available, cameras=%s", list(reader.image_keys.keys()))
    return {
        "available": True,
        "obs_scalar_keys": reader.obs_scalar_keys,
        "action_keys": reader.action_keys,
        "image_keys": reader.image_keys,
    }


@router.get("/obs-stream/state")
async def obs_stream_state() -> dict:
    """Return latest scalar observations and actions."""
    reader = _get_obs_reader()
    if reader is None:
        raise HTTPException(503, "Observation stream not available")
    obs_result = reader.read_obs()
    act_result = reader.read_action()
    return {
        "obs": obs_result[0] if obs_result else None,
        "obs_ts": obs_result[1] if obs_result else None,
        "action": act_result[0] if act_result else None,
        "action_ts": act_result[1] if act_result else None,
    }


@router.get("/urdf-viz/meta")
async def urdf_viz_meta() -> dict:
    """One-shot identity + advertised sources for the in-browser URDF viewer.

    The frontend fetches this once at iframe init to learn (1) which robot is
    on the run (so it can load the URDF and arrange arms in the scene) and
    (2) which pose sources are currently available. ``sources`` is a list of
    user-facing names — today ``"state"`` (always) and ``"action"`` (when the
    run is writing the commanded action to the obs stream). Future sources
    (e.g. ``"prediction"`` from a debug model) appear here without changing
    the rest of the contract.

    ``{"available": false}`` when no run is streaming or the robot has no
    vendored URDF.
    """
    from lerobot.gui.urdf_viz import resolve_robot

    reader = _get_obs_reader()
    if reader is None:
        return {"available": False}
    obs_result = reader.read_obs()
    if not obs_result:
        return {"available": False}
    spec = resolve_robot(obs_result[0].keys())
    if spec is None:
        return {"available": False}
    sources = ["state"]
    if reader.read_action():
        sources.append("action")
    return {
        "available": True,
        "name": spec.name,
        "urdf": f"/urdf-assets/{spec.urdf_url_path}",
        "bimanual": len(spec.arms) == 2,
        "sources": sources,
        # ee_link is None for descriptions that didn't declare one; the
        # frontend skips polyline rendering in that case.
        "ee_link": spec.ee_link,
    }


@router.get("/urdf-viz")
async def urdf_viz_source(source: str = "state") -> dict:
    """Per-arm URDF joint angles for one named source.

    ``source`` mirrors the user-facing names advertised by
    :func:`urdf_viz_meta` (``"state"`` / ``"action"``); the backend decides
    how to fulfil each one. The response is the same shape regardless of
    source — a list of arms, each with one or more ``frames``. A teleop's
    action is a single frame; a future chunk-output policy would return many
    with an ``fps`` for playback timing. The renderer applies frames[0] for
    a pose and steps through for a chunk; it does not need to know which
    backend produced the data.

    ``{"available": false}`` when no run is streaming, the robot has no
    vendored URDF, or the requested source has no current data.
    """
    from lerobot.gui.urdf_viz import compute_joint_angles, resolve_robot

    reader = _get_obs_reader()
    if reader is None:
        return {"available": False}
    obs_result = reader.read_obs()
    if not obs_result:
        return {"available": False}
    obs = obs_result[0]
    spec = resolve_robot(obs.keys())
    if spec is None:
        return {"available": False}

    if source == "state":
        sample = obs
    elif source == "action":
        act_result = reader.read_action()
        if not act_result:
            return {"available": False}
        sample = act_result[0]
    else:
        raise HTTPException(status_code=400, detail=f"unknown source: {source!r}")

    angles = compute_joint_angles(spec, sample)
    arms_payload = [
        {"prefix": a.obs_prefix, "frames": [{"joints": angles.get(a.obs_prefix, {})}]} for a in spec.arms
    ]
    return {"available": True, "arms": arms_payload}


@router.get("/obs-stream/image/{cam_key}")
async def obs_stream_image(cam_key: str) -> Response:
    """Return latest camera frame as JPEG (cached until new frame arrives)."""
    reader = _get_obs_reader()
    if reader is None:
        raise HTTPException(503, "Observation stream not available")

    # Skip re-encoding if the frame hasn't changed
    seq = reader.image_seq(cam_key)
    cached = _jpeg_cache.get(cam_key)
    if cached is not None and cached[0] == seq:
        return Response(
            content=cached[1],
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    result = reader.read_image(cam_key)
    if result is None:
        raise HTTPException(404, f"No image for camera '{cam_key}'")
    img, _ts = result

    import cv2

    def _encode() -> bytes:
        # cv2.cvtColor + imencode is 5–10 ms for a 720p frame. Push it off
        # the event loop so concurrent obs-stream image requests for other
        # cameras + the obs_stream_state poll don't queue up behind it.
        # cv2 releases the GIL during the C-side encode, so the default
        # thread pool scales naturally.
        _, jpeg = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
        return jpeg.tobytes()

    jpeg_bytes = await asyncio.get_event_loop().run_in_executor(None, _encode)
    _jpeg_cache[cam_key] = (seq, jpeg_bytes)
    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )
