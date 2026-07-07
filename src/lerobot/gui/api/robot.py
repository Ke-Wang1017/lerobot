"""Robot and teleop profile management, camera detection, and port scanning."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import platform
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

if TYPE_CHECKING:
    from lerobot.gui.state import AppState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/robot", tags=["robot"])

# Module-level state (same pattern as datasets.py)
_app_state: AppState = None  # type: ignore

# Config directories
ROBOT_PROFILES_DIR = Path.home() / ".config" / "lerobot" / "robots"
TELEOP_PROFILES_DIR = Path.home() / ".config" / "lerobot" / "teleops"

# Camera preview state
_preview_cameras: list = []
_preview_camera_info: list[dict] = []

# Rest-position recording state (holds robot connection between start/finish)
_rest_recording_robot = None

# Safe-trajectory recording state. Same idea as the rest-position session
# (one active connection held across HTTP calls) but with a background
# sampler thread accumulating frames in between.
_trajectory_recording_robot = None
_trajectory_recorder = None  # lerobot.robots.safe_trajectory.TrajectoryRecorder


def set_app_state(state: AppState) -> None:
    global _app_state
    _app_state = state


# ============================================================================
# Config loading helpers
# ============================================================================

_configs_loaded = False


def _ensure_configs_loaded():
    """Import all robot/teleop config modules to trigger ``@register_subclass``.

    Auto-discovers every submodule under ``lerobot.robots`` and
    ``lerobot.teleoperators`` via ``pkgutil.walk_packages``. Each import
    is wrapped in ``contextlib.suppress(Exception)`` so packages whose
    optional dependencies (hebi, pyrealsense2, reachy2_sdk, ...) aren't
    installed stay silent. The walk is one level deep — we want the
    per-package config modules (e.g. ``config_bi_so107_follower``), not
    every transitive file (motors, kinematics, etc.).

    Why auto-discovery: the previous hard-coded list silently dropped any
    new robot/teleop package whose author forgot to add an entry — the
    new type just didn't appear in the Robot tab dropdown with no error,
    no log, no hint. Caught on bi_so107_follower_predictive (2026-05-12).
    Mirrors the same pattern already used by ``_get_known_fields`` in
    ``run.py``.
    """
    global _configs_loaded
    if _configs_loaded:
        return

    import importlib
    import pkgutil

    import lerobot.robots
    import lerobot.teleoperators

    for pkg in (lerobot.robots, lerobot.teleoperators):
        for _importer, modname, _ispkg in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
            with contextlib.suppress(Exception):
                importlib.import_module(modname)

    _configs_loaded = True


# ============================================================================
# Schema introspection
# ============================================================================

# Fields to skip in schema output (handled separately or irrelevant to GUI)
_SKIP_FIELDS = {"cameras", "calibration_dir"}


def _stringify_type(annotation: Any) -> str:
    """Convert a type annotation to a simple string for the frontend."""
    s = str(annotation)
    # Clean up common patterns
    for prefix in ("typing.", "<class '", "pathlib."):
        s = s.replace(prefix, "")
    s = s.rstrip("'>")
    return s


def _literal_choices(annotation: Any) -> list[str] | None:
    """If ``annotation`` is ``Literal["a", "b", ...]``, return the values as
    strings so the frontend can render a dropdown. Otherwise return None.

    Generic-by-design — any dataclass field annotated as ``Literal[...]``
    automatically becomes a `<select>` in the robot config editor. To add
    a new enum-ish field, change its annotation; no GUI code edit needed.
    """
    import typing

    if typing.get_origin(annotation) is typing.Literal:
        return [str(v) for v in typing.get_args(annotation)]
    return None


def _introspect_fields(cls: type) -> list[dict]:
    """Extract field info from a dataclass config class."""
    # Resolve string annotations (PEP 563 / __future__ annotations) so
    # ``Literal["latest", "wait_for_new"]`` is the actual ``Literal``
    # generic, not a bare string. Without this, ``_literal_choices``
    # always returns None for forward-reference-style annotations.
    import typing

    try:
        type_hints = typing.get_type_hints(cls)
    except Exception:
        type_hints = {}
    result = []
    for f in dataclasses.fields(cls):
        if f.name in _SKIP_FIELDS:
            continue
        required = (
            f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING  # type: ignore[arg-type]
        )
        default = None
        if f.default is not dataclasses.MISSING:
            default = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[arg-type]
            # Don't call factory, just indicate it has a default
            default = None

        resolved_type = type_hints.get(f.name, f.type)
        entry = {
            "name": f.name,
            "type_str": _stringify_type(f.type),
            "required": required,
            "default": default,
        }
        choices = _literal_choices(resolved_type)
        if choices is not None:
            entry["choices"] = choices
        # Field-level documentation surfaced via ``dataclasses.field(
        # metadata={"description": "..."})``. Frontend renders this as a
        # tooltip on the label + control so users hover-discover what the
        # knob means without reading the source. Per-Literal-choice
        # documentation is forwarded as ``choice_descriptions`` so dropdown
        # options can each carry their own tooltip — useful for opaque
        # algorithm names like "stateful_lp" vs "amp_gated_lp".
        description = f.metadata.get("description")
        if description:
            entry["description"] = description
        choice_descs = f.metadata.get("choice_descriptions")
        if choice_descs:
            entry["choice_descriptions"] = dict(choice_descs)
        result.append(entry)
    return result


@router.get("/schemas")
async def get_robot_schemas() -> list[dict]:
    """Return field schemas for all registered robot config types."""
    from lerobot.robots.config import RobotConfig

    _ensure_configs_loaded()

    schemas = []
    for type_name, config_cls in sorted(RobotConfig.get_known_choices().items()):
        schemas.append(
            {
                "type_name": type_name,
                "fields": _introspect_fields(config_cls),
            }
        )
    return schemas


@router.get("/teleop-schemas")
async def get_teleop_schemas() -> list[dict]:
    """Return field schemas for all registered teleoperator config types."""
    from lerobot.teleoperators.config import TeleoperatorConfig

    _ensure_configs_loaded()

    schemas = []
    for type_name, config_cls in sorted(TeleoperatorConfig.get_known_choices().items()):
        schemas.append(
            {
                "type_name": type_name,
                "fields": _introspect_fields(config_cls),
            }
        )
    return schemas


# ============================================================================
# Profile CRUD
# ============================================================================


class ProfileData(BaseModel):
    type: str
    name: str
    fields: dict[str, Any] = {}
    cameras: dict[str, dict[str, Any]] = {}
    rest_position: dict[str, float] = {}


class RenameRequest(BaseModel):
    new_name: str


def _ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


def _is_profile_file(f: Path) -> bool:
    """True if *f* is a profile JSON, not a sibling artefact.

    The profiles dir also holds ``<name>.trajectory.json`` files (safe
    trajectories recorded against a profile). Those must not appear in
    profile listings — they have no ``type`` field and the GUI would
    fail to render them as robots.
    """
    return f.suffix == ".json" and not f.name.endswith(".trajectory.json")


def _list_profiles(profiles_dir: Path) -> list[dict]:
    _ensure_dir(profiles_dir)
    profiles = []
    for f in sorted(profiles_dir.glob("*.json")):
        if not _is_profile_file(f):
            continue
        try:
            data = json.loads(f.read_text())
            profiles.append(
                {
                    "name": data.get("name", f.stem),
                    "type": data.get("type", "unknown"),
                }
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read profile {f}: {e}")
    return profiles


def _read_profile(profiles_dir: Path, name: str) -> dict:
    path = profiles_dir / f"{name}.json"
    if not path.exists():
        raise HTTPException(404, f"Profile '{name}' not found")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise HTTPException(500, f"Failed to parse profile: {e}") from e


def _write_profile(profiles_dir: Path, data: ProfileData) -> None:
    _ensure_dir(profiles_dir)
    path = profiles_dir / f"{data.name}.json"
    # Atomic write — same rationale as _write_trajectory: a half-written
    # profile silently loses the user's saved port configuration.
    import os

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data.model_dump(), indent=2))
    os.replace(tmp, path)
    logger.info(f"Saved profile: {path}")


def _delete_profile(profiles_dir: Path, name: str) -> None:
    path = profiles_dir / f"{name}.json"
    if not path.exists():
        raise HTTPException(404, f"Profile '{name}' not found")
    # safe-destruct: user-confirmed delete via GUI dialog
    path.unlink()
    logger.info(f"Deleted profile: {path}")


def _rename_profile(profiles_dir: Path, old_name: str, new_name: str) -> None:
    old_path = profiles_dir / f"{old_name}.json"
    if not old_path.exists():
        raise HTTPException(404, f"Profile '{old_name}' not found")
    new_path = profiles_dir / f"{new_name}.json"
    if new_path.exists():
        raise HTTPException(409, f"Profile '{new_name}' already exists")
    data = json.loads(old_path.read_text())
    data["name"] = new_name
    new_path.write_text(json.dumps(data, indent=2))
    # safe-destruct: rename: drop old after writing new
    old_path.unlink()
    logger.info(f"Renamed profile: {old_path} -> {new_path}")


# Robot profiles
@router.get("/profiles")
async def list_robot_profiles() -> list[dict]:
    return _list_profiles(ROBOT_PROFILES_DIR)


@router.post("/profiles")
async def create_robot_profile(profile: ProfileData) -> dict:
    path = ROBOT_PROFILES_DIR / f"{profile.name}.json"
    if path.exists():
        raise HTTPException(409, f"Profile '{profile.name}' already exists")
    _write_profile(ROBOT_PROFILES_DIR, profile)
    return {"status": "created", "name": profile.name}


@router.get("/profiles/{name}")
async def get_robot_profile(name: str) -> dict:
    return _read_profile(ROBOT_PROFILES_DIR, name)


@router.put("/profiles/{name}")
async def update_robot_profile(name: str, profile: ProfileData) -> dict:
    _write_profile(ROBOT_PROFILES_DIR, profile)
    return {"status": "updated", "name": profile.name}


@router.delete("/profiles/{name}")
async def delete_robot_profile(name: str) -> dict:
    _delete_profile(ROBOT_PROFILES_DIR, name)
    return {"status": "deleted", "name": name}


@router.post("/profiles/{name}/rename")
async def rename_robot_profile(name: str, req: RenameRequest) -> dict:
    _rename_profile(ROBOT_PROFILES_DIR, name, req.new_name)
    return {"status": "renamed", "old_name": name, "new_name": req.new_name}


# Teleop profiles
@router.get("/teleop-profiles")
async def list_teleop_profiles() -> list[dict]:
    return _list_profiles(TELEOP_PROFILES_DIR)


@router.post("/teleop-profiles")
async def create_teleop_profile(profile: ProfileData) -> dict:
    path = TELEOP_PROFILES_DIR / f"{profile.name}.json"
    if path.exists():
        raise HTTPException(409, f"Profile '{profile.name}' already exists")
    _write_profile(TELEOP_PROFILES_DIR, profile)
    return {"status": "created", "name": profile.name}


@router.get("/teleop-profiles/{name}")
async def get_teleop_profile(name: str) -> dict:
    return _read_profile(TELEOP_PROFILES_DIR, name)


@router.put("/teleop-profiles/{name}")
async def update_teleop_profile(name: str, profile: ProfileData) -> dict:
    _write_profile(TELEOP_PROFILES_DIR, profile)
    return {"status": "updated", "name": profile.name}


@router.delete("/teleop-profiles/{name}")
async def delete_teleop_profile(name: str) -> dict:
    _delete_profile(TELEOP_PROFILES_DIR, name)
    return {"status": "deleted", "name": name}


@router.post("/teleop-profiles/{name}/rename")
async def rename_teleop_profile(name: str, req: RenameRequest) -> dict:
    _rename_profile(TELEOP_PROFILES_DIR, name, req.new_name)
    return {"status": "renamed", "old_name": name, "new_name": req.new_name}


# ============================================================================
# Camera detection and preview
# ============================================================================


def _detect_and_open_cameras() -> list[dict]:
    """Detect cameras and open them for live preview. Runs in thread pool.

    RealSense cameras are detected and opened first via pyrealsense2.
    This claims their V4L2 device nodes so that the subsequent OpenCV
    scan naturally skips them (they appear as busy / unopenable).
    """
    global _preview_cameras, _preview_camera_info
    _close_preview_cameras()

    from lerobot.cameras.configs import ColorMode

    all_cameras: list[dict] = []

    # --- Phase 1: detect + open RealSense cameras first ---
    # This claims /dev/video* nodes owned by RealSense so OpenCV skips them.
    try:
        from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

        realsense_cams = RealSenseCamera.find_cameras()
        logger.info(f"Found {len(realsense_cams)} RealSense camera(s)")

        for cam_info in realsense_cams:
            cam_id = cam_info.get("id")
            # `camera` holds an open handle until ownership is transferred to
            # `_preview_cameras`. The finally clause disconnects if anything
            # between connect() and the successful append raises — otherwise
            # the V4L2 / librealsense FD would leak (TODO 2026-05-11 root cause).
            camera = None
            try:
                config = RealSenseCameraConfig(
                    serial_number_or_name=str(cam_id),
                    color_mode=ColorMode.BGR,
                )
                camera = RealSenseCamera(config)
                camera.connect(warmup=True)
                _preview_cameras.append(camera)
                _preview_camera_info.append(cam_info)
                all_cameras.append(cam_info)
                camera = None  # ownership transferred
                logger.info(f"Opened preview camera: RealSense {cam_id}")
            except Exception as e:
                logger.warning(f"Failed to open RealSense {cam_id}: {e}")
                all_cameras.append(cam_info)
            finally:
                if camera is not None:
                    with contextlib.suppress(Exception):
                        camera.disconnect()
    except Exception as e:
        logger.warning(f"RealSense camera detection failed: {e}")

    # --- Phase 2: detect + open OpenCV cameras ---
    # RealSense V4L2 nodes are now busy, so find_cameras() skips most.
    # We also explicitly filter any remaining RealSense nodes via sysfs.
    try:
        from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

        opencv_cams = OpenCVCamera.find_cameras()
        logger.info(f"Found {len(opencv_cams)} OpenCV camera(s)")

        for cam_info in opencv_cams:
            cam_id = cam_info.get("id")
            # On Linux, skip V4L2 nodes that belong to RealSense
            if isinstance(cam_id, str) and cam_id.startswith("/dev/video"):
                devname = cam_id.split("/")[-1]  # e.g. "video6"
                sysfs_name = Path(f"/sys/class/video4linux/{devname}/name")
                if sysfs_name.exists():
                    hw_name = sysfs_name.read_text().strip()
                    if "RealSense" in hw_name:
                        logger.debug(f"Skipping {cam_id} (RealSense V4L2 node: {hw_name})")
                        continue
            # Same ownership-transfer pattern as the RealSense branch above —
            # `camera` is disconnected by the finally clause unless registered.
            camera = None
            try:
                config = OpenCVCameraConfig(
                    index_or_path=cam_id,
                    color_mode=ColorMode.BGR,
                )
                camera = OpenCVCamera(config)
                camera.connect(warmup=True)
                _preview_cameras.append(camera)
                _preview_camera_info.append(cam_info)
                all_cameras.append(cam_info)
                camera = None  # ownership transferred
                logger.info(f"Opened preview camera: OpenCV {cam_id}")
            except Exception as e:
                logger.warning(f"Failed to open OpenCV {cam_id}: {e}")
            finally:
                if camera is not None:
                    with contextlib.suppress(Exception):
                        camera.disconnect()
    except Exception as e:
        logger.warning(f"OpenCV camera detection failed: {e}")

    return all_cameras


def _close_preview_cameras() -> None:
    """Disconnect all preview cameras."""
    global _preview_cameras, _preview_camera_info
    for cam in _preview_cameras:
        with contextlib.suppress(Exception):
            cam.disconnect()
    _preview_cameras.clear()
    _preview_camera_info.clear()


def cleanup_in_process_resources() -> None:
    """Disconnect every in-process hardware resource the robot router holds.

    Called from the server's shutdown hook so cameras and any orphaned
    rest-position / safe-trajectory recording robot are released cleanly
    (motors detorqued, serial ports closed, V4L2 nodes freed) instead of
    relying on interpreter shutdown to drop the FDs.

    Defensive on every step — if disconnect raises we still null out the
    module global so a retry doesn't double-free.
    """
    global _rest_recording_robot, _trajectory_recording_robot, _trajectory_recorder

    _close_preview_cameras()

    if _trajectory_recorder is not None:
        with contextlib.suppress(Exception):
            _trajectory_recorder.stop()
        _trajectory_recorder = None

    if _trajectory_recording_robot is not None:
        with contextlib.suppress(Exception):
            _trajectory_recording_robot.disconnect()
        _trajectory_recording_robot = None

    if _rest_recording_robot is not None:
        with contextlib.suppress(Exception):
            _rest_recording_robot.disconnect()
        _rest_recording_robot = None


@router.post("/detect-cameras")
async def detect_cameras() -> list[dict]:
    """Detect available cameras and open them for preview."""
    loop = asyncio.get_event_loop()
    cameras = await loop.run_in_executor(None, _detect_and_open_cameras)
    return cameras


@router.get("/camera-frame/{index}")
async def get_camera_frame(index: int) -> Response:
    """Return a JPEG frame from a preview camera."""
    if index < 0 or index >= len(_preview_cameras):
        raise HTTPException(404, "Camera index out of range")
    camera = _preview_cameras[index]
    try:
        frame = camera.async_read()
        # Camera is opened with BGR color mode, so no conversion needed
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return Response(content=jpeg.tobytes(), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(500, f"Failed to read camera frame: {e}") from e


@router.post("/stop-cameras")
async def stop_cameras() -> dict:
    """Disconnect all preview cameras."""
    await asyncio.get_event_loop().run_in_executor(None, _close_preview_cameras)
    return {"status": "ok"}


# ============================================================================
# Port scanning and arm identification
# ============================================================================


@router.get("/ports")
async def scan_ports() -> list[dict]:
    """Scan for USB serial ports (ttyACM*, ttyUSB* on Linux).

    Only shows USB serial adapters, not legacy serial ports (ttyS*),
    virtual terminals (tty0-63), or kernel consoles (ttyprintk).
    Uses pyserial when available for richer device metadata.
    """
    ports = []
    try:
        from serial.tools import list_ports

        for port_info in sorted(list_ports.comports(), key=lambda p: p.device):
            # Filter to USB serial ports only
            dev = port_info.device
            devname = Path(dev).name
            if (
                not (devname.startswith("ttyACM") or devname.startswith("ttyUSB"))
                and platform.system() == "Linux"
            ):
                continue  # Skip non-USB ports on Linux
            ports.append(
                {
                    "path": dev,
                    "name": port_info.description or devname,
                    "manufacturer": port_info.manufacturer or "",
                    "vid_pid": f"{port_info.vid:04x}:{port_info.pid:04x}" if port_info.vid else "",
                }
            )
    except ImportError:
        # Fallback: glob for USB serial devices
        if platform.system() == "Linux":
            for pattern in ["ttyACM*", "ttyUSB*"]:
                for p in sorted(Path("/dev").glob(pattern)):
                    ports.append({"path": str(p), "name": p.name})
        else:
            logger.warning("pyserial not installed, cannot scan ports")
    return ports


def _wiggle_shoulder(port: str) -> dict:
    """Wiggle the shoulder motor on the given port. Runs in thread pool."""
    try:
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus
    except ImportError:
        return {"status": "error", "message": "lerobot motor modules not available"}

    motors = {
        "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
    }

    bus = None
    try:
        bus = FeetechMotorsBus(port=port, motors=motors)
        bus.connect()

        current_pos = bus.read("Present_Position", "shoulder_pan", normalize=False)
        ticks = 200

        bus.write("Goal_Position", "shoulder_pan", current_pos + ticks, normalize=False)
        time.sleep(0.5)
        bus.write("Goal_Position", "shoulder_pan", current_pos - ticks, normalize=False)
        time.sleep(0.5)
        bus.write("Goal_Position", "shoulder_pan", current_pos, normalize=False)
        time.sleep(0.3)

        return {"status": "ok", "port": port}
    except Exception as e:
        return {"status": "error", "port": port, "message": str(e)}
    finally:
        if bus:
            with contextlib.suppress(Exception):
                bus.disconnect()


def _collect_all_port_assignments() -> list[dict]:
    """Read all saved profiles and extract port field values.

    Returns a flat list of {port, profile_name, profile_kind, field_name}.
    """
    from lerobot.robots.config import RobotConfig
    from lerobot.teleoperators.config import TeleoperatorConfig

    _ensure_configs_loaded()

    assignments = []

    for profiles_dir, kind, base_cls in [
        (ROBOT_PROFILES_DIR, "robot", RobotConfig),
        (TELEOP_PROFILES_DIR, "teleop", TeleoperatorConfig),
    ]:
        _ensure_dir(profiles_dir)
        for f in profiles_dir.glob("*.json"):
            if not _is_profile_file(f):
                continue
            try:
                data = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            profile_name = data.get("name", f.stem)
            profile_type = data.get("type")
            fields_data = data.get("fields", {})

            # Find which fields are port fields from the schema
            choices = base_cls.get_known_choices()
            config_cls = choices.get(profile_type)
            if not config_cls:
                continue
            for field in dataclasses.fields(config_cls):
                if field.name in _SKIP_FIELDS:
                    continue
                if "port" in field.name and "str" in _stringify_type(field.type).lower():
                    port_value = fields_data.get(field.name)
                    if port_value and isinstance(port_value, str) and port_value.strip():
                        assignments.append(
                            {
                                "port": port_value.strip(),
                                "profile_name": profile_name,
                                "profile_kind": kind,
                                "field_name": field.name,
                            }
                        )

    return assignments


@router.get("/all-port-assignments")
async def get_all_port_assignments() -> list[dict]:
    """Return all port assignments across all saved profiles."""
    return _collect_all_port_assignments()


class IdentifyArmRequest(BaseModel):
    port: str


@router.post("/open-in-files")
async def open_in_file_manager(body: dict) -> dict:
    """Open a profile directory in the system file manager.

    The subprocess spawn happens in the default executor so a slow
    fork/exec (heavy desktop session, many open FDs) cannot stall the
    FastAPI event loop and block other in-flight requests.
    """
    import subprocess as _subprocess

    kind = body.get("kind", "robot")
    profiles_dir = ROBOT_PROFILES_DIR if kind == "robot" else TELEOP_PROFILES_DIR

    def _spawn() -> None:
        _ensure_dir(profiles_dir)
        _subprocess.Popen(["xdg-open", str(profiles_dir)])

    try:
        await asyncio.get_event_loop().run_in_executor(None, _spawn)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="xdg-open not found") from None

    return {"status": "ok"}


@router.post("/identify-arm")
async def identify_arm(request: IdentifyArmRequest) -> dict:
    """Wiggle the shoulder motor on the given port to identify which arm it is."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _wiggle_shoulder, request.port)
    return result


# ============================================================================
# Rest position
# ============================================================================


def _make_robot_from_profile(profile: dict):
    """Instantiate a Robot from GUI profile data (no subprocess).

    Builds the correct RobotConfig via draccus type dispatch, skipping cameras
    (not needed for motor-only operations like rest position).
    """
    from lerobot.robots.config import RobotConfig
    from lerobot.robots.utils import make_robot_from_config

    _ensure_configs_loaded()

    config_dict = {"type": profile["type"]}
    config_dict.update(profile.get("fields", {}))

    import draccus

    config = draccus.decode(RobotConfig, config_dict)
    return make_robot_from_config(config)


def _disable_torque(device) -> None:
    """Disable torque on all motors regardless of device type."""
    if hasattr(device, "bus"):
        device.bus.disable_torque()
    elif hasattr(device, "left_arm") and hasattr(device, "right_arm"):
        device.left_arm.bus.disable_torque()
        device.right_arm.bus.disable_torque()
    else:
        raise RuntimeError(f"Cannot disable torque: unsupported device type {type(device).__name__}")


def _do_start_rest_recording(profile: dict) -> dict:
    """Connect to robot and disable torque so user can move it. Runs in thread pool."""
    global _rest_recording_robot

    # Clean up any previous recording session
    _do_cancel_rest_recording()

    robot = _make_robot_from_profile(profile)
    try:
        robot.connect()
        _disable_torque(robot)
        _rest_recording_robot = robot
        return {"status": "ok"}
    except Exception as e:
        logger.exception("Failed to start rest position recording")
        try:
            if robot.is_connected:
                robot.disconnect()
        except Exception:
            pass
        return {"status": "error", "message": str(e)}


def _do_finish_rest_recording() -> dict:
    """Read current positions from the held robot, then disconnect. Runs in thread pool."""
    global _rest_recording_robot
    from lerobot.robots.rest_position import record_rest_position

    robot = _rest_recording_robot
    if robot is None:
        return {"status": "error", "message": "No recording session active"}

    try:
        rest_pos = record_rest_position(robot)
        return {"status": "ok", "rest_position": rest_pos}
    except Exception as e:
        logger.exception("Failed to finish rest position recording")
        return {"status": "error", "message": str(e)}
    finally:
        _rest_recording_robot = None
        try:
            if robot.is_connected:
                robot.disconnect()
        except Exception:
            pass


def _do_cancel_rest_recording() -> dict:
    """Disconnect the held robot without reading positions. Runs in thread pool."""
    global _rest_recording_robot

    robot = _rest_recording_robot
    _rest_recording_robot = None
    if robot is None:
        return {"status": "ok"}
    try:
        if robot.is_connected:
            robot.disconnect()
    except Exception:
        pass
    return {"status": "ok"}


def _do_move_to_rest_position(profile: dict, rest_position: dict, duration_s: float) -> dict:
    """Connect to robot, interpolate to rest, disconnect. Runs in thread pool."""
    from lerobot.robots.rest_position import move_to_rest_position

    robot = _make_robot_from_profile(profile)
    try:
        robot.connect()
        move_to_rest_position(robot, rest_position, duration_s=duration_s)
        _disable_torque(robot)
        return {"status": "ok"}
    except Exception as e:
        logger.exception("Failed to move to rest position")
        return {"status": "error", "message": str(e)}
    finally:
        try:
            if robot.is_connected:
                robot.disconnect()
        except Exception:
            pass


class RestPositionRecordRequest(BaseModel):
    robot: dict[str, Any]


class RestPositionMoveRequest(BaseModel):
    robot: dict[str, Any]
    rest_position: dict[str, float]
    duration_s: float = 3.0


@router.post("/start-rest-recording")
async def start_rest_recording_endpoint(req: RestPositionRecordRequest) -> dict:
    """Connect to robot and disable torque so user can move it to rest pose."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _do_start_rest_recording, req.robot)
    if result["status"] == "error":
        raise HTTPException(500, result["message"])
    return result


@router.post("/finish-rest-recording")
async def finish_rest_recording_endpoint() -> dict:
    """Read positions from the held robot connection and disconnect."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _do_finish_rest_recording)
    if result["status"] == "error":
        raise HTTPException(500, result["message"])
    return result


@router.post("/cancel-rest-recording")
async def cancel_rest_recording_endpoint() -> dict:
    """Cancel recording session — disconnect without reading."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _do_cancel_rest_recording)
    return result


@router.post("/move-to-rest-position")
async def move_to_rest_position_endpoint(req: RestPositionMoveRequest) -> dict:
    """Connect to robot, smoothly move to rest position, disconnect."""
    if not req.rest_position:
        raise HTTPException(400, "rest_position is empty")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, _do_move_to_rest_position, req.robot, req.rest_position, req.duration_s
    )
    if result["status"] == "error":
        raise HTTPException(500, result["message"])
    return result


# ============================================================================
# Safe trajectory recording / replay
# ============================================================================


def _trajectory_path(name: str) -> Path:
    """Sibling file next to the robot profile: ``<name>.trajectory.json``."""
    return ROBOT_PROFILES_DIR / f"{name}.trajectory.json"


def _read_trajectory(name: str) -> dict | None:
    path = _trajectory_path(name)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read trajectory %s: %s", path, e)
        return None


def _write_trajectory(name: str, trajectory: dict) -> Path:
    _ensure_dir(ROBOT_PROFILES_DIR)
    path = _trajectory_path(name)
    # Atomic write: a recorded safe-trajectory is often hundreds of KB
    # to several MB; a kill or panic between the file truncate and the
    # write completing would leave the user with an empty / half-written
    # JSON on the next session, silently losing the captured trajectory.
    import os

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(trajectory))
    os.replace(tmp, path)
    return path


def _trajectory_meta(name: str) -> dict:
    """Lightweight summary for the GUI: just enough to render the UI without
    paying the full trajectory load."""
    traj = _read_trajectory(name)
    if traj is None:
        return {"exists": False}
    timestamps = traj.get("timestamps") or []
    return {
        "exists": True,
        "fps": traj.get("fps"),
        "frame_count": len(timestamps),
        "duration_s": timestamps[-1] if timestamps else 0.0,
    }


def _do_start_trajectory_recording(profile: dict, rest_position: dict, fps: int) -> dict:
    """Connect, move to rest, disable torque, start sampler thread.

    The rest-position prerequisite is enforced by the caller — we receive
    it as an arg so we can interpolate to a known starting pose before the
    user begins hand-guiding the arm. Without that, replay's first frame
    would be wherever the user happened to put the arm at Record time, and
    we'd lose the "known starting point" guarantee.
    """
    global _trajectory_recording_robot, _trajectory_recorder
    from lerobot.robots.rest_position import move_to_rest_position
    from lerobot.robots.safe_trajectory import TrajectoryRecorder

    _do_cancel_trajectory_recording()  # clean up any leftover session

    robot = _make_robot_from_profile(profile)
    try:
        robot.connect()
        move_to_rest_position(robot, rest_position, duration_s=3.0)
        _disable_torque(robot)
        recorder = TrajectoryRecorder(robot, fps=fps)
        recorder.start()
        _trajectory_recording_robot = robot
        _trajectory_recorder = recorder
        return {"status": "ok"}
    except Exception as e:
        logger.exception("Failed to start trajectory recording")
        try:
            if robot.is_connected:
                robot.disconnect()
        except Exception:
            pass
        return {"status": "error", "message": str(e)}


def _do_stop_trajectory_recording(profile_name: str) -> dict:
    """Stop the sampler, persist the trajectory, disconnect.

    The profile name is needed so we know where to write the sibling
    trajectory file. The FE passes the same profile it used to start the
    session.
    """
    global _trajectory_recording_robot, _trajectory_recorder
    robot = _trajectory_recording_robot
    recorder = _trajectory_recorder
    if recorder is None or robot is None:
        return {"status": "error", "message": "No recording session active"}
    try:
        trajectory = recorder.stop()
        # Backfill robot_id from the profile name so the file is
        # self-describing even if the JSON is moved.
        trajectory["robot_id"] = profile_name
        path = _write_trajectory(profile_name, trajectory)
        timestamps = trajectory["timestamps"]
        return {
            "status": "ok",
            "path": str(path),
            "frame_count": len(timestamps),
            "duration_s": timestamps[-1] if timestamps else 0.0,
        }
    except Exception as e:
        logger.exception("Failed to stop trajectory recording")
        return {"status": "error", "message": str(e)}
    finally:
        _trajectory_recorder = None
        _trajectory_recording_robot = None
        try:
            if robot.is_connected:
                robot.disconnect()
        except Exception:
            pass


def _do_cancel_trajectory_recording() -> dict:
    """Stop the sampler and disconnect, discarding any captured frames."""
    global _trajectory_recording_robot, _trajectory_recorder
    recorder = _trajectory_recorder
    robot = _trajectory_recording_robot
    _trajectory_recorder = None
    _trajectory_recording_robot = None
    if recorder is not None:
        with contextlib.suppress(Exception):
            recorder.cancel()
    if robot is not None:
        with contextlib.suppress(Exception):
            if robot.is_connected:
                robot.disconnect()
    return {"status": "ok"}


def _do_replay_trajectory(profile: dict, rest_position: dict, profile_name: str) -> dict:
    """Connect, move to rest, replay the saved trajectory, disconnect."""
    from lerobot.robots.rest_position import move_to_rest_position
    from lerobot.robots.safe_trajectory import replay_trajectory

    trajectory = _read_trajectory(profile_name)
    if trajectory is None:
        return {"status": "error", "message": f"No safe trajectory recorded for profile {profile_name!r}"}

    robot = _make_robot_from_profile(profile)
    try:
        robot.connect()
        move_to_rest_position(robot, rest_position, duration_s=3.0)
        # Short ramp at the start of replay handles any small drift
        # between rest_position and trajectory[0] (typically <1°).
        replay_trajectory(robot, trajectory, ramp_to_start_s=0.5)
        # Park the arm back at rest before disconnecting: the recorded
        # trajectory's endpoint isn't guaranteed to be a stable pose
        # under gravity, and we don't want the arm to drift / sag when
        # torque is released on disconnect.
        move_to_rest_position(robot, rest_position, duration_s=2.0)
        return {"status": "ok", "frame_count": len(trajectory["timestamps"])}
    except Exception as e:
        logger.exception("Trajectory replay failed")
        return {"status": "error", "message": str(e)}
    finally:
        try:
            if robot.is_connected:
                robot.disconnect()
        except Exception:
            pass


class TrajectoryRecordRequest(BaseModel):
    robot: dict[str, Any]
    rest_position: dict[str, float]
    fps: int = 30


class TrajectoryStopRequest(BaseModel):
    profile_name: str


class TrajectoryReplayRequest(BaseModel):
    robot: dict[str, Any]
    rest_position: dict[str, float]
    profile_name: str


@router.get("/trajectory-meta/{name}")
async def trajectory_meta_endpoint(name: str) -> dict:
    """Lightweight metadata about the saved trajectory for *name* (if any)."""
    return _trajectory_meta(name)


@router.post("/start-trajectory-recording")
async def start_trajectory_recording_endpoint(req: TrajectoryRecordRequest) -> dict:
    if not req.rest_position:
        raise HTTPException(400, "rest_position is required — record a rest pose first")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, _do_start_trajectory_recording, req.robot, req.rest_position, req.fps
    )
    if result["status"] == "error":
        raise HTTPException(500, result["message"])
    return result


@router.post("/stop-trajectory-recording")
async def stop_trajectory_recording_endpoint(req: TrajectoryStopRequest) -> dict:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _do_stop_trajectory_recording, req.profile_name)
    if result["status"] == "error":
        raise HTTPException(500, result["message"])
    return result


@router.post("/cancel-trajectory-recording")
async def cancel_trajectory_recording_endpoint() -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _do_cancel_trajectory_recording)


@router.post("/replay-trajectory")
async def replay_trajectory_endpoint(req: TrajectoryReplayRequest) -> dict:
    if not req.rest_position:
        raise HTTPException(400, "rest_position is required — set one before replay")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, _do_replay_trajectory, req.robot, req.rest_position, req.profile_name
    )
    if result["status"] == "error":
        raise HTTPException(500, result["message"])
    return result


@router.delete("/trajectory/{name}")
async def delete_trajectory_endpoint(name: str) -> dict:
    path = _trajectory_path(name)
    if path.exists():
        # safe-destruct: user clicked Clear in the Safe Trajectory panel
        path.unlink()
    return {"status": "ok"}


def _do_recover(profile: dict) -> dict:
    """Build the robot from *profile* and run non-physical recovery.

    Pre: no other process / handler is currently holding the robot's port(s).
    Recovery opens each bus directly, bypassing the strict handshake that the
    normal ``connect()`` path uses, so it must own the serial port.
    """
    from lerobot.motors.recovery import recover_robot

    try:
        robot = _make_robot_from_profile(profile)
    except Exception as e:
        logger.exception("Failed to build robot for recover")
        return {"status": "error", "message": f"could not build robot: {e}"}

    try:
        reports = recover_robot(robot)
        return {
            "status": "ok",
            "reports": [r.to_dict() for r in reports],
        }
    except Exception as e:
        logger.exception("Recover failed")
        return {"status": "error", "message": str(e)}


class RecoverRequest(BaseModel):
    robot: dict[str, Any]


@router.post("/recover")
async def recover_endpoint(req: RecoverRequest) -> dict:
    """Attempt non-physical recovery of a wedged robot.

    Opens each bus without the strict handshake, pings every expected motor,
    disables torque on responders, and applies a driver-specific trick
    (Feetech: clear overload latch via direct ``Torque_Enable=0`` write) to
    non-responders. Returns one report per bus.

    Pre: caller must ensure no active teleop / record / replay session is
    holding the robot's serial port(s).
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _do_recover, req.robot)
    if result["status"] == "error":
        raise HTTPException(500, result["message"])
    return result
