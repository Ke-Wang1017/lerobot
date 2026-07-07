"""Model tab API: browse training outputs, inspect checkpoints, launch training."""

from __future__ import annotations

import contextlib
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from lerobot.gui.training.runs import RUNS_DIR as _RUNS_DIR

if TYPE_CHECKING:
    from lerobot.gui.state import AppState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/models", tags=["models"])

_app_state: AppState = None  # type: ignore

SOURCES_FILE = Path.home() / ".config" / "lerobot" / "model_sources.json"


def set_app_state(state: AppState) -> None:
    global _app_state
    _app_state = state


# ============================================================================
# Source folder persistence
# ============================================================================

_DEFAULT_SOURCE = str(Path.cwd() / "outputs")
_CONVERTED_SOURCE = str(Path.home() / ".cache" / "lerobot" / "converted")
# GUI-managed training runs (lerobot.gui.training.orchestrator) land under
# RUNS_DIR/<run_id>/output/checkpoints/<step>/pretrained_model/. Track the
# orchestrator's ACTUAL runs dir (honours LEROBOT_RUNS_DIR) rather than a
# hardcoded path, or a custom runs dir leaves trained models unscanned.
# Auto-register so they appear in the Models tab — closes the loop (DESIGN.md C3).
_GUI_RUNS_SOURCE = str(_RUNS_DIR)


def _read_sources() -> list[dict]:
    defaults = [
        {"path": _DEFAULT_SOURCE, "removable": False, "expanded": True},
    ]
    # Add converted checkpoints source if it exists
    if Path(_CONVERTED_SOURCE).is_dir():
        defaults.append({"path": _CONVERTED_SOURCE, "removable": False, "expanded": True})
    # GUI-managed training runs (auto-registered so newly-trained models
    # appear in the Models tab without the user having to add the dir).
    if Path(_GUI_RUNS_SOURCE).is_dir():
        defaults.append({"path": _GUI_RUNS_SOURCE, "removable": False, "expanded": True})

    if not SOURCES_FILE.exists():
        return defaults
    try:
        data = json.loads(SOURCES_FILE.read_text())
        sources = data.get("sources", [])
        # Ensure defaults are present
        for d in defaults:
            if not any(s["path"] == d["path"] for s in sources):
                sources.insert(0, d)
        return sources
    except Exception:
        logger.warning("Failed to read model sources, using defaults", exc_info=True)
        return defaults


def _write_sources(sources: list[dict]) -> None:
    """Persist model source folders to config (atomic write — same rationale
    as gui.api.datasets._write_sources / _write_opened).
    """
    import os

    SOURCES_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {"version": 1, "sources": sources}
    tmp = SOURCES_FILE.with_suffix(SOURCES_FILE.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, SOURCES_FILE)


# ============================================================================
# Model scanning
# ============================================================================


def _count_safetensor_params(filepath: Path) -> int | None:
    """Count total parameters from a safetensors file header (no weight loading)."""
    import struct

    try:
        with open(filepath, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        total = 0
        for name, info in header.items():
            if name == "__metadata__":
                continue
            params = 1
            for s in info["shape"]:
                params *= s
            total += params
        return total
    except Exception:
        return None


def _read_checkpoint_meta(ckpt_dir: Path) -> dict | None:
    """Read metadata from a single checkpoint directory."""
    pretrained = ckpt_dir / "pretrained_model"
    config_file = pretrained / "config.json"
    if not config_file.is_file():
        return None

    try:
        config = json.loads(config_file.read_text())
    except Exception:
        return None

    # Read training step
    step = None
    step_file = ckpt_dir / "training_state" / "training_step.json"
    if step_file.is_file():
        with contextlib.suppress(Exception):
            step = json.loads(step_file.read_text()).get("step")

    # Model file size and parameter count
    model_file = pretrained / "model.safetensors"
    model_size = model_file.stat().st_size if model_file.is_file() else 0
    num_params = _count_safetensor_params(model_file) if model_file.is_file() else None

    has_training_state = (ckpt_dir / "training_state").is_dir()

    return {
        "step": step,
        "model_size_bytes": model_size,
        "num_parameters": num_params,
        "has_training_state": has_training_state,
        "policy_type": config.get("type", ""),
        # The path lerobot-record / lerobot-eval / etc. feed into
        # ``--policy.path``. Synthesised server-side so the JS doesn't have
        # to know about layout (legacy ``<run>/checkpoints/<step>/`` vs
        # GUI-managed ``<run>/output/checkpoints/<step>/``). Computed
        # inside this function — after the config.json existence check —
        # so a checkpoint with a missing pretrained_model never gets a
        # policy_path that points at nothing.
        "policy_path": str(pretrained),
    }


def _is_step_dir(name: str) -> bool:
    """A checkpoint-step directory name. Accepts every convention real trainers write: a bare
    numeric step (lerobot-train, e.g. ``000005`` / ``50000``) and ``checkpoint-<N>`` (HF Trainer
    and the HVLA S1-standalone trainer, e.g. ``checkpoint-50000``). The ``last`` symlink and
    backups (``checkpoint-50000-backup``) are deliberately not steps."""
    return name.isdigit() or bool(re.fullmatch(r"checkpoint-\d+", name))


def _dir_has_step_subdirs(d: Path) -> bool:
    """True iff ``d`` exists AND contains at least one checkpoint-step subdir (see
    :func:`_is_step_dir`). Filters out empty/placeholder dirs that early code paths may have
    pre-created."""
    if not d.is_dir():
        return False
    return any(child.is_dir() and _is_step_dir(child.name) for child in d.iterdir())


def _scan_training_run(run_dir: Path) -> dict | None:
    """Scan a single training run directory for checkpoints.

    Recognizes two layouts (preferring whichever actually has step subdirs):
      - Standard lerobot-train output: ``<run_dir>/checkpoints/<NNNNNN>/...``
      - GUI-managed (docker recipe writes here): ``<run_dir>/output/checkpoints/<NNNNNN>/...``
        The extra ``output/`` level is because the GUI orchestrator bind-mounts
        the run_dir into the container and lerobot-train writes to a subdir
        (so the bind-mount target doesn't pre-exist and the FileExistsError
        validator passes). See scripts/training/README.md "2026-06-07 gotchas".
    """
    ckpts_dir = run_dir / "checkpoints"
    if not _dir_has_step_subdirs(ckpts_dir):
        # GUI-managed layout fallback
        ckpts_dir = run_dir / "output" / "checkpoints"
        if not _dir_has_step_subdirs(ckpts_dir):
            return None

    # Find checkpoint subdirs (numeric names or 'last')
    checkpoints = []
    last_target = None

    # Resolve 'last' symlink
    last_link = ckpts_dir / "last"
    if last_link.exists():
        with contextlib.suppress(Exception):
            last_target = last_link.resolve().name

    for child in sorted(ckpts_dir.iterdir()):
        if not child.is_dir() or child.name == "last":
            continue
        meta = _read_checkpoint_meta(child)
        if meta:
            meta["name"] = child.name
            meta["path"] = str(child)
            meta["is_last"] = child.name == last_target
            checkpoints.append(meta)

    if not checkpoints:
        return None

    # Use the latest checkpoint for run-level metadata
    latest = next((c for c in checkpoints if c["is_last"]), checkpoints[-1])

    # Read train_config.json from latest checkpoint
    train_config = _read_train_config(Path(latest["path"]))

    # The run dir is named by random run_id; prefer the human name the user
    # gave the run (stored on the GUI's run.json) so the Models tab is legible.
    # Falls back to the dir name for converted / non-GUI checkpoints.
    run_meta = _read_run_meta(run_dir)

    return {
        "name": run_meta.get("recipe_name") or run_dir.name,
        "run_id": run_dir.name,
        "created_at": run_meta.get("created_at"),
        "path": str(run_dir),
        "policy_type": latest["policy_type"],
        "dataset": train_config.get("dataset", {}).get("repo_id", "") if train_config else "",
        "dataset_root": train_config.get("dataset", {}).get("root") if train_config else None,
        "current_step": latest.get("step"),
        "total_steps": train_config.get("steps") if train_config else None,
        "batch_size": train_config.get("batch_size") if train_config else None,
        "model_size_bytes": latest["model_size_bytes"],
        "num_parameters": latest.get("num_parameters"),
        "num_checkpoints": len(checkpoints),
        "checkpoints": checkpoints,
        # Default path callers feed into ``--policy.path`` when they want
        # "the obvious choice" for this run (e.g. the "Test on robot"
        # button). Points at the RESOLVED last-checkpoint dir, not the
        # `last` symlink — symlinks rot when the run dir moves. None iff
        # no readable checkpoint was found (already filtered above, so
        # in practice always non-None for this branch).
        "default_policy_path": latest.get("policy_path"),
        "wandb_run_id": (train_config.get("wandb", {}) or {}).get("run_id") if train_config else None,
        "wandb_project": (train_config.get("wandb", {}) or {}).get("project") if train_config else None,
    }


def _read_run_meta(run_dir: Path) -> dict:
    """Run-level metadata from the GUI's ``run.json`` (the user-given name,
    created_at, …). Empty dict for converted / non-GUI checkpoint dirs that
    have no run.json."""
    f = run_dir / "run.json"
    if not f.is_file():
        return {}
    try:
        return json.loads(f.read_text())
    except Exception:
        return {}


def _read_train_config(ckpt_dir: Path) -> dict | None:
    """Read train_config.json from a checkpoint directory."""
    tc_file = ckpt_dir / "pretrained_model" / "train_config.json"
    if not tc_file.is_file():
        return None
    try:
        return json.loads(tc_file.read_text())
    except Exception:
        return None


def _read_flat_checkpoint(ckpt_dir: Path) -> dict | None:
    """Read metadata from a flat checkpoint directory (model.safetensors + config.json).

    Used for converted checkpoints that don't have the standard
    checkpoints/pretrained_model/ structure (e.g. HVLA S2 VLM).
    """
    config_file = ckpt_dir / "config.json"
    model_file = ckpt_dir / "model.safetensors"
    if not config_file.is_file() or not model_file.is_file():
        return None

    try:
        config = json.loads(config_file.read_text())
    except Exception:
        return None

    model_size = model_file.stat().st_size
    num_params = _count_safetensor_params(model_file)

    return {
        "name": ckpt_dir.name,
        "path": str(ckpt_dir),
        "policy_type": config.get("type", "unknown"),
        "dataset": "",
        "dataset_root": None,
        "current_step": None,
        "total_steps": None,
        "batch_size": None,
        "model_size_bytes": model_size,
        "num_parameters": num_params,
        "num_checkpoints": 1,
        "checkpoints": [
            {
                "name": ckpt_dir.name,
                "path": str(ckpt_dir),
                "step": None,
                "model_size_bytes": model_size,
                "num_parameters": num_params,
                "has_training_state": False,
                "is_last": True,
                "policy_type": config.get("type", "unknown"),
                # Flat layout has no nested pretrained_model/; weights live
                # directly in this dir, so policy_path == path. Lets the
                # frontend treat flat + standard layouts uniformly (just
                # use the field; no client-side heuristic needed).
                "policy_path": str(ckpt_dir),
            }
        ],
        "default_policy_path": str(ckpt_dir),
        "wandb_run_id": None,
        "wandb_project": None,
    }


def _find_highest_ep_snapshot(run_dir: Path) -> Path | None:
    """Return the ep_N/ subdir with the largest N inside ``run_dir``,
    or None if there are none. Used to surface a single rollback point
    in the GUI rather than enumerating every snapshot."""
    best: tuple[int, Path] | None = None
    try:
        for child in run_dir.iterdir():
            if not child.is_dir() or not child.name.startswith("ep_"):
                continue
            if not (child / "actor.pt").is_file():
                continue
            try:
                n = int(child.name[len("ep_") :])
            except ValueError:
                continue
            if best is None or n > best[0]:
                best = (n, child)
    except (PermissionError, OSError):
        return None
    return best[1] if best else None


def _scan_source(source_path: str, max_depth: int = 2) -> list[dict]:
    """Scan a directory for training runs (dirs containing checkpoints/)."""
    root = Path(source_path)
    if not root.is_dir():
        return []

    found = []
    _scan_recursive(root, root, found, max_depth, 0)
    found.sort(key=lambda d: d["name"])
    return found


def _scan_recursive(base: Path, current: Path, found: list[dict], max_depth: int, depth: int) -> None:
    if depth > max_depth:
        return
    try:
        # Check if this directory is a training run.
        # Standard layout: <dir>/checkpoints/<step>/...
        # GUI-managed (docker recipe): <dir>/output/checkpoints/<step>/...
        has_standard = _dir_has_step_subdirs(current / "checkpoints")
        has_gui = _dir_has_step_subdirs(current / "output" / "checkpoints")
        if has_standard or has_gui:
            run_meta = _scan_training_run(current)
            if run_meta:
                # _scan_training_run already prefers the run's human name
                # (run.json recipe_name). Only fall back to the relative path
                # (which disambiguates nested non-GUI runs) when there was no
                # name — don't clobber the name the user gave the run.
                if run_meta.get("name") == current.name:
                    with contextlib.suppress(ValueError):
                        run_meta["name"] = str(current.relative_to(base))
                found.append(run_meta)
            return  # Don't recurse into training run subdirs

        # Check if this is a flat checkpoint dir (model.safetensors + config.json, no checkpoints/)
        # Used by converted checkpoints (e.g. HVLA S2 VLM)
        if (current / "model.safetensors").is_file() and (current / "config.json").is_file():
            meta = _read_flat_checkpoint(current)
            if meta:
                with contextlib.suppress(ValueError):
                    meta["name"] = str(current.relative_to(base))
                found.append(meta)
            return

        # Check if this is an RLT run dir (latest/actor.pt inside it)
        # Mirrors how training runs are detected by checkpoints/ subdir.
        if (current / "latest" / "actor.pt").is_file():
            # Primary entry: the rolling latest/ checkpoint.
            meta = _read_rlt_checkpoint(current / "latest", base)
            if meta:
                meta["path"] = str(current)  # run dir, not checkpoint subdir
                meta["run_dir"] = str(current)  # output_dir target on resume
                with contextlib.suppress(ValueError):
                    meta["name"] = str(current.relative_to(base))
                found.append(meta)
            # Rollback entry: the highest-numbered ep_N/ snapshot, if any.
            # All snapshots stay on disk for manual recovery, but only the
            # most recent one is surfaced in the dropdown — the only one a
            # user typically wants when latest/ is in doubt. The run_dir
            # field keeps continued training writing to the parent.
            highest_snap = _find_highest_ep_snapshot(current)
            if highest_snap is not None:
                snap_meta = _read_rlt_checkpoint(highest_snap, base)
                if snap_meta:
                    snap_meta["path"] = str(highest_snap)
                    snap_meta["run_dir"] = str(current)
                    with contextlib.suppress(ValueError):
                        snap_meta["name"] = str(highest_snap.relative_to(base))
                    found.append(snap_meta)
            return

        # Recurse
        if depth < max_depth:
            for child in sorted(current.iterdir()):
                if child.is_dir() and not child.name.startswith("."):
                    _scan_recursive(base, child, found, max_depth, depth + 1)
    except PermissionError:
        pass
    except Exception:
        pass


def _read_rlt_checkpoint(ckpt_dir: Path, base: Path) -> dict | None:
    """Read metadata from an RLT checkpoint (actor.pt + optional training_state.pt)."""
    meta = {
        "type": "rlt",
        "path": str(ckpt_dir),
        "policyType": "rlt",
    }
    try:
        meta["name"] = str(ckpt_dir.relative_to(base))
    except ValueError:
        meta["name"] = str(ckpt_dir)

    # Read training state if available
    ts_path = ckpt_dir / "training_state.pt"
    if ts_path.exists():
        try:
            import torch

            ts = torch.load(str(ts_path), weights_only=True, map_location="cpu")
            meta["episode"] = ts.get("episode", 0)
            meta["updates"] = ts.get("total_updates", 0)
            meta["successes"] = sum(ts.get("successes", []))
        except Exception as e:
            logger.warning("Failed to read RLT training state from %s: %s", ts_path, e)

    # Check for RL token encoder in parent or sibling dirs
    # (convention: rlt_token checkpoint is separate from actor checkpoint)
    meta["has_replay_buffer"] = (ckpt_dir / "replay_buffer.pt").exists()
    meta["has_critic"] = (ckpt_dir / "critic.pt").exists()

    return meta


# ============================================================================
# Pydantic models
# ============================================================================


class SourceRequest(BaseModel):
    path: str


class SourceInfo(BaseModel):
    path: str
    removable: bool
    expanded: bool


class ModelSourceEntry(BaseModel):
    name: str
    # The run_id dir name + created_at, so the UI can show the id/date alongside
    # the human name (name is the run's recipe_name for GUI runs). None for
    # converted / non-GUI checkpoints with no run.json.
    run_id: str | None = None
    created_at: float | None = None
    path: str
    policy_type: str
    dataset: str
    current_step: int | None
    total_steps: int | None
    batch_size: int | None
    model_size_bytes: int
    num_parameters: int | None = None
    num_checkpoints: int
    # The path callers feed into ``--policy.path`` by default. Server-emitted
    # so the JS doesn't reconstruct it (and historically got the layout
    # wrong — see ``feat/training-prototype`` commit log). None iff no
    # readable checkpoint exists (RLT entries, corrupt runs, etc.).
    default_policy_path: str | None = None
    wandb_run_id: str | None = None
    wandb_project: str | None = None


class CheckpointInfo(BaseModel):
    name: str
    path: str
    step: int | None
    model_size_bytes: int
    num_parameters: int | None = None
    has_training_state: bool
    is_last: bool
    policy_type: str
    # Per-checkpoint policy path (= ``<ckpt>/pretrained_model`` for the
    # standard layout, == path for flat layouts). Lets the dropdown use
    # the same convention as ``default_policy_path`` when a future PR
    # exposes non-last checkpoints in the picker.
    policy_path: str | None = None


# ============================================================================
# Endpoints
# ============================================================================


@router.get("/sources")
async def list_sources() -> list[SourceInfo]:
    return [SourceInfo(**s) for s in _read_sources()]


@router.post("/sources")
async def add_source(req: SourceRequest) -> SourceInfo:
    path = str(Path(req.path).expanduser().resolve())
    if not Path(path).is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {path}")

    sources = _read_sources()
    if any(s["path"] == path for s in sources):
        raise HTTPException(status_code=409, detail="Source already exists")

    new_source = {"path": path, "removable": True, "expanded": True}
    sources.append(new_source)
    _write_sources(sources)
    logger.info(f"Added model source: {path}")
    return SourceInfo(**new_source)


@router.delete("/sources/{encoded_path:path}")
async def remove_source(encoded_path: str) -> dict[str, str]:
    path = unquote(encoded_path)
    sources = _read_sources()
    source = next((s for s in sources if s["path"] == path), None)
    if not source:
        raise HTTPException(status_code=404, detail=f"Source not found: {path}")
    if not source.get("removable", True):
        raise HTTPException(status_code=400, detail="Cannot remove default source")

    sources = [s for s in sources if s["path"] != path]
    _write_sources(sources)
    logger.info(f"Removed model source: {path}")
    return {"status": "ok"}


@router.put("/sources/{encoded_path:path}/expanded")
async def set_source_expanded(encoded_path: str, expanded: bool = True) -> dict[str, str]:
    path = unquote(encoded_path)
    sources = _read_sources()
    source = next((s for s in sources if s["path"] == path), None)
    if not source:
        raise HTTPException(status_code=404, detail=f"Source not found: {path}")

    source["expanded"] = expanded
    _write_sources(sources)
    return {"status": "ok"}


@router.get("/sources/{encoded_path:path}/models")
async def scan_source(encoded_path: str) -> list[ModelSourceEntry]:
    """Scan a source folder for training runs."""
    import asyncio

    path = unquote(encoded_path)
    sources = _read_sources()
    if not any(s["path"] == path for s in sources):
        raise HTTPException(status_code=404, detail=f"Source not found: {path}")

    loop = asyncio.get_event_loop()
    models = await loop.run_in_executor(None, _scan_source, path)
    # Filter out RLT checkpoints from main browser (they show in HVLA form instead)
    models = [m for m in models if m.get("type") != "rlt"]
    return [ModelSourceEntry(**{k: v for k, v in m.items() if k != "checkpoints"}) for m in models]


@router.get("/rlt-checkpoints")
async def list_rlt_checkpoints() -> list[dict]:
    """List RLT checkpoints across all source directories."""
    import asyncio

    sources = _read_sources()
    all_rlt = []

    loop = asyncio.get_event_loop()
    for s in sources:
        models = await loop.run_in_executor(None, _scan_source, s["path"])
        all_rlt.extend(m for m in models if m.get("type") == "rlt")

    return all_rlt


@router.get("/run/{encoded_path:path}/checkpoints")
async def list_checkpoints(encoded_path: str) -> list[CheckpointInfo]:
    """List checkpoints for a training run."""
    path = unquote(encoded_path)
    run_dir = Path(path)
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Run not found: {path}")

    run_meta = _scan_training_run(run_dir)
    if not run_meta:
        raise HTTPException(status_code=404, detail="No checkpoints found")

    return [CheckpointInfo(**c) for c in run_meta["checkpoints"]]


@router.get("/run/{encoded_path:path}/config")
async def get_train_config(encoded_path: str) -> dict:
    """Return train_config.json for a checkpoint."""
    path = unquote(encoded_path)
    ckpt_dir = Path(path)

    # Try as checkpoint dir first, then as training run (use last checkpoint)
    config = _read_train_config(ckpt_dir)
    if not config:
        last = ckpt_dir / "checkpoints" / "last"
        if last.exists():
            config = _read_train_config(last.resolve())
    if not config:
        raise HTTPException(status_code=404, detail="train_config.json not found")

    return config


@router.post("/open-in-files")
async def open_in_file_manager(body: dict) -> dict:
    """Open a directory in the system file manager.

    Spawns the subprocess in the default executor so a slow fork/exec
    (heavy desktop session, many open FDs) cannot stall the FastAPI
    event loop.
    """
    import asyncio
    import subprocess as _subprocess

    path = body.get("path", "")
    if not path or not Path(path).is_dir():
        raise HTTPException(status_code=400, detail=f"Not a valid directory: {path}")

    def _spawn() -> None:
        _subprocess.Popen(["xdg-open", path])

    try:
        await asyncio.get_event_loop().run_in_executor(None, _spawn)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="xdg-open not found") from None

    return {"status": "ok"}
