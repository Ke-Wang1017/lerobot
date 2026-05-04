# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""REST API endpoints for dataset editing operations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

if TYPE_CHECKING:
    from lerobot.gui.state import AppState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/edits", tags=["edits"])

# Will be set by server.py
_app_state: "AppState" = None  # type: ignore


def set_app_state(state: "AppState") -> None:
    """Set the application state for edit handlers."""
    global _app_state
    _app_state = state


def _require_unlocked(dataset_id: str) -> None:
    """Raise HTTP 423 if the dataset is locked by an in-progress operation."""
    if _app_state.is_locked(dataset_id):
        raise HTTPException(status_code=423, detail="Dataset is busy (operation in progress)")


def _save_edits_for_dataset(dataset_id: str) -> None:
    """Persist pending edits for a dataset to disk."""
    from lerobot.gui.state import save_edits_to_file

    if dataset_id not in _app_state.datasets:
        return

    dataset = _app_state.datasets[dataset_id]
    edits = _app_state.get_edits_for_dataset(dataset_id)
    save_edits_to_file(dataset.root, edits)


class DeleteRequest(BaseModel):
    """Request to mark an episode for deletion."""

    dataset_id: str
    episode_index: int


class TrimRequest(BaseModel):
    """Request to trim an episode."""

    dataset_id: str
    episode_index: int
    start_frame: int
    end_frame: int


class EditInfo(BaseModel):
    """Information about a pending edit."""

    index: int
    edit_type: str
    dataset_id: str
    episode_index: int
    params: dict
    created_at: str


class PendingEditsResponse(BaseModel):
    """Response containing all pending edits."""

    edits: list[EditInfo]
    total: int


@router.get("", response_model=PendingEditsResponse)
async def list_pending_edits(dataset_id: str | None = None):
    """List all pending edits, optionally filtered by dataset."""
    if dataset_id:
        edits = _app_state.get_edits_for_dataset(dataset_id)
    else:
        edits = _app_state.pending_edits

    return PendingEditsResponse(
        edits=[
            EditInfo(
                index=i,
                edit_type=e.edit_type,
                dataset_id=e.dataset_id,
                episode_index=e.episode_index,
                params=e.params,
                created_at=e.created_at.isoformat(),
            )
            for i, e in enumerate(_app_state.pending_edits)
            if dataset_id is None or e.dataset_id == dataset_id
        ],
        total=len(edits),
    )


@router.post("/delete")
async def mark_episode_deleted(request: DeleteRequest):
    """Mark an episode for deletion."""
    from lerobot.gui.state import PendingEdit

    _require_unlocked(request.dataset_id)

    if request.dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_id}")

    dataset = _app_state.datasets[request.dataset_id]
    if request.episode_index < 0 or request.episode_index >= dataset.meta.total_episodes:
        raise HTTPException(status_code=400, detail=f"Invalid episode index: {request.episode_index}")

    # Check if already marked for deletion
    if _app_state.is_episode_deleted(request.dataset_id, request.episode_index):
        raise HTTPException(status_code=400, detail="Episode already marked for deletion")

    edit = PendingEdit(
        edit_type="delete",
        dataset_id=request.dataset_id,
        episode_index=request.episode_index,
    )
    _app_state.add_edit(edit)
    _save_edits_for_dataset(request.dataset_id)

    logger.info(f"Marked episode {request.episode_index} for deletion in {request.dataset_id}")
    return {"status": "ok", "message": f"Episode {request.episode_index} marked for deletion"}


@router.post("/trim")
async def set_episode_trim(request: TrimRequest):
    """Set trim range for an episode."""
    from lerobot.gui.state import PendingEdit

    _require_unlocked(request.dataset_id)

    if request.dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_id}")

    dataset = _app_state.datasets[request.dataset_id]
    if request.episode_index < 0 or request.episode_index >= dataset.meta.total_episodes:
        raise HTTPException(status_code=400, detail=f"Invalid episode index: {request.episode_index}")

    # Get episode length
    episode = dataset.meta.episodes[request.episode_index]
    episode_length = episode["length"]

    if request.start_frame < 0 or request.end_frame > episode_length:
        raise HTTPException(
            status_code=400, detail=f"Invalid trim range: {request.start_frame}-{request.end_frame} (length={episode_length})"
        )

    if request.start_frame >= request.end_frame:
        raise HTTPException(status_code=400, detail="Start frame must be less than end frame")

    # Remove existing trim for this episode
    _app_state.pending_edits = [
        e
        for e in _app_state.pending_edits
        if not (e.dataset_id == request.dataset_id and e.episode_index == request.episode_index and e.edit_type == "trim")
    ]

    # Only add trim if it's not the full range
    if request.start_frame > 0 or request.end_frame < episode_length:
        edit = PendingEdit(
            edit_type="trim",
            dataset_id=request.dataset_id,
            episode_index=request.episode_index,
            params={"start_frame": request.start_frame, "end_frame": request.end_frame},
        )
        _app_state.add_edit(edit)
        logger.info(f"Set trim for episode {request.episode_index}: frames {request.start_frame}-{request.end_frame}")

    _save_edits_for_dataset(request.dataset_id)

    return {
        "status": "ok",
        "message": f"Episode {request.episode_index} will be trimmed to frames {request.start_frame}-{request.end_frame}",
    }


@router.delete("/{edit_index}")
async def remove_edit(edit_index: int):
    """Remove a pending edit by index."""
    if edit_index < 0 or edit_index >= len(_app_state.pending_edits):
        raise HTTPException(status_code=404, detail=f"Edit not found: {edit_index}")

    edit = _app_state.pending_edits[edit_index]
    dataset_id = edit.dataset_id
    _require_unlocked(dataset_id)
    _app_state.remove_edit(edit_index)
    _save_edits_for_dataset(dataset_id)

    logger.info(f"Removed {edit.edit_type} edit for episode {edit.episode_index}")
    return {"status": "ok", "message": "Edit removed"}


@router.post("/discard")
async def discard_edits(dataset_id: str | None = None):
    """Discard all pending edits, optionally for a specific dataset."""
    from lerobot.gui.state import clear_edits_file

    if dataset_id:
        _require_unlocked(dataset_id)
    else:
        # Discarding all — check no dataset is locked
        for ds_id in _app_state.datasets:
            _require_unlocked(ds_id)

    count = len(_app_state.pending_edits if dataset_id is None else _app_state.get_edits_for_dataset(dataset_id))

    # Clear the edits file(s)
    if dataset_id:
        if dataset_id in _app_state.datasets:
            clear_edits_file(_app_state.datasets[dataset_id].root)
    else:
        # Clear all datasets' edit files
        for ds_id, dataset in _app_state.datasets.items():
            clear_edits_file(dataset.root)

    _app_state.clear_edits(dataset_id)

    logger.info(f"Discarded {count} pending edits")
    return {"status": "ok", "message": f"Discarded {count} pending edits"}


@router.post("/apply")
async def apply_edits(dataset_id: str):
    """Apply all pending edits for a dataset to disk."""
    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    lock = _app_state.get_lock(dataset_id)
    if lock.locked():
        raise HTTPException(status_code=423, detail="Dataset is busy (operation in progress)")

    async with lock:
        return await _apply_edits_locked(dataset_id)


async def _apply_edits_locked(dataset_id: str):
    """Apply edits while holding the dataset lock."""
    from pathlib import Path

    from lerobot.datasets.dataset_tools import (
        delete_episodes_virtual,
        reaggregate_dataset_stats,
        trim_episode_virtual,
    )
    from lerobot.datasets.io_utils import load_episodes

    edits = _app_state.get_edits_for_dataset(dataset_id)
    if not edits:
        return {"status": "ok", "message": "No edits to apply", "applied": 0}

    dataset = _app_state.datasets[dataset_id]
    original_root = Path(dataset.root)
    applied = 0
    errors = []
    applied_edit_ids: set[int] = set()

    # Sort edits: apply trims first (they modify in place), then deletes
    trim_edits = [e for e in edits if e.edit_type == "trim"]
    delete_edits = sorted([e for e in edits if e.edit_type == "delete"], key=lambda e: e.episode_index, reverse=True)

    # Pre-flight: would deletes wipe out the entire dataset? Reject early so
    # the user keeps their pending edits and can adjust their selection.
    if delete_edits and len(delete_edits) >= dataset.meta.total_episodes:
        errors.append(
            f"Cannot delete all {dataset.meta.total_episodes} episodes — "
            "at least one episode must remain. Unmark one before saving."
        )
        return {
            "status": "partial",
            "message": errors[0],
            "applied": 0,
            "errors": errors,
            "warnings": [],
        }

    # Apply trims (these modify in-place)
    for edit in trim_edits:
        try:
            start_frame = edit.params["start_frame"]
            end_frame = edit.params["end_frame"]

            # Log original state before trim
            original_length = dataset.meta.episodes[edit.episode_index]["length"]
            logger.info(
                f"TRIM dataset={original_root} episode={edit.episode_index} "
                f"original_length={original_length} keep_frames=[{start_frame}, {end_frame})"
            )

            trim_episode_virtual(
                dataset=dataset,
                episode_index=edit.episode_index,
                start_frame=start_frame,
                end_frame=end_frame,
                recompute_stats=False,
            )

            # Reload metadata to see changes
            dataset.meta.episodes = load_episodes(original_root)
            applied += 1
            applied_edit_ids.add(id(edit))
            logger.info(f"Applied trim to episode {edit.episode_index}: keeping frames {start_frame}-{end_frame - 1}")
        except Exception as e:
            errors.append(f"Trim episode {edit.episode_index}: {e}")
            logger.exception(f"Failed to trim episode {edit.episode_index}")

    # Apply deletes - virtual delete works in-place (no video re-encoding)
    if delete_edits:
        try:
            episode_indices = [e.episode_index for e in delete_edits]

            # Log episode info before deletion
            for ep_idx in episode_indices:
                ep_length = dataset.meta.episodes[ep_idx]["length"]
                logger.info(f"DELETE dataset={original_root} episode={ep_idx} length={ep_length}")

            # Reload metadata to pick up any trim changes
            dataset.meta.episodes = load_episodes(original_root)

            # Virtual delete modifies metadata and parquet in-place (no video re-encoding)
            delete_episodes_virtual(dataset, episode_indices=episode_indices, recompute_stats=False)

            logger.info(f"Deleted episodes {episode_indices} (virtual - no video re-encoding)")

            applied += len(delete_edits)
            applied_edit_ids.update(id(e) for e in delete_edits)
        except Exception as e:
            errors.append(f"Delete episodes: {e}")
            logger.exception("Failed to delete episodes")

    # Re-aggregate stats once after all edits (O(E) instead of O(N*E))
    if applied > 0:
        try:
            reaggregate_dataset_stats(dataset)
            logger.info("Re-aggregated dataset stats after all edits")
        except Exception as e:
            logger.exception(f"Failed to re-aggregate stats: {e}")
            errors.append(f"Stats re-aggregation failed: {e}")

    # Clear only applied edits — keep failed ones pending so the user can
    # adjust and retry without re-marking everything from scratch.
    from lerobot.gui.state import save_edits_to_file

    remaining_edits = [e for e in edits if id(e) not in applied_edit_ids]
    _app_state.pending_edits = (
        [e for e in _app_state.pending_edits if e.dataset_id != dataset_id]
        + remaining_edits
    )
    save_edits_to_file(original_root, remaining_edits)

    # Invalidate all dataset-scoped caches. Edits change episode lengths,
    # so the cumulative-sum cache (_episode_start_indices) must also be
    # dropped — otherwise subsequent frame lookups use stale offsets.
    from lerobot.gui.cache_invalidation import invalidate_caches
    from lerobot.gui.api.datasets import _invalidate_episode_start_indices
    invalidate_caches(_app_state, dataset_id, invalidate_episode_indices=_invalidate_episode_start_indices)

    # Reload dataset from disk to get updated metadata
    # We reload in-place to avoid LeRobotDataset constructor trying to access HuggingFace Hub
    try:
        from lerobot.gui.dataset_reload import reload_dataset_from_disk

        old_dataset = _app_state.datasets[dataset_id]
        logger.info(f"Reloading dataset from {old_dataset.root}")
        reload_dataset_from_disk(old_dataset)
        logger.info(f"Reloaded dataset {dataset_id}: {old_dataset.meta.total_episodes} episodes, {old_dataset.meta.total_frames} frames")

    except Exception as e:
        logger.exception(f"Failed to reload dataset after edits: {e}")
        errors.append(f"Dataset reload failed: {e}")

    # Verify dataset integrity after edits
    warnings = []
    try:
        from lerobot.datasets.dataset_tools import verify_dataset

        verification = verify_dataset(old_dataset.root, check_videos=False, verbose=False)
        if not verification.is_valid:
            for err in verification.errors:
                logger.warning(f"Post-edit verification: {err.message}")
                warnings.append(err.message)
        for warn in verification.warnings:
            logger.warning(f"Post-edit verification warning: {warn.message}")
            warnings.append(warn.message)
    except Exception as e:
        logger.warning(f"Post-edit verification failed: {e}")

    if errors:
        return {
            "status": "partial",
            "message": f"Applied {applied} edits with {len(errors)} errors",
            "applied": applied,
            "errors": errors,
            "warnings": warnings,
        }

    return {"status": "ok", "message": f"Applied {applied} edits", "applied": applied, "warnings": warnings}


class MergeIntoRequest(BaseModel):
    source_dataset_id: str
    target_dataset_id: str
    force: bool = False


def _validate_merge_compat(source_meta, target_meta) -> list[dict]:
    """Compare two dataset metas and return a list of mismatches (may be empty)."""
    mismatches = []
    if target_meta.fps != source_meta.fps:
        mismatches.append({"field": "fps", "target": target_meta.fps, "source": source_meta.fps})
    if target_meta.robot_type != source_meta.robot_type:
        mismatches.append({
            "field": "robot_type",
            "target": target_meta.robot_type,
            "source": source_meta.robot_type,
        })
    tf = target_meta.features
    sf = source_meta.features
    if tf != sf:
        target_only = sorted(set(tf.keys()) - set(sf.keys()))
        source_only = sorted(set(sf.keys()) - set(tf.keys()))
        shared_diff = {}
        for k in sorted(set(tf.keys()) & set(sf.keys())):
            if tf[k] != sf[k]:
                shared_diff[k] = {"target": tf[k], "source": sf[k]}
        mismatches.append({
            "field": "features",
            "target_only": target_only,
            "source_only": source_only,
            "shared_diff": shared_diff,
        })
    return mismatches


@router.post("/merge-into/validate")
async def validate_merge(request: MergeIntoRequest):
    """Check compatibility between source and target datasets for merge."""
    source_id = request.source_dataset_id
    target_id = request.target_dataset_id

    if source_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Source dataset not found: {source_id}")
    if target_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Target dataset not found: {target_id}")

    mismatches = _validate_merge_compat(
        _app_state.datasets[source_id].meta,
        _app_state.datasets[target_id].meta,
    )
    return {"compatible": len(mismatches) == 0, "mismatches": mismatches}


@router.post("/merge-into")
async def merge_into_dataset(request: MergeIntoRequest):
    """Merge all episodes from source dataset into target dataset in-place."""
    from lerobot.datasets.dataset_tools import merge_into

    source_id = request.source_dataset_id
    target_id = request.target_dataset_id

    if source_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Source dataset not found: {source_id}")
    if target_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Target dataset not found: {target_id}")
    if source_id == target_id:
        raise HTTPException(status_code=400, detail="Cannot merge a dataset into itself")

    _require_unlocked(source_id)
    _require_unlocked(target_id)

    target_lock = _app_state.get_lock(target_id)
    source_lock = _app_state.get_lock(source_id)

    if target_lock.locked() or source_lock.locked():
        raise HTTPException(status_code=423, detail="One or both datasets are busy")

    async with target_lock:
        async with source_lock:
            return await _merge_into_locked(source_id, target_id, force=request.force)


async def _merge_into_locked(source_id: str, target_id: str, *, force: bool = False):
    """Execute merge while holding both dataset locks."""
    source_ds = _app_state.datasets[source_id]
    target_ds = _app_state.datasets[target_id]

    source_eps = source_ds.num_episodes
    source_frames = source_ds.num_frames
    target_eps_before = target_ds.num_episodes

    logger.info(
        f"Merging {source_eps} episodes from {source_id} into {target_id} "
        f"({target_eps_before} episodes) force={force}"
    )

    try:
        from lerobot.datasets.dataset_tools import merge_into

        merge_into(target_ds, source_ds, skip_validation=force)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Merge failed: {e}")
        raise HTTPException(status_code=500, detail=f"Merge failed: {e}")

    # Invalidate caches for the merge target (new episodes added — the
    # cumulative-sum cache must be dropped so subsequent frame lookups
    # pick up the grown dataset).
    from lerobot.gui.cache_invalidation import invalidate_caches
    from lerobot.gui.api.datasets import _invalidate_episode_start_indices
    invalidate_caches(_app_state, target_id, invalidate_episode_indices=_invalidate_episode_start_indices)

    logger.info(
        f"Merge complete: {target_ds.num_episodes} episodes, "
        f"{target_ds.num_frames} frames in target"
    )

    return {
        "status": "ok",
        "message": f"Merged {source_eps} episodes ({source_frames} frames) into {target_id}",
        "target_episodes": target_ds.num_episodes,
        "target_frames": target_ds.num_frames,
    }
