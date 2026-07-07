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
"""Shared business logic for dataset edits.

The FastAPI handlers in ``edits.py`` and the MCP tools in ``mcp/server.py``
both drive the same in-memory PendingEdit queue. Rather than have each
re-implement validation + queue mutation (or worse — have MCP self-call
FastAPI, see the README's "Don't auto-bind" anti-pattern), the
substantive logic lives here as sync free functions that take
``AppState`` and raise typed exceptions on failure.

Mapping:

- ``DatasetNotFoundError``  → FastAPI 404 / MCP error
- ``EditValidationError``   → FastAPI 400 / MCP error
- ``EditConflictError``     → FastAPI 409 with the carried ``detail`` /
                              MCP returns the structured detail to the AI
                              so it can retry with the appropriate confirm
                              flag.
- ``DatasetBusyError``      → FastAPI 423 / MCP error

Each function returns a plain dict that the FastAPI handler or MCP tool
forwards as the response. Persistence to disk (the per-dataset
``.lerobot_gui_edits.json``) is handled inside these functions so neither
caller has to remember to save.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lerobot.gui.state import AppState, PendingEdit

logger = logging.getLogger(__name__)


# ── Typed exceptions ──────────────────────────────────────────────────────


class DatasetNotFoundError(KeyError):
    """Dataset id not present in ``AppState.datasets``."""


class EditValidationError(ValueError):
    """Request arguments fail validation (bad range, unknown feature, etc.)."""


class EditConflictError(ValueError):
    """Edit conflicts with prior state (overlap, large-edit threshold).

    The ``detail`` attribute carries a structured dict the caller can
    surface verbatim — frontend / AI can read it to decide how to retry
    (typically with ``confirm_overlap=True`` or ``confirm_large=True``).
    """

    def __init__(self, detail: dict[str, Any]):
        self.detail = detail
        super().__init__(detail.get("message", "Edit conflicts with prior state"))


class DatasetBusyError(RuntimeError):
    """Dataset is locked by an in-progress operation."""


# ── Internal helpers ──────────────────────────────────────────────────────


def _require_dataset(app_state: AppState, dataset_id: str):
    if dataset_id not in app_state.datasets:
        raise DatasetNotFoundError(f"Dataset not found: {dataset_id}")
    return app_state.datasets[dataset_id]


def _require_unlocked(app_state: AppState, dataset_id: str) -> None:
    if app_state.is_locked(dataset_id):
        raise DatasetBusyError(f"Dataset {dataset_id} is busy (operation in progress)")


def _save_edits(app_state: AppState, dataset_id: str) -> None:
    from lerobot.gui.state import save_edits_to_file

    if dataset_id not in app_state.datasets:
        return
    dataset = app_state.datasets[dataset_id]
    save_edits_to_file(dataset.root, app_state.get_edits_for_dataset(dataset_id))


def _edit_info(index: int, edit: PendingEdit) -> dict[str, Any]:
    return {
        "index": index,
        "edit_type": edit.edit_type,
        "dataset_id": edit.dataset_id,
        "episode_index": edit.episode_index,
        "params": edit.params,
        "created_at": edit.created_at.isoformat(),
    }


# ── Public helpers — called by both FastAPI handlers and MCP tools ────────


def list_pending(app_state: AppState, dataset_id: str | None = None) -> dict[str, Any]:
    """List pending edits, optionally scoped to one dataset.

    Returns ``{"edits": [...], "total": N}``.
    """
    if dataset_id is not None:
        edits = app_state.get_edits_for_dataset(dataset_id)
        # Preserve original indices so a caller can remove by index later.
        infos = [
            _edit_info(i, e) for i, e in enumerate(app_state.pending_edits) if e.dataset_id == dataset_id
        ]
    else:
        edits = app_state.pending_edits
        infos = [_edit_info(i, e) for i, e in enumerate(app_state.pending_edits)]
    return {"edits": infos, "total": len(edits)}


def propose_delete(app_state: AppState, dataset_id: str, episode_index: int) -> dict[str, Any]:
    """Mark an episode for deletion. Idempotent only on the failure side —
    re-marking the same episode raises ``EditValidationError``.
    """
    from lerobot.gui.state import PendingEdit

    _require_unlocked(app_state, dataset_id)
    dataset = _require_dataset(app_state, dataset_id)
    if not (0 <= episode_index < dataset.meta.total_episodes):
        raise EditValidationError(
            f"Invalid episode index {episode_index} for dataset {dataset_id} "
            f"(total_episodes={dataset.meta.total_episodes})"
        )
    if app_state.is_episode_deleted(dataset_id, episode_index):
        raise EditValidationError(f"Episode {episode_index} is already marked for deletion in {dataset_id}")
    edit = PendingEdit(
        edit_type="delete",
        dataset_id=dataset_id,
        episode_index=episode_index,
    )
    app_state.add_edit(edit)
    _save_edits(app_state, dataset_id)
    logger.info(f"Marked episode {episode_index} for deletion in {dataset_id}")
    return {
        "status": "ok",
        "message": f"Episode {episode_index} marked for deletion",
        "dataset_id": dataset_id,
        "episode_index": episode_index,
    }


def propose_trim(
    app_state: AppState,
    dataset_id: str,
    episode_index: int,
    start_frame: int,
    end_frame: int,
) -> dict[str, Any]:
    """Set the trim range ``[start_frame, end_frame)`` for an episode.

    Replaces any prior trim on the same episode. A full-range trim
    (``start=0, end=ep_length``) clears the existing trim without
    adding one (= "untrim"); the response message still reads as a set.
    """
    from lerobot.gui.state import PendingEdit

    _require_unlocked(app_state, dataset_id)
    dataset = _require_dataset(app_state, dataset_id)
    if not (0 <= episode_index < dataset.meta.total_episodes):
        raise EditValidationError(
            f"Invalid episode index {episode_index} for dataset {dataset_id} "
            f"(total_episodes={dataset.meta.total_episodes})"
        )

    episode = dataset.meta.episodes[episode_index]
    episode_length = int(episode["length"])

    if start_frame < 0 or end_frame > episode_length:
        raise EditValidationError(
            f"Invalid trim range [{start_frame}, {end_frame}) for episode "
            f"{episode_index} (length={episode_length})"
        )
    if start_frame >= end_frame:
        raise EditValidationError(
            f"Invalid trim range [{start_frame}, {end_frame}): the keep window must have positive length"
        )

    # Drop any prior trim on this episode — replace semantics.
    app_state.pending_edits = [
        e
        for e in app_state.pending_edits
        if not (e.dataset_id == dataset_id and e.episode_index == episode_index and e.edit_type == "trim")
    ]

    # Only stage if it's not the full range; full-range = "untrim".
    if start_frame > 0 or end_frame < episode_length:
        edit = PendingEdit(
            edit_type="trim",
            dataset_id=dataset_id,
            episode_index=episode_index,
            params={"start_frame": start_frame, "end_frame": end_frame},
        )
        app_state.add_edit(edit)
        logger.info(f"Set trim for episode {episode_index}: frames [{start_frame}, {end_frame})")

    _save_edits(app_state, dataset_id)
    kept = end_frame - start_frame
    dropped = episode_length - kept
    return {
        "status": "ok",
        # The message states kept/dropped explicitly so a caller who got
        # the semantics backwards (thinking it means "remove this range")
        # sees the actual outcome in the response and can self-correct.
        "message": (
            f"Episode {episode_index}: keep frames [{start_frame}, {end_frame}) "
            f"({kept} of {episode_length} frames; dropping {dropped})"
        ),
        "dataset_id": dataset_id,
        "episode_index": episode_index,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "kept_frames": kept,
        "dropped_frames": dropped,
        "episode_length_before": episode_length,
    }


def discard_pending(app_state: AppState, dataset_id: str | None = None) -> dict[str, Any]:
    """Drop pending edits without touching disk. Scope is per-dataset
    when ``dataset_id`` is given, otherwise every dataset's queue is
    cleared (and every dataset checked for the busy guard first).
    """
    from lerobot.gui.state import clear_edits_file

    if dataset_id is not None:
        _require_unlocked(app_state, dataset_id)
        count = len(app_state.get_edits_for_dataset(dataset_id))
        if dataset_id in app_state.datasets:
            clear_edits_file(app_state.datasets[dataset_id].root)
        app_state.clear_edits(dataset_id)
    else:
        for ds_id in app_state.datasets:
            _require_unlocked(app_state, ds_id)
        count = len(app_state.pending_edits)
        for dataset in app_state.datasets.values():
            clear_edits_file(dataset.root)
        app_state.clear_edits(None)

    logger.info(f"Discarded {count} pending edits")
    return {
        "status": "ok",
        "message": f"Discarded {count} pending edits",
        "discarded": count,
    }


# ── Merge dataset-into-dataset helpers ────────────────────────────────────


def check_merge_compat(app_state: AppState, source_id: str, target_id: str) -> dict[str, Any]:
    """Compare schemas of two opened datasets; return a structured compat report.

    Pure read — no disk writes, no locks taken. Used by both the FastAPI
    ``/api/edits/merge-into/validate`` endpoint and the MCP
    ``validate_dataset_merge`` tool. Returns ``{"compatible": bool,
    "mismatches": [...]}`` where each mismatch dict names the field and
    carries the conflicting values.

    Raises ``DatasetNotFoundError`` if either id isn't opened in the GUI.
    """
    if source_id not in app_state.datasets:
        raise DatasetNotFoundError(f"Source dataset not found: {source_id}")
    if target_id not in app_state.datasets:
        raise DatasetNotFoundError(f"Target dataset not found: {target_id}")

    # Local import to avoid pulling FastAPI handlers into _edits_core.
    from lerobot.gui.api.edits import _validate_merge_compat

    mismatches = _validate_merge_compat(
        app_state.datasets[source_id].meta,
        app_state.datasets[target_id].meta,
    )
    return {"compatible": len(mismatches) == 0, "mismatches": mismatches}


async def merge_dataset_into(
    app_state: AppState,
    source_id: str,
    target_id: str,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Merge ``source_id``'s episodes into ``target_id`` (canonical write).

    Source is read-only; target's parquet + videos grow in place. Both
    dataset locks are acquired for the duration of the merge so concurrent
    edits / saves on either side can't race. The underlying
    ``dataset_tools.merge_into`` runs synchronously and can take minutes
    for large datasets; it's pushed onto the default executor so the
    event loop stays responsive.

    Outcome transparency: the response carries before/after counts on the
    target so the caller can see exactly how much grew. ``source_*``
    values describe what was copied.

    Raises:
        DatasetNotFoundError: source or target not opened in the GUI.
        EditValidationError: source == target (cannot self-merge) or
            ``merge_into`` raised ``ValueError`` (schema mismatch and
            ``force=False``).
        DatasetBusyError: source or target lock already held.
    """
    import asyncio

    if source_id not in app_state.datasets:
        raise DatasetNotFoundError(f"Source dataset not found: {source_id}")
    if target_id not in app_state.datasets:
        raise DatasetNotFoundError(f"Target dataset not found: {target_id}")
    if source_id == target_id:
        raise EditValidationError("Cannot merge a dataset into itself")

    _require_unlocked(app_state, source_id)
    _require_unlocked(app_state, target_id)

    target_lock = app_state.get_lock(target_id)
    source_lock = app_state.get_lock(source_id)
    if target_lock.locked() or source_lock.locked():
        raise DatasetBusyError("One or both datasets are busy")

    source_ds = app_state.datasets[source_id]
    target_ds = app_state.datasets[target_id]
    source_eps = source_ds.num_episodes
    source_frames = source_ds.num_frames
    target_eps_before = target_ds.num_episodes
    target_frames_before = target_ds.num_frames

    logger.info(
        f"Merging {source_eps} episodes from {source_id} into {target_id} "
        f"({target_eps_before} episodes) force={force}"
    )

    async with target_lock, source_lock:
        try:
            from lerobot.datasets.dataset_tools import merge_into

            await asyncio.get_event_loop().run_in_executor(
                None, lambda: merge_into(target_ds, source_ds, skip_validation=force)
            )
        except ValueError as e:
            raise EditValidationError(str(e)) from e

        # Invalidate target caches: new episodes added, the cumulative-sum
        # cache must be dropped so subsequent frame lookups pick up growth.
        from lerobot.gui.api.datasets import _invalidate_episode_start_indices
        from lerobot.gui.cache_invalidation import invalidate_caches

        invalidate_caches(app_state, target_id, invalidate_episode_indices=_invalidate_episode_start_indices)

    logger.info(
        f"Merge complete: {target_ds.num_episodes} episodes, "
        f"{target_ds.num_frames} frames in target {target_id}"
    )

    return {
        "status": "ok",
        "source_id": source_id,
        "target_id": target_id,
        "source_episodes_merged": source_eps,
        "source_frames_merged": source_frames,
        "target_episodes_before": target_eps_before,
        "target_episodes_after": target_ds.num_episodes,
        "target_frames_before": target_frames_before,
        "target_frames_after": target_ds.num_frames,
        "force_used": force,
    }


# ── Feature-set helpers (heavier — validation + overlap resolution) ───────


_DEFAULT_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
_READONLY_DTYPES = {"image", "video"}
_LARGE_SAVE_FRAME_THRESHOLD = 10_000


def _resolve_synthetic_feature(dataset, requested_feature: str) -> str:
    """Map a user-facing feature name to its storage feature name.

    Special case for the LeRobot 3.0 subtask format: ``subtask`` (string)
    → ``subtask_index`` (int64) when the dataset has a subtask lookup.
    Returns the input unchanged otherwise.
    """
    from lerobot.gui.api.datasets import (
        SUBTASK_DISPLAY_FEATURE,
        SUBTASK_STORAGE_FEATURE,
        _has_subtask_lookup,
    )

    if (
        requested_feature == SUBTASK_DISPLAY_FEATURE
        and SUBTASK_STORAGE_FEATURE in dataset.meta.features
        and _has_subtask_lookup(dataset)
    ):
        return SUBTASK_STORAGE_FEATURE
    return requested_feature


def _validate_value_against_declared_bounds(feature_name: str, feature_info: dict, value: Any) -> str:
    """Return error string (empty when valid) for declared-bounds / categorical check."""
    from lerobot.datasets.feature_utils import (
        is_categorical_feature,
        validate_feature_numeric_bounds,
    )

    has_bounds = feature_info.get("min") is not None or feature_info.get("max") is not None
    if not has_bounds and not is_categorical_feature(feature_info):
        return ""

    import numpy as np

    try:
        arr = np.asarray(value)
    except (TypeError, ValueError):
        return f"Could not interpret value {value!r} as numeric for bounds check"
    return validate_feature_numeric_bounds(feature_name, feature_info, arr)


def _validate_feature_edit(
    dataset,
    dataset_id: str,
    episode_index: int,
    feature: str,
    frame_from: int,
    frame_to: int,
    value: Any,
    confirm_large: bool,
) -> tuple[str, int, int, int, int, dict[str, Any]]:
    """Validate a feature-set request against the dataset schema + envelope.

    Returns ``(storage_feature, frame_from, frame_to, global_from, global_to,
    feature_info)``. ``frame_from``/``frame_to`` may be coerced to the full
    episode range when the feature is detected as per-episode-broadcast.

    Raises:
        EditValidationError: for schema / range failures (400-class).
        EditConflictError: for the large-edit threshold (409-class with
            structured detail so the caller knows to retry with
            ``confirm_large=True``).
    """
    feature_dict = dataset.meta.features
    storage_feature = _resolve_synthetic_feature(dataset, feature)

    if storage_feature not in feature_dict:
        raise EditValidationError(f"Unknown feature: {feature!r}")

    feature_info = feature_dict[storage_feature]
    dtype = feature_info.get("dtype", "")

    if storage_feature in _DEFAULT_FEATURES:
        raise EditValidationError(f"Feature {feature!r} is auto-managed and not editable")
    if dtype in _READONLY_DTYPES:
        raise EditValidationError(f"Feature {feature!r} has dtype {dtype!r} and is not editable in V1")
    if storage_feature == "action" or storage_feature.startswith("observation."):
        raise EditValidationError(
            f"Feature {feature!r} is recorded sensor / control data and is read-only in V1"
        )

    if not (0 <= episode_index < dataset.meta.total_episodes):
        raise EditValidationError(f"Invalid episode index: {episode_index}")

    ep = dataset.meta.episodes[episode_index]
    ep_length = int(ep["length"])
    if frame_from < 0 or frame_to > ep_length or frame_from >= frame_to:
        raise EditValidationError(
            f"Invalid range [{frame_from}, {frame_to}) for episode {episode_index} (length={ep_length})"
        )

    # Per-episode broadcast features: silently coerce sub-range to full episode.
    from lerobot.gui.api.datasets import _detect_per_episode_features, _get_episode_start_index

    per_episode = _detect_per_episode_features(dataset_id, dataset)
    if storage_feature in per_episode:
        frame_from, frame_to = 0, ep_length

    n_frames = frame_to - frame_from
    if n_frames > _LARGE_SAVE_FRAME_THRESHOLD and not confirm_large:
        raise EditConflictError(
            {
                "code": "large_edit_confirmation_required",
                "message": (
                    f"This edit touches {n_frames} frames (> {_LARGE_SAVE_FRAME_THRESHOLD}). "
                    "Re-send with confirm_large=true to proceed."
                ),
                "frames": n_frames,
            }
        )

    bounds_error = _validate_value_against_declared_bounds(feature, feature_info, value)
    if bounds_error:
        raise EditValidationError(bounds_error)

    episode_start = _get_episode_start_index(dataset_id, episode_index)
    global_from = episode_start + frame_from
    global_to = episode_start + frame_to
    return storage_feature, frame_from, frame_to, global_from, global_to, feature_info


def _find_overlapping_feature_edits(
    app_state: AppState,
    dataset_id: str,
    episode_index: int,
    feature: str,
    frame_from: int,
    frame_to: int,
) -> list[tuple[int, PendingEdit]]:
    """Return ``(index_in_pending_edits, edit)`` for prior edits that overlap."""
    overlaps: list[tuple[int, Any]] = []
    for i, e in enumerate(app_state.pending_edits):
        if (
            e.edit_type != "feature_set"
            or e.dataset_id != dataset_id
            or e.episode_index != episode_index
            or e.params.get("feature") != feature
        ):
            continue
        a = int(e.params.get("frame_from", 0))
        b = int(e.params.get("frame_to", 0))
        if frame_from < b and a < frame_to:
            overlaps.append((i, e))
    return overlaps


def _clip_overlapping_edits(
    app_state: AppState,
    overlaps: list[tuple[int, Any]],
    new_from: int,
    new_to: int,
) -> int:
    """Last-write-wins resolution: clip prior edits' ranges to the non-overlap.

    Iterates in reverse index order so removals don't shift indices we still
    need. Returns the count of fully-contained-and-removed edits. Mutates
    ``app_state.pending_edits`` in place.
    """
    from lerobot.gui.state import PendingEdit

    removed = 0
    for i, e in sorted(overlaps, key=lambda x: x[0], reverse=True):
        a = int(e.params["frame_from"])
        b = int(e.params["frame_to"])
        if new_from <= a and new_to >= b:
            app_state.pending_edits.pop(i)
            removed += 1
            continue
        left = (a, min(b, new_from))
        right = (max(a, new_to), b)
        left_keep = left[0] < left[1]
        right_keep = right[0] < right[1]
        if left_keep and right_keep:
            e.params["frame_from"] = left[0]
            e.params["frame_to"] = left[1]
            episode_start = e.params["global_from_index"] - a
            e.params["global_from_index"] = episode_start + left[0]
            e.params["global_to_index"] = episode_start + left[1]
            split = PendingEdit(
                edit_type="feature_set",
                dataset_id=e.dataset_id,
                episode_index=e.episode_index,
                params={
                    **e.params,
                    "frame_from": right[0],
                    "frame_to": right[1],
                    "global_from_index": episode_start + right[0],
                    "global_to_index": episode_start + right[1],
                },
            )
            app_state.pending_edits.append(split)
        elif left_keep:
            episode_start = e.params["global_from_index"] - a
            e.params["frame_from"] = left[0]
            e.params["frame_to"] = left[1]
            e.params["global_from_index"] = episode_start + left[0]
            e.params["global_to_index"] = episode_start + left[1]
        elif right_keep:
            episode_start = e.params["global_from_index"] - a
            e.params["frame_from"] = right[0]
            e.params["frame_to"] = right[1]
            e.params["global_from_index"] = episode_start + right[0]
            e.params["global_to_index"] = episode_start + right[1]
        else:
            app_state.pending_edits.pop(i)
            removed += 1
    return removed


def propose_feature_set(
    app_state: AppState,
    dataset_id: str,
    episode_index: int,
    feature: str,
    frame_from: int,
    frame_to: int,
    value: Any,
    confirm_large: bool = False,
    confirm_overlap: bool = False,
) -> dict[str, Any]:
    """Stage a per-frame feature-value edit.

    Raises:
        DatasetNotFoundError: dataset id not loaded.
        DatasetBusyError: dataset is locked by another operation.
        EditValidationError: schema / range / bounds failure.
        EditConflictError: large-edit threshold (retry with
            ``confirm_large=True``) or overlap with prior staged edits
            (retry with ``confirm_overlap=True``); ``detail`` carries
            structured info so the caller can present a useful prompt.
    """
    from lerobot.gui.state import PendingEdit

    _require_unlocked(app_state, dataset_id)
    dataset = _require_dataset(app_state, dataset_id)
    (
        storage_feature,
        eff_from,
        eff_to,
        global_from,
        global_to,
        _,
    ) = _validate_feature_edit(
        dataset, dataset_id, episode_index, feature, frame_from, frame_to, value, confirm_large
    )

    overlaps = _find_overlapping_feature_edits(
        app_state, dataset_id, episode_index, storage_feature, eff_from, eff_to
    )
    if overlaps and not confirm_overlap:
        raise EditConflictError(
            {
                "code": "overlapping_edit",
                "message": (
                    f"You already have {len(overlaps)} staged edit(s) on "
                    f"{feature!r} (episode {episode_index}) overlapping "
                    f"frames {eff_from}…{eff_to - 1}. "
                    "Re-send with confirm_overlap=true to clip the prior edit(s)."
                ),
                "feature": storage_feature,
                "episode_index": episode_index,
                "new_range": [eff_from, eff_to],
                "overlapping": [
                    {
                        "edit_index": i,
                        "frame_from": int(e.params["frame_from"]),
                        "frame_to": int(e.params["frame_to"]),
                        "value": e.params.get("value"),
                    }
                    for i, e in overlaps
                ],
            }
        )
    if overlaps and confirm_overlap:
        removed = _clip_overlapping_edits(app_state, overlaps, eff_from, eff_to)
        logger.info(
            f"Resolved {len(overlaps)} overlapping edit(s) on "
            f"{storage_feature} ep={episode_index}: {removed} removed, "
            f"{len(overlaps) - removed} clipped"
        )

    edit = PendingEdit(
        edit_type="feature_set",
        dataset_id=dataset_id,
        episode_index=episode_index,
        params={
            "feature": storage_feature,
            "frame_from": eff_from,
            "frame_to": eff_to,
            "global_from_index": global_from,
            "global_to_index": global_to,
            "value": value,
        },
    )
    app_state.add_edit(edit)
    _save_edits(app_state, dataset_id)
    logger.info(
        f"Staged feature-set edit: feature={storage_feature} ep={episode_index} "
        f"frames=[{eff_from}, {eff_to}) global=[{global_from}, {global_to})"
    )
    response: dict[str, Any] = {
        "status": "ok",
        "message": "Feature-set edit staged",
        "dataset_id": dataset_id,
        "episode_index": episode_index,
        "feature": storage_feature,
        "frame_from": eff_from,
        "frame_to": eff_to,
    }
    if eff_from != frame_from or eff_to != frame_to:
        response["coerced_range"] = [eff_from, eff_to]
        response["coerce_reason"] = "per_episode_broadcast"
    return response
