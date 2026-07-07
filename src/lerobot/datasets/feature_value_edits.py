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

"""In-place per-frame value editing for LeRobot datasets.

The peer of :func:`lerobot.datasets.dataset_tools.modify_features`: schema
stays the same, only cell values change. Used by the GUI's feature-edit
apply path and by notebooks/scripts that need to relabel a dataset.

Lives in its own module to isolate the surface from
``dataset_tools.py``'s busy refactor history.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from lerobot.datasets.feature_utils import validate_feature_numeric_bounds
from lerobot.datasets.utils import DATA_DIR

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset


_NON_EDITABLE_DTYPES = frozenset({"image", "video"})


class StatsRecomputationError(RuntimeError):
    """Raised when one or more episodes fail to have their stats recomputed
    after data shards were rewritten.

    The data is already on disk at this point — the rename pass-2 succeeded
    — so the dataset has new values but stats may be stale. Callers can
    catch this specifically and present "Saved, but stats are stale" to
    the user, or re-run stats computation manually.
    """

    def __init__(self, episodes: list[int], cause: BaseException) -> None:
        super().__init__(
            f"Stats recomputation failed for {len(episodes)} episode(s): {episodes!r} "
            f"(data shards were written successfully). Underlying error: {cause}"
        )
        self.episodes = episodes
        self.cause = cause


@dataclasses.dataclass(frozen=True)
class FeatureValueEdit:
    """A single value-overwrite edit targeting a contiguous frame range.

    Pre: ``feature`` exists in ``dataset.meta.features`` and isn't
    image/video. ``[from_index, to_index)`` is half-open (matches LeRobot's
    ``dataset_from_index`` / ``dataset_to_index`` convention) and lies
    within ``[0, dataset.meta.total_frames)``. ``value`` is the new value
    to write to every frame in the range — a Python scalar for shape
    ``[1]`` features, a list for vector features, or a string for
    ``string`` features.
    """

    feature: str
    from_index: int
    to_index: int
    value: Any


def _coerce_edit(edit) -> FeatureValueEdit:
    """Accept dicts (from API callers) or FeatureValueEdit instances."""
    if isinstance(edit, FeatureValueEdit):
        return edit
    return FeatureValueEdit(
        feature=edit["feature"],
        from_index=int(edit["from_index"]),
        to_index=int(edit["to_index"]),
        value=edit["value"],
    )


def _shape_value_for_column(value: Any, feature_info: dict, n_rows: int) -> list:
    """Broadcast ``value`` to a list of length ``n_rows`` matching the column dtype/shape.

    Pre: ``feature_info`` is the entry from ``dataset.meta.features`` for
    a non-image/video feature. ``n_rows`` ≥ 1.
    Post: returns a plain Python ``list`` of length ``n_rows`` with cells
    that pandas/pyarrow can serialize back into the parquet shard.
    """
    shape = list(feature_info.get("shape") or [1])
    dtype = str(feature_info.get("dtype", ""))

    is_scalar_shape = shape in ([], [1])

    # Strings are always scalar-shaped (shape [1] or scalar).
    if dtype == "string":
        return [str(value)] * n_rows

    if is_scalar_shape:
        # Pandas stores shape-[1] columns as plain scalars (not [v]) in v3 parquets,
        # matching what _copy_data_with_feature_changes does when adding a column.
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value)
            if arr.size != 1:
                raise ValueError(f"feature has shape {shape} but value has {arr.size} elements: {value!r}")
            scalar = arr.item()
        else:
            scalar = value
        if dtype == "bool":
            scalar = bool(scalar)
        elif dtype.startswith("int"):
            scalar = int(scalar)
        elif dtype.startswith("float"):
            scalar = float(scalar)
        return [scalar] * n_rows

    # Vector / matrix features. Coerce to numpy and broadcast.
    arr = np.asarray(value)
    expected = tuple(shape)
    if arr.shape != expected:
        # Allow scalar broadcast (single value applied to every component).
        if arr.shape == ():
            arr = np.broadcast_to(arr, expected).copy()
        else:
            raise ValueError(
                f"feature shape mismatch: expected {expected}, got {tuple(arr.shape)} for value {value!r}"
            )
    if dtype.startswith("int"):
        arr = arr.astype(np.int64, copy=False)
    elif dtype.startswith("float"):
        # Use the declared float width (float32 / float64) to keep the column dtype stable.
        arr = arr.astype(np.dtype(dtype), copy=False)
    return [arr.copy() for _ in range(n_rows)]


def _resolve_subtask_string_edits(
    work_root: Path,
    dataset,
    edits: list,
) -> list:
    """Resolve string values for ``subtask_index`` against ``meta/subtasks.parquet``.

    Pre: ``edits`` is a list of ``FeatureValueEdit``. Edits whose ``feature == "subtask_index"``
    and ``value`` is a string are resolved here; non-string values pass through.
    Post: returns a new list with strings replaced by their int64 indices.
    Side effect: appends new strings to ``meta/subtasks.parquet`` (rewriting the
    file once if anything was added).

    Done at Save time, not stage time, so concurrent new-string Saves on
    different sessions converge to one row in the lookup table.
    """
    needs_resolution = [e for e in edits if e.feature == "subtask_index" and isinstance(e.value, str)]
    if not needs_resolution:
        return list(edits)

    if "subtask_index" not in dataset.meta.features:
        raise ValueError("Cannot resolve string subtask_index edits: dataset has no subtask_index feature")

    subtasks_path = work_root / "meta" / "subtasks.parquet"
    if subtasks_path.exists():
        subtasks_df = pd.read_parquet(subtasks_path)
    else:
        # First subtask appearing in the dataset — initialize the lookup.
        subtasks_df = pd.DataFrame({"subtask_index": []}, dtype="int64")
        subtasks_df.index = pd.Index([], name="subtask")

    if subtasks_df.index.name != "subtask":
        subtasks_df.index.name = "subtask"

    existing: dict[str, int] = {str(name): int(row["subtask_index"]) for name, row in subtasks_df.iterrows()}
    next_idx = (max(existing.values()) + 1) if existing else 0
    new_strings: list[str] = []

    resolved: list = []
    for e in edits:
        if e.feature == "subtask_index" and isinstance(e.value, str):
            if e.value not in existing:
                existing[e.value] = next_idx
                new_strings.append(e.value)
                next_idx += 1
            resolved.append(
                FeatureValueEdit(
                    feature=e.feature,
                    from_index=e.from_index,
                    to_index=e.to_index,
                    value=existing[e.value],
                )
            )
        else:
            resolved.append(e)

    # Postcondition: no string subtask_index values survive resolution.
    # If one slipped through, downstream parquet writes would fail with
    # confusing dtype errors instead of pointing back at this layer.
    assert all(not (r.feature == "subtask_index" and isinstance(r.value, str)) for r in resolved), (
        "subtask_index resolution left at least one string value unresolved"
    )
    assert len(resolved) == len(edits), f"resolution dropped edits: in={len(edits)} out={len(resolved)}"

    if new_strings:
        new_rows = pd.DataFrame(
            {"subtask_index": [existing[s] for s in new_strings]},
            index=pd.Index(new_strings, name="subtask"),
        )
        updated = pd.concat([subtasks_df, new_rows]) if len(subtasks_df) else new_rows
        subtasks_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write — same temp+rename pattern as the data shards.
        tmp_path = subtasks_path.with_suffix(subtasks_path.suffix + ".tmp")
        try:
            updated.to_parquet(tmp_path)
            os.replace(tmp_path, subtasks_path)
        except Exception:
            if tmp_path.exists():
                # safe-destruct: our own .tmp lookup-table file
                tmp_path.unlink()
            raise
        logging.info(f"Appended {len(new_strings)} subtask(s) to {subtasks_path}")

        # Refresh in-memory meta so subsequent dataset[i] reads can decode
        # the newly-assigned indices. Without this, the data parquet has
        # subtask_index = N but meta.subtasks.iloc[N] is out-of-bounds — the
        # reader's auto-decode hits an IndexError on every frame written
        # with a brand-new subtask.
        dataset.meta.subtasks = updated

    return resolved


def set_feature_values(
    dataset: LeRobotDataset,
    edits: list,
    *,
    in_place: bool = True,
    output_dir: str | Path | None = None,
) -> None:
    """Overwrite values of EXISTING features for specific frame ranges.

    Schema is unchanged: this is a value-edit primitive, the peer of
    :func:`lerobot.datasets.dataset_tools.modify_features`. Affected
    ``data/chunk-*/file-*.parquet`` shards are rewritten with the listed
    cells overwritten; per-episode ``stats/<feature>/*`` columns in
    ``meta/episodes/*.parquet`` are recomputed for every episode whose
    frames were touched. Videos and ``info.json`` are NEVER modified by
    this function.

    Args:
        dataset: The dataset to edit. Must already be loaded.
        edits: List of edits. Each is either a ``FeatureValueEdit`` or a
            dict with keys ``feature`` (str), ``from_index`` (int),
            ``to_index`` (int, exclusive), ``value`` (any). Frame indices
            are GLOBAL ``index`` values (not per-episode frame_index).
        in_place: If True (default), rewrite the existing dataset's parquet
            files atomically (temp + rename). If False, copy the dataset to
            ``output_dir`` first and apply edits there.
        output_dir: When ``in_place=False``, target directory for the forked
            copy. Required when ``in_place=False``.

    Raises:
        ValueError: If any edit references a missing feature, an image/video
            feature, an out-of-range frame range, a value that doesn't match
            the feature's dtype/shape, or a value that violates declared
            ``min``/``max`` / categorical bounds.
        StatsRecomputationError: If data shards were written successfully
            but per-episode stats recomputation failed for one or more
            episodes. The new values ARE on disk; only stats are stale.
            Catch this specifically to report "Saved, stats stale" to the
            user without losing the data write.

    Example:
        set_feature_values(
            dataset,
            edits=[
                {"feature": "reward", "from_index": 120, "to_index": 130, "value": -0.1},
                {"feature": "success", "from_index": 380, "to_index": 385, "value": True},
            ],
        )
    """
    # Late import to avoid a circular dependency: dataset_tools imports from
    # this module via the re-export, and _recompute_episode_stats_from_data
    # is a generic helper that lives in dataset_tools.
    from lerobot.datasets.dataset_tools import _recompute_episode_stats_from_data

    if not in_place and output_dir is None:
        raise ValueError("output_dir is required when in_place=False")
    if in_place and output_dir is not None:
        raise ValueError("output_dir is not used when in_place=True (left as a guardrail)")

    edit_objs = [_coerce_edit(e) for e in edits]
    if not edit_objs:
        return

    feature_dict = dataset.meta.features
    total_frames = dataset.meta.total_frames
    assert total_frames > 0, f"dataset has total_frames={total_frames} — empty dataset cannot be edited"

    # ── Validate ────────────────────────────────────────────────────────
    # Enforce dtype, shape, declared bounds (``min``/``max``), and categorical
    # range (``names``) all in one place. Any caller of ``set_feature_values``
    # — GUI stage path, notebooks, scripts — funnels through these checks, so
    # there's no way to write a bounds-violating value to disk through the
    # public API. The GUI also checks bounds at stage time for early UX
    # feedback, but THIS is the source-of-truth gate.
    for e in edit_objs:
        if e.feature not in feature_dict:
            raise ValueError(f"Unknown feature: {e.feature!r}")
        ft = feature_dict[e.feature]
        if ft.get("dtype") in _NON_EDITABLE_DTYPES:
            raise ValueError(f"Feature {e.feature!r} has dtype {ft.get('dtype')!r} which is not editable")
        if e.from_index < 0 or e.to_index > total_frames or e.from_index >= e.to_index:
            raise ValueError(
                f"Invalid range [{e.from_index}, {e.to_index}) for {e.feature!r} "
                f"(total_frames={total_frames})"
            )
        # Bounds + categorical check. ``value`` may be a Python scalar, list,
        # or numpy array; the bounds checker flattens and inspects, shape-
        # agnostic. String values are skipped here because subtask resolution
        # converts them to ints in the next pass — at which point the
        # categorical range check is implicit (lookup index is always valid).
        if not isinstance(e.value, str):
            try:
                arr = np.asarray(e.value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Edit value for {e.feature!r} could not be coerced to a numpy array: {exc}"
                ) from exc
            err = validate_feature_numeric_bounds(e.feature, ft, arr)
            if err:
                raise ValueError(err.rstrip("\n"))

    # ── Resolve the work directory ─────────────────────────────────────
    if in_place:
        work_root = Path(dataset.root)
    else:
        out_root = Path(output_dir)
        if out_root.exists() and any(out_root.iterdir()):
            raise ValueError(f"output_dir {out_root} already exists and is non-empty")
        # Copy the entire dataset to out_root, then mutate there. Reuses the
        # full-rewrite machinery; for V1 this branch is intended for "fork on
        # save" workflows or test harnesses, not the GUI's hot path.
        out_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(dataset.root, out_root, dirs_exist_ok=True)
        work_root = out_root

    # ── Resolve string-keyed edits (subtask_index) to indices ─────────
    # Stage 1 of the Save sequence per the design doc: any edit whose ``value``
    # is a string targeting a sidecar-lookup feature (currently only
    # ``subtask_index``) gets resolved against the lookup table here. New
    # strings are appended to the table and the resolved int replaces the
    # staged string in-memory. Done before shard writes so cells go in as
    # int64s and so concurrent new-string Saves converge to one row.
    edit_objs = _resolve_subtask_string_edits(work_root, dataset, edit_objs)

    data_dir = work_root / DATA_DIR
    parquet_files = sorted(data_dir.glob("*/*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    # ── Apply edits per shard, two-pass for cross-file crash safety ────
    # Pass 1: load each shard, mutate in memory, write the .tmp sibling.
    # Pass 2: rename every .tmp → final in sequence. A crash after pass 1
    # but before any rename leaves only orphan .tmp files (cleanable);
    # only the rename loop has any partial-Save window.
    affected_episodes: set[int] = set()
    pending_renames: list[tuple[Path, Path]] = []  # [(tmp, final), ...]

    try:
        for shard_path in parquet_files:
            df = pd.read_parquet(shard_path)
            if "index" not in df.columns:
                raise ValueError(f"Data shard {shard_path} is missing the global 'index' column")

            idx_min = int(df["index"].min())
            idx_max = int(df["index"].max())  # inclusive

            shard_dirty = False
            for e in edit_objs:
                # Half-open intersect: [from, to) overlaps [idx_min, idx_max+1)
                lo = max(e.from_index, idx_min)
                hi = min(e.to_index, idx_max + 1)
                if lo >= hi:
                    continue
                mask = (df["index"] >= lo) & (df["index"] < hi)
                n = int(mask.sum())
                if n == 0:
                    continue
                df.loc[mask, e.feature] = pd.Series(
                    _shape_value_for_column(e.value, feature_dict[e.feature], n),
                    index=df.index[mask],
                )
                affected_episodes.update(int(x) for x in df.loc[mask, "episode_index"].unique())
                shard_dirty = True

            if not shard_dirty:
                continue

            tmp_path = shard_path.with_suffix(shard_path.suffix + ".tmp")
            df.to_parquet(tmp_path, compression="snappy", index=False)
            pending_renames.append((tmp_path, shard_path))
            logging.info(f"Wrote tmp shard {tmp_path} ({len(df)} rows)")
    except Exception:
        # Pass 1 failed — clean up any tmp we wrote so far.
        for tmp_path, _ in pending_renames:
            if tmp_path.exists():
                # safe-destruct: our own .tmp file we just wrote in this function.
                tmp_path.unlink()
        raise

    # Pass 2: rename all .tmp → final. Per-file atomic via os.replace.
    for tmp_path, final_path in pending_renames:
        os.replace(tmp_path, final_path)
        logging.info(f"Renamed {tmp_path} → {final_path}")

    # ── Recompute affected episode stats ───────────────────────────────
    # Failures here are surfaced via StatsRecomputationError rather than
    # silently logged: stale stats break the schema endpoint's
    # observed_min/max chip and break normalization at training time, which
    # are real silent-corruption paths. Data has already been written
    # successfully, so callers can recover by either rerunning stats
    # recomputation or accepting the stale-stats state.
    failed_episodes: list[int] = []
    first_cause: BaseException | None = None
    for ep_idx in sorted(affected_episodes):
        try:
            _recompute_episode_stats_from_data(work_root, ep_idx, feature_dict)
        except Exception as e:
            logging.warning(f"Failed to recompute stats for episode {ep_idx}: {e}")
            failed_episodes.append(ep_idx)
            if first_cause is None:
                first_cause = e

    # ── Finalize: required by v3 docs to flush metadata writers ────────
    try:
        dataset.finalize()
    except Exception as e:
        logging.warning(f"dataset.finalize() failed (non-fatal for in-place value edits): {e}")

    if failed_episodes:
        assert first_cause is not None, "failed_episodes non-empty but first_cause is None"
        raise StatsRecomputationError(failed_episodes, first_cause) from first_cause
