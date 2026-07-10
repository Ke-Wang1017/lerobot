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

"""Dataset tools utilities for LeRobotDataset.

This module provides utilities for:
- Deleting episodes from datasets
- Splitting datasets into multiple smaller datasets
- Adding/removing features from datasets
- Merging datasets (wrapper around aggregate functionality)
"""

import contextlib
import logging
import os
import shutil
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from lerobot.configs import (
    DepthEncoderConfig,
    RGBEncoderConfig,
    VideoEncoderConfig,
    depth_encoder_defaults,
    encoder_config_from_video_info,
    rgb_encoder_defaults,
)
from lerobot.configs.video import DEPTH_ENCODER_INFO_FIELD_NAMES
from lerobot.datasets.aggregate import (
    aggregate_data,
    aggregate_datasets,
    aggregate_metadata,
    aggregate_videos,
    validate_all_metadata,
)
from lerobot.datasets.compute_stats import (
    aggregate_stats,
    compute_episode_stats,
    compute_relative_action_stats,
)
from lerobot.datasets.image_writer import write_image
from lerobot.datasets.io_utils import (
    get_parquet_file_size_in_mb,
    load_episodes,
    load_info,
    load_stats,
    write_info,
    write_stats,
    write_tasks,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    DATA_DIR,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_DATA_PATH,
    DEFAULT_EPISODES_PATH,
    DEPTH_FILE_PATTERN,
    IMAGE_FILE_PATTERN,
    VIDEO_DIR,
    update_chunk_file_indices,
)
from lerobot.datasets.video_utils import encode_video_frames, reencode_video
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_IMAGE, OBS_STATE


def _load_episode_with_stats(src_dataset: LeRobotDataset, episode_idx: int) -> dict:
    """Load a single episode's metadata including stats from parquet file.

    Args:
        src_dataset: Source dataset
        episode_idx: Episode index to load

    Returns:
        dict containing episode metadata and stats
    """
    ep_meta = src_dataset.meta.episodes[episode_idx]
    chunk_idx = ep_meta["meta/episodes/chunk_index"]
    file_idx = ep_meta["meta/episodes/file_index"]

    parquet_path = src_dataset.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
    df = pd.read_parquet(parquet_path)

    episode_row = df[df["episode_index"] == episode_idx].iloc[0]

    return episode_row.to_dict()


def _extract_episode_stats_from_parquet(episode_row: dict, features: dict) -> dict:
    """Extract per-episode stats from a parquet row, handling nested numpy array deserialization.

    When pandas/pyarrow serializes numpy arrays with shape (3, 1, 1) to parquet,
    they can be deserialized as nested object arrays. This function flattens them back.

    Args:
        episode_row: Dictionary from a single episode's parquet row
        features: The dataset's features dict (to detect image/video dtypes)

    Returns:
        Dictionary mapping feature names to their stats dicts
    """
    episode_stats = {}
    for key in episode_row:
        if not key.startswith("stats/"):
            continue
        stat_key = key.replace("stats/", "")
        parts = stat_key.split("/")
        if len(parts) != 2:
            continue

        feature_name, stat_name = parts
        if feature_name not in episode_stats:
            episode_stats[feature_name] = {}

        value = episode_row[key]

        if feature_name in features:
            feature_dtype = features[feature_name].get("dtype", "")
            if feature_dtype in ["image", "video"] and stat_name != "count":
                if isinstance(value, np.ndarray) and value.dtype == object:
                    flat_values = []
                    for item in value:
                        while isinstance(item, np.ndarray):
                            item = item.flatten()[0]
                        flat_values.append(item)
                    value = np.array(flat_values, dtype=np.float64).reshape(3, 1, 1)
                elif isinstance(value, np.ndarray) and value.shape == (3,):
                    value = value.reshape(3, 1, 1)

        episode_stats[feature_name][stat_name] = value

    return episode_stats


def _reaggregate_and_write_stats(local_dir: Path, features: dict) -> None:
    """Re-aggregate stats from all per-episode parquet files and write stats.json.

    Reads stats/* columns from all episode parquet files, aggregates them
    using the parallel variance algorithm, and writes the result to meta/stats.json.

    Args:
        local_dir: Root directory of the dataset
        features: The dataset's features dict
    """
    episodes_dir = local_dir / "meta" / "episodes"
    all_stats = []

    for parquet_path in sorted(episodes_dir.rglob("*.parquet")):
        df = pd.read_parquet(parquet_path)
        for _, row in df.iterrows():
            episode_stats = _extract_episode_stats_from_parquet(row.to_dict(), features)
            if episode_stats:
                all_stats.append(episode_stats)

    if not all_stats:
        logging.warning("No per-episode statistics found to aggregate")
        return

    aggregated_stats = aggregate_stats(all_stats)
    filtered_stats = {k: v for k, v in aggregated_stats.items() if k in features}
    write_stats(filtered_stats, local_dir)


def reaggregate_dataset_stats(dataset: LeRobotDataset) -> None:
    """Re-aggregate stats.json from per-episode parquet stats.

    Call this once after batching multiple trim/delete operations with
    ``recompute_stats=False`` to avoid O(N*E) re-aggregation overhead.

    Args:
        dataset: The dataset whose stats.json should be rebuilt.
    """
    _reaggregate_and_write_stats(dataset.root, dataset.meta.features)


def _recompute_episode_stats_from_data(
    local_dir: Path,
    episode_index: int,
    features: dict,
) -> None:
    """Recompute per-episode stats from data parquet and update episode metadata.

    Loads the episode's data rows, computes stats for non-video features,
    and updates the stats/* columns in the episode parquet file.

    For video features, stats are not recomputed (virtual trim doesn't modify
    video pixels, and VISUAL normalization typically uses IDENTITY mode).
    Only the count is updated to reflect the new frame count.

    Args:
        local_dir: Root directory of the dataset
        episode_index: Index of the episode to recompute stats for
        features: The dataset's features dict
    """
    from lerobot.utils.utils import flatten_dict

    # Load the episode's data from data parquet
    data_dir = local_dir / DATA_DIR
    all_data = []
    for parquet_path in sorted(data_dir.rglob("*.parquet")):
        df = pd.read_parquet(parquet_path)
        ep_data = df[df["episode_index"] == episode_index]
        if len(ep_data) > 0:
            all_data.append(ep_data)

    if not all_data:
        logging.warning(f"No data found for episode {episode_index}")
        return

    ep_df = pd.concat(all_data)
    new_frame_count = len(ep_df)

    # Build episode_data dict for compute_episode_stats (non-video features only)
    episode_data = {}
    for key, feat_info in features.items():
        if feat_info["dtype"] in ["image", "video", "string"]:
            continue
        if key not in ep_df.columns:
            continue
        col_data = ep_df[key].values
        # Convert list-of-arrays to 2D numpy array
        if isinstance(col_data[0], (list, np.ndarray)):
            episode_data[key] = np.stack(col_data)
        else:
            episode_data[key] = col_data

    # Compute stats for non-video features
    non_video_features = {k: v for k, v in features.items() if v["dtype"] not in ["image", "video"]}
    new_stats = compute_episode_stats(episode_data, non_video_features)

    # Update the episode parquet file
    episodes_dir = local_dir / "meta" / "episodes"
    for parquet_path in sorted(episodes_dir.rglob("*.parquet")):
        df = pd.read_parquet(parquet_path)
        if episode_index not in df["episode_index"].values:
            continue

        row_idx = df.index[df["episode_index"] == episode_index][0]

        # Build a new stats dict for this row, preserving existing video stats
        flat_new_stats = flatten_dict({"stats": new_stats})

        # Update the row as a dict, then reconstruct the DataFrame row
        row_dict = df.loc[row_idx].to_dict()
        for col_name, value in flat_new_stats.items():
            if col_name in row_dict:
                # Match dtype of existing column to avoid pyarrow mixed-dtype errors
                existing = row_dict[col_name]
                if isinstance(existing, np.ndarray) and isinstance(value, np.ndarray):
                    value = value.astype(existing.dtype)
                row_dict[col_name] = value

        # Update video feature counts to match new frame count
        for key, feat_info in features.items():
            if feat_info["dtype"] in ["image", "video"]:
                count_col = f"stats/{key}/count"
                if count_col in row_dict:
                    row_dict[count_col] = np.array([new_frame_count])

        # Drop the old row and append the updated one
        df = df.drop(index=row_idx)
        new_row_df = pd.DataFrame([row_dict])
        df = pd.concat([df, new_row_df], ignore_index=True)
        df = df.sort_values("episode_index").reset_index(drop=True)

        df.to_parquet(parquet_path, index=False)
        break  # Episode only exists in one file


def delete_episodes(
    dataset: LeRobotDataset,
    episode_indices: list[int],
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
) -> LeRobotDataset:
    """Delete episodes from a LeRobotDataset and create a new dataset.

    Video segments that need re-encoding (because the source file mixes kept and
    deleted episodes) are re-encoded with the source dataset's existing encoder
    settings — read back from ``meta/info.json`` — so the output dataset stays
    consistent with its own metadata.

    Args:
        dataset: The source LeRobotDataset.
        episode_indices: List of episode indices to delete.
        output_dir: Root directory where the edited dataset will be stored. If not specified, defaults to $HF_LEROBOT_HOME/repo_id. Equivalent to new_root in EditDatasetConfig.
        repo_id: Edited dataset identifier. Equivalent to new_repo_id in EditDatasetConfig.
    """
    if not episode_indices:
        raise ValueError("No episodes to delete")

    valid_indices = set(range(dataset.meta.total_episodes))
    invalid = set(episode_indices) - valid_indices
    if invalid:
        raise ValueError(f"Invalid episode indices: {invalid}")

    logging.info(f"Deleting {len(episode_indices)} episodes from dataset")

    if repo_id is None:
        repo_id = f"{dataset.repo_id}_modified"
    output_dir = Path(output_dir) if output_dir is not None else HF_LEROBOT_HOME / repo_id

    episodes_to_keep = [i for i in range(dataset.meta.total_episodes) if i not in episode_indices]
    if not episodes_to_keep:
        raise ValueError("Cannot delete all episodes from dataset")

    new_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=dataset.meta.fps,
        features=dataset.meta.features,
        robot_type=dataset.meta.robot_type,
        root=output_dir,
        use_videos=len(dataset.meta.video_keys) > 0,
    )

    episode_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(episodes_to_keep)}

    video_metadata = None
    if dataset.meta.video_keys:
        video_metadata = _copy_and_reindex_videos(dataset, new_meta, episode_mapping)

    data_metadata = _copy_and_reindex_data(dataset, new_meta, episode_mapping)

    _copy_and_reindex_episodes_metadata(dataset, new_meta, episode_mapping, data_metadata, video_metadata)

    new_dataset = LeRobotDataset(
        repo_id=repo_id,
        root=output_dir,
        image_transforms=dataset.image_transforms,
        delta_timestamps=dataset.delta_timestamps,
        tolerance_s=dataset.tolerance_s,
    )

    logging.info(f"Created new dataset with {len(episodes_to_keep)} episodes")
    return new_dataset


def split_dataset(
    dataset: LeRobotDataset,
    splits: dict[str, float | list[int]],
    output_dir: str | Path | None = None,
) -> dict[str, LeRobotDataset]:
    """Split a LeRobotDataset into multiple smaller datasets.

    Video segments that need re-encoding (because the source file mixes episodes
    that fall into different splits) are re-encoded with the source dataset's
    existing encoder settings — read back from ``meta/info.json`` — so each
    output split stays consistent with its own metadata.

    Args:
        dataset: The source LeRobotDataset to split.
        splits: Either a dict mapping split names to episode indices, or a dict mapping
                split names to fractions (must sum to <= 1.0).
        output_dir: Root directory where the split datasets will be stored. If not specified, defaults to $HF_LEROBOT_HOME/repo_id.

    Examples:
      Split by specific episodes
        splits = {"train": [0, 1, 2], "val": [3, 4]}
        datasets = split_dataset(dataset, splits)

      Split by fractions
        splits = {"train": 0.8, "val": 0.2}
        datasets = split_dataset(dataset, splits)
    """
    if not splits:
        raise ValueError("No splits provided")

    if all(isinstance(v, float) for v in splits.values()):
        splits = _fractions_to_episode_indices(dataset.meta.total_episodes, splits)

    all_episodes = set()
    for split_name, episodes in splits.items():
        if not episodes:
            raise ValueError(f"Split '{split_name}' has no episodes")
        episode_set = set(episodes)
        if episode_set & all_episodes:
            raise ValueError("Episodes cannot appear in multiple splits")
        all_episodes.update(episode_set)

    valid_indices = set(range(dataset.meta.total_episodes))
    invalid = all_episodes - valid_indices
    if invalid:
        raise ValueError(f"Invalid episode indices: {invalid}")

    if output_dir is not None:
        output_dir = Path(output_dir)

    result_datasets = {}

    for split_name, episodes in splits.items():
        logging.info(f"Creating split '{split_name}' with {len(episodes)} episodes")

        split_repo_id = f"{dataset.repo_id}_{split_name}"

        split_output_dir = (
            output_dir / split_name if output_dir is not None else HF_LEROBOT_HOME / split_repo_id
        )

        episode_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(episodes))}

        new_meta = LeRobotDatasetMetadata.create(
            repo_id=split_repo_id,
            fps=dataset.meta.fps,
            features=dataset.meta.features,
            robot_type=dataset.meta.robot_type,
            root=split_output_dir,
            use_videos=len(dataset.meta.video_keys) > 0,
            chunks_size=dataset.meta.chunks_size,
            data_files_size_in_mb=dataset.meta.data_files_size_in_mb,
            video_files_size_in_mb=dataset.meta.video_files_size_in_mb,
        )

        video_metadata = None
        if dataset.meta.video_keys:
            video_metadata = _copy_and_reindex_videos(dataset, new_meta, episode_mapping)

        data_metadata = _copy_and_reindex_data(dataset, new_meta, episode_mapping)

        _copy_and_reindex_episodes_metadata(dataset, new_meta, episode_mapping, data_metadata, video_metadata)

        new_dataset = LeRobotDataset(
            repo_id=split_repo_id,
            root=split_output_dir,
            image_transforms=dataset.image_transforms,
            delta_timestamps=dataset.delta_timestamps,
            tolerance_s=dataset.tolerance_s,
        )

        result_datasets[split_name] = new_dataset

    return result_datasets


def merge_datasets(
    datasets: list[LeRobotDataset],
    output_repo_id: str,
    output_dir: str | Path | None = None,
    concatenate_videos: bool = True,
    concatenate_data: bool = True,
) -> LeRobotDataset:
    """Merge multiple LeRobotDatasets into a single dataset.

    This is a wrapper around the aggregate_datasets functionality with a cleaner API.

    Args:
        datasets: List of LeRobotDatasets to merge.
        output_repo_id: Merged dataset identifier.
        output_dir: Root directory where the merged dataset will be stored. If not specified, defaults to $HF_LEROBOT_HOME/output_repo_id.
        concatenate_videos: When False, keep one mp4 per source file instead of packing into shards.
        concatenate_data: When False, keep one parquet per source file instead of packing into shards.
    """
    if not datasets:
        raise ValueError("No datasets to merge")

    output_dir = Path(output_dir) if output_dir is not None else HF_LEROBOT_HOME / output_repo_id

    repo_ids = [ds.repo_id for ds in datasets]
    roots = [ds.root for ds in datasets]

    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id=output_repo_id,
        roots=roots,
        aggr_root=output_dir,
        concatenate_videos=concatenate_videos,
        concatenate_data=concatenate_data,
    )

    merged_dataset = LeRobotDataset(
        repo_id=output_repo_id,
        root=output_dir,
        image_transforms=datasets[0].image_transforms,
        delta_timestamps=datasets[0].delta_timestamps,
        tolerance_s=datasets[0].tolerance_s,
    )

    return merged_dataset


def _compute_next_file_indices(
    target_meta: LeRobotDatasetMetadata,
) -> tuple[dict, dict, dict]:
    """Compute next available (chunk, file) indices from the target dataset's episode metadata.

    Scans all episodes to find the maximum file indices for data, meta, and video files,
    then increments by one to get the starting point for new files.

    Returns:
        Tuple of (data_idx, meta_idx, videos_idx) ready for aggregate functions.
    """
    episodes = target_meta.episodes
    chunk_size = target_meta.chunks_size

    def _next_idx(chunk_col, file_col):
        chunks = [int(episodes[i][chunk_col]) for i in range(len(episodes))]
        files = [int(episodes[i][file_col]) for i in range(len(episodes))]
        max_chunk = max(chunks)
        max_file = max(f for c, f in zip(chunks, files, strict=True) if c == max_chunk)
        return update_chunk_file_indices(max_chunk, max_file, chunk_size)

    next_data_chunk, next_data_file = _next_idx("data/chunk_index", "data/file_index")
    data_idx = {"chunk": next_data_chunk, "file": next_data_file}

    next_meta_chunk, next_meta_file = _next_idx("meta/episodes/chunk_index", "meta/episodes/file_index")
    meta_idx = {"chunk": next_meta_chunk, "file": next_meta_file}

    videos_idx = {}
    for key in target_meta.video_keys:
        next_vid_chunk, next_vid_file = _next_idx(f"videos/{key}/chunk_index", f"videos/{key}/file_index")
        videos_idx[key] = {
            "chunk": next_vid_chunk,
            "file": next_vid_file,
            "latest_duration": 0,
            "episode_duration": 0,
        }

    return data_idx, meta_idx, videos_idx


def merge_into(
    target: LeRobotDataset,
    source: LeRobotDataset,
    skip_validation: bool = False,
) -> LeRobotDataset:
    """Merge source dataset episodes into target dataset in-place.

    Appends all episodes from source into target. Existing target data files,
    video files, and episode metadata files remain untouched — only new files
    are created for the incoming episodes, plus small metadata updates
    (info.json, tasks, stats).

    Prerequisites (checked automatically unless skip_validation=True):
    - Same FPS
    - Same robot_type
    - Same features (observation keys, action dimensions, dtypes)

    Args:
        target: The destination dataset (modified in-place).
        source: The source dataset whose episodes will be appended.
        skip_validation: If True, skip fps/robot_type/features validation.

    Returns:
        The target dataset, reloaded to reflect the merged state.
    """
    # Reload target metadata from disk to ensure fresh state
    target.meta.episodes = load_episodes(target.meta.root)
    target.meta.info = load_info(target.meta.root)

    # 1. Validate compatibility
    if not skip_validation:
        validate_all_metadata([target.meta, source.meta])

    # 2. Compute starting file indices (one past target's current max)
    data_idx, meta_idx, videos_idx = _compute_next_file_indices(target.meta)

    # 3. Merge tasks — preserve target's existing task indices, append only new
    existing_tasks = target.meta.tasks.copy()
    new_tasks = [t for t in source.meta.tasks.index if t not in existing_tasks.index]
    next_task_idx = len(existing_tasks)
    for task in new_tasks:
        existing_tasks.loc[task] = next_task_idx
        next_task_idx += 1
    target.meta.tasks = existing_tasks

    # 4. Run aggregate pipeline — writes new files into target directory
    src_meta = source.meta
    dst_meta = target.meta

    if dst_meta.video_keys:
        videos_idx = aggregate_videos(
            src_meta,
            dst_meta,
            videos_idx,
            dst_meta.video_files_size_in_mb,
            dst_meta.chunks_size,
        )

    data_idx = aggregate_data(
        src_meta,
        dst_meta,
        data_idx,
        dst_meta.data_files_size_in_mb,
        dst_meta.chunks_size,
    )

    meta_idx = aggregate_metadata(src_meta, dst_meta, meta_idx, data_idx, videos_idx)

    # 5. Update totals
    dst_meta.info["total_episodes"] += src_meta.total_episodes
    dst_meta.info["total_frames"] += src_meta.total_frames
    dst_meta.info["total_tasks"] = len(dst_meta.tasks)
    dst_meta.info["splits"] = {"train": f"0:{dst_meta.info['total_episodes']}"}
    write_info(dst_meta.info, dst_meta.root)

    # 6. Write tasks
    write_tasks(dst_meta.tasks, dst_meta.root)

    # 7. Reaggregate stats from all episode parquet files
    _reaggregate_and_write_stats(dst_meta.root, dst_meta.features)

    # 8. Reload target in-place (avoids LeRobotDataset constructor which makes Hub calls)
    target.meta.info = load_info(target.root)
    target.meta.episodes = load_episodes(target.root)
    target.meta.stats = load_stats(target.root)

    import datasets as hf_datasets

    if target.hf_dataset is not None:
        with contextlib.suppress(Exception):
            target.hf_dataset.cleanup_cache_files()

    hf_datasets.disable_caching()
    try:
        # Re-load the reader's hf_dataset; the facade's hf_dataset is a property
        # that delegates to reader.hf_dataset.
        if target.reader is not None:
            target.reader.load_and_activate()
        else:
            # Fallback: instantiate a fresh reader for write-mode datasets
            target._ensure_reader().load_and_activate()
    finally:
        hf_datasets.enable_caching()

    logging.info(
        f"Merged {src_meta.total_episodes} episodes ({src_meta.total_frames} frames) "
        f"into {target.repo_id}. New total: {dst_meta.info['total_episodes']} episodes."
    )
    return target


def modify_features(
    dataset: LeRobotDataset,
    add_features: dict[str, tuple[np.ndarray | torch.Tensor | Callable, dict]] | None = None,
    remove_features: str | list[str] | None = None,
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
) -> LeRobotDataset:
    """Modify a LeRobotDataset by adding and/or removing features in a single pass.

    This is the most efficient way to modify features, as it only copies the dataset once
    regardless of how many features are being added or removed.

    Args:
        dataset: The source LeRobotDataset.
        add_features: Optional dict mapping feature names to (feature_values, feature_info) tuples.
        remove_features: Optional feature name(s) to remove. Can be a single string or list.
        output_dir: Root directory where the edited dataset will be stored. If not specified, defaults to $HF_LEROBOT_HOME/repo_id. Equivalent to new_root in EditDatasetConfig.
        repo_id: Edited dataset identifier. Equivalent to new_repo_id in EditDatasetConfig.

    Returns:
        New dataset with features modified.

    Example:
        new_dataset = modify_features(
            dataset,
            add_features={
                "reward": (reward_array, {"dtype": "float32", "shape": [1], "names": None}),
            },
            remove_features=["old_feature"],
            output_dir="./output",
        )
    """
    if add_features is None and remove_features is None:
        raise ValueError("Must specify at least one of add_features or remove_features")

    remove_features_list: list[str] = []
    if remove_features is not None:
        remove_features_list = [remove_features] if isinstance(remove_features, str) else remove_features

    if add_features:
        required_keys = {"dtype", "shape"}
        for feature_name, (_, feature_info) in add_features.items():
            if feature_name in dataset.meta.features:
                raise ValueError(f"Feature '{feature_name}' already exists in dataset")

            if not required_keys.issubset(feature_info.keys()):
                raise ValueError(f"feature_info for '{feature_name}' must contain keys: {required_keys}")

    if remove_features_list:
        for name in remove_features_list:
            if name not in dataset.meta.features:
                raise ValueError(f"Feature '{name}' not found in dataset")

        required_features = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
        if any(name in required_features for name in remove_features_list):
            raise ValueError(f"Cannot remove required features: {required_features}")

    if repo_id is None:
        repo_id = f"{dataset.repo_id}_modified"
    output_dir = Path(output_dir) if output_dir is not None else HF_LEROBOT_HOME / repo_id

    new_features = dataset.meta.features.copy()

    if remove_features_list:
        for name in remove_features_list:
            new_features.pop(name, None)

    if add_features:
        for feature_name, (_, feature_info) in add_features.items():
            new_features[feature_name] = feature_info

    video_keys_to_remove = [name for name in remove_features_list if name in dataset.meta.video_keys]
    remaining_video_keys = [k for k in dataset.meta.video_keys if k not in video_keys_to_remove]

    new_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=dataset.meta.fps,
        features=new_features,
        robot_type=dataset.meta.robot_type,
        root=output_dir,
        use_videos=len(remaining_video_keys) > 0,
    )

    _copy_data_with_feature_changes(
        dataset=dataset,
        new_meta=new_meta,
        add_features=add_features,
        remove_features=remove_features_list if remove_features_list else None,
    )

    if new_meta.video_keys:
        _copy_videos(dataset, new_meta, exclude_keys=video_keys_to_remove if video_keys_to_remove else None)

    new_dataset = LeRobotDataset(
        repo_id=repo_id,
        root=output_dir,
        image_transforms=dataset.image_transforms,
        delta_timestamps=dataset.delta_timestamps,
        tolerance_s=dataset.tolerance_s,
    )

    return new_dataset


def add_features(
    dataset: LeRobotDataset,
    features: dict[str, tuple[np.ndarray | torch.Tensor | Callable, dict]],
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
) -> LeRobotDataset:
    """Add multiple features to a LeRobotDataset in a single pass.

    This is more efficient than calling add_feature() multiple times, as it only
    copies the dataset once regardless of how many features are being added.

    Args:
        dataset: The source LeRobotDataset.
        features: Dictionary mapping feature names to (feature_values, feature_info) tuples.
        output_dir: Root directory where the edited dataset will be stored. If not specified, defaults to $HF_LEROBOT_HOME/repo_id. Equivalent to new_root in EditDatasetConfig.
        repo_id: Edited dataset identifier. Equivalent to new_repo_id in EditDatasetConfig.

    Returns:
        New dataset with all features added.

    Example:
        features = {
            "task_embedding": (task_emb_array, {"dtype": "float32", "shape": [384], "names": None}),
            "cam1_embedding": (cam1_emb_array, {"dtype": "float32", "shape": [768], "names": None}),
            "cam2_embedding": (cam2_emb_array, {"dtype": "float32", "shape": [768], "names": None}),
        }
        new_dataset = add_features(dataset, features, output_dir="./output", repo_id="my_dataset")
    """
    if not features:
        raise ValueError("No features provided")

    return modify_features(
        dataset=dataset,
        add_features=features,
        remove_features=None,
        output_dir=output_dir,
        repo_id=repo_id,
    )


def remove_feature(
    dataset: LeRobotDataset,
    feature_names: str | list[str],
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
) -> LeRobotDataset:
    """Remove features from a LeRobotDataset.

    Args:
        dataset: The source LeRobotDataset.
        feature_names: Name(s) of features to remove. Can be a single string or list.
        output_dir: Root directory where the edited dataset will be stored. If not specified, defaults to $HF_LEROBOT_HOME/repo_id. Equivalent to new_root in EditDatasetConfig.
        repo_id: Edited dataset identifier. Equivalent to new_repo_id in EditDatasetConfig.

    Returns:
        New dataset with features removed.
    """
    return modify_features(
        dataset=dataset,
        add_features=None,
        remove_features=feature_names,
        output_dir=output_dir,
        repo_id=repo_id,
    )


def rename_feature(
    dataset: LeRobotDataset,
    old_name: str,
    new_name: str,
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
) -> LeRobotDataset:
    """Rename a feature in a LeRobotDataset.

    Args:
        dataset: The source LeRobotDataset.
        old_name: Current name of the feature.
        new_name: New name for the feature.
        output_dir: Directory to save the new dataset. If None, uses default location.
        repo_id: Repository ID for the new dataset. If None, appends "_renamed" to original.

    Returns:
        New dataset with feature renamed.

    Example:
        dataset = LeRobotDataset("my_dataset", root="/path/to/dataset")
        new_dataset = rename_feature(
            dataset,
            old_name="observation.images.cam1",
            new_name="observation.images.camera_left",
            output_dir="/path/to/output",
        )
    """
    # Validate
    if old_name not in dataset.meta.features:
        raise ValueError(f"Feature '{old_name}' not found in dataset")
    if new_name in dataset.meta.features:
        raise ValueError(f"Feature '{new_name}' already exists in dataset")

    logging.info(f"Renaming feature: {old_name} -> {new_name}")

    if repo_id is None:
        repo_id = f"{dataset.repo_id}_renamed"
    output_dir = Path(output_dir) if output_dir is not None else HF_LEROBOT_HOME / repo_id

    # Create new metadata with renamed feature
    new_features = dataset.meta.features.copy()
    new_features[new_name] = new_features.pop(old_name)

    new_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=dataset.meta.fps,
        features=new_features,
        robot_type=dataset.meta.robot_type,
        root=output_dir,
        use_videos=len(dataset.meta.video_keys) > 0,
        chunks_size=dataset.meta.chunks_size,
        data_files_size_in_mb=dataset.meta.data_files_size_in_mb,
        video_files_size_in_mb=dataset.meta.video_files_size_in_mb,
    )

    # Copy data with renamed columns
    _copy_data_with_feature_changes(
        dataset=dataset,
        new_meta=new_meta,
        rename_features={old_name: new_name},
    )

    # Copy videos with renamed directory
    if new_meta.video_keys:
        _copy_videos(
            dataset,
            new_meta,
            rename_keys={old_name: new_name} if old_name in dataset.meta.video_keys else None,
        )

    new_dataset = LeRobotDataset(
        repo_id=repo_id,
        root=output_dir,
        image_transforms=dataset.image_transforms,
        delta_timestamps=dataset.delta_timestamps,
        tolerance_s=dataset.tolerance_s,
    )

    logging.info(f"✓ Feature renamed successfully: {old_name} -> {new_name}")
    return new_dataset


def swap_features(
    dataset: LeRobotDataset,
    feature1: str,
    feature2: str,
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
) -> LeRobotDataset:
    """Swap two features in a LeRobotDataset.

    Args:
        dataset: The source LeRobotDataset.
        feature1: Name of the first feature to swap.
        feature2: Name of the second feature to swap.
        output_dir: Directory to save the new dataset. If None, uses default location.
        repo_id: Repository ID for the new dataset. If None, appends "_swapped" to original.

    Returns:
        New dataset with features swapped.

    Example:
        dataset = LeRobotDataset("my_dataset", root="/path/to/dataset")
        new_dataset = swap_features(
            dataset,
            feature1="observation.images.left_wrist",
            feature2="observation.images.right_wrist",
            output_dir="/path/to/output",
        )
    """
    import tempfile

    if repo_id is None:
        repo_id = f"{dataset.repo_id}_swapped"

    logging.info(f"Swapping features: {feature1} <-> {feature2}")

    # Use temporary directories for intermediate steps
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # Step 1: Rename feature1 to temporary name
        temp_name = f"__temp_swap_{feature1}_to_{feature2}__"
        dataset = rename_feature(
            dataset,
            old_name=feature1,
            new_name=temp_name,
            output_dir=temp_dir / "step1",
        )

        # Step 2: Rename feature2 to feature1
        dataset = rename_feature(
            dataset,
            old_name=feature2,
            new_name=feature1,
            output_dir=temp_dir / "step2",
        )

        # Step 3: Rename temp back to feature2 (final output)
        dataset = rename_feature(
            dataset,
            old_name=temp_name,
            new_name=feature2,
            output_dir=output_dir,
            repo_id=repo_id,
        )

    logging.info(f"✓ Features swapped successfully: {feature1} <-> {feature2}")
    return dataset


# ============================================================================
# set_feature_values — in-place per-frame value editing
#
# Implementation lives in :mod:`lerobot.datasets.feature_value_edits` to
# isolate it from this file's busy refactor history. Re-exported here so
# existing imports (``from lerobot.datasets.dataset_tools import
# set_feature_values, FeatureValueEdit``) keep working. The ``F401`` noqa
# is needed because ruff can't see cross-module callers and would
# otherwise auto-strip the apparently-unused import.
# ============================================================================

from typing import Any  # noqa: E402

from lerobot.datasets.feature_value_edits import (  # noqa: E402, F401
    FeatureValueEdit,
    StatsRecomputationError,
    _shape_value_for_column,
    set_feature_values,
)

# ──────────────────────────────────────────────────────────────────────
# add_features_inplace — schema-additive in-place column add
# ──────────────────────────────────────────────────────────────────────


def _sweep_orphan_tmp_shards(dataset_root: Path | str) -> int:
    """Delete ``.tmp`` siblings under ``data/`` and ``meta/`` left by a crashed save.

    Safe to call on dataset open. Only removes files matching ``*.tmp`` —
    real shards and metadata files are untouched. Returns the number of
    files deleted.
    """
    root = Path(dataset_root)
    removed = 0
    for sub in ("data", "meta"):
        sub_dir = root / sub
        if not sub_dir.exists():
            continue
        for tmp in sub_dir.rglob("*.tmp"):
            try:
                # safe-destruct: orphan .tmp files from interrupted atomic-write saves
                tmp.unlink()
                removed += 1
            except OSError:
                pass
    return removed


def _is_recorded_feature(name: str) -> bool:
    """True for ``action`` and ``observation.*`` features.

    These are recorded sensor / control data — modifying or removing them
    breaks the contract that downstream components (training pipelines,
    normalization stats, eval rollouts) rely on. Used by the in-place
    schema-mutation primitives (``remove_features_inplace``,
    ``rename_features_inplace``) to refuse operations that would corrupt
    these columns.

    Mirrors the GUI's ``isRecordedFeature`` predicate in
    ``src/lerobot/gui/static/feature_editing.js`` and the inline check in
    ``src/lerobot/gui/api/edits.py::_validate_feature_edit``. If you
    update one, update the others.
    """
    return name == "action" or name.startswith("observation.")


def _atomic_swap_files(rename_pairs: list[tuple[Path, Path]]) -> None:
    """Atomically swap a batch of files. All-or-nothing.

    Each pair is ``(src, dst)``: ``src`` (the freshly-written ``.tmp`` produced
    by the caller's Pass 1) replaces ``dst`` (the existing on-disk file). On
    any failure during the swap loop, all completed swaps are reverted by
    restoring the original ``dst`` from a hardlink-backed ``.bak`` sibling,
    and any unfinished ``src`` files are unlinked.

    Pre: every ``dst`` exists; every ``src`` exists at call time. The
    ``dst.with_suffix(dst.suffix + ".bak")`` path must be available (no
    pre-existing ``.bak`` collision).
    Post: on success, all dsts have new content and no ``.bak`` / ``.tmp``
    siblings remain. On failure, all dsts hold their original content and
    no ``.bak`` / ``.tmp`` siblings remain; the original exception propagates.

    Cost: one ``os.link`` (hardlink, near-zero) per pair before its
    ``os.replace``. Doubles inode count briefly during the swap window.
    """
    completed: list[tuple[Path, Path]] = []  # [(dst, bak), ...] — successfully swapped
    try:
        for src, dst in rename_pairs:
            bak = dst.with_suffix(dst.suffix + ".bak")
            os.link(dst, bak)  # hardlink: original survives at both dst and bak
            try:
                os.replace(src, dst)
                completed.append((dst, bak))
            except Exception:
                # Pass-through dst → bak hardlink wasn't committed yet; clean it up.
                # safe-destruct: just-created hardlink, original still at dst
                bak.unlink()
                raise
    except Exception:
        # Restore originals from .bak in reverse order. Best-effort: if any
        # individual restore fails, log and continue so we don't block other
        # restores from running.
        for dst, bak in reversed(completed):
            try:
                os.replace(bak, dst)
            except Exception as e:
                logging.warning(f"_atomic_swap_files: rollback restore failed for {dst}: {e}")
        # Clean up any unfinished sources (caller's .tmp files that were
        # never moved). For completed swaps, src no longer exists (os.replace
        # consumed it), so the existence check filters those out.
        for src, _ in rename_pairs:
            if src.exists():
                # safe-destruct: caller's .tmp file, never committed
                src.unlink()
        raise
    # All swaps succeeded — drop backups (originals no longer needed).
    for _, bak in completed:
        # safe-destruct: original superseded by committed new content
        bak.unlink()


def add_features_inplace(
    dataset: LeRobotDataset,
    features: dict[str, tuple],
    *,
    recompute_stats: bool = True,
) -> None:
    """Add new feature columns to an existing dataset in place.

    Each entry in ``features`` maps a feature name to ``(fill_value, feature_info)``:

    - ``fill_value`` is a scalar broadcast across every row of every shard.
    - ``feature_info`` follows the same schema used by :func:`modify_features`
      (``{"dtype": ..., "shape": ..., "names": ...}``) and may include an
      optional ``"per_episode": True`` hint. The hint is preserved verbatim
      through ``info.json`` so consumers (e.g. the GUI editor) can coerce
      sub-range edits to whole episodes.

    Videos are NOT touched. Only ``data/chunk-*/file-*.parquet`` shards and
    ``meta/info.json`` are rewritten. Per-episode stats columns for the new
    feature(s) are populated when ``recompute_stats=True`` (default).

    Atomicity: each parquet shard is written to a ``.tmp`` sibling and
    renamed via ``os.replace`` (per-file atomic). ``info.json`` is rewritten
    last. A crash mid-rename can leave a partially applied save; orphan
    ``.tmp`` files are removable via :func:`_sweep_orphan_tmp_shards`.

    Args:
        dataset: The dataset to extend. Must already be loaded.
        features: ``{name: (fill_value, feature_info)}`` for each new column.
        recompute_stats: When True (default), recompute per-episode stats
            for the new columns by re-aggregating each affected episode's
            data. Pass False to skip if stats will be computed separately.

    Raises:
        ValueError: For empty input, name collisions, ``DEFAULT_FEATURES``
            collisions, missing ``dtype``/``shape``, malformed ``shape``,
            or non-bool ``per_episode``.
    """
    from lerobot.datasets.utils import DATA_DIR, INFO_PATH
    from lerobot.utils.constants import DEFAULT_FEATURES
    from lerobot.utils.io_utils import write_json

    if not features:
        raise ValueError("features dict is empty")

    required_keys = {"dtype", "shape"}
    existing = dataset.meta.features
    for name, (_, info) in features.items():
        # DEFAULT_FEATURES check first — these names also appear in
        # `dataset.meta.features` (auto-merged at create time), so the
        # generic "already exists" message would otherwise mask them.
        if name in DEFAULT_FEATURES:
            raise ValueError(f"Feature '{name}' is a reserved DEFAULT_FEATURE")
        if name in existing:
            raise ValueError(f"Feature '{name}' already exists in dataset")
        if not required_keys.issubset(info.keys()):
            raise ValueError(f"feature_info for '{name}' must include keys: {required_keys}")
        shape = info["shape"]
        if (
            not isinstance(shape, (list, tuple))
            or len(shape) == 0
            or not all(isinstance(d, int) and d > 0 for d in shape)
        ):
            raise ValueError(f"feature_info['shape'] for '{name}' must be a non-empty list of positive ints")
        if "per_episode" in info and not isinstance(info["per_episode"], bool):
            raise ValueError(f"feature_info['per_episode'] for '{name}' must be a bool")

    work_root = Path(dataset.root)
    info_path = work_root / INFO_PATH
    data_dir = work_root / DATA_DIR

    parquet_files = sorted(data_dir.glob("*/*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    # Build the new feature dict (declared schema only — fills happen in shards).
    new_feature_specs = {name: dict(info) for name, (_, info) in features.items()}

    # ── Pass 1: write .tmp shards with appended columns ────────────────
    # For each new feature, build a numpy column with the declared dtype
    # so the on-disk parquet matches the schema. Without the explicit
    # ``astype``, pandas would auto-promote (float32 → float64, int8 →
    # int64) when constructing the Series from a Python list of scalars.
    pending_renames: list[tuple[Path, Path]] = []  # [(tmp, final), ...]
    try:
        for shard_path in parquet_files:
            df = pd.read_parquet(shard_path)
            n_rows = len(df)
            for name, (fill, info) in features.items():
                col_values = _shape_value_for_column(fill, info, n_rows)
                dtype_str = info.get("dtype", "")
                shape = list(info.get("shape") or [1])
                is_scalar = shape in ([], [1])
                if dtype_str == "string":
                    df[name] = pd.Series(col_values, dtype="object")
                elif is_scalar and dtype_str:
                    # Scalar columns: cast through numpy so the declared
                    # dtype lands on disk verbatim (no float32→double drift).
                    df[name] = pd.Series(col_values, dtype=np.dtype(dtype_str))
                else:
                    # Vector / matrix: pandas keeps these as object lists;
                    # parquet stores them as fixed-size lists. The element
                    # dtype is preserved by _shape_value_for_column already.
                    df[name] = col_values
            tmp_path = shard_path.with_suffix(shard_path.suffix + ".tmp")
            df.to_parquet(tmp_path, compression="snappy", index=False)
            pending_renames.append((tmp_path, shard_path))
    except Exception:
        for tmp_path, _ in pending_renames:
            if tmp_path.exists():
                # safe-destruct: our own .tmp file we just wrote in this function
                tmp_path.unlink()
        raise

    # ── Pass 1.5: write info.json.tmp (still no swaps yet) ─────────────
    # Materializing the new info.json before Pass 2 lets us bundle its swap
    # with the data-shard swaps in a single atomic batch — a crash at any
    # point (mid-Pass-2 or just before the info.json swap) rolls back to
    # the pre-call state instead of leaving a desync.
    import json as _json

    with info_path.open("r") as f:
        info_dict = _json.load(f)
    for name, spec in new_feature_specs.items():
        info_dict["features"][name] = spec
    info_tmp = info_path.with_suffix(info_path.suffix + ".tmp")
    try:
        write_json(info_dict, info_tmp)
    except Exception:
        for tmp_path, _ in pending_renames:
            if tmp_path.exists():
                # safe-destruct: our own .tmp file we just wrote in this function
                tmp_path.unlink()
        raise

    # ── Pass 2: atomic swap of all data shards + info.json ─────────────
    # Either every dst (data shards + info.json) gets the new content, or
    # they all stay at the original. _atomic_swap_files cleans up .tmp /
    # .bak siblings on both success and failure paths.
    _atomic_swap_files([*pending_renames, (info_tmp, info_path)])

    # Refresh the in-memory metadata so callers see the new schema.
    dataset.meta = LeRobotDatasetMetadata(repo_id=dataset.repo_id, root=dataset.root)
    feature_dict_after = dataset.meta.features

    # ── Compute + append stats columns for the NEW features only ───────
    # The general-purpose ``_recompute_episode_stats_from_data`` only
    # writes columns that already exist in the episodes parquet, so it
    # silently drops stats for newly-added features. Handle the
    # additive-only path here directly.
    if recompute_stats:
        new_feature_subset = {n: feature_dict_after[n] for n in new_feature_specs}
        _add_new_feature_stats_to_episodes(work_root, new_feature_subset)

    try:
        dataset.finalize()
    except Exception as e:
        logging.warning(f"dataset.finalize() failed (non-fatal for in-place schema add): {e}")


def bootstrap_subtask_format(
    dataset: "LeRobotDataset",
    initial_subtask: str = "",
) -> None:
    """Provision the LeRobot 3.0 subtask format on a dataset that lacks it.

    Creates the ``meta/subtasks.parquet`` lookup (string → index) and appends
    a per-frame ``subtask_index`` (int64) column, labeling every existing
    frame ``initial_subtask``. Afterwards, per-range subtask strings can be
    assigned via :func:`lerobot.datasets.feature_value_edits.set_feature_values`
    (new strings are appended to the lookup at Save time).

    The lookup is written before the column: a crash in between leaves a
    lookup-only dataset, which readers ignore and a retry reuses; the reverse
    partial state (column without lookup) would be undecodable. A pre-existing
    lookup (e.g. from a prior partial run) is reused — ``initial_subtask``
    resolves to its existing row when present.

    ``subtask_index`` is declared ``per_episode=False``: subtask labeling is
    per-frame-range by design, and without the explicit declaration the
    constant initial fill would be inferred as per-episode-uniform, coercing
    range edits to whole episodes.

    Args:
        dataset: The dataset to extend. Must already be loaded.
        initial_subtask: Label applied to every existing frame.

    Raises:
        ValueError: If the dataset already has a ``subtask_index`` feature or
            ``initial_subtask`` is not a string.
    """
    from lerobot.datasets.io_utils import load_subtasks
    from lerobot.datasets.utils import DEFAULT_SUBTASKS_PATH

    if not isinstance(initial_subtask, str):
        raise ValueError(f"initial_subtask must be a string, got {type(initial_subtask).__name__}")
    if "subtask_index" in dataset.meta.features:
        raise ValueError("Dataset already has a 'subtask_index' feature")

    subtasks_df = load_subtasks(dataset.root)
    if subtasks_df is None:
        subtasks_df = pd.DataFrame(
            {"subtask_index": pd.array([], dtype="int64")},
            index=pd.Index([], name="subtask"),
        )
    existing = {str(name): int(row["subtask_index"]) for name, row in subtasks_df.iterrows()}
    if initial_subtask in existing:
        fill_index = existing[initial_subtask]
    else:
        # Row position must equal the subtask_index value — readers decode
        # positionally (``meta.subtasks.iloc[idx].name``) — so append at the
        # end with the next index, same as the Save-time resolver.
        fill_index = (max(existing.values()) + 1) if existing else 0
        new_row = pd.DataFrame(
            {"subtask_index": [fill_index]},
            index=pd.Index([initial_subtask], name="subtask"),
        )
        updated = pd.concat([subtasks_df, new_row]) if len(subtasks_df) else new_row
        subtasks_path = Path(dataset.root) / DEFAULT_SUBTASKS_PATH
        subtasks_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = subtasks_path.with_suffix(subtasks_path.suffix + ".tmp")
        try:
            updated.to_parquet(tmp_path)
            os.replace(tmp_path, subtasks_path)
        except Exception:
            if tmp_path.exists():
                # safe-destruct: our own .tmp lookup-table file
                tmp_path.unlink()
            raise

    # add_features_inplace replaces dataset.meta with a fresh load, which
    # picks up the lookup written above — no manual meta.subtasks refresh.
    add_features_inplace(
        dataset,
        features={
            "subtask_index": (
                fill_index,
                {"dtype": "int64", "shape": [1], "names": None, "per_episode": False},
            )
        },
    )


def remove_features_inplace(
    dataset: "LeRobotDataset",
    names: list[str] | str,
) -> None:
    """Remove feature columns from an existing dataset in place.

    Drops the named columns from every parquet shard under ``data/``,
    drops the matching ``stats/<feature>/*`` columns from
    ``meta/episodes/*.parquet``, and removes the entries from
    ``info.json``. Videos are NOT touched — refuses to remove image /
    video features (those have on-disk filenames that need a real fork).

    Args:
        dataset: The dataset to mutate. Must already be loaded.
        names: One feature name or a list of feature names to drop.

    Raises:
        ValueError: For empty input, missing names, ``DEFAULT_FEATURES``
            collisions, or attempts to drop image/video features.
    """
    from lerobot.datasets.utils import DATA_DIR, INFO_PATH
    from lerobot.utils.constants import DEFAULT_FEATURES
    from lerobot.utils.io_utils import write_json

    name_list = [names] if isinstance(names, str) else list(names)
    if not name_list:
        raise ValueError("names list is empty")

    existing = dataset.meta.features
    for name in name_list:
        if name in DEFAULT_FEATURES:
            raise ValueError(f"Cannot remove DEFAULT_FEATURE '{name}'")
        if name not in existing:
            raise ValueError(f"Feature '{name}' not found in dataset")
        if existing[name].get("dtype") in ("image", "video"):
            raise ValueError(
                f"Cannot remove image/video feature '{name}' in-place "
                "— use the forked remove_feature() instead"
            )
        if _is_recorded_feature(name):
            raise ValueError(
                f"Cannot remove recorded feature '{name}' — action / observation.* "
                "are sensor / control data the rest of the pipeline depends on"
            )

    work_root = Path(dataset.root)
    info_path = work_root / INFO_PATH
    data_dir = work_root / DATA_DIR

    parquet_files = sorted(data_dir.glob("*/*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    # ── Pass 1: drop columns in each data shard via .tmp ───────────────
    pending_renames: list[tuple[Path, Path]] = []
    try:
        for shard_path in parquet_files:
            df = pd.read_parquet(shard_path)
            cols_to_drop = [n for n in name_list if n in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
            tmp_path = shard_path.with_suffix(shard_path.suffix + ".tmp")
            df.to_parquet(tmp_path, compression="snappy", index=False)
            pending_renames.append((tmp_path, shard_path))
    except Exception:
        for tmp_path, _ in pending_renames:
            if tmp_path.exists():
                # safe-destruct: our own .tmp file we just wrote in this function
                tmp_path.unlink()
        raise

    # ── Pass 1.5: drop stats columns in each episodes parquet ──────────
    # Stats parquets are written to .tmp here but NOT swapped — they're
    # bundled into the atomic batch below alongside the data shards and
    # info.json.
    stats_renames: list[tuple[Path, Path]] = []
    eps_dir = work_root / "meta" / "episodes"
    if eps_dir.exists():
        prefixes = [f"stats/{n}/" for n in name_list]
        try:
            for parquet_path in sorted(eps_dir.rglob("*.parquet")):
                edf = pd.read_parquet(parquet_path)
                cols_to_drop = [c for c in edf.columns if any(c.startswith(p) for p in prefixes)]
                if cols_to_drop:
                    edf = edf.drop(columns=cols_to_drop)
                    tmp_path = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
                    edf.to_parquet(tmp_path, compression="snappy", index=False)
                    stats_renames.append((tmp_path, parquet_path))
        except Exception:
            for tmp_path, _ in [*pending_renames, *stats_renames]:
                if tmp_path.exists():
                    # safe-destruct: our own .tmp file we just wrote in this function
                    tmp_path.unlink()
            raise

    # ── Pass 1.6: write info.json.tmp ──────────────────────────────────
    import json as _json

    with info_path.open("r") as f:
        info_dict = _json.load(f)
    for name in name_list:
        info_dict["features"].pop(name, None)

    info_tmp = info_path.with_suffix(info_path.suffix + ".tmp")
    try:
        write_json(info_dict, info_tmp)
    except Exception:
        for tmp_path, _ in [*pending_renames, *stats_renames]:
            if tmp_path.exists():
                # safe-destruct: our own .tmp file we just wrote in this function
                tmp_path.unlink()
        raise

    # ── Pass 2: atomic swap of all data shards + stats + info.json ─────
    _atomic_swap_files([*pending_renames, *stats_renames, (info_tmp, info_path)])

    dataset.meta = LeRobotDatasetMetadata(repo_id=dataset.repo_id, root=dataset.root)

    try:
        dataset.finalize()
    except Exception as e:
        logging.warning(f"dataset.finalize() failed (non-fatal for remove): {e}")


def rename_features_inplace(
    dataset: "LeRobotDataset",
    renames: dict[str, str],
    *,
    spec_overrides: dict[str, dict] | None = None,
) -> None:
    """Rename feature columns in an existing dataset in place.

    Renames every parquet column under ``data/`` and the matching
    ``stats/<feature>/*`` columns in ``meta/episodes/*.parquet``.
    Updates ``info.json`` last (atomic via ``.tmp + rename``). Videos
    are NOT touched — this function refuses to rename image / video
    features because their on-disk filenames also encode the feature
    name (use the forked :func:`rename_feature` for those).

    Args:
        dataset: The dataset to mutate. Must already be loaded.
        renames: ``{old_name: new_name}``. Multiple renames are applied
            in one pass.
        spec_overrides: Optional ``{new_name: {key: value, ...}}`` —
            keys merged into the renamed feature's spec in
            ``info.json``. Useful when the rename also flips a hint
            (e.g. ``per_episode``) that the source column didn't declare.

    Raises:
        ValueError: For empty input, missing ``old_name``, existing
            ``new_name``, ``DEFAULT_FEATURES`` collisions, or attempts
            to rename image/video features.
    """
    from lerobot.datasets.utils import DATA_DIR, INFO_PATH
    from lerobot.utils.constants import DEFAULT_FEATURES
    from lerobot.utils.io_utils import write_json

    if not renames:
        raise ValueError("renames dict is empty")

    spec_overrides = spec_overrides or {}
    existing = dataset.meta.features

    # ── Validate ───────────────────────────────────────────────────────
    new_names_seen: set[str] = set()
    for old, new in renames.items():
        # DEFAULT_FEATURES check first — these names also appear in
        # `dataset.meta.features` (auto-merged at create time), so the
        # "already exists" branch would otherwise mask them.
        if old in DEFAULT_FEATURES or new in DEFAULT_FEATURES:
            raise ValueError(f"Cannot rename DEFAULT_FEATURES (refused: {old} → {new})")
        if old not in existing:
            raise ValueError(f"Feature '{old}' not found in dataset")
        if new in existing and new != old:
            raise ValueError(f"Feature '{new}' already exists in dataset")
        if new in new_names_seen:
            raise ValueError(f"Duplicate rename target: '{new}'")
        if existing[old].get("dtype") in ("image", "video"):
            raise ValueError(
                f"Cannot rename image/video feature '{old}' in-place "
                "— use the forked rename_feature() instead"
            )
        if _is_recorded_feature(old) or _is_recorded_feature(new):
            raise ValueError(
                f"Cannot rename recorded feature in-place (refused: {old} → {new}) "
                "— action / observation.* are sensor / control data; "
                "use the forked rename_feature() instead if you really need this"
            )
        new_names_seen.add(new)

    work_root = Path(dataset.root)
    info_path = work_root / INFO_PATH
    data_dir = work_root / DATA_DIR

    parquet_files = sorted(data_dir.glob("*/*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    # ── Pass 1: rename columns in each data shard via .tmp ─────────────
    pending_renames: list[tuple[Path, Path]] = []
    try:
        for shard_path in parquet_files:
            df = pd.read_parquet(shard_path)
            df = df.rename(columns=renames)
            tmp_path = shard_path.with_suffix(shard_path.suffix + ".tmp")
            df.to_parquet(tmp_path, compression="snappy", index=False)
            pending_renames.append((tmp_path, shard_path))
    except Exception:
        for tmp_path, _ in pending_renames:
            if tmp_path.exists():
                # safe-destruct: our own .tmp file we just wrote in this function
                tmp_path.unlink()
        raise

    # ── Pass 1.5: rename stats columns to .tmp (no swap yet) ───────────
    stats_renames: list[tuple[Path, Path]] = []
    eps_dir = work_root / "meta" / "episodes"
    if eps_dir.exists():
        try:
            for parquet_path in sorted(eps_dir.rglob("*.parquet")):
                edf = pd.read_parquet(parquet_path)
                col_renames: dict[str, str] = {}
                for old, new in renames.items():
                    old_prefix = f"stats/{old}/"
                    new_prefix = f"stats/{new}/"
                    for col in edf.columns:
                        if col.startswith(old_prefix):
                            col_renames[col] = new_prefix + col[len(old_prefix) :]
                if col_renames:
                    edf = edf.rename(columns=col_renames)
                    tmp_path = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
                    edf.to_parquet(tmp_path, compression="snappy", index=False)
                    stats_renames.append((tmp_path, parquet_path))
        except Exception:
            for tmp_path, _ in [*pending_renames, *stats_renames]:
                if tmp_path.exists():
                    # safe-destruct: our own .tmp file we just wrote in this function
                    tmp_path.unlink()
            raise

    # ── Pass 1.6: write info.json.tmp ──────────────────────────────────
    import json as _json

    with info_path.open("r") as f:
        info_dict = _json.load(f)
    feats = info_dict["features"]
    for old, new in renames.items():
        spec = dict(feats.pop(old))
        if new in spec_overrides:
            spec.update(spec_overrides[new])
        feats[new] = spec

    info_tmp = info_path.with_suffix(info_path.suffix + ".tmp")
    try:
        write_json(info_dict, info_tmp)
    except Exception:
        for tmp_path, _ in [*pending_renames, *stats_renames]:
            if tmp_path.exists():
                # safe-destruct: our own .tmp file we just wrote in this function
                tmp_path.unlink()
        raise

    # ── Pass 2: atomic swap of all data shards + stats + info.json ─────
    _atomic_swap_files([*pending_renames, *stats_renames, (info_tmp, info_path)])

    # Refresh in-memory metadata.
    dataset.meta = LeRobotDatasetMetadata(repo_id=dataset.repo_id, root=dataset.root)

    try:
        dataset.finalize()
    except Exception as e:
        logging.warning(f"dataset.finalize() failed (non-fatal for rename): {e}")


def _add_new_feature_stats_to_episodes(
    dataset_root: Path,
    new_features: dict,
) -> None:
    """Compute and append per-episode ``stats/<feature>/*`` columns for newly-added features.

    Reads each episode's data slice for the new columns, computes stats via
    :func:`compute_episode_stats`, and writes the resulting columns into
    every ``meta/episodes/*.parquet`` shard. Columns that already exist are
    overwritten; columns that don't exist are added.
    """
    from lerobot.datasets.compute_stats import compute_episode_stats
    from lerobot.utils.utils import flatten_dict

    work_root = Path(dataset_root)
    data_dir = work_root / DATA_DIR
    eps_dir = work_root / "meta" / "episodes"

    # Skip image/video/string — compute_episode_stats can't compute on them
    # and the caller path doesn't add such features anyway.
    eligible = {
        n: spec for n, spec in new_features.items() if spec.get("dtype") not in ("image", "video", "string")
    }
    if not eligible:
        return

    # Collect per-episode data slices for the new columns only.
    per_ep_data: dict[int, dict[str, np.ndarray]] = {}
    for parquet_path in sorted(data_dir.glob("*/*.parquet")):
        df = pd.read_parquet(parquet_path, columns=["episode_index", *eligible.keys()])
        for ep_idx, group in df.groupby("episode_index"):
            ep_idx = int(ep_idx)
            ep_bucket = per_ep_data.setdefault(ep_idx, {})
            for name in eligible:
                col_data = group[name].values
                if len(col_data) == 0:
                    continue
                first = col_data[0]
                if isinstance(first, (list, np.ndarray)):
                    arr = np.stack([np.asarray(v) for v in col_data])
                else:
                    arr = np.asarray(col_data)
                if name in ep_bucket:
                    ep_bucket[name] = np.concatenate([ep_bucket[name], arr], axis=0)
                else:
                    ep_bucket[name] = arr

    if not per_ep_data:
        return

    # Compute stats per episode and gather flattened columns.
    per_ep_flat_stats: dict[int, dict[str, Any]] = {}
    for ep_idx, ep_arrays in per_ep_data.items():
        stats = compute_episode_stats(ep_arrays, eligible)
        per_ep_flat_stats[ep_idx] = flatten_dict({"stats": stats})

    # Write the new columns into each episodes parquet shard. Use the same
    # drop-and-rebuild pattern as ``_recompute_episode_stats_from_data`` so
    # pyarrow can infer the column dtype uniformly across rows (assigning
    # numpy arrays into freshly-added object columns confuses the writer).
    for parquet_path in sorted(eps_dir.rglob("*.parquet")):
        edf = pd.read_parquet(parquet_path)
        if "episode_index" not in edf.columns:
            continue
        relevant_eps = [ep for ep in per_ep_flat_stats if (edf["episode_index"] == ep).any()]
        if not relevant_eps:
            continue

        rebuilt_rows = []
        for ep_idx in relevant_eps:
            row_idx = edf.index[edf["episode_index"] == ep_idx][0]
            row_dict = edf.loc[row_idx].to_dict()
            for col_name, value in per_ep_flat_stats[ep_idx].items():
                row_dict[col_name] = value
            rebuilt_rows.append(row_dict)
            edf = edf.drop(index=row_idx)

        new_rows_df = pd.DataFrame(rebuilt_rows)
        edf = pd.concat([edf, new_rows_df], ignore_index=True)
        edf = edf.sort_values("episode_index").reset_index(drop=True)

        tmp_path = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
        edf.to_parquet(tmp_path, compression="snappy", index=False)
        os.replace(tmp_path, parquet_path)


def _fractions_to_episode_indices(
    total_episodes: int,
    splits: dict[str, float],
) -> dict[str, list[int]]:
    """Convert split fractions to episode indices."""
    if sum(splits.values()) > 1.0:
        raise ValueError("Split fractions must sum to <= 1.0")

    indices = list(range(total_episodes))
    result = {}
    start_idx = 0

    for split_name, fraction in splits.items():
        num_episodes = int(total_episodes * fraction)
        if num_episodes == 0:
            logging.warning(f"Split '{split_name}' has no episodes, skipping...")
            continue
        end_idx = start_idx + num_episodes
        if split_name == list(splits.keys())[-1]:
            end_idx = total_episodes
        result[split_name] = indices[start_idx:end_idx]
        start_idx = end_idx

    return result


def _copy_and_reindex_data(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    episode_mapping: dict[int, int],
) -> dict[int, dict]:
    """Copy and filter data files, only modifying files with deleted episodes.

    Args:
        src_dataset: Source dataset to copy from
        dst_meta: Destination metadata object
        episode_mapping: Mapping from old episode indices to new indices

    Returns:
        dict mapping episode index to its data file metadata (chunk_index, file_index, etc.)
    """
    if src_dataset.meta.episodes is None:
        src_dataset.meta.episodes = load_episodes(src_dataset.meta.root)

    file_to_episodes: dict[Path, set[int]] = {}
    for old_idx in episode_mapping:
        file_path = src_dataset.meta.get_data_file_path(old_idx)
        if file_path not in file_to_episodes:
            file_to_episodes[file_path] = set()
        file_to_episodes[file_path].add(old_idx)

    global_index = 0
    episode_data_metadata: dict[int, dict] = {}

    if dst_meta.tasks is None:
        all_task_indices = set()
        for src_path in file_to_episodes:
            df = pd.read_parquet(src_dataset.root / src_path)
            mask = df["episode_index"].isin(list(episode_mapping.keys()))
            task_series: pd.Series = df[mask]["task_index"]
            all_task_indices.update(task_series.unique().tolist())
        tasks = [src_dataset.meta.tasks.iloc[idx].name for idx in all_task_indices]
        dst_meta.save_episode_tasks(list(set(tasks)))

    task_mapping = {}
    for old_task_idx in range(len(src_dataset.meta.tasks)):
        task_name = src_dataset.meta.tasks.iloc[old_task_idx].name
        new_task_idx = dst_meta.get_task_index(task_name)
        if new_task_idx is not None:
            task_mapping[old_task_idx] = new_task_idx

    for src_path in tqdm(sorted(file_to_episodes.keys()), desc="Processing data files"):
        df = pd.read_parquet(src_dataset.root / src_path)

        all_episodes_in_file = set(df["episode_index"].unique())
        episodes_to_keep = file_to_episodes[src_path]

        if all_episodes_in_file == episodes_to_keep:
            df["episode_index"] = df["episode_index"].replace(episode_mapping)
            df["index"] = range(global_index, global_index + len(df))
            df["task_index"] = df["task_index"].replace(task_mapping)

            first_ep_old_idx = min(episodes_to_keep)
            src_ep = src_dataset.meta.episodes[first_ep_old_idx]
            chunk_idx = src_ep["data/chunk_index"]
            file_idx = src_ep["data/file_index"]
        else:
            mask = df["episode_index"].isin(list(episode_mapping.keys()))
            df = df[mask].copy().reset_index(drop=True)

            if len(df) == 0:
                continue

            df["episode_index"] = df["episode_index"].replace(episode_mapping)
            df["index"] = range(global_index, global_index + len(df))
            df["task_index"] = df["task_index"].replace(task_mapping)

            first_ep_old_idx = min(episodes_to_keep)
            src_ep = src_dataset.meta.episodes[first_ep_old_idx]
            chunk_idx = src_ep["data/chunk_index"]
            file_idx = src_ep["data/file_index"]

        dst_path = dst_meta.root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        _write_parquet(df, dst_path, dst_meta)

        for ep_old_idx in episodes_to_keep:
            ep_new_idx = episode_mapping[ep_old_idx]
            ep_df = df[df["episode_index"] == ep_new_idx]
            episode_data_metadata[ep_new_idx] = {
                "data/chunk_index": chunk_idx,
                "data/file_index": file_idx,
                "dataset_from_index": int(ep_df["index"].min()),
                "dataset_to_index": int(ep_df["index"].max() + 1),
            }

        global_index += len(df)

    return episode_data_metadata


def _keep_episodes_from_video_with_av(
    input_path: Path,
    output_path: Path,
    episodes_to_keep: list[tuple[int, int]],
    fps: float,
    video_encoder: VideoEncoderConfig,
) -> None:
    """Keep only specified episodes from a video file using PyAV.

    This function decodes frames from specified frame ranges and re-encodes them with
    properly reset timestamps to ensure monotonic progression.

    Args:
        input_path: Source video file path.
        output_path: Destination video file path.
        episodes_to_keep: List of (start_frame, end_frame) tuples for episodes to keep.
            Ranges are half-open intervals: [start_frame, end_frame), where start_frame
            is inclusive and end_frame is exclusive.
        fps: Frame rate of the video.
        video_encoder: Video encoder settings used to re-encode the kept frames.
    """
    from fractions import Fraction

    import av

    if not episodes_to_keep:
        raise ValueError("No episodes to keep")

    in_container = av.open(str(input_path))

    # Check if video stream exists.
    if not in_container.streams.video:
        raise ValueError(
            f"No video streams found in {input_path}. "
            "The video file may be corrupted or empty. "
            "Try re-downloading the dataset or checking the video file."
        )

    v_in = in_container.streams.video[0]

    out = av.open(str(output_path), mode="w")

    # Convert fps to Fraction for PyAV compatibility.
    fps_fraction = Fraction(fps).limit_denominator(1000)
    codec_options = video_encoder.get_codec_options(as_strings=True)
    v_out = out.add_stream(video_encoder.vcodec, rate=fps_fraction, options=codec_options)

    # PyAV type stubs don't distinguish video streams from audio/subtitle streams.
    v_out.width = v_in.codec_context.width
    v_out.height = v_in.codec_context.height
    v_out.pix_fmt = video_encoder.pix_fmt

    # Set time_base to match the frame rate for proper timestamp handling.
    v_out.time_base = Fraction(1, int(fps))

    out.start_encoding()

    # Create set of (start, end) ranges for fast lookup.
    # Convert to a sorted list for efficient checking.
    frame_ranges = sorted(episodes_to_keep)

    # Track frame index for setting PTS and current range being processed.
    src_frame_count = 0
    frame_count = 0
    range_idx = 0

    # Read through entire video once and filter frames.
    for packet in in_container.demux(v_in):
        for frame in packet.decode():
            if frame is None:
                continue

            # Check if frame is in any of our desired frame ranges.
            # Skip ranges that have already passed.
            while range_idx < len(frame_ranges) and src_frame_count >= frame_ranges[range_idx][1]:
                range_idx += 1

            # If we've passed all ranges, stop processing.
            if range_idx >= len(frame_ranges):
                break

            # Check if frame is in current range.
            start_frame = frame_ranges[range_idx][0]

            if src_frame_count < start_frame:
                src_frame_count += 1
                continue

            # Frame is in range - create a new frame with reset timestamps.
            # We need to create a copy to avoid modifying the original.
            new_frame = frame.reformat(width=v_out.width, height=v_out.height, format=v_out.pix_fmt)
            new_frame.pts = frame_count
            new_frame.time_base = Fraction(1, int(fps))

            # Encode and mux the frame.
            for pkt in v_out.encode(new_frame):
                out.mux(pkt)

            src_frame_count += 1
            frame_count += 1

    # Flush encoder.
    for pkt in v_out.encode():
        out.mux(pkt)

    out.close()
    in_container.close()


# TODO: Consolidate with _keep_episodes_from_video_with_av (frame-based) once
# our time-based trim callers are migrated to frame indices.
def _keep_episodes_from_video_by_time(
    input_path: Path,
    output_path: Path,
    episodes_to_keep: list[tuple[float, float]],
    fps: float,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
) -> None:
    """Keep only specified episodes from a video file using PyAV (time-based).

    Like _keep_episodes_from_video_with_av but uses timestamp ranges instead of
    frame index ranges. Used by trim_episode and related GUI editing functions.

    Args:
        input_path: Source video file path.
        output_path: Destination video file path.
        episodes_to_keep: List of (start_time, end_time) tuples in seconds.
        fps: Frame rate of the video.
        vcodec: Video codec to use for encoding.
        pix_fmt: Pixel format for output video.
    """
    from fractions import Fraction

    import av

    if not episodes_to_keep:
        raise ValueError("No episodes to keep")

    in_container = av.open(str(input_path))

    if not in_container.streams.video:
        raise ValueError(
            f"No video streams found in {input_path}. "
            "The video file may be corrupted or empty. "
            "Try re-downloading the dataset or checking the video file."
        )

    v_in = in_container.streams.video[0]

    out = av.open(str(output_path), mode="w")

    fps_fraction = Fraction(fps).limit_denominator(1000)

    encoder_options = {"preset": "12"} if vcodec == "libsvtav1" else {}
    v_out = out.add_stream(vcodec, rate=fps_fraction, options=encoder_options)

    v_out.width = v_in.codec_context.width
    v_out.height = v_in.codec_context.height
    v_out.pix_fmt = pix_fmt

    v_out.time_base = Fraction(1, int(fps))

    out.start_encoding()

    time_ranges = sorted(episodes_to_keep)

    frame_count = 0
    range_idx = 0

    # Half a frame duration epsilon for floating-point timestamp comparisons.
    eps = 0.5 / fps

    for packet in in_container.demux(v_in):
        for frame in packet.decode():
            if frame is None:
                continue

            frame_time = float(frame.pts * frame.time_base) if frame.pts is not None else 0.0

            while range_idx < len(time_ranges) and frame_time >= time_ranges[range_idx][1] - eps:
                range_idx += 1

            if range_idx >= len(time_ranges):
                break

            start_ts, end_ts = time_ranges[range_idx]
            if frame_time < start_ts - eps:
                continue

            new_frame = frame.reformat(width=v_out.width, height=v_out.height, format=v_out.pix_fmt)
            new_frame.pts = frame_count
            new_frame.time_base = Fraction(1, int(fps))

            for pkt in v_out.encode(new_frame):
                out.mux(pkt)

            frame_count += 1

    for pkt in v_out.encode():
        out.mux(pkt)

    out.close()
    in_container.close()


def _copy_and_reindex_videos(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    episode_mapping: dict[int, int],
) -> dict[int, dict]:
    """Copy and filter video files, only re-encoding files with deleted episodes.

    For video files that only contain kept episodes, we copy them directly.
    For files with mixed kept/deleted episodes, we use PyAV filters to efficiently
    re-encode only the desired segments. The encoder used for re-encoding is
    derived per video key from the source dataset's ``meta/info.json`` so the
    destination metadata keeps describing the videos accurately.

    Args:
        src_dataset: Source dataset to copy from
        dst_meta: Destination metadata object
        episode_mapping: Mapping from old episode indices to new indices

    Returns:
        dict mapping episode index to its video metadata (chunk_index, file_index, timestamps)
    """
    if src_dataset.meta.episodes is None:
        src_dataset.meta.episodes = load_episodes(src_dataset.meta.root)

    episodes_video_metadata: dict[int, dict] = {new_idx: {} for new_idx in episode_mapping.values()}

    for video_key in src_dataset.meta.video_keys:
        logging.info(f"Processing videos for {video_key}")
        video_encoder = encoder_config_from_video_info(
            src_dataset.meta.info.features.get(video_key, {}).get("info")
        )

        if dst_meta.video_path is None:
            raise ValueError("Destination metadata has no video_path defined")

        file_to_episodes: dict[tuple[int, int], list[int]] = {}
        for old_idx in episode_mapping:
            src_ep = src_dataset.meta.episodes[old_idx]
            chunk_idx = src_ep[f"videos/{video_key}/chunk_index"]
            file_idx = src_ep[f"videos/{video_key}/file_index"]
            file_key = (chunk_idx, file_idx)
            if file_key not in file_to_episodes:
                file_to_episodes[file_key] = []
            file_to_episodes[file_key].append(old_idx)

        for (src_chunk_idx, src_file_idx), episodes_in_file in tqdm(
            sorted(file_to_episodes.items()), desc=f"Processing {video_key} video files"
        ):
            all_episodes_in_file = [
                ep_idx
                for ep_idx in range(src_dataset.meta.total_episodes)
                if src_dataset.meta.episodes[ep_idx].get(f"videos/{video_key}/chunk_index") == src_chunk_idx
                and src_dataset.meta.episodes[ep_idx].get(f"videos/{video_key}/file_index") == src_file_idx
            ]

            episodes_to_keep_set = set(episodes_in_file)
            all_in_file_set = set(all_episodes_in_file)

            if all_in_file_set == episodes_to_keep_set:
                assert src_dataset.meta.video_path is not None
                src_video_path = src_dataset.root / src_dataset.meta.video_path.format(
                    video_key=video_key, chunk_index=src_chunk_idx, file_index=src_file_idx
                )
                dst_video_path = dst_meta.root / dst_meta.video_path.format(
                    video_key=video_key, chunk_index=src_chunk_idx, file_index=src_file_idx
                )
                dst_video_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src_video_path, dst_video_path)

                for old_idx in episodes_in_file:
                    new_idx = episode_mapping[old_idx]
                    src_ep = src_dataset.meta.episodes[old_idx]
                    episodes_video_metadata[new_idx][f"videos/{video_key}/chunk_index"] = src_chunk_idx
                    episodes_video_metadata[new_idx][f"videos/{video_key}/file_index"] = src_file_idx
                    episodes_video_metadata[new_idx][f"videos/{video_key}/from_timestamp"] = src_ep[
                        f"videos/{video_key}/from_timestamp"
                    ]
                    episodes_video_metadata[new_idx][f"videos/{video_key}/to_timestamp"] = src_ep[
                        f"videos/{video_key}/to_timestamp"
                    ]
            else:
                # Build list of frame ranges to keep, in sorted order.
                sorted_keep_episodes = sorted(episodes_in_file, key=lambda x: episode_mapping[x])
                episodes_to_keep_ranges: list[tuple[int, int]] = []
                for old_idx in sorted_keep_episodes:
                    src_ep = src_dataset.meta.episodes[old_idx]
                    from_frame = round(src_ep[f"videos/{video_key}/from_timestamp"] * src_dataset.meta.fps)
                    to_frame = round(src_ep[f"videos/{video_key}/to_timestamp"] * src_dataset.meta.fps)
                    assert src_ep["length"] == to_frame - from_frame, (
                        f"Episode length mismatch: {src_ep['length']} vs {to_frame - from_frame}"
                    )
                    episodes_to_keep_ranges.append((from_frame, to_frame))

                # Use PyAV filters to efficiently re-encode only the desired segments.
                assert src_dataset.meta.video_path is not None
                src_video_path = src_dataset.root / src_dataset.meta.video_path.format(
                    video_key=video_key, chunk_index=src_chunk_idx, file_index=src_file_idx
                )
                dst_video_path = dst_meta.root / dst_meta.video_path.format(
                    video_key=video_key, chunk_index=src_chunk_idx, file_index=src_file_idx
                )
                dst_video_path.parent.mkdir(parents=True, exist_ok=True)

                logging.info(
                    f"Re-encoding {video_key} (chunk {src_chunk_idx}, file {src_file_idx}) "
                    f"with {len(episodes_to_keep_ranges)} episodes"
                )
                _keep_episodes_from_video_with_av(
                    src_video_path,
                    dst_video_path,
                    episodes_to_keep_ranges,
                    src_dataset.meta.fps,
                    video_encoder,
                )

                cumulative_ts = 0.0
                for old_idx in sorted_keep_episodes:
                    new_idx = episode_mapping[old_idx]
                    src_ep = src_dataset.meta.episodes[old_idx]
                    ep_length = src_ep["length"]
                    ep_duration = ep_length / src_dataset.meta.fps

                    episodes_video_metadata[new_idx][f"videos/{video_key}/chunk_index"] = src_chunk_idx
                    episodes_video_metadata[new_idx][f"videos/{video_key}/file_index"] = src_file_idx
                    episodes_video_metadata[new_idx][f"videos/{video_key}/from_timestamp"] = cumulative_ts
                    episodes_video_metadata[new_idx][f"videos/{video_key}/to_timestamp"] = (
                        cumulative_ts + ep_duration
                    )

                    cumulative_ts += ep_duration

    return episodes_video_metadata


def _copy_and_reindex_episodes_metadata(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    episode_mapping: dict[int, int],
    data_metadata: dict[int, dict],
    video_metadata: dict[int, dict] | None = None,
) -> None:
    """Copy and reindex episodes metadata using provided data and video metadata.

    Args:
        src_dataset: Source dataset to copy from
        dst_meta: Destination metadata object
        episode_mapping: Mapping from old episode indices to new indices
        data_metadata: Dict mapping new episode index to its data file metadata
        video_metadata: Optional dict mapping new episode index to its video metadata
    """
    from lerobot.utils.utils import flatten_dict

    if src_dataset.meta.episodes is None:
        src_dataset.meta.episodes = load_episodes(src_dataset.meta.root)

    all_stats = []
    total_frames = 0

    for old_idx, new_idx in tqdm(
        sorted(episode_mapping.items(), key=lambda x: x[1]), desc="Processing episodes metadata"
    ):
        src_episode_full = _load_episode_with_stats(src_dataset, old_idx)

        src_episode = src_dataset.meta.episodes[old_idx]

        episode_meta = data_metadata[new_idx].copy()

        if video_metadata and new_idx in video_metadata:
            episode_meta.update(video_metadata[new_idx])

        # Extract episode statistics from parquet metadata.
        # When pandas/pyarrow serializes numpy arrays with shape (C, 1, 1) to parquet,
        # they are being deserialized as nested object arrays like:
        #   array([array([array([0.])]), array([array([0.])]), array([array([0.])])])
        # This happens particularly with image/video statistics. We need to detect and flatten
        # these nested structures back to proper (C, 1, 1) arrays so aggregate_stats can process them.
        episode_stats = {}
        for key in src_episode_full:
            if key.startswith("stats/"):
                stat_key = key.replace("stats/", "")
                parts = stat_key.split("/")
                if len(parts) == 2:
                    feature_name, stat_name = parts
                    if feature_name not in episode_stats:
                        episode_stats[feature_name] = {}

                    value = src_episode_full[key]

                    if feature_name in src_dataset.meta.features:
                        feature_dtype = src_dataset.meta.features[feature_name]["dtype"]
                        if feature_dtype in ["image", "video"] and stat_name != "count":
                            # Stats are channel-first (C, 1, 1)
                            if isinstance(value, np.ndarray) and value.dtype == object:
                                flat_values = []
                                for item in value:
                                    while isinstance(item, np.ndarray):
                                        item = item.flatten()[0]
                                    flat_values.append(item)
                                value = np.array(flat_values, dtype=np.float64).reshape(-1, 1, 1)
                            elif isinstance(value, np.ndarray) and value.ndim == 1:
                                value = value.reshape(-1, 1, 1)

                    episode_stats[feature_name][stat_name] = value

        all_stats.append(episode_stats)

        episode_dict = {
            "episode_index": new_idx,
            "tasks": src_episode["tasks"],
            "length": src_episode["length"],
        }
        episode_dict.update(episode_meta)
        episode_dict.update(flatten_dict({"stats": episode_stats}))
        dst_meta._save_episode_metadata(episode_dict)

        total_frames += src_episode["length"]

    dst_meta._close_writer()

    dst_meta.info.total_episodes = len(episode_mapping)
    dst_meta.info.total_frames = total_frames
    dst_meta.info.total_tasks = len(dst_meta.tasks) if dst_meta.tasks is not None else 0
    dst_meta.info.splits = {"train": f"0:{len(episode_mapping)}"}
    write_info(dst_meta.info, dst_meta.root)

    if not all_stats:
        logging.warning("No statistics found to aggregate")
        return

    logging.info(f"Aggregating statistics for {len(all_stats)} episodes")
    aggregated_stats = aggregate_stats(all_stats)
    filtered_stats = {k: v for k, v in aggregated_stats.items() if k in dst_meta.features}
    write_stats(filtered_stats, dst_meta.root)


def _write_parquet(df: pd.DataFrame, path: Path, meta: LeRobotDatasetMetadata) -> None:
    """Write DataFrame to parquet

    This ensures images are properly embedded and the file can be loaded correctly by HF datasets.
    """
    from lerobot.datasets.feature_utils import get_hf_features_from_features
    from lerobot.datasets.io_utils import embed_images

    hf_features = get_hf_features_from_features(meta.features)
    ep_dataset = datasets.Dataset.from_dict(df.to_dict(orient="list"), features=hf_features, split="train")

    if len(meta.image_keys) > 0:
        ep_dataset = embed_images(ep_dataset)

    table = ep_dataset.with_format("arrow")[:]
    writer = pq.ParquetWriter(path, schema=table.schema, compression="snappy", use_dictionary=True)
    writer.write_table(table)
    writer.close()


def _save_data_chunk(
    df: pd.DataFrame,
    meta: LeRobotDatasetMetadata,
    chunk_idx: int = 0,
    file_idx: int = 0,
) -> tuple[int, int, dict[int, dict]]:
    """Save a data chunk and return updated indices and episode metadata.

    Returns:
        tuple: (next_chunk_idx, next_file_idx, episode_metadata_dict)
            where episode_metadata_dict maps episode_index to its data file metadata
    """
    path = meta.root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
    path.parent.mkdir(parents=True, exist_ok=True)

    _write_parquet(df, path, meta)

    episode_metadata = {}
    for ep_idx in df["episode_index"].unique():
        ep_df = df[df["episode_index"] == ep_idx]
        episode_metadata[ep_idx] = {
            "data/chunk_index": chunk_idx,
            "data/file_index": file_idx,
            "dataset_from_index": int(ep_df["index"].min()),
            "dataset_to_index": int(ep_df["index"].max() + 1),
        }

    file_size = get_parquet_file_size_in_mb(path)
    if file_size >= DEFAULT_DATA_FILE_SIZE_IN_MB * 0.9:
        chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)

    return chunk_idx, file_idx, episode_metadata


def _copy_data_with_feature_changes(
    dataset: LeRobotDataset,
    new_meta: LeRobotDatasetMetadata,
    add_features: dict[str, tuple] | None = None,
    remove_features: list[str] | None = None,
    rename_features: dict[str, str] | None = None,
) -> None:
    """Copy data while adding, removing, or renaming features."""
    data_dir = dataset.root / DATA_DIR
    parquet_files = sorted(data_dir.glob("*/*.parquet"))

    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    frame_idx = 0

    for src_path in tqdm(parquet_files, desc="Processing data files"):
        df = pd.read_parquet(src_path).reset_index(drop=True)

        relative_path = src_path.relative_to(dataset.root)
        chunk_dir = relative_path.parts[1]
        file_name = relative_path.parts[2]

        chunk_idx = int(chunk_dir.split("-")[1])
        file_idx = int(file_name.split("-")[1].split(".")[0])

        if remove_features:
            df = df.drop(columns=remove_features, errors="ignore")

        if rename_features:
            df = df.rename(columns=rename_features)

        if add_features:
            end_idx = frame_idx + len(df)
            for feature_name, (values, _) in add_features.items():
                if callable(values):
                    feature_values = []
                    for _, row in df.iterrows():
                        ep_idx = row["episode_index"]
                        frame_in_ep = row["frame_index"]
                        value = values(row.to_dict(), ep_idx, frame_in_ep)
                        if isinstance(value, np.ndarray) and value.size == 1:
                            value = value.item()
                        feature_values.append(value)
                    df[feature_name] = feature_values
                else:
                    feature_slice = values[frame_idx:end_idx]
                    if len(feature_slice.shape) > 1 and feature_slice.shape[1] == 1:
                        df[feature_name] = feature_slice.flatten()
                    else:
                        df[feature_name] = feature_slice
            frame_idx = end_idx

        # Write using the same chunk/file structure as source
        dst_path = new_meta.root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        _write_parquet(df, dst_path, new_meta)

    _copy_episodes_metadata_and_stats(dataset, new_meta, rename_features=rename_features)


def _copy_videos(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    exclude_keys: list[str] | None = None,
    rename_keys: dict[str, str] | None = None,
) -> None:
    """Copy video files, optionally excluding or renaming certain keys."""
    if exclude_keys is None:
        exclude_keys = []

    for video_key in src_dataset.meta.video_keys:
        if video_key in exclude_keys:
            continue

        # Determine destination key (renamed or original)
        dst_key = rename_keys.get(video_key, video_key) if rename_keys else video_key

        video_files = set()
        for ep_idx in range(len(src_dataset.meta.episodes)):
            try:
                video_files.add(src_dataset.meta.get_video_file_path(ep_idx, video_key))
            except KeyError:
                continue

        for src_path in tqdm(sorted(video_files), desc=f"Copying {video_key} videos"):
            # Replace old key with new key in the path
            dst_path_str = str(src_path).replace(f"videos/{video_key}/", f"videos/{dst_key}/")
            dst_path = dst_meta.root / dst_path_str
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_dataset.root / src_path, dst_path)


def _copy_episodes_metadata_and_stats(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    rename_features: dict[str, str] | None = None,
) -> None:
    """Copy episodes metadata and recalculate stats, optionally renaming feature columns."""
    if src_dataset.meta.tasks is not None:
        write_tasks(src_dataset.meta.tasks, dst_meta.root)
        dst_meta.tasks = src_dataset.meta.tasks.copy()

    episodes_dir = src_dataset.root / "meta/episodes"
    dst_episodes_dir = dst_meta.root / "meta/episodes"

    if episodes_dir.exists():
        if rename_features:
            # Need to rename columns in episode metadata files
            episode_files = sorted(episodes_dir.glob("*/*.parquet"))
            for ep_file in tqdm(episode_files, desc="Renaming episode metadata columns"):
                df = pd.read_parquet(ep_file)

                # Build column mapping for stats/* and videos/* columns
                column_mapping = {}
                for old_name, new_name in rename_features.items():
                    for col in df.columns:
                        if col.startswith(f"stats/{old_name}/"):
                            column_mapping[col] = col.replace(f"stats/{old_name}/", f"stats/{new_name}/")
                        elif col.startswith(f"videos/{old_name}/"):
                            column_mapping[col] = col.replace(f"videos/{old_name}/", f"videos/{new_name}/")

                if column_mapping:
                    df = df.rename(columns=column_mapping)

                # Write to destination
                relative_path = ep_file.relative_to(episodes_dir)
                dst_file = dst_episodes_dir / relative_path
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(dst_file, index=False)
        else:
            # No renaming, just copy the directory
            shutil.copytree(episodes_dir, dst_episodes_dir, dirs_exist_ok=True)

    dst_meta.info.total_episodes = src_dataset.meta.total_episodes
    dst_meta.info.total_frames = src_dataset.meta.total_frames
    dst_meta.info.total_tasks = src_dataset.meta.total_tasks
    # Preserve original splits if available, otherwise create default
    dst_meta.info.splits = (
        src_dataset.meta.info.splits
        if src_dataset.meta.info.splits
        else {"train": f"0:{src_dataset.meta.total_episodes}"}
    )

    if dst_meta.video_keys and src_dataset.meta.video_keys:
        for key in dst_meta.video_keys:
            # Check in source features (with old name if renamed)
            src_key = key
            if rename_features:
                # Reverse lookup: find old name for this new name
                for old, new in rename_features.items():
                    if new == key:
                        src_key = old
                        break

            if src_key in src_dataset.meta.features:
                dst_meta.info.features[key]["info"] = deepcopy(
                    src_dataset.meta.info.features[src_key].get("info", {})
                )

    write_info(dst_meta.info, dst_meta.root)

    # Handle stats
    if set(dst_meta.features.keys()) != set(src_dataset.meta.features.keys()):
        # Features were added, removed, or renamed
        logging.info("Recalculating dataset statistics...")
        if src_dataset.meta.stats:
            new_stats = {}
            for key in dst_meta.features:
                # Map back to source key if renamed
                src_key = key
                if rename_features:
                    for old, new in rename_features.items():
                        if new == key:
                            src_key = old
                            break

                if src_key in src_dataset.meta.stats:
                    new_stats[key] = src_dataset.meta.stats[src_key]
            write_stats(new_stats, dst_meta.root)
    else:
        # No features changed, copy as-is
        if src_dataset.meta.stats:
            write_stats(src_dataset.meta.stats, dst_meta.root)


def _save_episode_images_for_video(
    dataset: LeRobotDataset,
    imgs_dir: Path,
    img_key: str,
    episode_index: int,
    num_workers: int = 4,
) -> None:
    """Save images from a specific episode and camera to disk for video encoding.

    Args:
        dataset: The LeRobot dataset to extract images from
        imgs_dir: Directory to save images to
        img_key: The image key (camera) to extract
        episode_index: Index of the episode to save
        num_workers: Number of threads for parallel image saving
    """
    # Create directory
    imgs_dir.mkdir(parents=True, exist_ok=True)

    # Get dataset without torch format for PIL image access
    hf_dataset = dataset.hf_dataset.with_format(None)

    # Select only this camera's images
    imgs_dataset = hf_dataset.select_columns(img_key)

    # Get episode start and end indices
    from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
    to_idx = dataset.meta.episodes["dataset_to_index"][episode_index]

    # Get all items for this episode
    episode_dataset = imgs_dataset.select(range(from_idx, to_idx))

    is_depth = img_key in dataset.meta.depth_keys
    frame_pattern = DEPTH_FILE_PATTERN if is_depth else IMAGE_FILE_PATTERN

    # Define function to save a single image
    def save_single_image(i_item_tuple):
        i, item = i_item_tuple
        write_image(item[img_key], imgs_dir / frame_pattern.format(frame_index=i))
        return i

    items = list(enumerate(episode_dataset))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(save_single_image, item) for item in items]
        for future in as_completed(futures):
            future.result()  # This will raise any exceptions that occurred


def _save_batch_episodes_images(
    dataset: LeRobotDataset,
    imgs_dir: Path,
    img_key: str,
    episode_indices: list[int],
    num_workers: int = 4,
) -> list[float]:
    """Save images from multiple episodes to disk for batch video encoding.

    Args:
        dataset: The LeRobot dataset to extract images from
        imgs_dir: Directory to save images to
        img_key: The image key (camera) to extract
        episode_indices: List of episode indices to save
        num_workers: Number of threads for parallel image saving

    Returns:
        List of episode durations in seconds
    """
    imgs_dir.mkdir(parents=True, exist_ok=True)
    hf_dataset = dataset.hf_dataset.with_format(None)
    imgs_dataset = hf_dataset.select_columns(img_key)

    is_depth = img_key in dataset.meta.depth_keys
    frame_pattern = DEPTH_FILE_PATTERN if is_depth else IMAGE_FILE_PATTERN

    # Define function to save a single image with global frame index
    # Defined once outside the loop to avoid repeated closure creation
    def save_single_image(i_item_tuple, base_frame_idx, img_key_param):
        i, item = i_item_tuple
        write_image(item[img_key_param], imgs_dir / frame_pattern.format(frame_index=base_frame_idx + i))
        return i

    episode_durations = []
    frame_idx = 0

    for ep_idx in episode_indices:
        # Get episode range
        from_idx = dataset.meta.episodes["dataset_from_index"][ep_idx]
        to_idx = dataset.meta.episodes["dataset_to_index"][ep_idx]
        episode_length = to_idx - from_idx
        episode_durations.append(episode_length / dataset.fps)

        # Get episode images
        episode_dataset = imgs_dataset.select(range(from_idx, to_idx))

        # Save images
        items = list(enumerate(episode_dataset))
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(save_single_image, item, frame_idx, img_key) for item in items]
            for future in as_completed(futures):
                future.result()

        frame_idx += episode_length

    return episode_durations


def _iter_episode_batches(
    episode_indices: list[int],
    episode_lengths: dict[int, int],
    size_per_frame_mb: float,
    video_file_size_limit: float,
    max_episodes: int | None,
    max_frames: int | None,
):
    """Generator that yields batches of episode indices for video encoding.

    Groups episodes into batches that respect size and memory constraints:
    - Stays under video file size limit
    - Respects maximum episodes per batch (if specified)
    - Respects maximum frames per batch (if specified)

    Args:
        episode_indices: List of episode indices to batch
        episode_lengths: Dictionary mapping episode index to episode length
        size_per_frame_mb: Estimated size per frame in MB
        video_file_size_limit: Maximum video file size in MB
        max_episodes: Maximum number of episodes per batch (None = no limit)
        max_frames: Maximum number of frames per batch (None = no limit)

    Yields:
        List of episode indices for each batch
    """
    batch_episodes = []
    estimated_size = 0.0
    total_frames = 0

    for ep_idx in episode_indices:
        ep_length = episode_lengths[ep_idx]
        ep_estimated_size = ep_length * size_per_frame_mb

        # we check if adding this episode would exceed any constraint
        would_exceed_size = estimated_size > 0 and estimated_size + ep_estimated_size >= video_file_size_limit
        would_exceed_episodes = max_episodes is not None and len(batch_episodes) >= max_episodes
        would_exceed_frames = max_frames is not None and total_frames + ep_length > max_frames

        if batch_episodes and (would_exceed_size or would_exceed_episodes or would_exceed_frames):
            # yield current batch before adding this episode
            yield batch_episodes
            # start a new batch with current episode
            batch_episodes = [ep_idx]
            estimated_size = ep_estimated_size
            total_frames = ep_length
        else:
            # add to current batch
            batch_episodes.append(ep_idx)
            estimated_size += ep_estimated_size
            total_frames += ep_length

    # yield final batch if not empty
    if batch_episodes:
        yield batch_episodes


def _estimate_frame_size_via_calibration(
    dataset: LeRobotDataset,
    img_key: str,
    episode_indices: list[int],
    temp_dir: Path,
    fps: int,
    video_encoder: VideoEncoderConfig,
    num_calibration_frames: int = 30,
) -> float:
    """Estimate MB per frame by encoding a small calibration sample.

    Encodes a representative sample of frames using the exact codec parameters
    to measure actual compression ratio, which is more accurate than heuristics.

    Args:
        dataset: Source dataset with images.
        img_key: Image key to calibrate (e.g., "observation.images.top").
        episode_indices: List of episode indices being processed.
        temp_dir: Temporary directory for calibration files.
        fps: Frames per second for video encoding.
        video_encoder: Video encoder settings used for calibration encoding.
        num_calibration_frames: Number of frames to use for calibration (default: 30).

    Returns:
        Estimated size in MB per frame based on actual encoding.
    """
    calibration_dir = temp_dir / "calibration" / img_key
    calibration_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Select a representative episode (prefer middle episode if available)
        calibration_ep_idx = episode_indices[len(episode_indices) // 2]

        # Get episode range
        from_idx = dataset.meta.episodes["dataset_from_index"][calibration_ep_idx]
        to_idx = dataset.meta.episodes["dataset_to_index"][calibration_ep_idx]
        episode_length = to_idx - from_idx

        # Use up to num_calibration_frames from this episode
        num_frames = min(num_calibration_frames, episode_length)

        # Get frames from dataset
        hf_dataset = dataset.hf_dataset.with_format(None)
        sample_indices = range(from_idx, from_idx + num_frames)

        # Save calibration frames using the suffix/format the encoder expects.
        is_depth = img_key in dataset.meta.depth_keys
        frame_pattern = DEPTH_FILE_PATTERN if is_depth else IMAGE_FILE_PATTERN
        for i, idx in enumerate(sample_indices):
            write_image(hf_dataset[idx][img_key], calibration_dir / frame_pattern.format(frame_index=i))

        # Encode calibration video
        calibration_video_path = calibration_dir / "calibration.mp4"
        encode_video_frames(
            imgs_dir=calibration_dir,
            video_path=calibration_video_path,
            fps=fps,
            video_encoder=video_encoder,
            overwrite=True,
        )

        # Measure actual compressed size
        video_size_bytes = calibration_video_path.stat().st_size
        video_size_mb = video_size_bytes / BYTES_PER_MIB
        size_per_frame_mb = video_size_mb / num_frames

        logging.info(
            f"  Calibration: {num_frames} frames -> {video_size_mb:.2f} MB "
            f"= {size_per_frame_mb:.4f} MB/frame for {img_key}"
        )

        return size_per_frame_mb

    finally:
        # Clean up calibration files
        if calibration_dir.exists():
            # safe-destruct: internal calibration scratch dir
            shutil.rmtree(calibration_dir)


def _copy_data_without_images(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    episode_indices: list[int],
    img_keys: list[str],
) -> None:
    """Copy data files without image columns.

    Args:
        src_dataset: Source dataset
        dst_meta: Destination metadata
        episode_indices: Episodes to include
        img_keys: Image keys to remove
    """
    from lerobot.datasets.utils import DATA_DIR

    data_dir = src_dataset.root / DATA_DIR
    parquet_files = sorted(data_dir.glob("*/*.parquet"))

    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    episode_set = set(episode_indices)

    for src_path in tqdm(parquet_files, desc="Processing data files"):
        df = pd.read_parquet(src_path).reset_index(drop=True)

        # Filter to only include selected episodes
        df = df[df["episode_index"].isin(episode_set)].copy()

        if len(df) == 0:
            continue

        # Remove image columns
        columns_to_drop = [col for col in img_keys if col in df.columns]
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)

        # Get chunk and file indices from path
        relative_path = src_path.relative_to(src_dataset.root)
        chunk_dir = relative_path.parts[1]
        file_name = relative_path.parts[2]
        chunk_idx = int(chunk_dir.split("-")[1])
        file_idx = int(file_name.split("-")[1].split(".")[0])

        # Write to destination without pandas index
        dst_path = dst_meta.root / f"data/chunk-{chunk_idx:03d}/file-{file_idx:03d}.parquet"
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst_path, index=False)


# Video conversion constants
BYTES_PER_KIB = 1024
BYTES_PER_MIB = BYTES_PER_KIB * BYTES_PER_KIB


def modify_tasks(
    dataset: LeRobotDataset,
    new_task: str | None = None,
    episode_tasks: dict[int, str] | None = None,
) -> LeRobotDataset:
    """Modify tasks in a LeRobotDataset.

    This function allows you to either:
    1. Set a single task for the entire dataset (using `new_task`)
    2. Set specific tasks for specific episodes (using `episode_tasks`)

    You can combine both: `new_task` sets the default, and `episode_tasks` overrides
    specific episodes.

    The dataset is modified in-place, updating only the task-related files:
    - meta/tasks.parquet
    - data/**/*.parquet (task_index column)
    - meta/episodes/**/*.parquet (tasks column)
    - meta/info.json (total_tasks)

    Args:
        dataset: The source LeRobotDataset to modify.
        new_task: A single task string to apply to all episodes. If None and episode_tasks
            is also None, raises an error.
        episode_tasks: Optional dict mapping episode indices to their task strings.
            Overrides `new_task` for specific episodes.


    Examples:
        Set a single task for all episodes:
            dataset = modify_tasks(dataset, new_task="Pick up the cube")

        Set different tasks for specific episodes:
            dataset = modify_tasks(
                dataset,
                episode_tasks={0: "Task A", 1: "Task B", 2: "Task A"}
            )

        Set a default task with overrides:
            dataset = modify_tasks(
                dataset,
                new_task="Default task",
                episode_tasks={5: "Special task for episode 5"}
            )
    """
    if new_task is None and episode_tasks is None:
        raise ValueError("Must specify at least one of new_task or episode_tasks")

    if episode_tasks is not None:
        valid_indices = set(range(dataset.meta.total_episodes))
        invalid = set(episode_tasks.keys()) - valid_indices
        if invalid:
            raise ValueError(f"Invalid episode indices: {invalid}")

    # Ensure episodes metadata is loaded
    if dataset.meta.episodes is None:
        dataset.meta.episodes = load_episodes(dataset.root)

    # Build the mapping from episode index to task string
    episode_to_task: dict[int, str] = {}
    for ep_idx in range(dataset.meta.total_episodes):
        if episode_tasks and ep_idx in episode_tasks:
            episode_to_task[ep_idx] = episode_tasks[ep_idx]
        elif new_task is not None:
            episode_to_task[ep_idx] = new_task
        else:
            # Keep original task if not overridden and no default provided
            original_tasks = dataset.meta.episodes[ep_idx]["tasks"]
            if not original_tasks:
                raise ValueError(f"Episode {ep_idx} has no tasks and no default task was provided")
            episode_to_task[ep_idx] = original_tasks[0]

    # Collect all unique tasks and create new task mapping
    unique_tasks = sorted(set(episode_to_task.values()))
    new_task_df = pd.DataFrame(
        {"task_index": list(range(len(unique_tasks)))}, index=pd.Index(unique_tasks, name="task")
    )
    task_to_index = {task: idx for idx, task in enumerate(unique_tasks)}

    logging.info(f"Modifying tasks in {dataset.repo_id}")
    logging.info(f"New tasks: {unique_tasks}")

    root = dataset.root

    # Update data files - modify task_index column
    logging.info("Updating data files...")
    data_dir = root / DATA_DIR

    for parquet_path in tqdm(sorted(data_dir.rglob("*.parquet")), desc="Updating data"):
        df = pd.read_parquet(parquet_path)

        # Build a mapping from episode_index to new task_index for rows in this file
        episode_indices_in_file = df["episode_index"].unique()
        ep_to_new_task_idx = {
            ep_idx: task_to_index[episode_to_task[ep_idx]] for ep_idx in episode_indices_in_file
        }

        # Update task_index column
        df["task_index"] = df["episode_index"].map(ep_to_new_task_idx)
        df.to_parquet(parquet_path, index=False)

    # Update episodes metadata - modify tasks column
    logging.info("Updating episodes metadata...")
    episodes_dir = root / "meta" / "episodes"

    for parquet_path in tqdm(sorted(episodes_dir.rglob("*.parquet")), desc="Updating episodes"):
        df = pd.read_parquet(parquet_path)

        # Update tasks column
        df["tasks"] = df["episode_index"].apply(lambda ep_idx: [episode_to_task[ep_idx]])
        df.to_parquet(parquet_path, index=False)

    # Write new tasks.parquet
    write_tasks(new_task_df, root)

    # Update info.json
    dataset.meta.info.total_tasks = len(unique_tasks)
    write_info(dataset.meta.info, root)

    # Reload metadata to reflect changes
    dataset.meta.tasks = new_task_df
    dataset.meta.episodes = load_episodes(root)

    logging.info(f"Tasks: {unique_tasks}")

    return dataset


def trim_episode(
    dataset: LeRobotDataset,
    episode_index: int,
    trim_start_s: float = 0.0,
    trim_end_s: float = 0.0,
) -> LeRobotDataset:
    """Trim an episode in-place by removing seconds from the start and/or end.

    This is a convenience wrapper around trim_episode_by_frames for time-based trimming.
    Use this for CLI/human-friendly interfaces where specifying seconds is natural.

    Args:
        dataset: The LeRobotDataset to modify.
        episode_index: Index of the episode to trim.
        trim_start_s: Duration in seconds to remove from the start of the episode.
        trim_end_s: Duration in seconds to remove from the end of the episode.

    Returns:
        The modified dataset (same instance, but files are updated on disk).

    Examples:
        Trim 1.5 seconds from the start:
            dataset = trim_episode(dataset, episode_index=0, trim_start_s=1.5)

        Trim 2 seconds from the end:
            dataset = trim_episode(dataset, episode_index=0, trim_end_s=2.0)

        Trim both ends:
            dataset = trim_episode(dataset, episode_index=0, trim_start_s=0.5, trim_end_s=1.0)
    """
    if trim_start_s < 0 or trim_end_s < 0:
        raise ValueError("trim_start_s and trim_end_s must be non-negative")

    if trim_start_s == 0 and trim_end_s == 0:
        logging.info("No trimming requested, returning dataset unchanged")
        return dataset

    # Validate episode index
    if episode_index < 0 or episode_index >= dataset.meta.total_episodes:
        raise ValueError(
            f"Invalid episode_index {episode_index}. "
            f"Dataset has {dataset.meta.total_episodes} episodes (0-{dataset.meta.total_episodes - 1})"
        )

    # Convert seconds to frames
    frames_to_trim_start = int(trim_start_s * dataset.fps)
    frames_to_trim_end = int(trim_end_s * dataset.fps)

    # Ensure episodes metadata is loaded for length calculation
    if dataset.meta.episodes is None:
        dataset.meta.episodes = load_episodes(dataset.meta.root)

    episode_length = dataset.meta.episodes[episode_index]["length"]
    start_frame = frames_to_trim_start
    end_frame = episode_length - frames_to_trim_end

    # Check if trimming too much
    if start_frame >= end_frame:
        total_trim_s = trim_start_s + trim_end_s
        episode_duration_s = episode_length / dataset.fps
        raise ValueError(
            f"At least one frame must remain after trimming. "
            f"Episode has {episode_length} frames ({episode_duration_s:.2f}s), "
            f"but trying to trim {total_trim_s:.2f}s total."
        )

    return trim_episode_by_frames(dataset, episode_index, start_frame, end_frame)


def trim_episode_by_frames(
    dataset: LeRobotDataset,
    episode_index: int,
    start_frame: int,
    end_frame: int,
    recompute_stats: bool = True,
) -> LeRobotDataset:
    """Trim an episode in-place by specifying the frame range to keep.

    This function modifies the dataset in-place, updating:
    - data/**/*.parquet (frame data)
    - videos/**/*.mp4 (re-encodes video if needed)
    - meta/episodes/**/*.parquet (episode metadata)
    - meta/info.json (total_frames)

    Note: After calling this function, you should reload the dataset to see the changes
    reflected in hf_dataset and episodes metadata.

    Args:
        dataset: The LeRobotDataset to modify.
        episode_index: Index of the episode to trim.
        start_frame: First frame to keep (0-indexed, inclusive).
        end_frame: Last frame to keep (0-indexed, exclusive).

    Returns:
        The modified dataset (same instance, but files are updated on disk).

    Examples:
        Keep only frames 30-90:
            dataset = trim_episode_by_frames(dataset, episode_index=0, start_frame=30, end_frame=90)

        Remove first 10 frames:
            dataset = trim_episode_by_frames(dataset, episode_index=0, start_frame=10, end_frame=100)
    """
    if episode_index < 0 or episode_index >= dataset.meta.total_episodes:
        raise ValueError(
            f"Invalid episode_index {episode_index}. "
            f"Dataset has {dataset.meta.total_episodes} episodes (0-{dataset.meta.total_episodes - 1})"
        )

    # Always reload episodes metadata from disk to ensure we have current data.
    # This is critical when multiple trim operations are applied sequentially,
    # as each trim modifies the episode indices (dataset_from_index, dataset_to_index)
    # for all subsequent episodes. Using stale in-memory metadata causes data corruption.
    dataset.meta.episodes = load_episodes(dataset.meta.root)
    dataset.meta.info = load_info(dataset.meta.root)

    # Get episode info
    episode_meta = dataset.meta.episodes[episode_index]
    episode_length = episode_meta["length"]

    # Validate frame range
    if start_frame < 0 or end_frame > episode_length:
        raise ValueError(
            f"Invalid frame range [{start_frame}, {end_frame}) for episode with {episode_length} frames"
        )
    if start_frame >= end_frame:
        raise ValueError(f"start_frame ({start_frame}) must be less than end_frame ({end_frame})")

    # No trimming needed if keeping full range
    if start_frame == 0 and end_frame == episode_length:
        logging.info("No trimming requested, returning dataset unchanged")
        return dataset

    # Calculate frames to trim from each end
    frames_to_trim_start = start_frame
    frames_to_trim_end = episode_length - end_frame
    new_episode_length = end_frame - start_frame

    if new_episode_length < 1:
        raise ValueError(
            f"At least one frame must remain after trimming. "
            f"Episode has {episode_length} frames, trying to trim {frames_to_trim_start + frames_to_trim_end}."
        )

    logging.info(
        f"Trimming episode {episode_index}: removing {frames_to_trim_start} frames from start, "
        f"{frames_to_trim_end} frames from end. New length: {new_episode_length} frames"
    )

    # Step 1: Update parquet data files
    _trim_episode_parquet_data(
        dataset=dataset,
        episode_index=episode_index,
        frames_to_trim_start=frames_to_trim_start,
        frames_to_trim_end=frames_to_trim_end,
    )

    # Step 2: Update videos if present
    if dataset.meta.video_keys:
        # Convert frames to seconds for video trimming (FFmpeg uses timestamps)
        trim_start_s = frames_to_trim_start / dataset.fps
        trim_end_s = frames_to_trim_end / dataset.fps
        _trim_episode_videos(
            dataset=dataset,
            episode_index=episode_index,
            trim_start_s=trim_start_s,
            trim_end_s=trim_end_s,
        )

    # Step 3: Update episode metadata and info.json
    frames_removed = frames_to_trim_start + frames_to_trim_end
    _trim_episode_metadata(
        dataset=dataset,
        episode_index=episode_index,
        new_length=new_episode_length,
        frames_removed=frames_removed,
    )

    # Step 4: Recompute per-episode stats and re-aggregate stats.json
    if recompute_stats:
        _recompute_episode_stats_from_data(dataset.root, episode_index, dataset.meta.features)
        _reaggregate_and_write_stats(dataset.root, dataset.meta.features)
    else:
        _recompute_episode_stats_from_data(dataset.root, episode_index, dataset.meta.features)

    logging.info(f"Episode {episode_index} trimmed successfully")
    return dataset


def _trim_episode_parquet_data(
    dataset: LeRobotDataset,
    episode_index: int,
    frames_to_trim_start: int,
    frames_to_trim_end: int,
) -> None:
    """Update parquet data files to trim frames from an episode.

    Also updates global indices for all subsequent frames/episodes.
    """
    data_dir = dataset.root / DATA_DIR
    parquet_files = sorted(data_dir.glob("*/*.parquet"))

    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    frames_removed = frames_to_trim_start + frames_to_trim_end

    # Get the episode's current data bounds
    episode_meta = dataset.meta.episodes[episode_index]
    episode_length = episode_meta["length"]
    ep_from_idx = episode_meta["dataset_from_index"]

    # Calculate frame_index range to keep (0-based within episode)
    # Use frame_index for filtering since it's always reliable (0 to length-1),
    # and works correctly whether the data index is global or per-file
    keep_frame_from = frames_to_trim_start
    keep_frame_to = episode_length - frames_to_trim_end

    for parquet_path in tqdm(sorted(parquet_files), desc="Updating data files"):
        df = pd.read_parquet(parquet_path)

        # Check what episodes are in this file
        episodes_in_file = set(df["episode_index"].unique())

        # Skip files that don't contain target episode or any subsequent episodes
        # (subsequent episodes need their global indices shifted)
        if episode_index not in episodes_in_file and all(ep < episode_index for ep in episodes_in_file):
            continue

        modified = False

        if episode_index in episodes_in_file:
            # Filter out trimmed frames from target episode using frame_index
            # Keep frames where: not in target episode, OR in kept frame_index range
            target_ep_mask = df["episode_index"] == episode_index
            keep_mask = ~target_ep_mask | (
                (df["frame_index"] >= keep_frame_from) & (df["frame_index"] < keep_frame_to)
            )
            df = df[keep_mask].copy()

            # Reset frame_index and timestamps for the trimmed episode
            ep_rows = df["episode_index"] == episode_index
            if ep_rows.sum() > 0:
                df.loc[ep_rows, "frame_index"] = range(ep_rows.sum())
                df.loc[ep_rows, "timestamp"] = [i / dataset.fps for i in range(ep_rows.sum())]
                # Recalculate indices for the trimmed episode (starts at same from_index)
                df.loc[ep_rows, "index"] = range(ep_from_idx, ep_from_idx + ep_rows.sum())

            modified = True

        # Shift global indices for frames in subsequent episodes
        if frames_removed > 0:
            subsequent_mask = df["episode_index"] > episode_index
            if subsequent_mask.any():
                df.loc[subsequent_mask, "index"] -= frames_removed
                modified = True

        if modified:
            df = df.reset_index(drop=True)
            df.to_parquet(parquet_path, index=False)


def _trim_episode_videos(
    dataset: LeRobotDataset,
    episode_index: int,
    trim_start_s: float,
    trim_end_s: float,
) -> None:
    """Re-encode video files to apply trimming to an episode."""
    import tempfile

    if dataset.meta.episodes is None:
        dataset.meta.episodes = load_episodes(dataset.meta.root)

    for video_key in dataset.meta.video_keys:
        logging.info(f"Processing video: {video_key}")

        # Get codec settings from the dataset's video info
        video_info = dataset.meta.features[video_key].get("info") or {}
        # Map canonical codec names to encoder names
        codec_map = {"av1": "libsvtav1", "h264": "libx264", "hevc": "libx265"}
        source_codec = video_info.get("video.codec", "av1") if video_info else "av1"
        vcodec = codec_map.get(source_codec, "libsvtav1")
        pix_fmt = video_info.get("video.pix_fmt", "yuv420p")

        # Get video file info for this episode
        episode_meta = dataset.meta.episodes[episode_index]
        chunk_idx = episode_meta[f"videos/{video_key}/chunk_index"]
        file_idx = episode_meta[f"videos/{video_key}/file_index"]

        # Find all episodes in this video file
        episodes_in_file = []
        for ep_idx in range(dataset.meta.total_episodes):
            ep_meta = dataset.meta.episodes[ep_idx]
            if (
                ep_meta.get(f"videos/{video_key}/chunk_index") == chunk_idx
                and ep_meta.get(f"videos/{video_key}/file_index") == file_idx
            ):
                episodes_in_file.append(ep_idx)

        # Build time ranges for all episodes in this file
        time_ranges = []
        for ep_idx in episodes_in_file:
            ep_meta = dataset.meta.episodes[ep_idx]
            from_ts = ep_meta[f"videos/{video_key}/from_timestamp"]
            to_ts = ep_meta[f"videos/{video_key}/to_timestamp"]

            if ep_idx == episode_index:
                # Apply trimming to target episode
                new_from_ts = from_ts + trim_start_s
                new_to_ts = to_ts - trim_end_s
                if new_from_ts < new_to_ts:
                    time_ranges.append((new_from_ts, new_to_ts))
            else:
                time_ranges.append((from_ts, to_ts))

        if not time_ranges:
            continue

        # Re-encode the video with the new time ranges
        assert dataset.meta.video_path is not None
        video_path = dataset.root / dataset.meta.video_path.format(
            video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
        )

        # Create a temporary file for the new video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            _keep_episodes_from_video_by_time(
                input_path=video_path,
                output_path=tmp_path,
                episodes_to_keep=time_ranges,
                fps=dataset.fps,
                vcodec=vcodec,
                pix_fmt=pix_fmt,
            )

            # Replace original with new video
            # safe-destruct: in-place trim: replace user video with trimmed; user-confirmed
            shutil.move(str(tmp_path), str(video_path))
        finally:
            if tmp_path.exists():
                # safe-destruct: in-place trim: drop temp on error path
                tmp_path.unlink()


def _trim_episode_metadata(
    dataset: LeRobotDataset,
    episode_index: int,
    new_length: int,
    frames_removed: int,
) -> None:
    """Update episode metadata and info.json after trimming."""
    episodes_dir = dataset.root / "meta" / "episodes"

    for parquet_path in sorted(episodes_dir.rglob("*.parquet")):
        df = pd.read_parquet(parquet_path)

        modified = False

        # Update length for the trimmed episode
        if episode_index in df["episode_index"].values:
            mask = df["episode_index"] == episode_index
            df.loc[mask, "length"] = new_length
            modified = True

        # Update dataset_from_index and dataset_to_index for all episodes
        for i, row in df.iterrows():
            ep_idx = row["episode_index"]
            if ep_idx == episode_index:
                # Trimmed episode: from_index stays same, to_index = from_index + new_length
                df.at[i, "dataset_to_index"] = row["dataset_from_index"] + new_length
                modified = True
            elif ep_idx > episode_index:
                # Subsequent episodes: shift down by frames_removed
                df.at[i, "dataset_from_index"] = row["dataset_from_index"] - frames_removed
                df.at[i, "dataset_to_index"] = row["dataset_to_index"] - frames_removed
                modified = True

        # Update video timestamps if present
        for video_key in dataset.meta.video_keys:
            from_ts_col = f"videos/{video_key}/from_timestamp"
            to_ts_col = f"videos/{video_key}/to_timestamp"

            if from_ts_col in df.columns:
                mask = df["episode_index"] == episode_index
                if mask.any():
                    # Recalculate timestamps for all episodes in the same video file
                    chunk_col = f"videos/{video_key}/chunk_index"
                    file_col = f"videos/{video_key}/file_index"

                    target_chunk = df.loc[mask, chunk_col].iloc[0]
                    target_file = df.loc[mask, file_col].iloc[0]

                    same_file_mask = (df[chunk_col] == target_chunk) & (df[file_col] == target_file)
                    cumulative_ts = 0.0

                    for idx in df[same_file_mask].index:
                        ep_length = df.at[idx, "length"]
                        ep_duration = ep_length / dataset.fps

                        df.at[idx, from_ts_col] = cumulative_ts
                        df.at[idx, to_ts_col] = cumulative_ts + ep_duration
                        cumulative_ts += ep_duration

                    modified = True

        if modified:
            df.to_parquet(parquet_path, index=False)

    # Update info.json
    dataset.meta.info["total_frames"] -= frames_removed
    write_info(dataset.meta.info, dataset.root)


def trim_episode_virtual(
    dataset: LeRobotDataset,
    episode_index: int,
    start_frame: int,
    end_frame: int,
    recompute_stats: bool = True,
) -> LeRobotDataset:
    """Trim an episode WITHOUT re-encoding video files.

    This is a "virtual" trim that only updates metadata and parquet data.
    Video files remain unchanged - only the from_timestamp/to_timestamp
    pointers are adjusted.

    Benefits:
    - Instant (no video processing)
    - Lossless (no re-encoding quality loss)
    - Reversible (original video data still exists)

    Trade-off:
    - Trimmed frames still exist in video files (uses disk space)

    Args:
        dataset: The LeRobotDataset to modify.
        episode_index: Index of the episode to trim.
        start_frame: First frame to keep (0-indexed, inclusive).
        end_frame: Last frame to keep (0-indexed, exclusive).
        recompute_stats: If True (default), recompute per-episode stats and
            re-aggregate stats.json. Set to False when batching multiple trims,
            then call ``reaggregate_dataset_stats()`` once at the end.

    Returns:
        The modified dataset.
    """
    if episode_index < 0 or episode_index >= dataset.meta.total_episodes:
        raise ValueError(
            f"Invalid episode_index {episode_index}. "
            f"Dataset has {dataset.meta.total_episodes} episodes (0-{dataset.meta.total_episodes - 1})"
        )

    # Always reload metadata from disk
    dataset.meta.episodes = load_episodes(dataset.meta.root)
    dataset.meta.info = load_info(dataset.meta.root)

    episode_meta = dataset.meta.episodes[episode_index]
    episode_length = episode_meta["length"]

    # Validate frame range
    if start_frame < 0 or end_frame > episode_length:
        raise ValueError(
            f"Invalid frame range [{start_frame}, {end_frame}) for episode with {episode_length} frames"
        )
    if start_frame >= end_frame:
        raise ValueError(f"start_frame ({start_frame}) must be less than end_frame ({end_frame})")

    if start_frame == 0 and end_frame == episode_length:
        logging.info("No trimming requested, returning dataset unchanged")
        return dataset

    frames_to_trim_start = start_frame
    frames_to_trim_end = episode_length - end_frame
    new_episode_length = end_frame - start_frame

    logging.info(
        f"Virtual trim episode {episode_index}: removing {frames_to_trim_start} frames from start, "
        f"{frames_to_trim_end} frames from end. New length: {new_episode_length} frames"
    )

    # Step 1: Update parquet data (same as regular trim)
    _trim_episode_parquet_data(
        dataset=dataset,
        episode_index=episode_index,
        frames_to_trim_start=frames_to_trim_start,
        frames_to_trim_end=frames_to_trim_end,
    )

    # Step 2: Update metadata with adjusted video timestamps (NO video re-encoding)
    frames_removed = frames_to_trim_start + frames_to_trim_end
    _trim_episode_metadata_virtual(
        dataset=dataset,
        episode_index=episode_index,
        new_length=new_episode_length,
        frames_removed=frames_removed,
        frames_trimmed_from_start=frames_to_trim_start,
    )

    # Step 3: Recompute per-episode stats and re-aggregate stats.json
    if recompute_stats:
        _recompute_episode_stats_from_data(dataset.root, episode_index, dataset.meta.features)
        _reaggregate_and_write_stats(dataset.root, dataset.meta.features)
    else:
        # Always recompute the individual episode's stats (cheap, O(episode_data)),
        # but skip the O(total_episodes) re-aggregation for the caller to do once.
        _recompute_episode_stats_from_data(dataset.root, episode_index, dataset.meta.features)

    logging.info(f"Episode {episode_index} virtually trimmed successfully")
    return dataset


def _trim_episode_metadata_virtual(
    dataset: LeRobotDataset,
    episode_index: int,
    new_length: int,
    frames_removed: int,
    frames_trimmed_from_start: int,
) -> None:
    """Update episode metadata for virtual trim (adjusts video timestamps, no re-encode)."""
    episodes_dir = dataset.root / "meta" / "episodes"
    trim_start_s = frames_trimmed_from_start / dataset.fps
    trim_end_s = (frames_removed - frames_trimmed_from_start) / dataset.fps

    for parquet_path in sorted(episodes_dir.rglob("*.parquet")):
        df = pd.read_parquet(parquet_path)
        modified = False

        # Update length for the trimmed episode
        if episode_index in df["episode_index"].values:
            mask = df["episode_index"] == episode_index
            df.loc[mask, "length"] = new_length
            modified = True

        # Update dataset_from_index and dataset_to_index
        for i, row in df.iterrows():
            ep_idx = row["episode_index"]
            if ep_idx == episode_index:
                df.at[i, "dataset_to_index"] = row["dataset_from_index"] + new_length
                modified = True
            elif ep_idx > episode_index:
                df.at[i, "dataset_from_index"] = row["dataset_from_index"] - frames_removed
                df.at[i, "dataset_to_index"] = row["dataset_to_index"] - frames_removed
                modified = True

        # Update video timestamps for the trimmed episode ONLY
        # Unlike regular trim, we DON'T recalculate all episodes - just shift this one's boundaries
        for video_key in dataset.meta.video_keys:
            from_ts_col = f"videos/{video_key}/from_timestamp"
            to_ts_col = f"videos/{video_key}/to_timestamp"

            if from_ts_col in df.columns:
                mask = df["episode_index"] == episode_index
                if mask.any():
                    # Shift from_timestamp forward by trim_start_s
                    # Shift to_timestamp backward by trim_end_s
                    current_from = df.loc[mask, from_ts_col].iloc[0]
                    current_to = df.loc[mask, to_ts_col].iloc[0]

                    df.loc[mask, from_ts_col] = current_from + trim_start_s
                    df.loc[mask, to_ts_col] = current_to - trim_end_s
                    modified = True

        if modified:
            df.to_parquet(parquet_path, index=False)

    # Update info.json
    dataset.meta.info["total_frames"] -= frames_removed
    write_info(dataset.meta.info, dataset.root)


def delete_episodes_virtual(
    dataset: LeRobotDataset,
    episode_indices: list[int],
    recompute_stats: bool = True,
) -> LeRobotDataset:
    """Delete episodes WITHOUT re-encoding video files.

    This is a "virtual" delete that only updates metadata and parquet data.
    Video files remain unchanged - deleted episode data still exists in the
    video but will not be accessed.

    Benefits:
    - Instant (no video processing)
    - Lossless (no re-encoding quality loss)

    Trade-off:
    - Deleted episode video data still exists in files (uses disk space)

    Args:
        dataset: The LeRobotDataset to modify in-place.
        episode_indices: List of episode indices to delete.
        recompute_stats: If True (default), re-aggregate stats.json from
            remaining episodes. Set to False when batching with other edits,
            then call ``reaggregate_dataset_stats()`` once at the end.

    Returns:
        The modified dataset.
    """
    if not episode_indices:
        raise ValueError("No episodes to delete")

    # Validate indices
    valid_indices = set(range(dataset.meta.total_episodes))
    invalid = set(episode_indices) - valid_indices
    if invalid:
        raise ValueError(f"Invalid episode indices: {invalid}")

    # Reload metadata
    dataset.meta.episodes = load_episodes(dataset.meta.root)
    dataset.meta.info = load_info(dataset.meta.root)

    episodes_to_keep = [i for i in range(dataset.meta.total_episodes) if i not in episode_indices]
    if not episodes_to_keep:
        raise ValueError("Cannot delete all episodes")

    logging.info(f"Virtual delete: removing episodes {episode_indices}")

    # Calculate total frames being removed
    frames_removed = sum(dataset.meta.episodes[i]["length"] for i in episode_indices)

    # Step 1: Update parquet data - remove rows and reindex
    _delete_episodes_parquet_data_virtual(dataset, episode_indices, episodes_to_keep)

    # Step 2: Update episode metadata
    _delete_episodes_metadata_virtual(dataset, episode_indices, episodes_to_keep, frames_removed)

    # Step 3: Re-aggregate stats from remaining episodes
    if recompute_stats:
        _reaggregate_and_write_stats(dataset.root, dataset.meta.features)

    logging.info(f"Virtually deleted {len(episode_indices)} episodes")
    return dataset


def _delete_episodes_parquet_data_virtual(
    dataset: LeRobotDataset,
    episode_indices: list[int],
    episodes_to_keep: list[int],
) -> None:
    """Update parquet data files to remove deleted episodes and reindex."""
    data_dir = dataset.root / "data"
    episode_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(episodes_to_keep))}

    # Track cumulative offset across files for global reindexing
    global_index_offset = 0

    for parquet_path in sorted(data_dir.rglob("*.parquet")):
        df = pd.read_parquet(parquet_path)

        # Filter out deleted episodes
        mask = df["episode_index"].isin(episodes_to_keep)
        df = df[mask].copy()

        if len(df) == 0:
            # All data in this file was deleted - remove the file
            # safe-destruct: in-place delete: drop empty parquet after filter
            parquet_path.unlink()
            continue

        # Remap episode indices
        df["episode_index"] = df["episode_index"].map(episode_mapping)

        # Reindex frame_index within each episode (should already be correct)
        # Reindex global index with cumulative offset across files
        df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
        df["index"] = range(global_index_offset, global_index_offset + len(df))
        global_index_offset += len(df)

        df.to_parquet(parquet_path, index=False)


def check_episode_video_duration(ep: dict, fps: float) -> dict[str, int]:
    """Check if an episode's video timestamps match its data length.

    Detects video-data duration mismatches such as re-recording artifacts
    (video too long) or truncated recordings (video too short).

    Args:
        ep: Episode metadata dict (from dataset.meta.episodes[i]).
        fps: Dataset frame rate.

    Returns:
        Dict mapping video_key -> frame count difference (video - expected).
        Positive = extra frames (e.g. re-recording artifact),
        negative = missing frames (e.g. truncated video).
        Zero = OK.  Returns ``{}`` for non-video episodes.
    """
    result: dict[str, int] = {}
    length = ep.get("length", 0)
    if length <= 0 or fps <= 0:
        return result

    expected_s = length / fps

    for key in ep:
        if not key.endswith("/from_timestamp"):
            continue
        video_key = key.rsplit("/from_timestamp", 1)[0].removeprefix("videos/")
        to_key = f"videos/{video_key}/to_timestamp"
        if to_key not in ep:
            continue
        from_ts = ep[key]
        to_ts = ep[to_key]
        if to_ts <= from_ts:
            continue
        video_span = to_ts - from_ts
        diff = round((video_span - expected_s) * fps)
        # 1-frame rounding is normal for video encoding
        result[video_key] = diff if abs(diff) > 1 else 0

    return result


def repair_episode_indices(dataset_root: Path) -> int:
    """Check and repair episode metadata dataset_from_index values.

    Some datasets have broken metadata where dataset_from_index resets to 0
    at file boundaries instead of being globally continuous. This function
    recomputes correct cumulative indices from episode lengths.

    WARNING: This function MODIFIES the dataset on disk if repair is needed.
    Specifically, it rewrites: meta/episodes/*.parquet files with corrected
    dataset_from_index and dataset_to_index values.

    Args:
        dataset_root: Path to the dataset root directory.

    Returns:
        Number of episodes that were repaired (0 if already correct).
    """
    episodes_dir = dataset_root / "meta" / "episodes"
    if not episodes_dir.exists():
        return 0

    # Load all episode metadata into a single dataframe
    all_dfs = []
    parquet_files = sorted(episodes_dir.rglob("*.parquet"))
    for parquet_path in parquet_files:
        df = pd.read_parquet(parquet_path)
        df["_source_file"] = str(parquet_path)
        all_dfs.append(df)

    if not all_dfs:
        return 0

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = combined_df.sort_values("episode_index").reset_index(drop=True)

    # Check if repair is needed by computing expected indices.
    # Track which source files actually contain repaired rows so we only
    # rewrite the files that need it (and can short-circuit on read-only).
    repaired_count = 0
    cumulative_idx = 0
    files_with_changes: set[str] = set()
    for i, row in combined_df.iterrows():
        expected_from = cumulative_idx
        expected_to = cumulative_idx + row["length"]

        if row["dataset_from_index"] != expected_from or row["dataset_to_index"] != expected_to:
            combined_df.at[i, "dataset_from_index"] = expected_from
            combined_df.at[i, "dataset_to_index"] = expected_to
            repaired_count += 1
            files_with_changes.add(row["_source_file"])

        cumulative_idx = expected_to

    if repaired_count == 0:
        return 0

    # Pre-flight: refuse to start if any target file isn't writable. A partial
    # rewrite would leave dataset_from_index inconsistent across files.
    unwritable_meta = [p for p in files_with_changes if not os.access(p, os.W_OK)]
    if unwritable_meta:
        raise PermissionError(
            f"Cannot repair episode indices in {dataset_root}: "
            f"{len(unwritable_meta)} metadata parquet file(s) are read-only "
            f"(e.g. {unwritable_meta[0]})"
        )

    # Write back only the files whose rows actually changed.
    for parquet_path in parquet_files:
        if str(parquet_path) not in files_with_changes:
            continue
        file_mask = combined_df["_source_file"] == str(parquet_path)
        file_df = combined_df[file_mask].drop(columns=["_source_file"]).copy()
        if len(file_df) > 0:
            file_df.to_parquet(parquet_path, index=False)

    # Also repair the data parquet's index column to be globally continuous
    # Build a mapping from (episode_index, frame_index) -> global_index
    episode_start_indices = {}
    cumulative = 0
    for _, row in combined_df.sort_values("episode_index").iterrows():
        episode_start_indices[int(row["episode_index"])] = cumulative
        cumulative += int(row["length"])

    # Update data parquet files
    data_dir = dataset_root / "data"
    if data_dir.exists():
        data_files = sorted(data_dir.rglob("*.parquet"))
        for data_path in tqdm(data_files, desc="Repairing data indices"):
            df = pd.read_parquet(data_path)
            if (
                "index" not in df.columns
                or "episode_index" not in df.columns
                or "frame_index" not in df.columns
            ):
                continue

            # Compute correct global index for each row
            def compute_global_index(row):
                ep_start = episode_start_indices.get(row["episode_index"], 0)
                return ep_start + row["frame_index"]

            new_indices = df.apply(compute_global_index, axis=1)
            if not df["index"].equals(new_indices):
                df["index"] = new_indices
                df.to_parquet(data_path, index=False)

    logging.info(f"Repaired {repaired_count} episode indices in {dataset_root}")
    return repaired_count


def _delete_episodes_metadata_virtual(
    dataset: LeRobotDataset,
    episode_indices: list[int],
    episodes_to_keep: list[int],
    frames_removed: int,
) -> None:
    """Update episode metadata to remove deleted episodes."""
    episodes_dir = dataset.root / "meta" / "episodes"
    episode_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(episodes_to_keep))}

    # Track cumulative index ACROSS all files (not reset per-file)
    cumulative_idx = 0

    for parquet_path in sorted(episodes_dir.rglob("*.parquet")):
        df = pd.read_parquet(parquet_path)

        # Filter out deleted episodes
        mask = df["episode_index"].isin(episodes_to_keep)
        df = df[mask].copy()

        if len(df) == 0:
            # safe-destruct: in-place delete: drop empty parquet after filter
            parquet_path.unlink()
            continue

        # Remap episode indices
        df["episode_index"] = df["episode_index"].map(episode_mapping)

        # Recalculate dataset_from_index and dataset_to_index
        df = df.sort_values("episode_index").reset_index(drop=True)
        for i, row in df.iterrows():
            length = row["length"]
            df.at[i, "dataset_from_index"] = cumulative_idx
            df.at[i, "dataset_to_index"] = cumulative_idx + length
            cumulative_idx += length

        df.to_parquet(parquet_path, index=False)

    # Update info.json
    dataset.meta.info["total_episodes"] = len(episodes_to_keep)
    dataset.meta.info["total_frames"] -= frames_removed
    write_info(dataset.meta.info, dataset.root)


def recompute_stats(
    dataset: LeRobotDataset,
    skip_image_video: bool = True,
    relative_action: bool = False,
    relative_exclude_joints: list[str] | None = None,
    chunk_size: int = 50,
    num_workers: int = 0,
) -> LeRobotDataset:
    """Recompute stats.json from scratch by iterating all episodes.

    Args:
        dataset: The LeRobotDataset to recompute stats for.
        skip_image_video: If True (default), only recompute stats for numeric features
            (action, state, etc.) and keep existing image/video stats unchanged.
        relative_action: If True, compute action stats in relative space by
            iterating all valid action chunks and subtracting the current state.
            This matches the normalization distribution the model sees during
            training with ``use_relative_actions=True``.
        relative_exclude_joints: Joint names to exclude from relative conversion when
            relative_action=True. These dims keep absolute stats.
        chunk_size: Action chunk size used for relative stats computation. Should match
            ``policy.chunk_size``. Only used when ``relative_action=True``.
        num_workers: Number of parallel threads for relative action stats computation.
            Values ≤1 mean single-threaded. Only used when ``relative_action=True``.

    Returns:
        The same dataset with updated stats.
    """
    features = dataset.meta.features
    meta_keys = {"index", "episode_index", "task_index", "frame_index", "timestamp"}
    numeric_features = {
        k: v
        for k, v in features.items()
        if v["dtype"] not in ["image", "video", "string"] and k not in meta_keys
    }

    if skip_image_video:
        features_to_compute = numeric_features
    else:
        features_to_compute = {
            k: v for k, v in features.items() if v["dtype"] != "string" and k not in meta_keys
        }

    relative_action_stats = None
    if relative_action and ACTION in features and OBS_STATE in features:
        if relative_exclude_joints is None:
            relative_exclude_joints = ["gripper"]
        relative_action_stats = compute_relative_action_stats(
            hf_dataset=dataset.hf_dataset,
            features=features,
            chunk_size=chunk_size,
            exclude_joints=relative_exclude_joints,
            num_workers=num_workers,
        )
        features_to_compute.pop(ACTION, None)

    logging.info(f"Recomputing stats for features: {list(features_to_compute.keys())}")

    data_dir = dataset.root / DATA_DIR
    parquet_files = sorted(data_dir.glob("*/*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    all_episode_stats = []
    # TODO: enable image and video stats re-computation
    numeric_keys = [k for k, v in features_to_compute.items() if v["dtype"] not in ["image", "video"]]

    for parquet_path in tqdm(parquet_files, desc="Computing stats from data files"):
        df = pd.read_parquet(parquet_path)

        for ep_idx in sorted(df["episode_index"].unique()):
            ep_df = df[df["episode_index"] == ep_idx]
            episode_data = {}
            for key in numeric_keys:
                if key in ep_df.columns:
                    values = ep_df[key].values
                    if hasattr(values[0], "__len__"):
                        episode_data[key] = np.stack(values)
                    else:
                        episode_data[key] = np.array(values)

            ep_stats = compute_episode_stats(episode_data, features_to_compute)
            all_episode_stats.append(ep_stats)

    if features_to_compute and not all_episode_stats:
        logging.warning("No episode stats computed")
        return dataset

    new_stats = aggregate_stats(all_episode_stats) if all_episode_stats else {}

    if relative_action_stats is not None:
        new_stats[ACTION] = relative_action_stats

    if dataset.meta.stats:
        for key, value in dataset.meta.stats.items():
            if key not in new_stats:
                new_stats[key] = value

    write_stats(new_stats, dataset.root)
    dataset.meta.stats = new_stats

    logging.info("Stats recomputed successfully")
    return dataset


def convert_image_to_video_dataset(
    dataset: LeRobotDataset,
    output_dir: Path | None = None,
    repo_id: str | None = None,
    rgb_encoder: RGBEncoderConfig | None = None,
    depth_encoder: DepthEncoderConfig | None = None,
    episode_indices: list[int] | None = None,
    num_workers: int = 4,
    max_episodes_per_batch: int | None = None,
    max_frames_per_batch: int | None = None,
) -> LeRobotDataset:
    """Convert image-to-video dataset.

    Creates a new LeRobotDataset with images encoded as videos, following the proper
    LeRobot dataset structure with videos stored in chunked MP4 files.

    Args:
        dataset: The source LeRobot dataset with images.
        output_dir: Root directory where the converted dataset will be stored. When
            ``None``, defaults to ``$HF_LEROBOT_HOME/repo_id``. Equivalent to
            ``new_root`` in ``EditDatasetConfig``.
        repo_id: Converted dataset identifier. Equivalent to ``new_repo_id`` in
            ``EditDatasetConfig``.
        rgb_encoder: Video encoder settings applied to RGB cameras. When ``None``,
            :func:`~lerobot.configs.video.rgb_encoder_defaults` is used.
        depth_encoder: Video encoder settings applied to depth-map cameras, including
            the quantization parameters persisted to the dataset metadata. When
            ``None``, :func:`~lerobot.configs.video.depth_encoder_defaults` is used.
        episode_indices: Episode indices to convert. When ``None``, all episodes are
            converted.
        num_workers: Number of threads for parallel processing.
        max_episodes_per_batch: Maximum episodes per video batch, to bound memory use.
            ``None`` means no limit.
        max_frames_per_batch: Maximum frames per video batch, to bound memory use.
            ``None`` means no limit.

    Returns:
        A new :class:`LeRobotDataset` with images encoded as videos.
    """
    if rgb_encoder is None:
        rgb_encoder = rgb_encoder_defaults()
    if depth_encoder is None:
        depth_encoder = depth_encoder_defaults()

    # Check that it's an image dataset
    if len(dataset.meta.video_keys) > 0:
        raise ValueError(
            f"This operation is for image datasets only. Video dataset provided: {dataset.repo_id}"
        )

    # Get all image keys
    hf_dataset = dataset.hf_dataset.with_format(None)
    img_keys = [key for key in hf_dataset.features if key.startswith(OBS_IMAGE)]

    if len(img_keys) == 0:
        raise ValueError(f"No image keys found in dataset {dataset.repo_id}")

    # Determine which episodes to process
    if episode_indices is None:
        episode_indices = list(range(dataset.meta.total_episodes))

    if repo_id is None:
        repo_id = f"{dataset.repo_id}_video"

    logging.info(
        f"Converting {len(episode_indices)} episodes with {len(img_keys)} cameras from {dataset.repo_id}"
    )
    logging.info(f"RGB video encoder: {rgb_encoder}, depth video encoder: {depth_encoder}")

    # Create new features dict, converting image features to video features
    new_features = {}
    for key, value in dataset.meta.features.items():
        if key not in img_keys:
            new_features[key] = value
        else:
            # Convert image key to video format
            new_features[key] = value.copy()
            new_features[key]["dtype"] = "video"  # Change dtype from "image" to "video"
            # Video info will be updated after episodes are encoded

    # Create new metadata for video dataset
    output_dir = Path(output_dir) if output_dir is not None else HF_LEROBOT_HOME / repo_id
    new_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=dataset.meta.fps,
        features=new_features,
        robot_type=dataset.meta.robot_type,
        root=output_dir,
        use_videos=True,
        chunks_size=dataset.meta.chunks_size,
        data_files_size_in_mb=dataset.meta.data_files_size_in_mb,
        video_files_size_in_mb=dataset.meta.video_files_size_in_mb,
    )

    # Create temporary directory for image extraction
    temp_dir = output_dir / "temp_images"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Process all episodes and batch encode videos
    # Use dictionary for O(1) episode metadata lookups instead of O(n) linear search
    all_episode_metadata = {}
    fps = int(dataset.fps)

    try:
        # Build episode metadata entries first
        logging.info("Building episode metadata...")
        cumulative_frame_idx = 0
        for ep_idx in episode_indices:
            src_episode = dataset.meta.episodes[ep_idx]
            ep_length = src_episode["length"]
            ep_meta = {
                "episode_index": ep_idx,
                "length": ep_length,
                "dataset_from_index": cumulative_frame_idx,
                "dataset_to_index": cumulative_frame_idx + ep_length,
            }
            if "data/chunk_index" in src_episode:
                ep_meta["data/chunk_index"] = src_episode["data/chunk_index"]
                ep_meta["data/file_index"] = src_episode["data/file_index"]
            all_episode_metadata[ep_idx] = ep_meta
            cumulative_frame_idx += ep_length

        # Process each camera and batch encode multiple episodes together
        video_file_size_limit = new_meta.video_files_size_in_mb

        # Pre-compute episode lengths for batching
        episode_lengths = {ep_idx: dataset.meta.episodes["length"][ep_idx] for ep_idx in episode_indices}

        for img_key in tqdm(img_keys, desc="Processing cameras"):
            target_encoder = depth_encoder if img_key in dataset.meta.depth_keys else rgb_encoder

            # Estimate size per frame by encoding a small calibration sample
            # This provides accurate compression ratio for the specific codec parameters
            size_per_frame_mb = _estimate_frame_size_via_calibration(
                dataset=dataset,
                img_key=img_key,
                episode_indices=episode_indices,
                temp_dir=temp_dir,
                fps=fps,
                video_encoder=target_encoder,
            )

            logging.info(f"Processing camera: {img_key}")
            chunk_idx, file_idx = 0, 0
            cumulative_timestamp = 0.0

            # Process episodes in batches to stay under size limit
            for batch_episodes in _iter_episode_batches(
                episode_indices=episode_indices,
                episode_lengths=episode_lengths,
                size_per_frame_mb=size_per_frame_mb,
                video_file_size_limit=video_file_size_limit,
                max_episodes=max_episodes_per_batch,
                max_frames=max_frames_per_batch,
            ):
                total_frames_in_batch = sum(episode_lengths[idx] for idx in batch_episodes)
                logging.info(
                    f"  Encoding batch of {len(batch_episodes)} episodes "
                    f"({batch_episodes[0]}-{batch_episodes[-1]}) = {total_frames_in_batch} frames"
                )

                # Save images for all episodes in this batch
                imgs_dir = temp_dir / f"batch_{chunk_idx}_{file_idx}" / img_key
                episode_durations = _save_batch_episodes_images(
                    dataset=dataset,
                    imgs_dir=imgs_dir,
                    img_key=img_key,
                    episode_indices=batch_episodes,
                    num_workers=num_workers,
                )

                # Encode all batched episodes into single video
                video_path = new_meta.root / new_meta.video_path.format(
                    video_key=img_key, chunk_index=chunk_idx, file_index=file_idx
                )
                video_path.parent.mkdir(parents=True, exist_ok=True)

                encode_video_frames(
                    imgs_dir=imgs_dir,
                    video_path=video_path,
                    fps=fps,
                    video_encoder=target_encoder,
                    overwrite=True,
                )

                # Clean up temporary images
                # safe-destruct: convert_image_to_video: drop temp images after encode
                shutil.rmtree(imgs_dir)

                # Update metadata for each episode in the batch
                for ep_idx, duration in zip(batch_episodes, episode_durations, strict=True):
                    from_timestamp = cumulative_timestamp
                    to_timestamp = cumulative_timestamp + duration
                    cumulative_timestamp = to_timestamp

                    # Find episode metadata entry and add video metadata (O(1) dictionary lookup)
                    ep_meta = all_episode_metadata[ep_idx]
                    ep_meta[f"videos/{img_key}/chunk_index"] = chunk_idx
                    ep_meta[f"videos/{img_key}/file_index"] = file_idx
                    ep_meta[f"videos/{img_key}/from_timestamp"] = from_timestamp
                    ep_meta[f"videos/{img_key}/to_timestamp"] = to_timestamp

                # Move to next video file for next batch
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, new_meta.chunks_size)
                cumulative_timestamp = 0.0

        # Copy and transform data files (removing image columns)
        _copy_data_without_images(dataset, new_meta, episode_indices, img_keys)

        # Save episode metadata
        episodes_df = pd.DataFrame(list(all_episode_metadata.values()))
        episodes_path = new_meta.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        episodes_path.parent.mkdir(parents=True, exist_ok=True)
        episodes_df.to_parquet(episodes_path, index=False)

        # Update metadata info
        new_meta.info.total_episodes = len(episode_indices)
        new_meta.info.total_frames = sum(ep["length"] for ep in all_episode_metadata.values())
        new_meta.info.total_tasks = dataset.meta.total_tasks
        new_meta.info.splits = {"train": f"0:{len(episode_indices)}"}

        # Update video info for all image keys (now videos). They are registered as
        # video features above, so update_video_info populates their (still-empty) info.
        for img_key in img_keys:
            target_encoder = depth_encoder if img_key in dataset.meta.depth_keys else rgb_encoder
            new_meta.update_video_info(video_key=img_key, video_encoder=target_encoder)

        write_info(new_meta.info, new_meta.root)

        # Copy stats and tasks
        if dataset.meta.stats is not None:
            # Remove image stats
            new_stats = {k: v for k, v in dataset.meta.stats.items() if k not in img_keys}
            write_stats(new_stats, new_meta.root)

        if dataset.meta.tasks is not None:
            write_tasks(dataset.meta.tasks, new_meta.root)

    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            # safe-destruct: convert_image_to_video: drop temp dir on completion
            shutil.rmtree(temp_dir)

    logging.info(f"Completed converting {dataset.repo_id} to video format")
    logging.info(f"New dataset saved to: {output_dir}")

    # Return new dataset
    return LeRobotDataset(repo_id=repo_id, root=output_dir)


# =============================================================================
# Dataset Verification (LeRobot v3.0)
# =============================================================================


class DatasetVerificationError:
    """Represents a single verification error."""

    def __init__(self, category: str, message: str, details: dict | None = None):
        self.category = category
        self.message = message
        self.details = details or {}

    def __repr__(self) -> str:
        if self.details:
            return f"{self.category}: {self.message} ({self.details})"
        return f"{self.category}: {self.message}"


class DatasetVerificationResult:
    """Result of dataset verification containing all errors found."""

    def __init__(self):
        self.errors: list[DatasetVerificationError] = []
        self.warnings: list[DatasetVerificationError] = []
        self.stats: dict = {}

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, category: str, message: str, details: dict | None = None) -> None:
        self.errors.append(DatasetVerificationError(category, message, details))

    def add_warning(self, category: str, message: str, details: dict | None = None) -> None:
        self.warnings.append(DatasetVerificationError(category, message, details))

    def __repr__(self) -> str:
        if self.is_valid:
            return f"DatasetVerificationResult(valid=True, warnings={len(self.warnings)})"
        return f"DatasetVerificationResult(valid=False, errors={len(self.errors)}, warnings={len(self.warnings)})"

    def summary(self) -> str:
        """Return a human-readable summary of verification results."""
        lines = []
        if self.is_valid:
            lines.append("✓ Dataset verification passed")
        else:
            lines.append(f"✗ Dataset verification failed with {len(self.errors)} error(s)")

        if self.warnings:
            lines.append(f"  {len(self.warnings)} warning(s)")

        if self.stats:
            lines.append(f"  Stats: {self.stats}")

        if self.errors:
            lines.append("\nErrors:")
            for err in self.errors[:10]:  # Limit to first 10
                lines.append(f"  - {err}")
            if len(self.errors) > 10:
                lines.append(f"  ... and {len(self.errors) - 10} more errors")

        if self.warnings:
            lines.append("\nWarnings:")
            for warn in self.warnings[:5]:
                lines.append(f"  - {warn}")
            if len(self.warnings) > 5:
                lines.append(f"  ... and {len(self.warnings) - 5} more warnings")

        return "\n".join(lines)


def verify_dataset(
    dataset_root: str | Path,
    check_videos: bool = True,
    verbose: bool = False,
) -> DatasetVerificationResult:
    """Verify dataset integrity and correctness for LeRobot v3.0 format.

    Performs comprehensive checks on dataset structure, indices, and consistency
    according to the LeRobot Dataset v3.0 specification.

    **Structure checks**:
    - `meta/info.json` exists and contains required fields
    - `meta/episodes/*.parquet` exist with episode metadata
    - `data/*.parquet` exist with frame data
    - `meta/stats.json` exists (warning if missing)
    - `meta/tasks.parquet` exists if total_tasks > 0
    - `videos/` directory structure matches video keys

    **info.json checks**:
    - Required fields: codebase_version, total_episodes, total_frames, fps, features
    - total_episodes matches actual episode count
    - total_frames matches actual frame count
    - features dict describes expected data columns

    **Episode metadata checks**:
    - `episode_index` is sequential (0, 1, 2, ..., N-1)
    - `length` is positive for all episodes
    - `dataset_from_index`/`dataset_to_index` are globally continuous
    - Sum of episode lengths equals total_frames
    - `data/chunk_index`, `data/file_index` reference valid files
    - For video datasets:
      - `videos/*/chunk_index`, `videos/*/file_index` reference valid files
      - `from_timestamp` < `to_timestamp`
      - Video duration approximately matches `length / fps`

    **Data parquet checks**:
    - Row count matches total_frames from info.json
    - Required columns exist: `index`, `episode_index`, `frame_index`
    - `index` column is compact (0 to N-1, no gaps, no duplicates)
    - `index` = `episode_start + frame_index` for each row
    - `frame_index` within each episode is 0 to length-1
    - All `episode_index` values are valid
    - All `task_index` values are valid (< total_tasks)

    **Cross-validation checks**:
    - Each episode's frame count in data matches metadata length
    - All data files referenced by episodes exist
    - All video files referenced by episodes exist (if check_videos=True)

    Args:
        dataset_root: Path to the dataset root directory
        check_videos: Whether to verify video files exist and timestamps are correct
        verbose: Whether to log progress

    Returns:
        DatasetVerificationResult containing all errors and warnings found

    Example:
        >>> result = verify_dataset("/path/to/dataset")
        >>> if result.is_valid:
        ...     print("Dataset is valid!")
        >>> else:
        ...     print(result.summary())
    """
    dataset_root = Path(dataset_root)
    result = DatasetVerificationResult()

    if verbose:
        logging.info(f"Verifying dataset at {dataset_root}")

    # ==========================================================================
    # 1. Verify directory structure exists
    # ==========================================================================
    if not dataset_root.exists():
        result.add_error("structure", f"Dataset root does not exist: {dataset_root}")
        return result

    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        result.add_error("structure", "Missing meta/info.json")
        return result

    episodes_dir = dataset_root / "meta" / "episodes"
    if not episodes_dir.exists():
        result.add_error("structure", "Missing meta/episodes directory")
        return result

    data_dir = dataset_root / "data"
    if not data_dir.exists():
        result.add_error("structure", "Missing data directory")
        return result

    # Check optional files
    stats_path = dataset_root / "meta" / "stats.json"
    if not stats_path.exists():
        result.add_warning("structure", "Missing meta/stats.json (normalization stats)")

    # ==========================================================================
    # 2. Load and verify info.json
    # ==========================================================================
    try:
        info = load_info(dataset_root)
    except Exception as e:
        result.add_error("info", f"Failed to load info.json: {e}")
        return result

    # Check required fields
    required_info_fields = ["total_episodes", "total_frames", "fps", "features"]
    for field in required_info_fields:
        if field not in info:
            result.add_error("info", f"Missing required field in info.json: {field}")

    # Check codebase_version (warning only)
    if "codebase_version" not in info:
        result.add_warning("info", "Missing codebase_version in info.json")
    elif not info["codebase_version"].startswith("v3"):
        result.add_warning("info", f"Unexpected codebase_version: {info['codebase_version']}")

    expected_total_episodes = info.get("total_episodes", 0)
    expected_total_frames = info.get("total_frames", 0)
    expected_total_tasks = info.get("total_tasks", 0)
    fps = info.get("fps", 30)
    features = info.get("features", {})

    result.stats["expected_episodes"] = expected_total_episodes
    result.stats["expected_frames"] = expected_total_frames
    result.stats["expected_tasks"] = expected_total_tasks
    result.stats["fps"] = fps

    # Check tasks.parquet exists if needed
    if expected_total_tasks > 0:
        tasks_path = dataset_root / "meta" / "tasks.parquet"
        if not tasks_path.exists():
            result.add_warning(
                "structure", f"Missing meta/tasks.parquet (expected {expected_total_tasks} tasks)"
            )

    # Identify video keys from features
    video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
    result.stats["video_keys"] = video_keys

    # ==========================================================================
    # 3. Load and verify episode metadata
    # ==========================================================================
    episode_files = sorted(episodes_dir.rglob("*.parquet"))
    if not episode_files:
        result.add_error("episodes", "No parquet files found in meta/episodes")
        return result

    try:
        episodes = load_episodes(dataset_root)
    except Exception as e:
        result.add_error("episodes", f"Failed to load episodes: {e}")
        return result

    if not episodes:
        result.add_error("episodes", "No episodes found in metadata")
        return result

    actual_episode_count = len(episodes)
    result.stats["actual_episodes"] = actual_episode_count

    # Check episode count matches info.json
    if actual_episode_count != expected_total_episodes:
        result.add_error(
            "episodes",
            f"Episode count mismatch: metadata has {actual_episode_count}, "
            f"info.json says {expected_total_episodes}",
        )

    # Check episode indices are sequential (0, 1, 2, ..., N-1)
    episode_indices = [ep["episode_index"] for ep in episodes]
    expected_indices = list(range(actual_episode_count))
    if episode_indices != expected_indices:
        missing = set(expected_indices) - set(episode_indices)
        extra = set(episode_indices) - set(expected_indices)
        duplicates = len(episode_indices) - len(set(episode_indices))
        result.add_error(
            "episodes",
            "Episode indices are not sequential 0..N-1",
            {
                "missing": sorted(missing)[:10],
                "extra": sorted(extra)[:10],
                "duplicates": duplicates,
            },
        )

    # Check dataset_from_index / dataset_to_index are globally continuous
    cumulative = 0
    total_length_from_episodes = 0
    referenced_data_files = set()

    for ep in episodes:
        ep_idx = ep["episode_index"]
        length = ep["length"]
        from_idx = ep.get("dataset_from_index", 0)
        to_idx = ep.get("dataset_to_index", length)

        # Check length is positive
        if length <= 0:
            result.add_error("episodes", f"Episode {ep_idx} has invalid length: {length}")
            continue

        total_length_from_episodes += length

        # Check global indices are continuous
        if from_idx != cumulative:
            result.add_error(
                "episodes",
                f"Episode {ep_idx}: dataset_from_index={from_idx}, expected {cumulative}",
            )

        expected_to = cumulative + length
        if to_idx != expected_to:
            result.add_error(
                "episodes",
                f"Episode {ep_idx}: dataset_to_index={to_idx}, expected {expected_to}",
            )

        cumulative += length

        # Track referenced data files
        data_chunk = ep.get("data/chunk_index", 0)
        data_file = ep.get("data/file_index", 0)
        referenced_data_files.add((data_chunk, data_file))

    result.stats["total_length_from_episodes"] = total_length_from_episodes

    # Check total frames from episode lengths matches info.json
    if total_length_from_episodes != expected_total_frames:
        result.add_error(
            "episodes",
            f"Total frames mismatch: sum of episode lengths is {total_length_from_episodes}, "
            f"info.json says {expected_total_frames}",
        )

    # ==========================================================================
    # 4. Load and verify data parquet files
    # ==========================================================================
    data_files = sorted(data_dir.rglob("*.parquet"))
    if not data_files:
        result.add_error("data", "No parquet files found in data directory")
        return result

    # Check referenced data files exist
    data_path_template = info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
    for chunk_idx, file_idx in referenced_data_files:
        data_path = dataset_root / data_path_template.format(chunk_index=chunk_idx, file_index=file_idx)
        if not data_path.exists():
            result.add_error(
                "data",
                f"Referenced data file missing: chunk-{chunk_idx:03d}/file-{file_idx:03d}.parquet",
            )

    # Load all data
    all_data = []
    for data_path in data_files:
        try:
            df = pd.read_parquet(data_path)
            all_data.append(df)
        except Exception as e:
            result.add_error("data", f"Failed to read {data_path.name}: {e}")

    if not all_data:
        result.add_error("data", "Could not load any data parquet files")
        return result

    data = pd.concat(all_data, ignore_index=True)
    actual_frame_count = len(data)
    result.stats["actual_frames"] = actual_frame_count

    # Check total row count matches info.json
    if actual_frame_count != expected_total_frames:
        result.add_error(
            "data",
            f"Row count mismatch: data has {actual_frame_count} rows, info.json says {expected_total_frames}",
        )

    # Check required columns exist
    required_columns = ["index", "episode_index", "frame_index"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        result.add_error("data", f"Missing required columns: {missing_columns}")
        return result

    # Check feature columns exist (warning only for non-video features)
    for feature_name, feature_info in features.items():
        if feature_info.get("dtype") == "video":
            continue  # Video features are stored in separate files
        if feature_name not in data.columns:
            result.add_warning("data", f"Missing feature column: {feature_name}")

    # Check index column is compact (0 to N-1)
    indices = data["index"].values
    sorted_indices = np.sort(indices)
    expected_index_array = np.arange(len(data))

    if not np.array_equal(sorted_indices, expected_index_array):
        unique_indices = np.unique(indices)
        num_duplicates = len(indices) - len(unique_indices)

        if num_duplicates > 0:
            result.add_error("data", f"Index column has {num_duplicates} duplicate values")

        if len(unique_indices) > 0:
            if unique_indices[0] != 0:
                result.add_error("data", f"Index column doesn't start at 0, starts at {unique_indices[0]}")
            if unique_indices[-1] != len(data) - 1:
                result.add_error(
                    "data",
                    f"Index column doesn't end at {len(data) - 1}, ends at {unique_indices[-1]}",
                )

            # Check for gaps
            gaps = np.where(np.diff(sorted_indices) > 1)[0]
            if len(gaps) > 0:
                result.add_error(
                    "data", f"Index column has {len(gaps)} gap(s)", {"first_gap_at": int(gaps[0])}
                )

    # Check episode_index values are valid
    data_episode_indices = set(data["episode_index"].unique())
    valid_episode_set = set(episode_indices)
    invalid_episodes = data_episode_indices - valid_episode_set
    if invalid_episodes:
        result.add_error(
            "data",
            f"Data contains invalid episode indices: {sorted(invalid_episodes)[:10]}",
        )

    # Check task_index values are valid
    if "task_index" in data.columns and expected_total_tasks > 0:
        max_task_index = data["task_index"].max()
        if max_task_index >= expected_total_tasks:
            result.add_error(
                "data",
                f"task_index has values >= total_tasks: max={max_task_index}, total_tasks={expected_total_tasks}",
            )

    # Build episode start index mapping
    episode_starts = {}
    episode_lengths = {}
    cumulative = 0
    for ep in episodes:
        episode_starts[ep["episode_index"]] = cumulative
        episode_lengths[ep["episode_index"]] = ep["length"]
        cumulative += ep["length"]

    # Check index = episode_start + frame_index for each row
    # And check frame_index is 0..length-1 for each episode
    index_errors = 0
    frame_index_errors = 0
    length_mismatches = 0

    for ep_idx in episode_indices:
        if ep_idx not in episode_starts:
            continue

        ep_start = episode_starts[ep_idx]
        ep_length = episode_lengths[ep_idx]
        ep_data = data[data["episode_index"] == ep_idx]

        if len(ep_data) == 0:
            result.add_error("alignment", f"Episode {ep_idx} has no data rows")
            continue

        # Check frame count matches metadata length
        if len(ep_data) != ep_length:
            length_mismatches += 1
            if length_mismatches <= 3:
                result.add_error(
                    "alignment",
                    f"Episode {ep_idx}: metadata says {ep_length} frames, data has {len(ep_data)}",
                )

        # Check frame_index is 0 to length-1
        frame_indices = np.sort(ep_data["frame_index"].values)
        expected_frame_indices = np.arange(len(ep_data))
        if not np.array_equal(frame_indices, expected_frame_indices):
            frame_index_errors += 1
            if frame_index_errors <= 3:
                result.add_error(
                    "data",
                    f"Episode {ep_idx}: frame_index not sequential 0..{len(ep_data) - 1}",
                    {"actual_range": f"{frame_indices.min()}-{frame_indices.max()}"},
                )

        # Check index = episode_start + frame_index
        for _, row in ep_data.iterrows():
            expected_index = ep_start + row["frame_index"]
            if row["index"] != expected_index:
                index_errors += 1
                if index_errors <= 3:
                    result.add_error(
                        "data",
                        f"Index mismatch: episode={ep_idx}, frame={row['frame_index']}, "
                        f"expected index={expected_index}, got {row['index']}",
                    )

    if length_mismatches > 3:
        result.add_warning("alignment", f"... and {length_mismatches - 3} more episode length mismatches")
    if index_errors > 3:
        result.add_warning("data", f"... and {index_errors - 3} more index mismatches")

    # Action-quality heuristic: an episode whose action column is identically
    # zero across every frame is almost always a recording-flow bug rather
    # than legit data. Common failure mode: gym-hil / pynput intervention flag
    # never engaged during teleop, so the policy's neutral [0,...,0] passed
    # through env.step and got recorded. Surface as a warning so users notice
    # silently-rotted episodes when opening the dataset, not at training time
    # when the loss curve is mysteriously flat.
    from lerobot.utils.constants import ACTION

    if ACTION in data.columns:
        bad_episodes = []
        for ep_idx in episode_indices:
            ep_actions = data.loc[data["episode_index"] == ep_idx, ACTION].tolist()
            if not ep_actions:
                continue
            try:
                if all(all(v == 0 for v in a) for a in ep_actions):
                    bad_episodes.append(int(ep_idx))
            except (TypeError, ValueError):
                # Malformed action data — skip this episode rather than crash
                # the whole verification run.
                continue
        if bad_episodes:
            listed = ", ".join(str(i) for i in bad_episodes[:5])
            more = f" (and {len(bad_episodes) - 5} more)" if len(bad_episodes) > 5 else ""
            result.add_warning(
                "data",
                f"{len(bad_episodes)} episode(s) have all-zero actions across every frame: "
                f"[{listed}]{more}. Common cause: teleop intervention never engaged "
                f"during recording — the policy's neutral action got recorded instead. "
                f"These episodes are useless for training/replay.",
                details={"episode_indices": bad_episodes},
            )
    if frame_index_errors > 3:
        result.add_warning("data", f"... and {frame_index_errors - 3} more frame_index issues")

    # ==========================================================================
    # 5. Verify video metadata and files (optional)
    # ==========================================================================
    if check_videos and video_keys:
        videos_dir = dataset_root / "videos"
        if not videos_dir.exists():
            result.add_error("videos", "Missing videos directory but dataset has video features")
        else:
            video_path_template = info.get(
                "video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
            )
            video_file_missing = 0
            video_timestamp_errors = 0

            for video_key in video_keys:
                for ep in episodes:
                    ep_idx = ep["episode_index"]
                    chunk_key = f"videos/{video_key}/chunk_index"
                    file_key = f"videos/{video_key}/file_index"
                    from_ts_key = f"videos/{video_key}/from_timestamp"
                    to_ts_key = f"videos/{video_key}/to_timestamp"

                    # Check metadata fields exist
                    if chunk_key not in ep or file_key not in ep:
                        result.add_warning(
                            "videos",
                            f"Episode {ep_idx} missing video location for {video_key}",
                        )
                        continue

                    # Check video file exists
                    chunk_idx = ep[chunk_key]
                    file_idx = ep[file_key]
                    video_path = dataset_root / video_path_template.format(
                        video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
                    )

                    if not video_path.exists():
                        video_file_missing += 1
                        if video_file_missing <= 3:
                            result.add_error(
                                "videos",
                                f"Missing video file for episode {ep_idx}, {video_key}: {video_path.name}",
                            )
                        continue

                    # Check timestamp consistency
                    if from_ts_key in ep and to_ts_key in ep:
                        from_ts = ep[from_ts_key]
                        to_ts = ep[to_ts_key]

                        if from_ts >= to_ts:
                            result.add_error(
                                "videos",
                                f"Episode {ep_idx}, {video_key}: from_timestamp ({from_ts}) >= to_timestamp ({to_ts})",
                            )
                            continue

                        frame_diff = check_episode_video_duration(ep, fps)
                        diff = frame_diff.get(video_key, 0)
                        if diff != 0:
                            video_timestamp_errors += 1
                            actual_duration = to_ts - from_ts
                            expected_duration = ep["length"] / fps
                            detail = (
                                f"+{diff} extra frames, possible re-recording artifact"
                                if diff > 0
                                else f"{diff} missing frames, possible truncation"
                            )
                            if video_timestamp_errors <= 3:
                                result.add_warning(
                                    "videos",
                                    f"Episode {ep_idx}, {video_key}: duration {actual_duration:.3f}s "
                                    f"doesn't match expected {expected_duration:.3f}s "
                                    f"({detail})",
                                )

            if video_file_missing > 3:
                result.add_warning("videos", f"... and {video_file_missing - 3} more missing video files")
            if video_timestamp_errors > 3:
                result.add_warning(
                    "videos", f"... and {video_timestamp_errors - 3} more timestamp mismatches"
                )

    # ==========================================================================
    # 6. Verify stats.json consistency with per-episode stats
    # ==========================================================================
    stats_path = dataset_root / "meta" / "stats.json"
    if stats_path.exists():
        try:
            stored_stats = load_stats(dataset_root)
            if stored_stats is not None:
                # Re-aggregate from per-episode parquet files
                all_ep_stats = []
                for ep_file in sorted(episodes_dir.rglob("*.parquet")):
                    ep_df = pd.read_parquet(ep_file)
                    for _, row in ep_df.iterrows():
                        ep_stats = _extract_episode_stats_from_parquet(row.to_dict(), features)
                        if ep_stats:
                            all_ep_stats.append(ep_stats)

                if all_ep_stats:
                    recomputed_stats = aggregate_stats(all_ep_stats)
                    for feature_key in stored_stats:
                        if feature_key not in recomputed_stats:
                            continue
                        for stat_key in stored_stats[feature_key]:
                            if stat_key not in recomputed_stats[feature_key]:
                                continue
                            stored_val = np.asarray(stored_stats[feature_key][stat_key], dtype=np.float64)
                            recomp_val = np.asarray(recomputed_stats[feature_key][stat_key], dtype=np.float64)
                            if not np.allclose(stored_val, recomp_val, atol=1e-4, equal_nan=True):
                                result.add_warning(
                                    "stats",
                                    f"stats.json mismatch for {feature_key}/{stat_key}: "
                                    f"stored={stored_val.tolist()}, recomputed={recomp_val.tolist()}",
                                )
        except Exception as e:
            result.add_warning("stats", f"Failed to verify stats consistency: {e}")

    if verbose:
        logging.info(result.summary())

    return result


def verify_dataset_quick(dataset_root: str | Path) -> bool:
    """Quick verification that returns True if dataset is valid.

    This is a convenience wrapper around verify_dataset() for simple checks.
    Does not check video files for speed.

    Args:
        dataset_root: Path to the dataset root directory

    Returns:
        True if dataset passes all checks, False otherwise
    """
    result = verify_dataset(dataset_root, check_videos=False, verbose=False)
    return result.is_valid


def _reencode_video_worker(args: tuple) -> Path:
    """Picklable worker for :func:`reencode_dataset`'s process pool."""
    video_path, video_encoder, encoder_threads = args
    reencode_video(
        input_video_path=video_path,
        output_video_path=video_path,
        video_encoder=video_encoder,
        encoder_threads=encoder_threads,
        overwrite=True,
    )
    return video_path


def reencode_dataset(
    dataset: LeRobotDataset,
    rgb_encoder: RGBEncoderConfig | None = None,
    depth_encoder: DepthEncoderConfig | None = None,
    encoder_threads: int | None = None,
    num_workers: int | None = None,
) -> LeRobotDataset:
    """Re-encode every video in a dataset with a new set of encoding parameters.

    Videos are re-encoded in-place and the video information in ``info.json`` is refreshed.

    Args:
        dataset: An existing :class:`LeRobotDataset` whose videos will be
            re-encoded.
        rgb_encoder: Target encoder configuration applied to every RGB video
            file. If ``None``, re-encoding is skipped for RGB videos.
        depth_encoder: Target encoder configuration applied to every depth video
            file. If ``None``, re-encoding is skipped for depth videos.
            Quantization parameters will not override the ones in the current dataset.
        encoder_threads: Per-encoder thread count forwarded to
            :func:`reencode_video`. ``None`` lets the codec decide.
        num_workers: Number of parallel processes. ``None`` or ``0`` means
            sequential (no multiprocessing); ``1+`` spawns a
            :class:`~concurrent.futures.ProcessPoolExecutor`.

    Returns:
        The same :class:`LeRobotDataset` instance with its metadata updated
        on disk.
    """
    meta = dataset.meta
    video_keys_encoders_dict = {}
    video_keys_paths_dict = {}

    if rgb_encoder is None and depth_encoder is None:
        raise ValueError("Either rgb_encoder or depth_encoder must be provided")

    # Only re-encode if the videos are not already encoded with the given video encoding parameters
    for video_key in meta.video_keys:
        current_info = meta.info.features[video_key].get("info", {})
        current_encoder = encoder_config_from_video_info(current_info)
        target_encoder = depth_encoder if video_key in meta.depth_keys else rgb_encoder
        if target_encoder is None:
            logging.info(f"No encoder provided for {video_key} video. Skipping re-encoding.")
        elif current_encoder != target_encoder:
            video_keys_paths_dict[video_key] = list((meta.root / VIDEO_DIR / video_key).rglob("*.mp4"))
            video_keys_encoders_dict[video_key] = target_encoder
        else:
            logging.info(f"{video_key} videos are already encoded with {target_encoder}. Nothing to do.")

    if len(video_keys_paths_dict) == 0:
        logging.warning("Dataset has no videos to re-encode.")
        return dataset
    logging.info(f"Re-encoding {sum(len(paths) for paths in video_keys_paths_dict.values())} video file(s).")

    worker_args = [
        (path, encoder, encoder_threads)
        for video_key, encoder in video_keys_encoders_dict.items()
        for path in video_keys_paths_dict[video_key]
    ]
    if num_workers and num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = [pool.submit(_reencode_video_worker, args) for args in worker_args]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Re-encoding videos",
            ):
                future.result()
    else:
        for args in tqdm(worker_args, desc="Re-encoding videos"):
            _reencode_video_worker(args)

    # Refresh video info in metadata for every re-encoded key. Re-encoding only
    # changes codec/container params, so for depth videos we preserve ``is_depth_map``
    # and the depth quantization params (``video.depth_min`` / ``video.depth_max`` /
    # ...), which describe the data rather than the codec and must survive a transcode.
    # RGB videos pass an empty set: still a refresh, but nothing to preserve.
    depth_preserve_keys = {"is_depth_map", *(f"video.{n}" for n in DEPTH_ENCODER_INFO_FIELD_NAMES)}
    for video_key, encoder in video_keys_encoders_dict.items():
        preserve_keys = depth_preserve_keys if video_key in meta.depth_keys else set()
        meta.update_video_info(video_key=video_key, video_encoder=encoder, preserve_keys=preserve_keys)

    write_info(meta.info, meta.root)
    logging.info("Dataset metadata updated.")

    return dataset
