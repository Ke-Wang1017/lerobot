#!/usr/bin/env python
"""Integration test for the stale metadata bug in trim operations.

Run with: pytest tests/test_trim_stale_metadata_bug.py -v

BUG SUMMARY:
When multiple trim operations are applied sequentially (like in the GUI),
each trim modifies the episode indices (dataset_from_index, dataset_to_index)
for all subsequent episodes. However, the in-memory metadata was not reloaded
between trims, causing subsequent trims to use stale indices.

Example:
- After trimming episode 0 (removing 10 frames), episode 2's from_index shifts
  from 714 -> 704 on disk
- But in-memory metadata still says 714
- When trimming episode 2 next, it uses wrong indices
- This causes frames to be incorrectly filtered, leading to data corruption

Root cause: trim_episode_by_frames() didn't reload metadata from disk before
each operation when called multiple times.

Fix: Always reload dataset.meta.episodes and dataset.meta.info at the start
of each trim_episode_by_frames() call.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.dataset_tools import delete_episodes, trim_episode_by_frames
from lerobot.datasets.io_utils import (
    load_episodes,
    load_info,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tests.fixtures.constants import DUMMY_REPO_ID

try:
    import datasets as hf_datasets
except ImportError:
    hf_datasets = None


def verify_dataset_integrity(root: Path) -> tuple[bool, str]:
    """Verify that all episodes have consistent data and metadata.

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Load metadata
    meta_dir = root / "meta" / "episodes"
    meta_files = sorted(meta_dir.rglob("*.parquet"))
    if not meta_files:
        return False, "No metadata files found"

    meta_dfs = [pd.read_parquet(f) for f in meta_files]
    meta_df = pd.concat(meta_dfs, ignore_index=True)

    # Load data
    data_dir = root / "data"
    data_files = sorted(data_dir.rglob("*.parquet"))
    if not data_files:
        return False, "No data files found"

    data_dfs = [pd.read_parquet(f) for f in data_files]
    data_df = pd.concat(data_dfs, ignore_index=True)

    # Check each episode
    errors = []
    for _, row in meta_df.iterrows():
        ep_idx = int(row["episode_index"])
        meta_len = int(row["length"])
        data_len = len(data_df[data_df["episode_index"] == ep_idx])

        if data_len != meta_len:
            errors.append(f"Episode {ep_idx}: data has {data_len} frames, metadata says {meta_len}")

    if errors:
        return False, "\n".join(errors)

    return True, ""


def test_multiple_trims_then_delete(tmp_path, lerobot_dataset_factory):
    """Test that multiple trims followed by delete works correctly.

    This reproduces the stale metadata bug where:
    1. Trim episode 0 (shifts indices for episodes 1, 2, ...)
    2. Trim episode 2 (uses stale in-memory indices -> data corruption)
    3. Delete episode 0 (fails or produces corrupted dataset)

    Without the fix, episode 2 would have mismatched data/metadata.
    With the fix, all operations succeed and data remains consistent.
    """
    # Create two source datasets
    ds_a = lerobot_dataset_factory(
        root=tmp_path / "ds_a",
        repo_id=f"{DUMMY_REPO_ID}_a",
        total_episodes=10,
        total_frames=500,
        use_videos=False,
    )
    ds_b = lerobot_dataset_factory(
        root=tmp_path / "ds_b",
        repo_id=f"{DUMMY_REPO_ID}_b",
        total_episodes=5,
        total_frames=250,
        use_videos=False,
    )

    # Merge datasets
    merged_root = tmp_path / "ds_merged"
    aggregate_datasets(
        repo_ids=[ds_a.repo_id, ds_b.repo_id],
        roots=[ds_a.root, ds_b.root],
        aggr_repo_id=f"{DUMMY_REPO_ID}_merged",
        aggr_root=merged_root,
    )

    # Load merged dataset
    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock1,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock2,
    ):
        mock1.return_value = "v3.0"
        mock2.return_value = str(merged_root)
        ds = LeRobotDataset(f"{DUMMY_REPO_ID}_merged", root=merged_root)

    initial_episodes = ds.meta.total_episodes
    initial_frames = ds.meta.total_frames

    # Verify initial integrity
    is_valid, error = verify_dataset_integrity(merged_root)
    assert is_valid, f"Initial dataset is corrupted: {error}"

    # Get episode lengths for trimming
    ep0_len = ds.meta.episodes[0]["length"]
    ep2_len = ds.meta.episodes[2]["length"]

    # Record episode 2's initial bounds
    ep2_from_before = ds.meta.episodes[2]["dataset_from_index"]

    # --- Simulate GUI workflow: multiple trims then delete ---

    # Step 1: Trim episode 0 - remove first 5 frames
    frames_to_remove_from_ep0 = 5
    trim_episode_by_frames(
        ds,
        episode_index=0,
        start_frame=frames_to_remove_from_ep0,
        end_frame=ep0_len,
    )

    # Verify episode 2's from_index was shifted on disk
    fresh_episodes = load_episodes(merged_root)
    ep2_from_after_trim1 = fresh_episodes[2]["dataset_from_index"]
    assert ep2_from_after_trim1 == ep2_from_before - frames_to_remove_from_ep0, (
        f"Episode 2's from_index should have shifted by {frames_to_remove_from_ep0}"
    )

    # Step 2: Trim episode 2 - remove first 3 frames
    # This is where the bug would occur without the fix:
    # The stale in-memory metadata would have wrong indices
    frames_to_remove_from_ep2 = 3
    trim_episode_by_frames(
        ds,
        episode_index=2,
        start_frame=frames_to_remove_from_ep2,
        end_frame=ep2_len,
    )

    # Verify integrity after trims (this is where the bug manifests)
    is_valid, error = verify_dataset_integrity(merged_root)
    assert is_valid, f"Dataset corrupted after trims: {error}"

    # Step 3: Reload dataset (like GUI does before delete)
    ds.meta.info = load_info(merged_root)
    ds.meta.episodes = load_episodes(merged_root)

    if hf_datasets is not None:
        hf_datasets.disable_caching()
        try:
            ds._ensure_reader().load_and_activate()
        finally:
            hf_datasets.enable_caching()

    # Step 4: Delete episode 0
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "dataset"
        new_ds = delete_episodes(
            dataset=ds,
            episode_indices=[0],
            output_dir=temp_path,
            repo_id=ds.repo_id,
        )

        # Verify the new dataset has correct episode count
        assert new_ds.meta.total_episodes == initial_episodes - 1

        # Verify integrity of new dataset
        is_valid, error = verify_dataset_integrity(temp_path)
        assert is_valid, f"Dataset corrupted after delete: {error}"

        # Verify we can iterate through all frames
        count = 0
        for _ in new_ds:
            count += 1

        expected_frames = (
            initial_frames
            - ep0_len  # Episode 0 deleted entirely
            - frames_to_remove_from_ep2  # Episode 2 trimmed
            # Note: ep0 was already trimmed before delete, but delete removes entire episode
        )
        # Actually, the expected frames calculation:
        # - Initial: initial_frames
        # - After trim ep0: initial_frames - frames_to_remove_from_ep0
        # - After trim ep2: initial_frames - frames_to_remove_from_ep0 - frames_to_remove_from_ep2
        # - After delete ep0: (initial_frames - frames_to_remove_from_ep0 - frames_to_remove_from_ep2) - (ep0_len - frames_to_remove_from_ep0)
        # Simplify: initial_frames - ep0_len - frames_to_remove_from_ep2
        expected_frames = initial_frames - ep0_len - frames_to_remove_from_ep2

        assert count == expected_frames, f"Expected {expected_frames} frames, got {count}"


def test_trim_same_episode_multiple_times(tmp_path, lerobot_dataset_factory):
    """Test trimming the same episode multiple times (edge case)."""
    ds = lerobot_dataset_factory(
        root=tmp_path / "ds",
        repo_id=DUMMY_REPO_ID,
        total_episodes=5,
        total_frames=250,
        use_videos=False,
    )

    ep0_len = ds.meta.episodes[0]["length"]

    # Trim episode 0 multiple times
    # First trim: remove first 5 frames
    trim_episode_by_frames(ds, episode_index=0, start_frame=5, end_frame=ep0_len)

    # Reload to get updated length
    ds.meta.episodes = load_episodes(ds.root)
    ep0_len_after = ds.meta.episodes[0]["length"]
    assert ep0_len_after == ep0_len - 5

    # Second trim: remove last 3 frames
    trim_episode_by_frames(ds, episode_index=0, start_frame=0, end_frame=ep0_len_after - 3)

    # Verify integrity
    is_valid, error = verify_dataset_integrity(ds.root)
    assert is_valid, f"Dataset corrupted after multiple trims to same episode: {error}"

    # Verify final length
    ds.meta.episodes = load_episodes(ds.root)
    final_len = ds.meta.episodes[0]["length"]
    assert final_len == ep0_len - 5 - 3


def test_trim_all_episodes_sequentially(tmp_path, lerobot_dataset_factory):
    """Test trimming all episodes in sequence (stress test for stale metadata bug)."""
    ds = lerobot_dataset_factory(
        root=tmp_path / "ds",
        repo_id=DUMMY_REPO_ID,
        total_episodes=5,
        total_frames=250,
        use_videos=False,
    )

    # Record original lengths
    original_lengths = [ds.meta.episodes[i]["length"] for i in range(ds.meta.total_episodes)]

    # Trim 2 frames from the start of each episode
    frames_to_trim = 2
    for ep_idx in range(ds.meta.total_episodes):
        # Reload to get current length (may have changed due to length updates)
        ds.meta.episodes = load_episodes(ds.root)
        current_len = ds.meta.episodes[ep_idx]["length"]

        if current_len > frames_to_trim:
            trim_episode_by_frames(
                ds,
                episode_index=ep_idx,
                start_frame=frames_to_trim,
                end_frame=current_len,
            )

    # Verify integrity after all trims
    is_valid, error = verify_dataset_integrity(ds.root)
    assert is_valid, f"Dataset corrupted after trimming all episodes: {error}"

    # Verify each episode was trimmed correctly
    ds.meta.episodes = load_episodes(ds.root)
    for ep_idx in range(ds.meta.total_episodes):
        expected_len = original_lengths[ep_idx] - frames_to_trim
        actual_len = ds.meta.episodes[ep_idx]["length"]
        assert actual_len == expected_len, (
            f"Episode {ep_idx}: expected length {expected_len}, got {actual_len}"
        )
