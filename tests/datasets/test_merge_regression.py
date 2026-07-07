#!/usr/bin/env python
"""Test for the merge bug where merging large datasets causes missing meta/episodes files.

Run with: pytest tests/test_merge_bug.py -v

BUG SUMMARY:
When merging datasets where source datasets have multiple meta/episodes files
(e.g., file-000.parquet, file-001.parquet, file-002.parquet), the merged dataset's
episode metadata incorrectly preserves the source file indices instead of remapping
them to the actual destination files.

Example from real data:
- Source dataset has 142 episodes across 3 meta files (file-000, file-001, file-002)
- After merge, only file-000.parquet exists on disk
- But episodes still reference file-001 and file-002 → FileNotFoundError

Root cause: aggregate.py update_meta_data() line 133-134 just adds offsets:
    df["meta/episodes/chunk_index"] = df["meta/episodes/chunk_index"] + meta_idx["chunk"]
    df["meta/episodes/file_index"] = df["meta/episodes/file_index"] + meta_idx["file"]

This should use src_to_dst mapping like data files do (line 136+).
"""

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_EPISODES_PATH
from tests.fixtures.constants import DUMMY_REPO_ID


def split_meta_episodes_into_multiple_files(dataset_root: Path, episodes_per_file: int = 5):
    """Manually split a dataset's meta/episodes into multiple files to simulate large datasets.

    This is needed because the test fixture creates small datasets that fit in a single file.
    Real-world datasets with many episodes get split across multiple meta/episodes files.

    Args:
        dataset_root: Path to the dataset root directory
        episodes_per_file: Number of episodes per file

    Returns:
        Number of meta files created
    """
    meta_dir = dataset_root / "meta" / "episodes" / "chunk-000"
    original_file = meta_dir / "file-000.parquet"

    if not original_file.exists():
        raise FileNotFoundError(f"Expected meta file at {original_file}")

    # Read all episodes
    df = pd.read_parquet(original_file)
    total_episodes = len(df)

    if total_episodes <= episodes_per_file:
        # Not enough episodes to split
        return 1

    # Split into multiple files
    file_idx = 0
    files_created = 0
    for start_idx in range(0, total_episodes, episodes_per_file):
        end_idx = min(start_idx + episodes_per_file, total_episodes)
        chunk_df = df.iloc[start_idx:end_idx].copy()

        # Update the self-referential file_index for each episode
        chunk_df["meta/episodes/file_index"] = file_idx

        # Write to new file
        out_file = meta_dir / f"file-{file_idx:03d}.parquet"
        chunk_df.to_parquet(out_file)
        file_idx += 1
        files_created += 1

    # Remove original if we created new files (file-000 was rewritten)
    return files_created


def test_merge_datasets_with_multiple_meta_files(tmp_path, lerobot_dataset_factory):
    """Test that merging datasets with multiple meta/episodes files works correctly.

    This reproduces the bug where source datasets have episodes spread across
    multiple meta/episodes/*.parquet files, but after merging, the episode metadata
    still references the original file indices instead of the merged destination.

    The test manually splits the source datasets' meta files to simulate the
    real-world scenario where large datasets have multiple meta files.
    """
    # Create datasets
    ds_a = lerobot_dataset_factory(
        root=tmp_path / "ds_a",
        repo_id=f"{DUMMY_REPO_ID}_a",
        total_episodes=15,
        total_frames=750,
        use_videos=False,
    )
    ds_b = lerobot_dataset_factory(
        root=tmp_path / "ds_b",
        repo_id=f"{DUMMY_REPO_ID}_b",
        total_episodes=10,
        total_frames=500,
        use_videos=False,
    )

    # Manually split meta files to simulate large datasets
    # ds_a: 15 episodes -> 3 files (5 episodes each): file-000, file-001, file-002
    # ds_b: 10 episodes -> 2 files (5 episodes each): file-000, file-001
    num_files_a = split_meta_episodes_into_multiple_files(ds_a.root, episodes_per_file=5)
    num_files_b = split_meta_episodes_into_multiple_files(ds_b.root, episodes_per_file=5)

    print(f"\nSource ds_a: {ds_a.num_episodes} episodes across {num_files_a} meta files")
    print(f"Source ds_b: {ds_b.num_episodes} episodes across {num_files_b} meta files")

    # Verify source datasets have multiple files
    assert num_files_a == 3, f"Expected 3 meta files for ds_a, got {num_files_a}"
    assert num_files_b == 2, f"Expected 2 meta files for ds_b, got {num_files_b}"

    # Merge datasets - the bug occurs here because update_meta_data() uses
    # simple offset addition instead of src_to_dst mapping for meta files
    aggregate_datasets(
        repo_ids=[ds_a.repo_id, ds_b.repo_id],
        roots=[ds_a.root, ds_b.root],
        aggr_repo_id=f"{DUMMY_REPO_ID}_merged",
        aggr_root=tmp_path / "ds_merged",
        data_files_size_in_mb=250,  # Default size, meta files should merge into fewer files
    )

    # Load merged dataset
    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock1,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock2,
    ):
        mock1.return_value = "v3.0"
        mock2.return_value = str(tmp_path / "ds_merged")
        ds_merged = LeRobotDataset(f"{DUMMY_REPO_ID}_merged", root=tmp_path / "ds_merged")

    # List actual meta files on disk
    meta_files_on_disk = set()
    for f in (tmp_path / "ds_merged" / "meta" / "episodes").rglob("*.parquet"):
        rel = f.relative_to(tmp_path / "ds_merged")
        meta_files_on_disk.add(str(rel))

    print(f"\nMerged dataset: {ds_merged.num_episodes} episodes")
    print(f"Meta files on disk: {sorted(meta_files_on_disk)}")

    # Check what files are referenced in metadata
    referenced_files = set()
    missing_files = []
    for ep_idx in range(ds_merged.num_episodes):
        ep = ds_merged.meta.episodes[ep_idx]
        chunk_idx = int(ep["meta/episodes/chunk_index"])
        file_idx = int(ep["meta/episodes/file_index"])
        meta_path = ds_merged.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        rel_path = str(meta_path.relative_to(ds_merged.root))
        referenced_files.add(rel_path)

        if not meta_path.exists():
            missing_files.append((ep_idx, chunk_idx, file_idx))

    print(f"Files referenced by metadata: {sorted(referenced_files)}")

    # Show file_index distribution
    file_indices = [
        int(ds_merged.meta.episodes[i]["meta/episodes/file_index"]) for i in range(ds_merged.num_episodes)
    ]
    print(f"File indices in metadata: {sorted(set(file_indices))} (unique values)")
    print(f"File index counts: {[(i, file_indices.count(i)) for i in sorted(set(file_indices))]}")

    if missing_files:
        print(f"\nBUG REPRODUCED: {len(missing_files)} episodes reference missing meta files!")
        for ep_idx, chunk, file in missing_files[:10]:
            print(f"  Episode {ep_idx} -> chunk-{chunk:03d}/file-{file:03d}.parquet (MISSING)")

    assert len(missing_files) == 0, (
        f"{len(missing_files)} episodes reference non-existent meta/episodes files.\n"
        f"Files on disk: {sorted(meta_files_on_disk)}\n"
        f"Files referenced: {sorted(referenced_files)}\n"
        f"This is the meta/episodes src_to_dst mapping bug in aggregate.py"
    )

    # Also verify data files
    for ep_idx in range(ds_merged.num_episodes):
        data_file_path = ds_merged.root / ds_merged.meta.get_data_file_path(ep_idx)
        assert data_file_path.exists(), f"Episode {ep_idx} references missing data file"

    # Verify iteration works
    count = 0
    for _item in ds_merged:
        count += 1
    expected = ds_a.num_frames + ds_b.num_frames
    assert count == expected, f"Expected {expected} frames, got {count}"
