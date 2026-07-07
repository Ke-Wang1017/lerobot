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
"""Tests for virtual (non-destructive) dataset editing operations.

Virtual edits modify metadata and parquet data but do NOT re-encode video files.
This preserves video quality and is much faster than re-encoding.
"""

import shutil
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from lerobot.datasets.io_utils import load_episodes, load_info
from lerobot.datasets.lerobot_dataset import LeRobotDataset


@pytest.fixture
def video_dataset(tmp_path):
    """Create a copy of pusht dataset for testing with only 3 episodes."""
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    source_dataset = LeRobotDataset("lerobot/pusht", episodes=[0, 1, 2])

    test_root = tmp_path / "pusht_test"
    shutil.copytree(source_dataset.root, test_root)

    # Filter episodes parquet to only include the 3 episodes we loaded
    episodes_parquet_path = test_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if episodes_parquet_path.exists():
        table = pq.read_table(episodes_parquet_path)
        df = table.to_pandas()
        # Keep only first 3 episodes
        df_filtered = df[df["episode_index"].isin([0, 1, 2])].copy()
        pq.write_table(pa.Table.from_pandas(df_filtered, preserve_index=False), episodes_parquet_path)

    # Update info.json with correct totals
    info_path = test_root / "meta" / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        info["total_episodes"] = 3
        # Calculate total frames from the 3 episodes
        total_frames = sum(ep["length"] for ep in source_dataset.meta.episodes)
        info["total_frames"] = total_frames
        info["splits"] = {"train": "0:3"}
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(test_root)

        dataset = LeRobotDataset("lerobot/pusht", root=test_root)

    # Ensure episodes are loaded
    if dataset.meta.episodes is None:
        dataset.meta.episodes = load_episodes(dataset.root)

    return dataset


class TestVirtualTrim:
    """Tests for virtual trim operation (no video re-encoding)."""

    def test_trim_updates_metadata(self, video_dataset):
        """Verify trim updates episode metadata correctly."""
        from lerobot.datasets.dataset_tools import trim_episode_virtual

        original_length = video_dataset.meta.episodes[0]["length"]
        original_total_frames = video_dataset.meta.total_frames

        # Trim 10 frames from start, 5 from end
        start_frame = 10
        end_frame = original_length - 5

        trim_episode_virtual(
            video_dataset,
            episode_index=0,
            start_frame=start_frame,
            end_frame=end_frame,
        )

        # Reload metadata
        video_dataset.meta.episodes = load_episodes(video_dataset.root)
        video_dataset.meta.info = load_info(video_dataset.root)

        new_length = video_dataset.meta.episodes[0]["length"]
        expected_length = end_frame - start_frame

        assert new_length == expected_length
        assert video_dataset.meta.total_frames == original_total_frames - (original_length - expected_length)

    def test_trim_updates_video_timestamps(self, video_dataset):
        """Verify trim updates video from/to timestamps correctly."""
        from lerobot.datasets.dataset_tools import trim_episode_virtual

        video_key = video_dataset.meta.video_keys[0]
        original_from_ts = video_dataset.meta.episodes[0][f"videos/{video_key}/from_timestamp"]
        original_to_ts = video_dataset.meta.episodes[0][f"videos/{video_key}/to_timestamp"]
        original_length = video_dataset.meta.episodes[0]["length"]

        # Trim 10 frames from start
        start_frame = 10
        end_frame = original_length
        trim_start_s = start_frame / video_dataset.fps

        trim_episode_virtual(
            video_dataset,
            episode_index=0,
            start_frame=start_frame,
            end_frame=end_frame,
        )

        video_dataset.meta.episodes = load_episodes(video_dataset.root)

        new_from_ts = video_dataset.meta.episodes[0][f"videos/{video_key}/from_timestamp"]
        new_to_ts = video_dataset.meta.episodes[0][f"videos/{video_key}/to_timestamp"]

        # from_timestamp should increase by trim amount
        assert abs(new_from_ts - (original_from_ts + trim_start_s)) < 0.001
        # to_timestamp should stay the same (we only trimmed start)
        assert abs(new_to_ts - original_to_ts) < 0.001

    def test_trim_does_not_affect_other_episodes_timestamps(self, video_dataset):
        """Verify trimming episode 0 does NOT change other episodes' timestamps."""
        from lerobot.datasets.dataset_tools import trim_episode_virtual

        video_key = video_dataset.meta.video_keys[0]

        # Save original timestamps for episodes 1 and 2
        ep1_from = video_dataset.meta.episodes[1][f"videos/{video_key}/from_timestamp"]
        ep1_to = video_dataset.meta.episodes[1][f"videos/{video_key}/to_timestamp"]
        ep2_from = video_dataset.meta.episodes[2][f"videos/{video_key}/from_timestamp"]
        ep2_to = video_dataset.meta.episodes[2][f"videos/{video_key}/to_timestamp"]

        original_length = video_dataset.meta.episodes[0]["length"]
        trim_episode_virtual(video_dataset, episode_index=0, start_frame=10, end_frame=original_length)

        video_dataset.meta.episodes = load_episodes(video_dataset.root)

        # Other episodes' timestamps should be UNCHANGED
        assert video_dataset.meta.episodes[1][f"videos/{video_key}/from_timestamp"] == ep1_from
        assert video_dataset.meta.episodes[1][f"videos/{video_key}/to_timestamp"] == ep1_to
        assert video_dataset.meta.episodes[2][f"videos/{video_key}/from_timestamp"] == ep2_from
        assert video_dataset.meta.episodes[2][f"videos/{video_key}/to_timestamp"] == ep2_to

    def test_trim_does_not_modify_video_file(self, video_dataset):
        """Verify virtual trim does NOT modify video files."""
        from lerobot.datasets.dataset_tools import trim_episode_virtual

        video_key = video_dataset.meta.video_keys[0]
        video_path = video_dataset.root / video_dataset.meta.get_video_file_path(0, video_key)

        # Get original video file stats
        original_size = video_path.stat().st_size
        original_mtime = video_path.stat().st_mtime

        original_length = video_dataset.meta.episodes[0]["length"]
        trim_episode_virtual(
            video_dataset,
            episode_index=0,
            start_frame=10,
            end_frame=original_length - 10,
        )

        # Video file should be UNCHANGED
        assert video_path.stat().st_size == original_size
        assert video_path.stat().st_mtime == original_mtime

    def test_trim_updates_parquet_data(self, video_dataset):
        """Verify trim removes correct rows from parquet data."""
        from lerobot.datasets.dataset_tools import trim_episode_virtual

        original_length = video_dataset.meta.episodes[0]["length"]
        original_total_rows = len(video_dataset.hf_dataset)

        start_frame = 5
        end_frame = original_length - 5
        frames_removed = original_length - (end_frame - start_frame)

        trim_episode_virtual(
            video_dataset,
            episode_index=0,
            start_frame=start_frame,
            end_frame=end_frame,
        )

        # Reload dataset
        video_dataset._ensure_reader().load_and_activate()

        assert len(video_dataset.hf_dataset) == original_total_rows - frames_removed

    def test_trim_shifts_subsequent_episodes(self, video_dataset):
        """Verify trimming episode 0 shifts dataset indices of subsequent episodes."""
        from lerobot.datasets.dataset_tools import trim_episode_virtual

        original_ep1_from = video_dataset.meta.episodes[1]["dataset_from_index"]
        original_length = video_dataset.meta.episodes[0]["length"]

        start_frame = 10
        end_frame = original_length - 10
        frames_removed = original_length - (end_frame - start_frame)

        trim_episode_virtual(
            video_dataset,
            episode_index=0,
            start_frame=start_frame,
            end_frame=end_frame,
        )

        video_dataset.meta.episodes = load_episodes(video_dataset.root)

        new_ep1_from = video_dataset.meta.episodes[1]["dataset_from_index"]
        assert new_ep1_from == original_ep1_from - frames_removed


class TestVirtualDelete:
    """Tests for virtual delete operation (no video re-encoding)."""

    def test_delete_removes_episode_metadata(self, video_dataset):
        """Verify delete removes episode from metadata."""
        from lerobot.datasets.dataset_tools import delete_episodes_virtual

        original_total_episodes = video_dataset.meta.total_episodes

        delete_episodes_virtual(video_dataset, episode_indices=[1])

        video_dataset.meta.episodes = load_episodes(video_dataset.root)
        video_dataset.meta.info = load_info(video_dataset.root)

        assert video_dataset.meta.total_episodes == original_total_episodes - 1
        assert len(video_dataset.meta.episodes) == original_total_episodes - 1

    def test_delete_renumbers_episodes(self, video_dataset):
        """Verify remaining episodes are renumbered correctly."""
        from lerobot.datasets.dataset_tools import delete_episodes_virtual

        # Delete episode 1 (middle episode)
        delete_episodes_virtual(video_dataset, episode_indices=[1])

        video_dataset.meta.episodes = load_episodes(video_dataset.root)

        # Episode indices should be 0, 1 (was 0, 2)
        episode_indices = [ep["episode_index"] for ep in video_dataset.meta.episodes]
        assert episode_indices == [0, 1]

    def test_delete_does_not_modify_video_files(self, video_dataset):
        """Verify virtual delete does NOT modify video files."""
        from lerobot.datasets.dataset_tools import delete_episodes_virtual

        video_key = video_dataset.meta.video_keys[0]

        # Get all video file stats before delete
        video_files = list((video_dataset.root / "videos" / video_key).rglob("*.mp4"))
        original_stats = {str(f): (f.stat().st_size, f.stat().st_mtime) for f in video_files}

        delete_episodes_virtual(video_dataset, episode_indices=[1])

        # All video files should be UNCHANGED
        for f in video_files:
            size, mtime = original_stats[str(f)]
            assert f.stat().st_size == size
            assert f.stat().st_mtime == mtime

    def test_delete_updates_parquet_data(self, video_dataset):
        """Verify delete removes correct rows from parquet data."""
        from lerobot.datasets.dataset_tools import delete_episodes_virtual

        ep1_length = video_dataset.meta.episodes[1]["length"]
        # Calculate original row count from episode metadata, not hf_dataset
        # (since the data parquet may contain more rows than selected episodes)
        original_total_rows = sum(ep["length"] for ep in video_dataset.meta.episodes)

        delete_episodes_virtual(video_dataset, episode_indices=[1])

        video_dataset._ensure_reader().load_and_activate()

        assert len(video_dataset.hf_dataset) == original_total_rows - ep1_length

    def test_delete_multiple_episodes(self, video_dataset):
        """Verify deleting multiple episodes works correctly."""
        from lerobot.datasets.dataset_tools import delete_episodes_virtual

        original_total_episodes = video_dataset.meta.total_episodes
        ep0_length = video_dataset.meta.episodes[0]["length"]
        ep2_length = video_dataset.meta.episodes[2]["length"]
        original_total_frames = video_dataset.meta.total_frames

        delete_episodes_virtual(video_dataset, episode_indices=[0, 2])

        video_dataset.meta.episodes = load_episodes(video_dataset.root)
        video_dataset.meta.info = load_info(video_dataset.root)

        assert video_dataset.meta.total_episodes == original_total_episodes - 2
        assert video_dataset.meta.total_frames == original_total_frames - ep0_length - ep2_length

    def test_delete_updates_dataset_indices(self, video_dataset):
        """Verify dataset_from_index and dataset_to_index are updated correctly."""
        from lerobot.datasets.dataset_tools import delete_episodes_virtual

        # Delete episode 0, episode 1 (was index 1) should become index 0
        # and its dataset_from_index should start at 0
        delete_episodes_virtual(video_dataset, episode_indices=[0])

        video_dataset.meta.episodes = load_episodes(video_dataset.root)

        # First episode should now start at index 0
        assert video_dataset.meta.episodes[0]["dataset_from_index"] == 0


class TestVirtualEditIntegration:
    """Integration tests for virtual edits."""

    def test_trim_then_delete(self, video_dataset):
        """Verify trim followed by delete works correctly."""
        from lerobot.datasets.dataset_tools import delete_episodes_virtual, trim_episode_virtual

        original_total_frames = video_dataset.meta.total_frames
        ep0_length = video_dataset.meta.episodes[0]["length"]
        ep1_length = video_dataset.meta.episodes[1]["length"]

        # Trim episode 0
        trim_episode_virtual(
            video_dataset,
            episode_index=0,
            start_frame=5,
            end_frame=ep0_length - 5,
        )

        # Reload
        video_dataset.meta.episodes = load_episodes(video_dataset.root)

        # Delete episode 1
        delete_episodes_virtual(video_dataset, episode_indices=[1])

        video_dataset.meta.episodes = load_episodes(video_dataset.root)
        video_dataset.meta.info = load_info(video_dataset.root)

        # Should have 2 episodes left
        assert video_dataset.meta.total_episodes == 2

        # Total frames = original - 10 (trim) - ep1_length (delete)
        expected_frames = original_total_frames - 10 - ep1_length
        assert video_dataset.meta.total_frames == expected_frames

    def test_dataset_still_loadable_after_edits(self, video_dataset):
        """Verify dataset can be reloaded after virtual edits."""
        from lerobot.datasets.dataset_tools import delete_episodes_virtual, trim_episode_virtual

        ep0_length = video_dataset.meta.episodes[0]["length"]

        trim_episode_virtual(
            video_dataset,
            episode_index=0,
            start_frame=5,
            end_frame=ep0_length - 5,
        )
        delete_episodes_virtual(video_dataset, episode_indices=[1])

        # Try to reload the dataset
        with (
            patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
            patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
        ):
            mock_get_safe_version.return_value = "v3.0"
            mock_snapshot_download.return_value = str(video_dataset.root)

            reloaded = LeRobotDataset("lerobot/pusht", root=video_dataset.root)

        assert reloaded.meta.total_episodes == 2
        assert len(reloaded) == reloaded.meta.total_frames


class TestMultiFileDatasetEdits:
    """Tests for virtual edits on multi-file datasets.

    Multi-file datasets have parquet data split across multiple files (e.g., chunk-000/file-000.parquet,
    chunk-000/file-001.parquet). In these datasets, the episode metadata's `dataset_from_index` field
    resets to 0 at each file boundary.

    These tests verify that virtual edits correctly handle this by recomputing indices from
    cumulative episode lengths rather than relying on per-file `dataset_from_index` values.
    """

    @pytest.fixture
    def multifile_dataset(self, tmp_path, lerobot_dataset_factory):
        """Create a multi-file dataset by aggregating datasets with small file size limits."""
        from lerobot.datasets.aggregate import aggregate_datasets

        # Create two source datasets
        ds_0 = lerobot_dataset_factory(
            root=tmp_path / "src_0",
            repo_id="test/src_0",
            total_episodes=5,
            total_frames=250,
        )
        ds_1 = lerobot_dataset_factory(
            root=tmp_path / "src_1",
            repo_id="test/src_1",
            total_episodes=5,
            total_frames=250,
        )

        # Aggregate with small file size to force multiple parquet files
        aggr_root = tmp_path / "multifile"
        aggregate_datasets(
            repo_ids=[ds_0.repo_id, ds_1.repo_id],
            roots=[ds_0.root, ds_1.root],
            aggr_repo_id="test/multifile",
            aggr_root=aggr_root,
            data_files_size_in_mb=0.01,  # Force file rotation
        )

        # Verify we actually have multiple files
        data_dir = aggr_root / "data"
        parquet_files = list(data_dir.rglob("*.parquet"))
        assert len(parquet_files) > 1, "Test setup failed: expected multiple parquet files"

        # Load the aggregated dataset
        with (
            patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
            patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
        ):
            mock_get_safe_version.return_value = "v3.0"
            mock_snapshot_download.return_value = str(aggr_root)
            dataset = LeRobotDataset("test/multifile", root=aggr_root)

        # Ensure episodes are loaded
        if dataset.meta.episodes is None:
            dataset.meta.episodes = load_episodes(dataset.root)

        return dataset

    def test_multifile_trim_updates_indices_correctly(self, multifile_dataset):
        """Verify trim on multi-file dataset updates indices correctly.

        The key assertion is that after trimming, subsequent episodes' dataset_from_index
        values are shifted by exactly the number of frames removed, regardless of which
        file they reside in.
        """
        from lerobot.datasets.dataset_tools import trim_episode_virtual

        # Get original state - pick an episode that's likely in the second file
        mid_episode = len(multifile_dataset.meta.episodes) // 2
        original_subsequent_from = multifile_dataset.meta.episodes[mid_episode + 1]["dataset_from_index"]
        original_length = multifile_dataset.meta.episodes[mid_episode]["length"]

        # Trim 10 frames from mid episode
        start_frame = 5
        end_frame = original_length - 5
        frames_removed = original_length - (end_frame - start_frame)

        trim_episode_virtual(
            multifile_dataset,
            episode_index=mid_episode,
            start_frame=start_frame,
            end_frame=end_frame,
        )

        multifile_dataset.meta.episodes = load_episodes(multifile_dataset.root)

        new_subsequent_from = multifile_dataset.meta.episodes[mid_episode + 1]["dataset_from_index"]
        assert new_subsequent_from == original_subsequent_from - frames_removed

    def test_multifile_delete_renumbers_correctly(self, multifile_dataset):
        """Verify delete on multi-file dataset renumbers episodes correctly."""
        from lerobot.datasets.dataset_tools import delete_episodes_virtual

        original_total_episodes = multifile_dataset.meta.total_episodes

        # Delete an episode from the middle
        mid_episode = len(multifile_dataset.meta.episodes) // 2
        delete_episodes_virtual(multifile_dataset, episode_indices=[mid_episode])

        multifile_dataset.meta.episodes = load_episodes(multifile_dataset.root)
        multifile_dataset.meta.info = load_info(multifile_dataset.root)

        # Check episode count
        assert multifile_dataset.meta.total_episodes == original_total_episodes - 1

        # Check episode indices are sequential
        episode_indices = [ep["episode_index"] for ep in multifile_dataset.meta.episodes]
        assert episode_indices == list(range(original_total_episodes - 1))

    def test_multifile_dataset_reloadable_after_edits(self, multifile_dataset):
        """Verify multi-file dataset can be reloaded and iterated after edits."""
        from lerobot.datasets.dataset_tools import delete_episodes_virtual, trim_episode_virtual

        original_total_frames = multifile_dataset.meta.total_frames
        ep0_length = multifile_dataset.meta.episodes[0]["length"]

        # Trim episode 0
        trim_episode_virtual(
            multifile_dataset,
            episode_index=0,
            start_frame=5,
            end_frame=ep0_length - 5,
        )

        # Delete episode 1
        multifile_dataset.meta.episodes = load_episodes(multifile_dataset.root)
        ep1_length = multifile_dataset.meta.episodes[1]["length"]
        delete_episodes_virtual(multifile_dataset, episode_indices=[1])

        # Reload dataset
        with (
            patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
            patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
        ):
            mock_get_safe_version.return_value = "v3.0"
            mock_snapshot_download.return_value = str(multifile_dataset.root)
            reloaded = LeRobotDataset("test/multifile", root=multifile_dataset.root)

        # Verify frame count
        expected_frames = original_total_frames - 10 - ep1_length  # 10 trimmed + ep1 deleted
        assert reloaded.meta.total_frames == expected_frames
        assert len(reloaded) == reloaded.meta.total_frames

        # Verify we can iterate through all frames without errors
        for i in range(len(reloaded)):
            _ = reloaded.hf_dataset[i]

    def test_multifile_dataset_indices_are_compact_after_delete(self, multifile_dataset):
        """Verify dataset indices remain compact after delete."""
        from lerobot.datasets.dataset_tools import delete_episodes_virtual

        delete_episodes_virtual(multifile_dataset, episode_indices=[2])

        multifile_dataset._ensure_reader().load_and_activate()

        indices = [item["index"].item() for item in multifile_dataset.hf_dataset]
        expected_indices = list(range(len(indices)))

        assert indices == expected_indices, "Dataset indices should be compact 0..N-1 with no holes"

    def test_multifile_dataset_indices_are_compact_after_trim(self, multifile_dataset):
        """Verify dataset indices remain compact after trim only (no delete)."""
        from lerobot.datasets.dataset_tools import trim_episode_virtual

        # Trim an episode in the middle (likely spans files)
        mid_episode = len(multifile_dataset.meta.episodes) // 2
        ep_length = multifile_dataset.meta.episodes[mid_episode]["length"]

        trim_episode_virtual(
            multifile_dataset,
            episode_index=mid_episode,
            start_frame=5,
            end_frame=ep_length - 5,
        )

        multifile_dataset._ensure_reader().load_and_activate()

        indices = [item["index"].item() for item in multifile_dataset.hf_dataset]
        expected_indices = list(range(len(indices)))

        assert indices == expected_indices, "Dataset indices should be compact 0..N-1 with no holes"


class TestRepairEpisodeIndices:
    """Tests for the repair_episode_indices function."""

    def test_repair_fixes_broken_indices(self, video_dataset):
        """Verify repair fixes broken dataset_from_index values."""
        from lerobot.datasets.dataset_tools import repair_episode_indices

        # Manually corrupt the episode metadata by setting wrong indices
        episodes_dir = video_dataset.root / "meta" / "episodes"
        parquet_files = list(episodes_dir.rglob("*.parquet"))
        assert len(parquet_files) > 0

        # Read and corrupt the metadata
        for parquet_path in parquet_files:
            df = pd.read_parquet(parquet_path)
            # Set all dataset_from_index to 0 (simulating per-file reset bug)
            df["dataset_from_index"] = 0
            df["dataset_to_index"] = df["length"]
            df.to_parquet(parquet_path, index=False)

        # Verify corruption
        video_dataset.meta.episodes = load_episodes(video_dataset.root)
        for ep in video_dataset.meta.episodes:
            assert ep["dataset_from_index"] == 0, "Setup failed: indices not corrupted"

        # Run repair
        repaired = repair_episode_indices(video_dataset.root)

        # Should have repaired all episodes except the first one (which starts at 0)
        assert repaired >= len(video_dataset.meta.episodes) - 1

        # Reload and verify indices are now cumulative
        video_dataset.meta.episodes = load_episodes(video_dataset.root)
        cumulative = 0
        for ep in video_dataset.meta.episodes:
            assert ep["dataset_from_index"] == cumulative, (
                f"Episode {ep['episode_index']}: expected dataset_from_index={cumulative}, "
                f"got {ep['dataset_from_index']}"
            )
            assert ep["dataset_to_index"] == cumulative + ep["length"]
            cumulative += ep["length"]

    def test_repair_no_op_when_correct(self, video_dataset):
        """Verify repair returns 0 when indices are already correct."""
        from lerobot.datasets.dataset_tools import repair_episode_indices

        # First call should return 0 (dataset created correctly)
        repaired = repair_episode_indices(video_dataset.root)
        assert repaired == 0, "Expected no repairs on fresh dataset"

        # Second call should also return 0
        repaired = repair_episode_indices(video_dataset.root)
        assert repaired == 0, "Expected no repairs on second call"

    def test_repair_preserves_other_fields(self, video_dataset):
        """Verify repair doesn't modify other episode metadata fields."""
        from lerobot.datasets.dataset_tools import repair_episode_indices

        # Get original values
        video_dataset.meta.episodes = load_episodes(video_dataset.root)
        original_lengths = [ep["length"] for ep in video_dataset.meta.episodes]
        original_indices = [ep["episode_index"] for ep in video_dataset.meta.episodes]

        # Corrupt and repair
        episodes_dir = video_dataset.root / "meta" / "episodes"
        for parquet_path in episodes_dir.rglob("*.parquet"):
            df = pd.read_parquet(parquet_path)
            df["dataset_from_index"] = 0
            df.to_parquet(parquet_path, index=False)

        repair_episode_indices(video_dataset.root)

        # Verify other fields unchanged
        video_dataset.meta.episodes = load_episodes(video_dataset.root)
        assert [ep["length"] for ep in video_dataset.meta.episodes] == original_lengths
        assert [ep["episode_index"] for ep in video_dataset.meta.episodes] == original_indices

    def test_repair_fixes_data_parquet_indices(self, video_dataset):
        """Verify repair also fixes the data parquet's index column.

        This is critical for multi-file datasets where the data index column
        may have per-file indices that don't match the metadata's global indices.
        """
        from lerobot.datasets.dataset_tools import repair_episode_indices

        video_dataset.meta.episodes = load_episodes(video_dataset.root)

        # Get the valid episode indices from metadata
        valid_episodes = {ep["episode_index"] for ep in video_dataset.meta.episodes}

        # Corrupt both episode metadata AND data parquet indices to be per-file (0-based)
        episodes_dir = video_dataset.root / "meta" / "episodes"
        for parquet_path in episodes_dir.rglob("*.parquet"):
            df = pd.read_parquet(parquet_path)
            df["dataset_from_index"] = 0
            df["dataset_to_index"] = df["length"]
            df.to_parquet(parquet_path, index=False)

        data_dir = video_dataset.root / "data"
        for data_path in data_dir.rglob("*.parquet"):
            df = pd.read_parquet(data_path)
            # Filter to only valid episodes (fixture may have extra data)
            df = df[df["episode_index"].isin(valid_episodes)].copy()
            # Reset index to be per-file (0-based within each file)
            df["index"] = range(len(df))
            df.to_parquet(data_path, index=False)

        # Run repair
        repaired = repair_episode_indices(video_dataset.root)
        assert repaired > 0, "Expected some repairs"

        # Reload and verify data parquet indices are now globally continuous
        video_dataset.meta.episodes = load_episodes(video_dataset.root)

        # Build expected global index for each (episode_index, frame_index)
        episode_starts = {}
        cumulative = 0
        for ep in video_dataset.meta.episodes:
            episode_starts[ep["episode_index"]] = cumulative
            cumulative += ep["length"]

        # Verify data parquet indices match
        for data_path in data_dir.rglob("*.parquet"):
            df = pd.read_parquet(data_path)
            for _, row in df.iterrows():
                ep_idx = row["episode_index"]
                if ep_idx not in episode_starts:
                    continue  # Skip episodes not in metadata
                expected_index = episode_starts[ep_idx] + row["frame_index"]
                assert row["index"] == expected_index, (
                    f"Data index mismatch: episode={ep_idx}, "
                    f"frame={row['frame_index']}, expected={expected_index}, got={row['index']}"
                )


class TestVerifyDataset:
    """Tests for the verify_dataset function."""

    def test_verify_valid_dataset(self, tmp_path, lerobot_dataset_factory):
        """Verify that a valid dataset passes verification."""
        from lerobot.datasets.dataset_tools import verify_dataset

        # Create a properly structured dataset using the factory
        dataset = lerobot_dataset_factory(
            root=tmp_path / "test_verify",
            repo_id="test/verify_valid",
            total_episodes=5,
            total_frames=250,
        )

        result = verify_dataset(dataset.root, check_videos=False)

        assert result.is_valid, f"Expected valid dataset, got errors: {result.errors}"

    def test_verify_detects_missing_info_json(self, tmp_path, lerobot_dataset_factory):
        """Verify detection of missing info.json."""
        from lerobot.datasets.dataset_tools import verify_dataset

        dataset = lerobot_dataset_factory(
            root=tmp_path / "test_missing_info",
            repo_id="test/missing_info",
            total_episodes=3,
            total_frames=150,
        )

        # Remove info.json
        info_path = dataset.root / "meta" / "info.json"
        info_path.unlink()

        result = verify_dataset(dataset.root, check_videos=False)

        assert not result.is_valid
        assert any("info.json" in str(e) for e in result.errors)

    def test_verify_detects_episode_count_mismatch(self, tmp_path, lerobot_dataset_factory):
        """Verify detection of episode count mismatch between info.json and metadata."""
        import json

        from lerobot.datasets.dataset_tools import verify_dataset

        dataset = lerobot_dataset_factory(
            root=tmp_path / "test_ep_count",
            repo_id="test/ep_count",
            total_episodes=3,
            total_frames=150,
        )

        # Corrupt info.json to have wrong episode count
        info_path = dataset.root / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        info["total_episodes"] = 999
        info_path.write_text(json.dumps(info))

        result = verify_dataset(dataset.root, check_videos=False)

        assert not result.is_valid
        assert any("Episode count mismatch" in str(e) for e in result.errors)

    def test_verify_warns_on_all_zero_action_episode(self, tmp_path, lerobot_dataset_factory):
        """An episode whose action column is identically zero across every
        frame is almost always a recording-flow bug, not legit data. Verify
        that opening such a dataset produces a clear warning naming the
        affected episode indices."""

        from lerobot.datasets.dataset_tools import verify_dataset
        from lerobot.utils.constants import ACTION

        dataset = lerobot_dataset_factory(
            root=tmp_path / "test_zero_actions",
            repo_id="test/zero_actions",
            total_episodes=4,
            total_frames=200,
        )

        # Zero out actions for episodes 1 and 3 only (mimicking the gym-hil /
        # intervention-flag bug we just fixed: some episodes are clean, others
        # are silently no-op).
        for parquet_path in (dataset.root / "data").rglob("*.parquet"):
            df = pd.read_parquet(parquet_path)
            mask = df["episode_index"].isin([1, 3])
            if mask.any():
                action_dim = len(df.loc[mask, ACTION].iloc[0])
                df.loc[mask, ACTION] = pd.Series(
                    [np.zeros(action_dim, dtype=np.float32) for _ in range(mask.sum())],
                    index=df[mask].index,
                )
                df.to_parquet(parquet_path, index=False)

        result = verify_dataset(dataset.root, check_videos=False)

        # Structural integrity is fine; only a *warning* is raised.
        assert result.is_valid
        zero_warnings = [w for w in result.warnings if "all-zero actions" in w.message]
        assert len(zero_warnings) == 1, f"Expected one all-zero warning, got: {result.warnings}"
        assert zero_warnings[0].details["episode_indices"] == [1, 3]

    def test_verify_no_warning_when_actions_are_normal(self, tmp_path, lerobot_dataset_factory):
        """The factory's default datasets have random non-zero actions — make
        sure we don't fire false positives on healthy data."""
        from lerobot.datasets.dataset_tools import verify_dataset

        dataset = lerobot_dataset_factory(
            root=tmp_path / "test_normal_actions",
            repo_id="test/normal_actions",
            total_episodes=3,
            total_frames=150,
        )

        result = verify_dataset(dataset.root, check_videos=False)

        assert not any("all-zero actions" in w.message for w in result.warnings)

    def test_verify_detects_broken_indices(self, tmp_path, lerobot_dataset_factory):
        """Verify detection of non-continuous dataset_from_index values."""
        from lerobot.datasets.dataset_tools import verify_dataset

        dataset = lerobot_dataset_factory(
            root=tmp_path / "test_broken_idx",
            repo_id="test/broken_idx",
            total_episodes=5,
            total_frames=250,
        )

        # Corrupt episode metadata
        episodes_dir = dataset.root / "meta" / "episodes"
        for parquet_path in episodes_dir.rglob("*.parquet"):
            df = pd.read_parquet(parquet_path)
            df["dataset_from_index"] = 0  # All episodes start at 0 (wrong)
            df.to_parquet(parquet_path, index=False)

        result = verify_dataset(dataset.root, check_videos=False)

        assert not result.is_valid
        assert any("dataset_from_index" in str(e) for e in result.errors)

    def test_verify_detects_frame_count_mismatch(self, tmp_path, lerobot_dataset_factory):
        """Verify detection of frame count mismatch."""
        import json

        from lerobot.datasets.dataset_tools import verify_dataset

        dataset = lerobot_dataset_factory(
            root=tmp_path / "test_frame_count",
            repo_id="test/frame_count",
            total_episodes=3,
            total_frames=150,
        )

        # Corrupt info.json to have wrong frame count
        info_path = dataset.root / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        info["total_frames"] = 999999
        info_path.write_text(json.dumps(info))

        result = verify_dataset(dataset.root, check_videos=False)

        assert not result.is_valid
        assert any("frames" in str(e).lower() or "mismatch" in str(e).lower() for e in result.errors)

    def test_verify_quick_returns_bool(self, tmp_path, lerobot_dataset_factory):
        """Verify that verify_dataset_quick returns a boolean."""
        from lerobot.datasets.dataset_tools import verify_dataset_quick

        dataset = lerobot_dataset_factory(
            root=tmp_path / "test_quick",
            repo_id="test/quick",
            total_episodes=2,
            total_frames=100,
        )

        result = verify_dataset_quick(dataset.root)

        assert isinstance(result, bool)
        assert result is True

    @pytest.mark.parametrize("repo_id", ["lerobot/pusht"])
    def test_verify_real_dataset_from_hub(self, repo_id):
        """Verify that a real dataset from the Hub passes verification.

        This test downloads the actual dataset and verifies its integrity,
        ensuring the verification function works on real-world data.
        """
        from lerobot.datasets.dataset_tools import verify_dataset
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        # Load dataset from Hub (this will download if not cached)
        dataset = LeRobotDataset(repo_id)

        # Run verification (skip video checks for speed)
        result = verify_dataset(dataset.root, check_videos=False)

        assert result.is_valid, f"Expected {repo_id} to be valid, got errors: {result.errors}"
        assert result.stats["actual_episodes"] == result.stats["expected_episodes"]
        assert result.stats["actual_frames"] == result.stats["expected_frames"]
