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
"""Tests for stats recomputation after virtual trim/delete operations.

These tests verify that dataset statistics (meta/stats.json and per-episode stats
in meta/episodes/*.parquet) are correctly recomputed when episodes are trimmed
or deleted using virtual (non-destructive) operations.
"""

import json

import numpy as np
import pytest

from lerobot.datasets.dataset_tools import (
    delete_episodes_virtual,
    reaggregate_dataset_stats,
    trim_episode_virtual,
    verify_dataset,
)
from lerobot.datasets.io_utils import load_stats


@pytest.fixture
def stats_dataset(tmp_path, empty_lerobot_dataset_factory):
    """Create a dataset with 3 episodes having distinct, controlled values.

    Episode 0: action values all 0.1, state values all 0.1 (10 frames)
    Episode 1: action values all 0.5, state values all 0.5 (10 frames)
    Episode 2: action values all 0.9, state values all 0.9 (10 frames)

    This makes stats predictable and allows us to verify recomputation.
    """
    features = {
        "action": {"dtype": "float32", "shape": (2,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
    }

    dataset = empty_lerobot_dataset_factory(
        root=tmp_path / "stats_test_dataset",
        features=features,
    )

    values_per_episode = [0.1, 0.5, 0.9]

    for ep_idx in range(3):
        val = values_per_episode[ep_idx]
        for _ in range(10):
            frame = {
                "action": np.full(2, val, dtype=np.float32),
                "observation.state": np.full(2, val, dtype=np.float32),
                "task": "test_task",
            }
            dataset.add_frame(frame)
        dataset.save_episode()

    dataset.finalize()
    return dataset


@pytest.fixture
def trim_dataset(tmp_path, empty_lerobot_dataset_factory):
    """Create a dataset with 2 episodes having distinct frame values.

    Episode 0: 20 frames. First 10 frames have value 0.0, last 10 have value 1.0
    Episode 1: 10 frames. All frames have value 0.5

    This allows testing that trimming episode 0 to keep only the last 10 frames
    (value 1.0) causes stats to change.
    """
    features = {
        "action": {"dtype": "float32", "shape": (2,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
    }

    dataset = empty_lerobot_dataset_factory(
        root=tmp_path / "trim_test_dataset",
        features=features,
    )

    # Episode 0: first 10 frames = 0.0, last 10 frames = 1.0
    for i in range(20):
        val = 0.0 if i < 10 else 1.0
        frame = {
            "action": np.full(2, val, dtype=np.float32),
            "observation.state": np.full(2, val, dtype=np.float32),
            "task": "test_task",
        }
        dataset.add_frame(frame)
    dataset.save_episode()

    # Episode 1: all frames = 0.5
    for _ in range(10):
        frame = {
            "action": np.full(2, 0.5, dtype=np.float32),
            "observation.state": np.full(2, 0.5, dtype=np.float32),
            "task": "test_task",
        }
        dataset.add_frame(frame)
    dataset.save_episode()

    dataset.finalize()
    return dataset


class TestDeleteEpisodesVirtualStats:
    """Tests for stats recomputation after virtual episode deletion."""

    def test_delete_reaggregates_stats_json(self, stats_dataset):
        """After deleting the episode with extreme values, stats.json must change.

        We delete episode 2 (values=0.9). The remaining episodes have values 0.1 and 0.5,
        so the aggregated max should decrease from 0.9 to 0.5.
        """
        stats_before = load_stats(stats_dataset.root)
        assert stats_before is not None

        # Before: max should be ~0.9 (from episode 2)
        action_max_before = stats_before["action"]["max"]
        assert np.allclose(action_max_before, 0.9, atol=0.01)

        delete_episodes_virtual(stats_dataset, episode_indices=[2])

        stats_after = load_stats(stats_dataset.root)
        assert stats_after is not None

        # After: max should be ~0.5 (episode 2 with 0.9 values was deleted)
        action_max_after = stats_after["action"]["max"]
        assert np.allclose(action_max_after, 0.5, atol=0.01), (
            f"After deleting episode with max values, stats.json max should have decreased "
            f"from ~0.9 to ~0.5, but got {action_max_after}"
        )

        # min should still be ~0.1 (episode 0 still exists)
        action_min_after = stats_after["action"]["min"]
        assert np.allclose(action_min_after, 0.1, atol=0.01)

    def test_delete_updates_mean(self, stats_dataset):
        """After deleting an episode, the aggregated mean must be recomputed.

        Original mean of [0.1, 0.5, 0.9] with equal counts = 0.5
        After deleting episode 0 (value=0.1), mean of [0.5, 0.9] = 0.7
        """
        delete_episodes_virtual(stats_dataset, episode_indices=[0])

        stats_after = load_stats(stats_dataset.root)
        assert stats_after is not None

        expected_mean = 0.7  # (0.5 + 0.9) / 2
        action_mean_after = stats_after["action"]["mean"]
        assert np.allclose(action_mean_after, expected_mean, atol=0.01), (
            f"After deleting episode 0, mean should be ~{expected_mean}, got {action_mean_after}"
        )


class TestTrimEpisodeVirtualStats:
    """Tests for stats recomputation after virtual episode trimming."""

    def test_trim_recomputes_per_episode_stats(self, trim_dataset):
        """After trimming, per-episode stats in parquet must reflect the trimmed data.

        Episode 0 originally has values [0.0]*10 + [1.0]*10, mean=0.5
        After trimming to keep only last 10 frames (start_frame=10, end_frame=20),
        episode 0 has only values [1.0]*10, mean=1.0
        """
        import pandas as pd

        # Read per-episode stats before trim
        episodes_dir = trim_dataset.root / "meta" / "episodes"
        parquet_files = sorted(episodes_dir.rglob("*.parquet"))
        df_before = pd.concat([pd.read_parquet(p) for p in parquet_files])
        ep0_before = df_before[df_before["episode_index"] == 0].iloc[0]

        # Check that stats/action/mean exists and is ~0.5 before trim
        if "stats/action/mean" in ep0_before.index:
            mean_before = ep0_before["stats/action/mean"]
            if isinstance(mean_before, np.ndarray):
                assert np.allclose(mean_before, 0.5, atol=0.01)

        # Trim episode 0: keep only last 10 frames (all value 1.0)
        trim_episode_virtual(trim_dataset, episode_index=0, start_frame=10, end_frame=20)

        # Read per-episode stats after trim
        df_after = pd.concat([pd.read_parquet(p) for p in parquet_files])
        ep0_after = df_after[df_after["episode_index"] == 0].iloc[0]

        if "stats/action/mean" in ep0_after.index:
            mean_after = ep0_after["stats/action/mean"]
            if isinstance(mean_after, np.ndarray):
                assert np.allclose(mean_after, 1.0, atol=0.01), (
                    f"After trimming episode 0 to only value-1.0 frames, "
                    f"per-episode mean should be ~1.0, got {mean_after}"
                )

    def test_trim_reaggregates_stats_json(self, trim_dataset):
        """After trimming, stats.json must be recomputed from updated per-episode stats.

        Before: episode 0 has mean=0.5 (10 frames of 0.0 + 10 of 1.0), episode 1 has mean=0.5
        Overall mean = 0.5

        After trimming episode 0 to [1.0]*10: episode 0 mean=1.0, episode 1 mean=0.5
        Overall mean = (1.0*10 + 0.5*10) / 20 = 0.75
        """
        stats_before = load_stats(trim_dataset.root)
        assert stats_before is not None
        # Overall mean should be about (0.0*10 + 1.0*10 + 0.5*10) / 30 = 0.5
        stats_before["action"]["mean"]

        # Trim episode 0: keep only last 10 frames (all value 1.0)
        trim_episode_virtual(trim_dataset, episode_index=0, start_frame=10, end_frame=20)

        stats_after = load_stats(trim_dataset.root)
        assert stats_after is not None

        # After: mean should be (1.0*10 + 0.5*10) / 20 = 0.75
        action_mean_after = stats_after["action"]["mean"]
        assert np.allclose(action_mean_after, 0.75, atol=0.01), (
            f"After trimming episode 0 to keep only value-1.0 frames, "
            f"overall mean should be ~0.75, got {action_mean_after}"
        )

    def test_trim_updates_min_max(self, trim_dataset):
        """After trimming away frames with min values, the overall min must change.

        Episode 0 has min=0.0 from first 10 frames. After trimming those away,
        the dataset min should increase.
        """
        stats_before = load_stats(trim_dataset.root)
        assert stats_before is not None
        action_min_before = stats_before["action"]["min"]
        assert np.allclose(action_min_before, 0.0, atol=0.01)

        # Trim episode 0: remove first 10 frames (value 0.0), keep last 10 (value 1.0)
        trim_episode_virtual(trim_dataset, episode_index=0, start_frame=10, end_frame=20)

        stats_after = load_stats(trim_dataset.root)
        assert stats_after is not None

        # After: min should be ~0.5 (from episode 1), not 0.0 anymore
        action_min_after = stats_after["action"]["min"]
        assert np.allclose(action_min_after, 0.5, atol=0.01), (
            f"After trimming away value-0.0 frames, min should be ~0.5, got {action_min_after}"
        )


class TestVerifyDatasetStats:
    """Tests for stats verification in verify_dataset()."""

    def test_verify_catches_stale_stats(self, stats_dataset):
        """verify_dataset should warn when stats.json doesn't match per-episode stats."""
        # Corrupt stats.json by zeroing out all values
        stats_path = stats_dataset.root / "meta" / "stats.json"
        with open(stats_path) as f:
            stats = json.load(f)

        # Zero out the action mean
        if "action" in stats and "mean" in stats["action"]:
            stats["action"]["mean"] = [0.0, 0.0]

        with open(stats_path, "w") as f:
            json.dump(stats, f)

        result = verify_dataset(stats_dataset.root, check_videos=False)

        # Should have a warning about stats inconsistency
        stats_warnings = [w for w in result.warnings if "stats" in w.category.lower()]
        assert len(stats_warnings) > 0, (
            f"verify_dataset should warn about corrupted stats.json, but got warnings: {result.warnings}"
        )

    def test_verify_passes_with_correct_stats(self, stats_dataset):
        """verify_dataset should not warn when stats.json matches per-episode stats."""
        result = verify_dataset(stats_dataset.root, check_videos=False)

        stats_warnings = [w for w in result.warnings if "stats" in w.category.lower()]
        assert len(stats_warnings) == 0, (
            f"verify_dataset should not warn about correct stats, but got warnings: {stats_warnings}"
        )


class TestBatchedStatsRecomputation:
    """Tests for deferred stats recomputation (recompute_stats=False)."""

    @pytest.fixture
    def batch_dataset(self, tmp_path, empty_lerobot_dataset_factory):
        """4 episodes, each 20 frames, with distinct first/second halves.

        Ep 0: first 10 frames = 0.0, last 10 = 0.2
        Ep 1: first 10 frames = 0.3, last 10 = 0.5
        Ep 2: first 10 frames = 0.6, last 10 = 0.8
        Ep 3: all 20 frames = 1.0
        """
        features = {
            "action": {"dtype": "float32", "shape": (2,), "names": None},
        }
        dataset = empty_lerobot_dataset_factory(
            root=tmp_path / "batch_dataset",
            features=features,
        )
        specs = [(0.0, 0.2), (0.3, 0.5), (0.6, 0.8), (1.0, 1.0)]
        for lo, hi in specs:
            for i in range(20):
                val = lo if i < 10 else hi
                dataset.add_frame(
                    {
                        "action": np.full(2, val, dtype=np.float32),
                        "task": "test",
                    }
                )
            dataset.save_episode()
        dataset.finalize()
        return dataset

    def test_batch_trim_deferred_reaggregate(self, batch_dataset):
        """Trim 3 episodes with recompute_stats=False, then reaggregate once.

        Trim eps 0-2 to keep only the second half (high values).
        Before trim: overall mean ≈ (0+.2+.3+.5+.6+.8+1+1)/8 * (10/20 each) = 0.55
        After trim:  ep0=0.2, ep1=0.5, ep2=0.8, ep3=1.0 (all 10 frames each)
                     mean = (0.2+0.5+0.8+1.0)/4 = 0.625
        """
        from lerobot.datasets.io_utils import load_episodes

        stats_before = load_stats(batch_dataset.root)
        assert stats_before is not None

        # Trim 3 episodes without re-aggregating
        for ep_idx in range(3):
            trim_episode_virtual(
                batch_dataset,
                episode_index=ep_idx,
                start_frame=10,
                end_frame=20,
                recompute_stats=False,
            )
            batch_dataset.meta.episodes = load_episodes(batch_dataset.root)

        # stats.json should still be STALE (not updated yet)
        stats_mid = load_stats(batch_dataset.root)
        assert np.allclose(stats_mid["action"]["mean"], stats_before["action"]["mean"]), (
            "stats.json should not have changed yet (recompute_stats=False)"
        )

        # Now re-aggregate once
        reaggregate_dataset_stats(batch_dataset)

        stats_after = load_stats(batch_dataset.root)
        assert stats_after is not None

        # After: mean = (0.2*10 + 0.5*10 + 0.8*10 + 1.0*10) / 40 = 0.625
        # But ep3 has 20 frames (untrimmed), so:
        # mean = (0.2*10 + 0.5*10 + 0.8*10 + 1.0*20) / 50 = (2+5+8+20)/50 = 0.7
        expected_mean = (0.2 * 10 + 0.5 * 10 + 0.8 * 10 + 1.0 * 20) / 50
        assert np.allclose(stats_after["action"]["mean"], expected_mean, atol=0.01), (
            f"After batch trim + single reaggregate, mean should be ~{expected_mean}, "
            f"got {stats_after['action']['mean']}"
        )

        # min should now be 0.2 (ep 0 lost its 0.0 frames)
        assert np.allclose(stats_after["action"]["min"], 0.2, atol=0.01), (
            f"min should be ~0.2, got {stats_after['action']['min']}"
        )

    def test_batch_trim_then_delete_deferred(self, batch_dataset):
        """Trim + delete with recompute_stats=False, single reaggregate at end."""
        from lerobot.datasets.io_utils import load_episodes

        # Trim ep 0 to keep only high half (0.2), skip reaggregate
        trim_episode_virtual(
            batch_dataset,
            episode_index=0,
            start_frame=10,
            end_frame=20,
            recompute_stats=False,
        )
        batch_dataset.meta.episodes = load_episodes(batch_dataset.root)

        # Delete ep 3 (all 1.0 values), skip reaggregate
        delete_episodes_virtual(
            batch_dataset,
            episode_indices=[3],
            recompute_stats=False,
        )
        batch_dataset.meta.episodes = load_episodes(batch_dataset.root)

        # Reaggregate once
        reaggregate_dataset_stats(batch_dataset)

        stats = load_stats(batch_dataset.root)
        # Remaining: ep0=0.2 (10fr), ep1=[0.3,0.5] (20fr), ep2=[0.6,0.8] (20fr)
        # mean = (0.2*10 + 0.3*10 + 0.5*10 + 0.6*10 + 0.8*10) / 50 = 24/50 = 0.48
        expected_mean = (0.2 * 10 + 0.3 * 10 + 0.5 * 10 + 0.6 * 10 + 0.8 * 10) / 50
        assert np.allclose(stats["action"]["mean"], expected_mean, atol=0.01), (
            f"mean should be ~{expected_mean}, got {stats['action']['mean']}"
        )

        # max should be 0.8 (ep 3 with 1.0 was deleted)
        assert np.allclose(stats["action"]["max"], 0.8, atol=0.01), (
            f"max should be ~0.8, got {stats['action']['max']}"
        )
