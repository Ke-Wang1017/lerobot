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
"""Tests for episode start index computation in the GUI API."""

from unittest.mock import MagicMock

import pytest

from lerobot.gui.api import datasets as datasets_module


@pytest.fixture
def mock_app_state():
    """Create a mock app state with fake datasets."""
    mock_state = MagicMock()
    mock_state.datasets = {}
    return mock_state


@pytest.fixture
def setup_datasets_module(mock_app_state):
    """Set up the datasets module with mock state and clean cache."""
    # Save original state
    original_app_state = datasets_module._app_state
    original_cache = datasets_module._episode_start_indices.copy()

    # Set mock state
    datasets_module._app_state = mock_app_state
    datasets_module._episode_start_indices.clear()

    yield mock_app_state

    # Restore original state
    datasets_module._app_state = original_app_state
    datasets_module._episode_start_indices.clear()
    datasets_module._episode_start_indices.update(original_cache)


def create_mock_dataset(episode_lengths: list[int]):
    """Create a mock dataset with given episode lengths."""
    mock_dataset = MagicMock()
    mock_dataset.meta.episodes = [{"length": length} for length in episode_lengths]
    return mock_dataset


class TestGetEpisodeStartIndex:
    """Tests for _get_episode_start_index function."""

    def test_single_episode(self, setup_datasets_module):
        """Episode 0 should always start at index 0."""
        mock_state = setup_datasets_module
        mock_state.datasets["test_ds"] = create_mock_dataset([100])

        result = datasets_module._get_episode_start_index("test_ds", 0)

        assert result == 0

    def test_two_episodes(self, setup_datasets_module):
        """Episode 1 should start after episode 0's length."""
        mock_state = setup_datasets_module
        mock_state.datasets["test_ds"] = create_mock_dataset([100, 50])

        assert datasets_module._get_episode_start_index("test_ds", 0) == 0
        assert datasets_module._get_episode_start_index("test_ds", 1) == 100

    def test_multiple_episodes(self, setup_datasets_module):
        """Test cumulative sum for multiple episodes."""
        mock_state = setup_datasets_module
        # Episodes with lengths: 100, 50, 75, 25
        mock_state.datasets["test_ds"] = create_mock_dataset([100, 50, 75, 25])

        assert datasets_module._get_episode_start_index("test_ds", 0) == 0
        assert datasets_module._get_episode_start_index("test_ds", 1) == 100
        assert datasets_module._get_episode_start_index("test_ds", 2) == 150  # 100 + 50
        assert datasets_module._get_episode_start_index("test_ds", 3) == 225  # 100 + 50 + 75

    def test_cumulative_sum_many_episodes(self, setup_datasets_module):
        """Test cumulative sum computation for many episodes.

        In multi-file datasets, the metadata's dataset_from_index resets to 0
        at each file boundary. Our fix ignores dataset_from_index entirely
        and computes the correct global index by summing episode lengths.

        Example from real bug: episode 59 had dataset_from_index=0 (per-file),
        but correct global index was 7389 (sum of episodes 0-58 lengths).
        """
        mock_state = setup_datasets_module
        mock_state.datasets["test_ds"] = create_mock_dataset([100, 150, 120, 80, 90, 110])

        # Verify cumulative sums are computed correctly
        assert datasets_module._get_episode_start_index("test_ds", 0) == 0
        assert datasets_module._get_episode_start_index("test_ds", 1) == 100
        assert datasets_module._get_episode_start_index("test_ds", 2) == 250  # 100 + 150
        assert datasets_module._get_episode_start_index("test_ds", 3) == 370  # 100 + 150 + 120
        assert datasets_module._get_episode_start_index("test_ds", 4) == 450  # 370 + 80
        assert datasets_module._get_episode_start_index("test_ds", 5) == 540  # 370 + 80 + 90

    def test_caches_results(self, setup_datasets_module):
        """Verify that results are cached for performance."""
        mock_state = setup_datasets_module
        mock_state.datasets["test_ds"] = create_mock_dataset([100, 50])

        # First call computes
        datasets_module._get_episode_start_index("test_ds", 0)

        # Should be cached
        assert "test_ds" in datasets_module._episode_start_indices

        # Second call should use cache (dataset.meta.episodes not accessed again)
        mock_state.datasets["test_ds"].meta.episodes = None  # Would fail if accessed
        result = datasets_module._get_episode_start_index("test_ds", 1)
        assert result == 100

    def test_unknown_dataset_returns_zero(self, setup_datasets_module):
        """Unknown dataset should return 0."""
        result = datasets_module._get_episode_start_index("nonexistent", 5)

        assert result == 0

    def test_out_of_range_episode_returns_zero(self, setup_datasets_module):
        """Out of range episode index should return 0."""
        mock_state = setup_datasets_module
        mock_state.datasets["test_ds"] = create_mock_dataset([100, 50])

        result = datasets_module._get_episode_start_index("test_ds", 10)

        assert result == 0


class TestInvalidateEpisodeStartIndices:
    """Tests for _invalidate_episode_start_indices function."""

    def test_invalidates_cache(self, setup_datasets_module):
        """Invalidation should clear the cache for a dataset."""
        mock_state = setup_datasets_module
        mock_state.datasets["test_ds"] = create_mock_dataset([100, 50])

        # Populate cache
        datasets_module._get_episode_start_index("test_ds", 0)
        assert "test_ds" in datasets_module._episode_start_indices

        # Invalidate
        datasets_module._invalidate_episode_start_indices("test_ds")

        assert "test_ds" not in datasets_module._episode_start_indices

    def test_invalidate_nonexistent_no_error(self, setup_datasets_module):
        """Invalidating a non-cached dataset should not raise."""
        # Should not raise
        datasets_module._invalidate_episode_start_indices("nonexistent")

    def test_invalidate_only_affects_specified_dataset(self, setup_datasets_module):
        """Invalidation should only affect the specified dataset."""
        mock_state = setup_datasets_module
        mock_state.datasets["ds_a"] = create_mock_dataset([100])
        mock_state.datasets["ds_b"] = create_mock_dataset([200])

        # Populate cache for both
        datasets_module._get_episode_start_index("ds_a", 0)
        datasets_module._get_episode_start_index("ds_b", 0)

        # Invalidate only ds_a
        datasets_module._invalidate_episode_start_indices("ds_a")

        assert "ds_a" not in datasets_module._episode_start_indices
        assert "ds_b" in datasets_module._episode_start_indices
