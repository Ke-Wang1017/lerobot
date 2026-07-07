"""Safety regression tests for in-place dataset mutation primitives.

Covers ``trim_episode``, ``delete_episodes``, and ``LeRobotDataset.resume``
edge cases where a bad input could destroy more than intended.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tests.fixtures.dataset_snapshot import assert_no_data_loss, snapshot_tree


def test_trim_episode_invalid_range_does_not_mutate(tmp_path, lerobot_dataset_factory):
    """trim with start ≥ end (or both 0) must reject without touching disk."""
    from lerobot.datasets.dataset_tools import trim_episode

    dataset = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=3, total_frames=30)
    snapshot = snapshot_tree(dataset.root)

    # Start ≥ end is invalid input.
    with pytest.raises((ValueError, AssertionError, IndexError)):
        # trim_start_s = 999.0 is way longer than episode → start frame ≥ end frame
        trim_episode(dataset, episode_index=0, trim_start_s=999.0, trim_end_s=0.0)

    assert_no_data_loss(snapshot, snapshot_tree(dataset.root))


def test_delete_episodes_empty_list_does_not_destroy(tmp_path, lerobot_dataset_factory):
    """delete_episodes([]) must be either no-op or raise — never silently destroy.

    The current implementation raises ValueError; either contract is acceptable
    as long as no files are removed.
    """
    from lerobot.datasets.dataset_tools import delete_episodes

    dataset = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=3, total_frames=30)
    snapshot = snapshot_tree(dataset.root)

    try:
        delete_episodes(dataset, episode_indices=[])
    except (ValueError, IndexError):
        pass  # raising is fine; only silent destruction is the bug

    after = snapshot_tree(dataset.root)
    removed = set(snapshot) - set(after)
    assert not removed, f"Empty delete removed files: {sorted(removed)}"


def test_delete_episodes_invalid_index_does_not_mutate(tmp_path, lerobot_dataset_factory):
    """delete with out-of-range episode index must reject without mutation."""
    from lerobot.datasets.dataset_tools import delete_episodes

    dataset = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=3, total_frames=30)
    snapshot = snapshot_tree(dataset.root)

    with pytest.raises((ValueError, IndexError, KeyError)):
        delete_episodes(dataset, episode_indices=[999])

    assert_no_data_loss(snapshot, snapshot_tree(dataset.root))


def test_lerobot_dataset_resume_does_not_probe_hub(tmp_path, lerobot_dataset_factory):
    """LeRobotDataset.resume(root=...) must NOT make a Hub call for local-only datasets.

    If it did, a network 404 (transient or permanent) could trigger destructive
    fallback behavior in callers that catch broadly.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_root = tmp_path / "local_ds"
    lerobot_dataset_factory(root=dataset_root, total_episodes=2, total_frames=20)

    with (
        patch("lerobot.datasets.dataset_metadata.snapshot_download") as m_meta_download,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as m_facade_download,
    ):
        m_meta_download.side_effect = AssertionError("Hub probe leaked through to metadata")
        m_facade_download.side_effect = AssertionError("Hub probe leaked through to facade")

        ds = LeRobotDataset.resume("test/local_ds", root=dataset_root)
        assert ds.num_episodes == 2
