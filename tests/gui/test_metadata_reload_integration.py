"""Integration regression: GUI must successfully reload hf_dataset after recording.

The May 2026 incident: ``_check_and_reload_metadata`` (and ``reload_hf_dataset``)
tried to do ``dataset.hf_dataset = ...``, which raises ``AttributeError`` post-
refactor (hf_dataset became a read-only property on the facade). The exception
was caught and logged, so:

  1. Recording adds an episode (info.json mtime updates)
  2. GUI auto-detection runs ``_check_and_reload_metadata``
  3. Metadata reloads OK, but hf_dataset reload silently fails
  4. GUI shows N+1 episodes in the episode list (metadata says so)
  5. User clicks the new episode → backend tries ``dataset[2117]``
  6. hf_dataset still has 2023 rows → ``IndexError: out of bounds for size 2023``

Refresh of the GUI re-instantiated the dataset, masking the bug.

These tests guard against that shape regressing.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def real_dataset(tmp_path, lerobot_dataset_factory):
    """A real LeRobotDataset on disk with 3 episodes (not a mock)."""
    return lerobot_dataset_factory(
        root=tmp_path / "ds",
        total_episodes=3,
        total_frames=30,
    )


def test_reload_keeps_metadata_and_data_consistent(tmp_path, lerobot_dataset_factory):
    """Public-surface contract: after ``reload_dataset_from_disk``,
    ``dataset.num_frames == len(dataset.hf_dataset)`` must hold.

    The May 2026 incident: GUI auto-reload bumped ``num_frames`` from 2023 to
    2219 (info.json reload OK) but ``len(hf_dataset)`` stayed at 2023 (data
    reload silently failed via swallowed AttributeError). User clicks new
    episode → backend reads frame 2117 → ``IndexError: out of bounds for size 2023``.

    Test strategy: open a dataset, then re-create it on disk with the same root
    but more frames. The on-disk state now has both updated info.json and
    updated parquet — but the in-memory handle is stale. Reload should resync
    both halves of the dataset. The assertion is on the public surface
    (``num_frames`` vs ``len(hf_dataset)``) so it stays meaningful even if the
    internal layout changes again.
    """
    from lerobot.gui.dataset_reload import reload_dataset_from_disk

    ds_root = tmp_path / "ds"
    initial = lerobot_dataset_factory(root=ds_root, total_episodes=2, total_frames=20)
    assert initial.num_frames == len(initial.hf_dataset)  # baseline invariant

    # Re-create the dataset on disk at the same root with more episodes.
    # The factory writes fresh metadata + parquet, simulating what a recording
    # session would do externally.
    lerobot_dataset_factory(root=ds_root, total_episodes=4, total_frames=40)

    # GUI's auto-detect-and-reload helper.
    reload_dataset_from_disk(initial)

    # The fundamental invariant: the indexable data range and the metadata
    # must agree. If they diverge, GUI playback hits IndexError.
    assert initial.num_frames == len(initial.hf_dataset), (
        f"BUG: metadata says {initial.num_frames} frames but indexable data has "
        f"{len(initial.hf_dataset)} — GUI playback would IndexError on frames "
        f"in the gap. This is the May 2026 incident shape."
    )
