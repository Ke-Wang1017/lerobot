"""Safety regression tests for the GUI edits flow.

The GUI applies trim+delete edits to a user's dataset on disk. A bug in the
apply path (or downstream tools) could destroy the dataset. These tests
guard against partial failure leaving the user in a worse state than they
started.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from tests.fixtures.dataset_snapshot import assert_no_data_loss, snapshot_tree


@pytest.fixture
def gui_app_state():
    """Provide a fresh AppState for each test, isolated from globals."""
    from lerobot.gui.frame_cache import FrameCache
    from lerobot.gui.state import AppState

    return AppState(frame_cache=FrameCache())


def test_apply_edits_invalid_episode_does_not_destroy(
    tmp_path, lerobot_dataset_factory, gui_app_state, monkeypatch
):
    """An apply() call referencing a non-existent episode must not destroy the dataset."""
    import lerobot.gui.api.edits as edits_module
    from lerobot.gui.api.edits import _apply_edits_locked
    from lerobot.gui.state import PendingEdit

    dataset = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=3, total_frames=30)
    snapshot = snapshot_tree(dataset.root)

    dataset_id = "test_ds"
    gui_app_state.datasets[dataset_id] = dataset
    gui_app_state.pending_edits.append(
        PendingEdit(
            edit_type="delete",
            dataset_id=dataset_id,
            episode_index=999,  # out of range
        )
    )
    monkeypatch.setattr(edits_module, "_app_state", gui_app_state)

    try:
        result = asyncio.run(_apply_edits_locked(dataset_id))
        if isinstance(result, dict) and "errors" in result:
            assert result["errors"], "Expected errors but got none"
    except Exception:
        pass  # raising is acceptable — only silent destruction fails

    assert_no_data_loss(snapshot, snapshot_tree(dataset.root))


def test_apply_edits_partial_failure_does_not_corrupt(
    tmp_path, lerobot_dataset_factory, gui_app_state, monkeypatch
):
    """Invalid edits must not destroy files unrelated to their target episodes."""
    import lerobot.gui.api.edits as edits_module
    from lerobot.gui.api.edits import _apply_edits_locked
    from lerobot.gui.state import PendingEdit

    dataset = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=3, total_frames=30)
    snapshot = snapshot_tree(dataset.root)

    dataset_id = "test_ds"
    gui_app_state.datasets[dataset_id] = dataset
    gui_app_state.pending_edits.extend(
        [
            PendingEdit(edit_type="delete", dataset_id=dataset_id, episode_index=999),
            PendingEdit(
                edit_type="trim",
                dataset_id=dataset_id,
                episode_index=0,
                params={"start_frame": 9999, "end_frame": 99999},
            ),
        ]
    )
    monkeypatch.setattr(edits_module, "_app_state", gui_app_state)

    with contextlib.suppress(Exception):
        asyncio.run(_apply_edits_locked(dataset_id))

    after = snapshot_tree(dataset.root)
    removed = set(snapshot) - set(after)
    unrelated_removed = [f for f in removed if "episode_000000" not in f and "ep0" not in f]
    assert not unrelated_removed, (
        f"Files unrelated to the failed edits were removed: {sorted(unrelated_removed)[:10]}"
    )
