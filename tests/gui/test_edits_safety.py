"""Safety regression tests for the GUI edits flow.

The GUI applies trim+delete edits to a user's dataset on disk. A bug in the
apply path (or downstream tools) could destroy the dataset. These tests
guard against partial failure leaving the user in a worse state than they
started.
"""

from __future__ import annotations

import asyncio

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
    gui_app_state.pending_edits.extend([
        PendingEdit(edit_type="delete", dataset_id=dataset_id, episode_index=999),
        PendingEdit(
            edit_type="trim",
            dataset_id=dataset_id,
            episode_index=0,
            params={"start_frame": 9999, "end_frame": 99999},
        ),
    ])
    monkeypatch.setattr(edits_module, "_app_state", gui_app_state)

    try:
        asyncio.run(_apply_edits_locked(dataset_id))
    except Exception:
        pass

    after = snapshot_tree(dataset.root)
    removed = set(snapshot) - set(after)
    unrelated_removed = [
        f for f in removed if "episode_000000" not in f and "ep0" not in f
    ]
    assert not unrelated_removed, (
        f"Files unrelated to the failed edits were removed: {sorted(unrelated_removed)[:10]}"
    )


def test_apply_edits_delete_all_keeps_pending(
    tmp_path, lerobot_dataset_factory, gui_app_state, monkeypatch
):
    """Marking every episode for deletion must not silently clear the pending edits.

    Regression: previously, hitting "Cannot delete all episodes" caught the error
    but still wiped the pending-edits list, leaving the user with a confusing
    "save changes did nothing" reset. The user should keep their edits so they
    can unmark one and retry.
    """
    import lerobot.gui.api.edits as edits_module
    from lerobot.gui.api.edits import _apply_edits_locked
    from lerobot.gui.state import PendingEdit

    dataset = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=3, total_frames=30)
    snapshot = snapshot_tree(dataset.root)

    dataset_id = "test_ds"
    gui_app_state.datasets[dataset_id] = dataset
    gui_app_state.pending_edits.extend([
        PendingEdit(edit_type="delete", dataset_id=dataset_id, episode_index=0),
        PendingEdit(edit_type="delete", dataset_id=dataset_id, episode_index=1),
        PendingEdit(edit_type="delete", dataset_id=dataset_id, episode_index=2),
    ])
    monkeypatch.setattr(edits_module, "_app_state", gui_app_state)

    result = asyncio.run(_apply_edits_locked(dataset_id))

    assert result["status"] == "partial"
    assert result["applied"] == 0
    assert result["errors"], "Expected an error reported back to the caller"
    assert "all" in result["errors"][0].lower()

    # Pending edits must survive the failed apply so the user can fix and retry.
    assert len(gui_app_state.pending_edits) == 3, (
        "Pending edits were silently cleared after a failed apply"
    )

    # Dataset on disk must be untouched.
    assert_no_data_loss(snapshot, snapshot_tree(dataset.root))


def test_apply_edits_partial_success_keeps_failed_pending(
    tmp_path, lerobot_dataset_factory, gui_app_state, monkeypatch
):
    """When some edits succeed and others fail, only the successful ones get cleared."""
    import lerobot.gui.api.edits as edits_module
    from lerobot.gui.api.edits import _apply_edits_locked
    from lerobot.gui.state import PendingEdit

    dataset = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=4, total_frames=40)

    dataset_id = "test_ds"
    gui_app_state.datasets[dataset_id] = dataset
    # One deletable edit + one trim with bogus range that will fail.
    delete_edit = PendingEdit(edit_type="delete", dataset_id=dataset_id, episode_index=0)
    bad_trim = PendingEdit(
        edit_type="trim",
        dataset_id=dataset_id,
        episode_index=1,
        params={"start_frame": 9999, "end_frame": 99999},
    )
    gui_app_state.pending_edits.extend([delete_edit, bad_trim])
    monkeypatch.setattr(edits_module, "_app_state", gui_app_state)

    asyncio.run(_apply_edits_locked(dataset_id))

    # Successful delete should be gone, failed trim should remain.
    remaining_types = [e.edit_type for e in gui_app_state.pending_edits]
    assert remaining_types == ["trim"], (
        f"Expected only the failed trim to remain, got {remaining_types}"
    )
