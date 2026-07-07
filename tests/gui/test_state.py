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
"""Tests for the AppState class and persistence functions."""

import asyncio
import json
from datetime import datetime
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import FastAPI

from lerobot.gui.api import edits as edits_module
from lerobot.gui.frame_cache import FrameCache
from lerobot.gui.state import (
    EDITS_FILENAME,
    AppState,
    PendingEdit,
    clear_edits_file,
    load_edits_from_file,
    save_edits_to_file,
)


@pytest.fixture
def app_state():
    """Create an AppState instance for testing."""
    return AppState(frame_cache=FrameCache(max_bytes=1_000_000))


class TestEditFiltering:
    """Tests for filtering edits by dataset."""

    def test_get_edits_for_dataset_filters_correctly(self, app_state):
        """Verify that get_edits_for_dataset returns only matching edits."""
        app_state.add_edit(PendingEdit("trim", "ds_a", 0, {"start_frame": 5, "end_frame": 10}))
        app_state.add_edit(PendingEdit("delete", "ds_b", 1))
        app_state.add_edit(PendingEdit("trim", "ds_a", 2, {"start_frame": 0, "end_frame": 5}))
        app_state.add_edit(PendingEdit("delete", "ds_a", 3))

        edits_a = app_state.get_edits_for_dataset("ds_a")

        assert len(edits_a) == 3
        assert all(e.dataset_id == "ds_a" for e in edits_a)

    def test_get_edits_for_dataset_empty_result(self, app_state):
        """Verify that filtering returns empty list for non-existent dataset."""
        app_state.add_edit(PendingEdit("trim", "ds_a", 0, {}))

        edits = app_state.get_edits_for_dataset("ds_nonexistent")

        assert edits == []

    def test_get_edits_for_dataset_preserves_order(self, app_state):
        """Verify that edits are returned in the order they were added."""
        app_state.add_edit(PendingEdit("trim", "ds_a", 0, {}))
        app_state.add_edit(PendingEdit("trim", "ds_a", 1, {}))
        app_state.add_edit(PendingEdit("trim", "ds_a", 2, {}))

        edits = app_state.get_edits_for_dataset("ds_a")

        assert [e.episode_index for e in edits] == [0, 1, 2]


class TestTrimReplacement:
    """Tests for trim replacement behavior."""

    def test_trim_replacement_logic(self, app_state):
        """Verify that the trim replacement logic works correctly.

        This tests the pattern used in the API where existing trims
        are removed before adding a new one for the same episode.
        """
        # Add initial trim
        app_state.add_edit(PendingEdit("trim", "ds_a", 0, {"start_frame": 0, "end_frame": 50}))
        app_state.add_edit(PendingEdit("delete", "ds_a", 1))  # Different edit type
        app_state.add_edit(PendingEdit("trim", "ds_a", 2, {"start_frame": 0, "end_frame": 30}))

        # Simulate replacement: remove existing trim for episode 0
        app_state.pending_edits = [
            e
            for e in app_state.pending_edits
            if not (e.dataset_id == "ds_a" and e.episode_index == 0 and e.edit_type == "trim")
        ]
        # Add new trim for episode 0
        app_state.add_edit(PendingEdit("trim", "ds_a", 0, {"start_frame": 10, "end_frame": 40}))

        # Verify only one trim for episode 0
        trims_ep0 = [
            e
            for e in app_state.get_edits_for_dataset("ds_a")
            if e.edit_type == "trim" and e.episode_index == 0
        ]
        assert len(trims_ep0) == 1
        assert trims_ep0[0].params["start_frame"] == 10
        assert trims_ep0[0].params["end_frame"] == 40

        # Verify other edits are untouched
        assert len(app_state.pending_edits) == 3  # delete + trim ep2 + new trim ep0

    def test_trim_replacement_different_episode_untouched(self, app_state):
        """Verify that trim replacement only affects the specific episode."""
        app_state.add_edit(PendingEdit("trim", "ds_a", 0, {"start_frame": 0, "end_frame": 50}))
        app_state.add_edit(PendingEdit("trim", "ds_a", 1, {"start_frame": 5, "end_frame": 45}))

        # Replace trim for episode 0 only
        app_state.pending_edits = [
            e
            for e in app_state.pending_edits
            if not (e.dataset_id == "ds_a" and e.episode_index == 0 and e.edit_type == "trim")
        ]
        app_state.add_edit(PendingEdit("trim", "ds_a", 0, {"start_frame": 10, "end_frame": 40}))

        # Episode 1's trim should be unchanged
        trims_ep1 = [e for e in app_state.pending_edits if e.episode_index == 1 and e.edit_type == "trim"]
        assert len(trims_ep1) == 1
        assert trims_ep1[0].params["start_frame"] == 5


class TestBasicEditOperations:
    """Tests for basic edit add/remove/clear operations."""

    def test_add_edit(self, app_state):
        """Verify that add_edit appends to the list."""
        edit = PendingEdit("delete", "ds_a", 0)
        app_state.add_edit(edit)

        assert len(app_state.pending_edits) == 1
        assert app_state.pending_edits[0] == edit

    def test_remove_edit_valid_index(self, app_state):
        """Verify that remove_edit removes the correct entry."""
        app_state.add_edit(PendingEdit("delete", "ds_a", 0))
        app_state.add_edit(PendingEdit("delete", "ds_a", 1))
        app_state.add_edit(PendingEdit("delete", "ds_a", 2))

        app_state.remove_edit(1)

        assert len(app_state.pending_edits) == 2
        assert app_state.pending_edits[0].episode_index == 0
        assert app_state.pending_edits[1].episode_index == 2

    def test_remove_edit_invalid_index(self, app_state):
        """Verify that invalid index is a no-op."""
        app_state.add_edit(PendingEdit("delete", "ds_a", 0))

        app_state.remove_edit(5)  # Invalid index
        app_state.remove_edit(-1)  # Negative index

        assert len(app_state.pending_edits) == 1

    def test_clear_edits_all(self, app_state):
        """Verify that clear_edits without dataset clears all."""
        app_state.add_edit(PendingEdit("delete", "ds_a", 0))
        app_state.add_edit(PendingEdit("delete", "ds_b", 1))

        app_state.clear_edits()

        assert len(app_state.pending_edits) == 0

    def test_clear_edits_by_dataset(self, app_state):
        """Verify that clear_edits with dataset clears only that dataset."""
        app_state.add_edit(PendingEdit("delete", "ds_a", 0))
        app_state.add_edit(PendingEdit("delete", "ds_b", 1))
        app_state.add_edit(PendingEdit("trim", "ds_a", 2, {}))

        app_state.clear_edits("ds_a")

        assert len(app_state.pending_edits) == 1
        assert app_state.pending_edits[0].dataset_id == "ds_b"


class TestEpisodeStatusChecks:
    """Tests for is_episode_deleted and get_episode_trim."""

    def test_is_episode_deleted_true(self, app_state):
        """Verify is_episode_deleted returns True for deleted episodes."""
        app_state.add_edit(PendingEdit("delete", "ds_a", 5))

        assert app_state.is_episode_deleted("ds_a", 5) is True

    def test_is_episode_deleted_false(self, app_state):
        """Verify is_episode_deleted returns False for non-deleted episodes."""
        app_state.add_edit(PendingEdit("delete", "ds_a", 5))

        assert app_state.is_episode_deleted("ds_a", 0) is False
        assert app_state.is_episode_deleted("ds_b", 5) is False

    def test_is_episode_deleted_trim_not_counted(self, app_state):
        """Verify that trim edits don't count as deletions."""
        app_state.add_edit(PendingEdit("trim", "ds_a", 5, {}))

        assert app_state.is_episode_deleted("ds_a", 5) is False

    def test_get_episode_trim_exists(self, app_state):
        """Verify get_episode_trim returns the trim range."""
        app_state.add_edit(PendingEdit("trim", "ds_a", 0, {"start_frame": 10, "end_frame": 50}))

        result = app_state.get_episode_trim("ds_a", 0)

        assert result == (10, 50)

    def test_get_episode_trim_not_found(self, app_state):
        """Verify get_episode_trim returns None when no trim exists."""
        app_state.add_edit(PendingEdit("delete", "ds_a", 0))

        assert app_state.get_episode_trim("ds_a", 0) is None
        assert app_state.get_episode_trim("ds_a", 1) is None


class TestDatasetLocking:
    """Tests for per-dataset locking."""

    def test_is_locked_false_by_default(self, app_state):
        """Verify that datasets are unlocked by default."""
        assert app_state.is_locked("ds_a") is False

    def test_get_lock_returns_asyncio_lock(self, app_state):
        """Verify that get_lock returns an asyncio.Lock."""
        lock = app_state.get_lock("ds_a")
        assert isinstance(lock, asyncio.Lock)

    def test_get_lock_returns_same_instance(self, app_state):
        """Verify that get_lock returns the same lock for the same dataset."""
        lock1 = app_state.get_lock("ds_a")
        lock2 = app_state.get_lock("ds_a")
        assert lock1 is lock2

    def test_get_lock_different_datasets(self, app_state):
        """Verify that different datasets get different locks."""
        lock_a = app_state.get_lock("ds_a")
        lock_b = app_state.get_lock("ds_b")
        assert lock_a is not lock_b

    def test_is_locked_true_when_acquired(self, app_state):
        """Verify that is_locked returns True when the lock is held."""
        lock = app_state.get_lock("ds_a")

        async def check():
            async with lock:
                assert app_state.is_locked("ds_a") is True
            assert app_state.is_locked("ds_a") is False

        asyncio.run(check())

    def test_is_locked_independent_per_dataset(self, app_state):
        """Verify that locking one dataset doesn't affect another."""
        lock_a = app_state.get_lock("ds_a")

        async def check():
            async with lock_a:
                assert app_state.is_locked("ds_a") is True
                assert app_state.is_locked("ds_b") is False

        asyncio.run(check())

    def test_discard_lock_removes_entry(self, app_state):
        """`discard_lock` drops the asyncio.Lock so closed datasets don't
        accumulate entries in `_dataset_locks` for the life of the session.
        """
        app_state.get_lock("ds_a")
        app_state.get_lock("ds_b")
        # Internal: there are now two cached locks.
        assert "ds_a" in app_state._dataset_locks
        assert "ds_b" in app_state._dataset_locks

        app_state.discard_lock("ds_a")
        assert "ds_a" not in app_state._dataset_locks
        assert "ds_b" in app_state._dataset_locks
        # is_locked still reports False (no entry == no lock held).
        assert app_state.is_locked("ds_a") is False

    def test_discard_lock_noop_on_missing(self, app_state):
        """Calling `discard_lock` on a dataset that was never locked is a no-op."""
        app_state.discard_lock("never_seen")  # must not raise


class TestEditPersistence:
    """Tests for saving and loading edits to/from disk."""

    def test_save_edits_creates_file(self, tmp_path):
        """Verify that save_edits_to_file creates the edits file."""
        edits = [
            PendingEdit("trim", "ds_a", 0, {"start_frame": 5, "end_frame": 100}),
            PendingEdit("delete", "ds_a", 3),
        ]

        save_edits_to_file(tmp_path, edits)

        edits_file = tmp_path / EDITS_FILENAME
        assert edits_file.exists()

        # Verify file content structure
        data = json.loads(edits_file.read_text())
        assert "version" in data
        assert "edits" in data
        assert len(data["edits"]) == 2

    def test_save_empty_edits_removes_file(self, tmp_path):
        """Verify that saving empty edits removes existing file."""
        edits_file = tmp_path / EDITS_FILENAME
        edits_file.write_text('{"version": 1, "edits": []}')

        save_edits_to_file(tmp_path, [])

        assert not edits_file.exists()

    def test_save_empty_edits_no_file_no_error(self, tmp_path):
        """Verify saving empty edits when no file exists doesn't raise."""
        save_edits_to_file(tmp_path, [])

        edits_file = tmp_path / EDITS_FILENAME
        assert not edits_file.exists()

    def test_load_edits_returns_saved_data(self, tmp_path):
        """Verify that load_edits_from_file returns saved edits."""
        original_edits = [
            PendingEdit("trim", "ds_a", 5, {"start_frame": 10, "end_frame": 50}),
            PendingEdit("delete", "ds_a", 7),
        ]
        save_edits_to_file(tmp_path, original_edits)

        loaded_edits = load_edits_from_file(tmp_path, "ds_a")

        assert len(loaded_edits) == 2
        assert loaded_edits[0].edit_type == "trim"
        assert loaded_edits[0].episode_index == 5
        assert loaded_edits[0].params == {"start_frame": 10, "end_frame": 50}
        assert loaded_edits[1].edit_type == "delete"
        assert loaded_edits[1].episode_index == 7

    def test_load_edits_sets_dataset_id(self, tmp_path):
        """Verify that loaded edits have correct dataset_id set."""
        original_edits = [PendingEdit("delete", "ds_a", 0)]
        save_edits_to_file(tmp_path, original_edits)

        loaded_edits = load_edits_from_file(tmp_path, "different_ds")

        assert loaded_edits[0].dataset_id == "different_ds"

    def test_load_edits_preserves_created_at(self, tmp_path):
        """Verify that loaded edits preserve creation timestamp."""
        original_time = datetime(2026, 1, 15, 10, 30, 0)
        original_edit = PendingEdit("delete", "ds_a", 0, created_at=original_time)
        save_edits_to_file(tmp_path, [original_edit])

        loaded_edits = load_edits_from_file(tmp_path, "ds_a")

        assert loaded_edits[0].created_at == original_time

    def test_load_edits_missing_file_returns_empty(self, tmp_path):
        """Verify that load returns empty list when file doesn't exist."""
        edits = load_edits_from_file(tmp_path, "ds_a")

        assert edits == []

    def test_load_edits_invalid_json_returns_empty(self, tmp_path):
        """Verify that load returns empty list for corrupted file."""
        edits_file = tmp_path / EDITS_FILENAME
        edits_file.write_text("not valid json {{{")

        edits = load_edits_from_file(tmp_path, "ds_a")

        assert edits == []

    def test_clear_edits_file_removes_file(self, tmp_path):
        """Verify that clear_edits_file removes the file."""
        edits_file = tmp_path / EDITS_FILENAME
        edits_file.write_text('{"version": 1, "edits": []}')

        clear_edits_file(tmp_path)

        assert not edits_file.exists()

    def test_clear_edits_file_missing_file_no_error(self, tmp_path):
        """Verify clear_edits_file doesn't raise when file doesn't exist."""
        clear_edits_file(tmp_path)

        # Should not raise
        edits_file = tmp_path / EDITS_FILENAME
        assert not edits_file.exists()

    def test_roundtrip_multiple_edits(self, tmp_path):
        """Verify that multiple edits survive save/load roundtrip."""
        original_edits = [
            PendingEdit("trim", "ds_a", 0, {"start_frame": 0, "end_frame": 50}),
            PendingEdit("delete", "ds_a", 1),
            PendingEdit("trim", "ds_a", 2, {"start_frame": 10, "end_frame": 20}),
            PendingEdit("delete", "ds_a", 5),
        ]

        save_edits_to_file(tmp_path, original_edits)
        loaded_edits = load_edits_from_file(tmp_path, "ds_a")

        assert len(loaded_edits) == 4
        for i, (orig, loaded) in enumerate(zip(original_edits, loaded_edits, strict=False)):
            assert loaded.edit_type == orig.edit_type, f"Edit {i} type mismatch"
            assert loaded.episode_index == orig.episode_index, f"Edit {i} episode mismatch"
            assert loaded.params == orig.params, f"Edit {i} params mismatch"


# ============================================================================
# Dataset Locking - API Integration Tests
# ============================================================================


@pytest.fixture
def locked_app():
    """Create a FastAPI app with edits router and a pre-created lock.

    No real dataset needed: _require_unlocked runs before dataset existence
    checks, so a locked dataset_id is enough to trigger 423.
    """
    app = FastAPI()
    app.include_router(edits_module.router)

    state = AppState(frame_cache=FrameCache(max_bytes=1_000_000))
    edits_module.set_app_state(state)

    lock = state.get_lock("ds_locked")
    return app, state, lock


class TestLockingEndpoints:
    """Verify that write endpoints return 423 when locked and pass when unlocked."""

    @pytest.mark.parametrize(
        "path,json_body",
        [
            ("/api/edits/delete", {"dataset_id": "ds_locked", "episode_index": 0}),
            (
                "/api/edits/trim",
                {"dataset_id": "ds_locked", "episode_index": 0, "start_frame": 0, "end_frame": 10},
            ),
            ("/api/edits/discard", None),
        ],
        ids=["delete", "trim", "discard"],
    )
    def test_returns_423_when_locked(self, locked_app, path, json_body):
        app, _state, lock = locked_app

        async def run():
            async with (
                lock,
                httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
            ):
                params = {"dataset_id": "ds_locked"} if path == "/api/edits/discard" else None
                resp = await client.post(path, json=json_body, params=params)
                assert resp.status_code == 423

        asyncio.run(run())

    def test_apply_returns_423_when_locked(self, locked_app):
        """Apply checks dataset existence first, so we need a fake entry."""
        app, state, lock = locked_app
        state.datasets["ds_locked"] = MagicMock()

        async def run():
            async with (
                lock,
                httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
            ):
                resp = await client.post("/api/edits/apply", params={"dataset_id": "ds_locked"})
                assert resp.status_code == 423

        asyncio.run(run())

    def test_remove_edit_returns_423_when_locked(self, locked_app):
        app, state, lock = locked_app
        state.add_edit(PendingEdit("delete", "ds_locked", 0))

        async def run():
            async with (
                lock,
                httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
            ):
                resp = await client.delete("/api/edits/0")
                assert resp.status_code == 423

        asyncio.run(run())

    def test_passes_lock_check_when_unlocked(self, locked_app):
        """Without the lock held, request passes the guard and hits 404 (dataset not found)."""
        app, _state, _lock = locked_app

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/edits/delete",
                    json={"dataset_id": "ds_locked", "episode_index": 0},
                )
                assert resp.status_code == 404

        asyncio.run(run())
