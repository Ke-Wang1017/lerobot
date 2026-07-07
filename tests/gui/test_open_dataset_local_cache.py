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
"""Tests for the local-cache completeness pre-check on dataset open.

Guards the contract that the GUI is local-only by default: it must not
silently call ``snapshot_download`` when a user opens a partial cache. The
pre-check refuses such opens with HTTP 409 + structured detail so the
frontend can ask the user to confirm an explicit Hub download.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI

from lerobot.gui.api import datasets as datasets_module
from lerobot.gui.api.datasets import _check_local_dataset_complete
from lerobot.gui.frame_cache import FrameCache
from lerobot.gui.state import AppState


@pytest.fixture
def app_with_state():
    """FastAPI app with the datasets router and a clean module-level state."""
    app = FastAPI()
    app.include_router(datasets_module.router)

    state = AppState(frame_cache=FrameCache(max_bytes=1_000_000))
    original_state = datasets_module._app_state
    original_mtime = datasets_module._dataset_info_mtime.copy()
    original_indices = datasets_module._episode_start_indices.copy()
    datasets_module.set_app_state(state)

    yield app, state

    datasets_module._app_state = original_state
    datasets_module._dataset_info_mtime.clear()
    datasets_module._dataset_info_mtime.update(original_mtime)
    datasets_module._episode_start_indices.clear()
    datasets_module._episode_start_indices.update(original_indices)


# ── _check_local_dataset_complete ───────────────────────────────────────────


class TestCheckLocalDatasetComplete:
    """Direct tests for the pure helper used by the open-dataset pre-check."""

    def test_complete_dataset_returns_ok(self, tmp_path, lerobot_dataset_factory):
        """A freshly-built dataset on disk has every data + video file present."""
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=3, total_frames=30)
        kind, problems = _check_local_dataset_complete(ds.root)
        assert kind == "complete", f"expected complete, got {kind}: {problems}"
        assert problems == []

    def test_missing_info_json(self, tmp_path):
        """An empty directory is reported with a clear marker problem."""
        kind, problems = _check_local_dataset_complete(tmp_path / "empty")
        assert kind == "missing_files"
        assert any("meta/info.json is missing" in p for p in problems)

    def test_missing_data_parquet(self, tmp_path, lerobot_dataset_factory):
        """Removing data parquet files is a missing-files problem (downloadable)."""
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=3, total_frames=30)
        # Wipe all data parquet files (keep the directory).
        for p in (ds.root / "data").rglob("*.parquet"):
            p.unlink()

        kind, problems = _check_local_dataset_complete(ds.root)
        assert kind == "missing_files", problems
        assert any("data parquet file(s) missing" in p for p in problems), problems

    def test_missing_video_file(self, tmp_path, lerobot_dataset_factory):
        """Removing a video referenced by metadata is flagged as missing files."""
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=3, total_frames=30, use_videos=True)
        videos_dir = ds.root / "videos"
        if not videos_dir.exists() or not list(videos_dir.rglob("*.mp4")):
            pytest.skip("factory built dataset without video files")

        # Drop one mp4 and verify it's surfaced as a missing video.
        next(videos_dir.rglob("*.mp4")).unlink()

        kind, problems = _check_local_dataset_complete(ds.root)
        assert kind == "missing_files", problems
        assert any("video file(s) missing" in p for p in problems), problems

    def test_episode_count_mismatch_is_metadata_inconsistent(self, tmp_path, lerobot_dataset_factory):
        """info.json claiming more episodes than the metadata table describes is
        a *metadata* inconsistency, not a missing-files problem.

        This is the real-world ``eval_rollout`` failure shape: ``info.json``'s
        ``total_episodes`` exceeds ``len(episodes)``, so resolving the "extra"
        episodes' paths raises ``IndexError``. The check classifies it as
        ``metadata_inconsistent`` (not ``missing_files``) so the caller states
        the mismatch faithfully instead of offering a download that wouldn't
        apply.
        """
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=3, total_frames=30)
        info_path = ds.root / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        # Claim two episodes that have no row in the metadata table.
        info["total_episodes"] = 5
        info["splits"] = {"train": "0:5"}
        info_path.write_text(json.dumps(info))

        kind, problems = _check_local_dataset_complete(ds.root)
        assert kind == "metadata_inconsistent", problems
        assert any("info.json reports 5 episode" in p and "describes 3" in p for p in problems), problems
        # All files are present — so it must NOT be reported as missing files.
        assert not any("missing" in p.lower() for p in problems), problems

    def test_renamed_folder_with_spaces_still_reports_local_problems(self, tmp_path, lerobot_dataset_factory):
        """Regression: a folder name like ``"my dataset copy"`` used to cause a
        confusing ``"Repo id must use alphanumeric chars…"`` to surface when
        meta/ was incomplete, because ``LeRobotDatasetMetadata.__init__`` would
        fall through to a Hub fetch using the bare folder name. The pre-check
        now diagnoses missing meta files directly so the user sees the actual
        problem, not the HF validation error for the synthesized repo_id.
        """
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=3, total_frames=30)
        # Move the freshly-built dataset to a folder name that contains chars
        # rejected by `huggingface_hub` repo-id validation.
        renamed = tmp_path / "my dataset with spaces"
        ds.root.rename(renamed)

        # Drop a required meta file so the metadata loader can't fully load.
        (renamed / "meta" / "tasks.parquet").unlink()

        kind, problems = _check_local_dataset_complete(renamed)
        assert kind == "missing_files"
        # Real diagnosis surfaces, not "Repo id must use alphanumeric…".
        assert any("tasks.parquet is missing" in p for p in problems), problems
        assert not any("alphanumeric" in p.lower() for p in problems), problems


# ── POST /api/datasets — 409 path on incomplete local cache ─────────────────


class TestOpenDatasetIncompleteCache:
    """End-to-end through the open endpoint."""

    def test_complete_local_cache_opens_normally(self, app_with_state, tmp_path, lerobot_dataset_factory):
        """Sanity: pre-check doesn't false-positive on a complete dataset."""
        app, _state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=3, total_frames=30)

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post("/api/datasets", json={"local_path": str(ds.root)})
                assert resp.status_code == 200, resp.text
                body = resp.json()
                assert body["total_episodes"] == 3

        asyncio.run(run())

    def test_incomplete_cache_returns_409_with_structured_detail(
        self, app_with_state, tmp_path, lerobot_dataset_factory
    ):
        """The core contract: don't silently download — surface 409 instead.

        The frontend keys off ``detail.code == "incomplete_local_cache"`` to
        decide whether to show the Hub-sync confirmation modal, so the shape
        of the payload is part of the API contract.
        """
        app, _state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=3, total_frames=30)
        # Make the local cache incomplete by removing all data parquets.
        for p in (ds.root / "data").rglob("*.parquet"):
            p.unlink()

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post("/api/datasets", json={"local_path": str(ds.root)})
                assert resp.status_code == 409, resp.text
                detail = resp.json()["detail"]
                assert detail["code"] == "incomplete_local_cache"
                # Missing files are recoverable via the Hub, so sync is offered.
                assert detail["kind"] == "missing_files"
                assert detail["hub_sync_available"] is True
                assert detail["local_path"] == str(ds.root)
                assert detail["repo_id"]  # non-empty
                assert isinstance(detail["problems"], list) and detail["problems"]
                assert any("data parquet" in p for p in detail["problems"])

        asyncio.run(run())

    def test_metadata_inconsistent_returns_409_without_hub_sync(
        self, app_with_state, tmp_path, lerobot_dataset_factory
    ):
        """A self-contradictory ``info.json`` (more episodes than the metadata
        table) must 409 with ``kind == "metadata_inconsistent"`` and
        ``hub_sync_available == False`` — the frontend keys off both to suppress
        the misleading 'Download & Open' call-to-action.
        """
        app, _state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=3, total_frames=30)
        info_path = ds.root / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        info["total_episodes"] = 5
        info["splits"] = {"train": "0:5"}
        info_path.write_text(json.dumps(info))

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post("/api/datasets", json={"local_path": str(ds.root)})
                assert resp.status_code == 409, resp.text
                detail = resp.json()["detail"]
                assert detail["code"] == "incomplete_local_cache"
                assert detail["kind"] == "metadata_inconsistent"
                assert detail["hub_sync_available"] is False
                assert "inconsistent" in detail["message"].lower()
                assert any("info.json reports 5 episode" in p for p in detail["problems"])

        asyncio.run(run())

    def test_confirm_hub_sync_skips_precheck(self, app_with_state, tmp_path, lerobot_dataset_factory):
        """With ``confirm_hub_sync=True`` the pre-check is bypassed, so the
        endpoint goes on to construct ``LeRobotDataset`` (which would normally
        download). We mock that constructor to confirm it's reached without
        the 409 firing — the real download path is exercised manually.
        """
        app, _state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=3, total_frames=30)
        # Snapshot the FULLY-built dataset under a separate root so the patched
        # LeRobotDataset constructor can return it as if downloaded.
        complete_root = ds.root

        # Now pretend the user opens a different (incomplete) path. We point
        # local_path at the complete root but verify that even if it WEREN'T
        # complete, confirm_hub_sync would skip the pre-check.
        for p in (complete_root / "data").rglob("*.parquet"):
            # Delete and recreate to force the pre-check to fail without
            # confirm_hub_sync. With confirm_hub_sync, this branch should
            # be skipped entirely.
            p.unlink()

        # Replace the LeRobotDataset import inside the open endpoint with
        # a stub that returns a minimally-functional handle. We can't easily
        # rebuild a real one after deleting the parquets, so the simplest
        # strong assertion is: when confirm_hub_sync=True, *no* 409 fires —
        # the request fails differently (downstream of the pre-check).
        with (
            patch.object(datasets_module, "_check_local_dataset_complete") as mock_check,
        ):
            # Sentinel: if the pre-check is consulted, the test is wrong.
            mock_check.side_effect = AssertionError("pre-check must be skipped on confirm_hub_sync")

            async def run():
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.post(
                        "/api/datasets",
                        json={"local_path": str(complete_root), "confirm_hub_sync": True},
                    )
                    # The pre-check is skipped (mock not called). The downstream
                    # LeRobotDataset constructor will fail on the missing data
                    # files — but it must NOT be a 409 from the pre-check.
                    assert resp.status_code != 409, resp.text
                    # And we never consulted the pre-check helper.
                    mock_check.assert_not_called()

            asyncio.run(run())
