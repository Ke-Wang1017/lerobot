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
"""Tests for metadata reload on re-open and episode list refresh."""

import asyncio
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI

from lerobot.gui.api import datasets as datasets_module
from lerobot.gui.frame_cache import FrameCache
from lerobot.gui.state import AppState


def _make_mock_dataset(total_episodes: int, ep_length: int = 100):
    """Create a mock dataset with the given number of episodes."""
    from unittest.mock import MagicMock

    ds = MagicMock()
    ds.repo_id = "test/dataset"
    ds.root = "/fake/path"
    ds.fps = 30
    ds.meta.total_episodes = total_episodes
    ds.meta.total_frames = total_episodes * ep_length
    ds.meta.robot_type = "test_robot"
    ds.meta.camera_keys = ["observation.images.front"]
    ds.meta.video_keys = ["observation.images.front"]
    ds.meta.features = {"observation.images.front": {}, "action": {}}
    ds.meta.episodes = [{"length": ep_length} for _ in range(total_episodes)]
    return ds


@pytest.fixture
def app_with_state():
    """Create a FastAPI app with datasets router and clean module state."""
    app = FastAPI()
    app.include_router(datasets_module.router)

    state = AppState(frame_cache=FrameCache(max_bytes=1_000_000))
    original_state = datasets_module._app_state
    original_mtime = datasets_module._dataset_info_mtime.copy()
    original_indices = datasets_module._episode_start_indices.copy()
    datasets_module.set_app_state(state)

    yield app, state

    # Restore
    datasets_module._app_state = original_state
    datasets_module._dataset_info_mtime.clear()
    datasets_module._dataset_info_mtime.update(original_mtime)
    datasets_module._episode_start_indices.clear()
    datasets_module._episode_start_indices.update(original_indices)


class TestMetadataReloadOnReopen:
    """Verify that re-opening an already-open dataset refreshes metadata."""

    def test_reopen_local_dataset_calls_check_and_reload(self, app_with_state, tmp_path):
        """Re-opening a local dataset should call _check_and_reload_metadata."""
        app, state = app_with_state
        dataset_id = str(tmp_path)
        state.datasets[dataset_id] = _make_mock_dataset(165)

        async def run():
            with patch.object(datasets_module, "_check_and_reload_metadata") as mock_reload:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.post(
                        "/api/datasets",
                        json={"local_path": str(tmp_path)},
                    )
                    assert resp.status_code == 200
                    mock_reload.assert_called_once_with(dataset_id)

        asyncio.run(run())

    def test_reopen_repo_dataset_calls_check_and_reload(self, app_with_state):
        """Re-opening a repo dataset should call _check_and_reload_metadata."""
        app, state = app_with_state
        dataset_id = "test/dataset"
        state.datasets[dataset_id] = _make_mock_dataset(165)

        async def run():
            with patch.object(datasets_module, "_check_and_reload_metadata") as mock_reload:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.post(
                        "/api/datasets",
                        json={"repo_id": "test/dataset"},
                    )
                    assert resp.status_code == 200
                    mock_reload.assert_called_once_with(dataset_id)

        asyncio.run(run())

    def test_reopen_returns_updated_episode_count(self, app_with_state, tmp_path):
        """After metadata reload updates total_episodes, re-open should return the new count."""
        app, state = app_with_state
        dataset_id = str(tmp_path)
        mock_ds = _make_mock_dataset(165)
        state.datasets[dataset_id] = mock_ds

        def simulate_reload(ds_id):
            """Simulate _check_and_reload_metadata updating the dataset."""
            mock_ds.meta.total_episodes = 216
            mock_ds.meta.total_frames = 216 * 100
            mock_ds.meta.episodes = [{"length": 100} for _ in range(216)]

        async def run():
            with patch.object(datasets_module, "_check_and_reload_metadata", side_effect=simulate_reload):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.post(
                        "/api/datasets",
                        json={"local_path": str(tmp_path)},
                    )
                    assert resp.status_code == 200
                    data = resp.json()
                    assert data["total_episodes"] == 216

        asyncio.run(run())

    def test_reopen_without_changes_returns_original_count(self, app_with_state, tmp_path):
        """Without metadata changes, the original count is returned."""
        app, state = app_with_state
        dataset_id = str(tmp_path)
        state.datasets[dataset_id] = _make_mock_dataset(165)

        async def run():
            with patch.object(datasets_module, "_check_and_reload_metadata"):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.post(
                        "/api/datasets",
                        json={"local_path": str(tmp_path)},
                    )
                    assert resp.status_code == 200
                    data = resp.json()
                    assert data["total_episodes"] == 165

        asyncio.run(run())


class TestListEpisodesReloadsMetadata:
    """Verify that listing episodes triggers metadata reload check."""

    def test_list_episodes_calls_check_and_reload(self, app_with_state):
        """list_episodes should call _check_and_reload_metadata."""
        app, state = app_with_state
        dataset_id = "test/dataset"
        state.datasets[dataset_id] = _make_mock_dataset(10)

        async def run():
            with patch.object(datasets_module, "_check_and_reload_metadata") as mock_reload:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.get(f"/api/datasets/{dataset_id}/episodes")
                    assert resp.status_code == 200
                    mock_reload.assert_called_once_with(dataset_id)

        asyncio.run(run())

    def test_list_episodes_returns_updated_count_after_reload(self, app_with_state):
        """After metadata reload adds episodes, list_episodes should return them all."""
        app, state = app_with_state
        dataset_id = "test/dataset"
        mock_ds = _make_mock_dataset(5, ep_length=50)
        state.datasets[dataset_id] = mock_ds

        def simulate_reload(ds_id):
            """Simulate new episodes being added."""
            mock_ds.meta.total_episodes = 8
            mock_ds.meta.total_frames = 8 * 50
            mock_ds.meta.episodes = [{"length": 50} for _ in range(8)]

        async def run():
            with patch.object(datasets_module, "_check_and_reload_metadata", side_effect=simulate_reload):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.get(f"/api/datasets/{dataset_id}/episodes")
                    assert resp.status_code == 200
                    episodes = resp.json()
                    assert len(episodes) == 8

        asyncio.run(run())


class TestListEpisodesActionStats:
    """The per-episode action stats must reach the GUI so each affected
    episode can be flagged in the timeline list — not just one aggregate
    warning at the top of the panel. The API exposes raw characteristics
    (min/max/mean/std per action component); the frontend derives flags
    like all-zero / static / jittery on top, so new checks can land
    without touching the API."""

    def test_action_stats_per_episode(self, app_with_state):
        from lerobot.gui.api.datasets import EpisodeActionStats

        app, state = app_with_state
        dataset_id = "test/action_stats"
        mock_ds = _make_mock_dataset(4)
        mock_ds.meta.episodes = [{"length": 50, "tasks": ["pick"]} for _ in range(4)]
        state.datasets[dataset_id] = mock_ds

        # Mix: episodes 1 and 3 are all-zero, episodes 0 and 2 have real
        # action variation. Frontend will derive its own flags on top.
        zero = EpisodeActionStats(
            min=[0.0, 0.0, 0.0, 0.0],
            max=[0.0, 0.0, 0.0, 0.0],
            mean=[0.0, 0.0, 0.0, 0.0],
            std=[0.0, 0.0, 0.0, 0.0],
        )
        nonzero = EpisodeActionStats(
            min=[-0.5, -0.3, -0.2, 0.0],
            max=[0.5, 0.3, 0.2, 2.0],
            mean=[0.01, 0.02, 0.0, 1.0],
            std=[0.2, 0.1, 0.08, 0.5],
        )
        cache = {0: nonzero, 1: zero, 2: nonzero, 3: zero}

        async def run():
            with (
                patch.object(datasets_module, "_check_and_reload_metadata"),
                patch.dict(datasets_module._episode_action_stats, {dataset_id: cache}),
            ):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.get(f"/api/datasets/{dataset_id}/episodes")
                    assert resp.status_code == 200
                    episodes = resp.json()
                    by_idx = {ep["episode_index"]: ep["action_stats"] for ep in episodes}
                    # Stats round-trip cleanly per episode.
                    assert by_idx[1]["max"] == [0.0, 0.0, 0.0, 0.0]
                    assert by_idx[2]["max"] == [0.5, 0.3, 0.2, 2.0]
                    assert by_idx[2]["std"] == [0.2, 0.1, 0.08, 0.5]

        asyncio.run(run())

    def test_action_stats_absent_for_dataset_without_action_feature(self, app_with_state):
        """When the dataset has no action feature (or stats not pre-computed),
        action_stats is None and the frontend renders no quality badge."""
        app, state = app_with_state
        dataset_id = "test/no_action"
        mock_ds = _make_mock_dataset(2)
        mock_ds.meta.episodes = [{"length": 50, "tasks": ["pick"]} for _ in range(2)]
        state.datasets[dataset_id] = mock_ds

        async def run():
            with (
                patch.object(datasets_module, "_check_and_reload_metadata"),
                # Empty stats dict → no episode has stats → all None
                patch.dict(datasets_module._episode_action_stats, {dataset_id: {}}),
            ):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.get(f"/api/datasets/{dataset_id}/episodes")
                    assert resp.status_code == 200
                    episodes = resp.json()
                    assert all(ep["action_stats"] is None for ep in episodes)

        asyncio.run(run())
