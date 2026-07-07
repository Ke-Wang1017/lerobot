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
"""Tests for episode quality checks (video-data duration mismatch)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI

from lerobot.datasets.dataset_tools import check_episode_video_duration
from lerobot.gui.api import datasets as datasets_module
from lerobot.gui.frame_cache import FrameCache
from lerobot.gui.state import AppState

# ---------------------------------------------------------------------------
# Unit tests for check_episode_video_duration
# ---------------------------------------------------------------------------


class TestCheckEpisodeVideoDuration:
    """Tests for the standalone check_episode_video_duration function."""

    def test_normal_episode_returns_zero(self):
        """An episode with matching video and data duration should return 0."""
        ep = {
            "length": 300,
            "videos/cam_front/from_timestamp": 0.0,
            "videos/cam_front/to_timestamp": 10.0,  # 300 / 30 = 10.0
        }
        result = check_episode_video_duration(ep, fps=30)
        assert result == {"cam_front": 0}

    def test_one_frame_rounding_returns_zero(self):
        """1-frame rounding difference should be treated as normal."""
        ep = {
            "length": 200,
            "videos/cam/from_timestamp": 5.0,
            # Expected: 5.0 + (200-1)/30 = 11.633..., actual: 11.667 (+1 frame)
            "videos/cam/to_timestamp": 11.667,
        }
        result = check_episode_video_duration(ep, fps=30)
        assert result == {"cam": 0}

    def test_rerecording_artifact_detected(self):
        """Episode with extra video frames from re-recording should be flagged."""
        # 286 frames at 30fps -> expected 9.533s, but video spans 11.233s (+51 frames)
        ep = {
            "length": 286,
            "videos/observation.images.front/from_timestamp": 0.0,
            "videos/observation.images.front/to_timestamp": 11.233,
        }
        result = check_episode_video_duration(ep, fps=30)
        extra = result["observation.images.front"]
        assert extra > 0
        assert extra == 51  # round((11.233 - 9.533) * 30)

    def test_multiple_cameras(self):
        """Extra frames should be detected per camera."""
        ep = {
            "length": 286,
            "videos/front/from_timestamp": 0.0,
            "videos/front/to_timestamp": 11.233,  # mismatched
            "videos/wrist/from_timestamp": 0.0,
            "videos/wrist/to_timestamp": 9.533,  # normal
        }
        result = check_episode_video_duration(ep, fps=30)
        assert result["front"] == 51
        assert result["wrist"] == 0

    def test_truncated_video_returns_negative(self):
        """Episode with shorter video than data should return negative."""
        # 300 frames at 30fps -> expected 10.0s, but video only spans 8.0s
        ep = {
            "length": 300,
            "videos/cam/from_timestamp": 0.0,
            "videos/cam/to_timestamp": 8.0,
        }
        result = check_episode_video_duration(ep, fps=30)
        assert result["cam"] < 0
        assert result["cam"] == round((8.0 - 10.0) * 30)  # -60

    def test_no_video_keys_returns_empty(self):
        """Non-video episode should return empty dict."""
        ep = {"length": 100}
        result = check_episode_video_duration(ep, fps=30)
        assert result == {}

    def test_invalid_timestamps_skipped(self):
        """Episode with from_ts >= to_ts should be skipped (not included in result)."""
        ep = {
            "length": 100,
            "videos/cam/from_timestamp": 5.0,
            "videos/cam/to_timestamp": 5.0,
        }
        result = check_episode_video_duration(ep, fps=30)
        assert result == {}

    def test_zero_length_returns_empty(self):
        """Episode with zero length should return empty."""
        ep = {
            "length": 0,
            "videos/cam/from_timestamp": 0.0,
            "videos/cam/to_timestamp": 1.0,
        }
        result = check_episode_video_duration(ep, fps=30)
        assert result == {}


# ---------------------------------------------------------------------------
# Integration tests: list_episodes returns video_extra_frames
# ---------------------------------------------------------------------------


def _make_mock_dataset(episodes_meta: list[dict], fps: int = 30):
    """Create a mock dataset with specific episode metadata."""
    ds = MagicMock()
    ds.repo_id = "test/dataset"
    ds.root = "/fake/path"
    ds.fps = fps
    ds.meta.total_episodes = len(episodes_meta)
    ds.meta.total_frames = sum(ep["length"] for ep in episodes_meta)
    ds.meta.camera_keys = ["observation.images.front"]
    ds.meta.video_keys = ["observation.images.front"]
    ds.meta.features = {"observation.images.front": {}, "action": {}}
    ds.meta.episodes = dict(enumerate(episodes_meta))
    return ds


@pytest.fixture
def app_with_state():
    """Create a FastAPI app with datasets router and clean module state."""
    app = FastAPI()
    app.include_router(datasets_module.router)

    state = AppState(frame_cache=FrameCache(max_bytes=1_000_000))
    original_state = datasets_module._app_state
    original_mtime = datasets_module._dataset_info_mtime.copy()
    datasets_module.set_app_state(state)

    yield app, state

    datasets_module._app_state = original_state
    datasets_module._dataset_info_mtime.clear()
    datasets_module._dataset_info_mtime.update(original_mtime)


class TestListEpisodesQuality:
    """Test that list_episodes surfaces video_extra_frames."""

    def test_normal_episodes_have_zero_extra_frames(self, app_with_state):
        """Normal episodes should have video_extra_frames=0."""
        app, state = app_with_state
        fps = 30
        state.datasets["test/ds"] = _make_mock_dataset(
            [
                {
                    "length": 300,
                    "tasks": ["pick"],
                    "videos/observation.images.front/from_timestamp": 0.0,
                    "videos/observation.images.front/to_timestamp": 300 / fps,
                },
            ],
            fps=fps,
        )

        async def run():
            with patch.object(datasets_module, "_check_and_reload_metadata"):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.get("/api/datasets/test/ds/episodes")
                    assert resp.status_code == 200
                    episodes = resp.json()
                    assert len(episodes) == 1
                    assert episodes[0]["video_extra_frames"] == 0

        asyncio.run(run())

    def test_rerecording_episode_has_extra_frames(self, app_with_state):
        """Episode with re-recording artifact should have video_extra_frames > 0."""
        app, state = app_with_state
        fps = 30
        state.datasets["test/ds"] = _make_mock_dataset(
            [
                {
                    "length": 286,
                    "tasks": ["pick"],
                    "videos/observation.images.front/from_timestamp": 0.0,
                    "videos/observation.images.front/to_timestamp": 11.233,  # +51 extra
                },
                {
                    "length": 207,
                    "tasks": ["pick"],
                    "videos/observation.images.front/from_timestamp": 11.233,
                    "videos/observation.images.front/to_timestamp": 18.100,  # normal
                },
            ],
            fps=fps,
        )

        async def run():
            with patch.object(datasets_module, "_check_and_reload_metadata"):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.get("/api/datasets/test/ds/episodes")
                    assert resp.status_code == 200
                    episodes = resp.json()
                    assert len(episodes) == 2
                    assert episodes[0]["video_extra_frames"] == 51
                    assert episodes[1]["video_extra_frames"] == 0

        asyncio.run(run())
