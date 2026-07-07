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
"""Tests for background episode prefetching."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from lerobot.gui.frame_cache import FrameCache


def _mock_decode_fn(_video_path, timestamps, _tolerance_s, **_kwargs):
    """Mock decode that returns correctly-shaped tensors for any batch size."""
    n = len(timestamps)
    return torch.zeros((n, 3, 4, 4), dtype=torch.uint8)


def _make_mock_app_state(camera_keys: list[str], episode_length: int, num_episodes: int = 1):
    """Create a mock AppState with a fake dataset for prefetch testing.

    The prefetch function calls decode_video_frames_torchcodec directly
    (not dataset[idx]), so we only need metadata, not a real dataset.
    """
    frame_cache = FrameCache(max_bytes=100_000_000)

    # Build episode metadata with video paths
    episodes = {}
    for i in range(num_episodes):
        ep = {"length": episode_length}
        for cam in camera_keys:
            ep[f"videos/{cam}/from_timestamp"] = 0.0
            ep[f"videos/{cam}/chunk_index"] = 0
            ep[f"videos/{cam}/file_index"] = 0
        episodes[i] = ep

    meta = MagicMock()
    meta.camera_keys = camera_keys
    meta.video_keys = camera_keys  # For video datasets, video_keys == camera_keys
    meta.total_episodes = num_episodes
    meta.episodes = episodes
    meta.get_video_file_path = MagicMock(return_value=Path("videos/cam/0000.mp4"))

    dataset = MagicMock()
    dataset.meta = meta
    dataset.fps = 30
    dataset.root = Path("/tmp/fake_dataset")

    app_state = MagicMock()
    app_state.datasets = {"test_ds": dataset}
    app_state.frame_cache = frame_cache

    return app_state, dataset


def _setup_prefetch_module(app_state):
    """Configure the datasets module's global state for testing."""
    import lerobot.gui.api.datasets as ds_mod

    ds_mod._app_state = app_state
    ds_mod._episode_start_indices.clear()
    # Pre-compute episode start indices (frame 0 of each episode)
    episodes = app_state.datasets["test_ds"].meta.episodes
    starts = [0]
    for i in range(len(episodes) - 1):
        starts.append(starts[-1] + episodes[i]["length"])
    ds_mod._episode_start_indices["test_ds"] = starts
    # Reset prefetch state
    ds_mod._prefetch_generation = 0
    ds_mod._prefetch_current = None
    ds_mod._prefetch_last_frame = 0
    return ds_mod


class TestPrefetchEpisode:
    """Tests for _prefetch_episode."""

    @patch("lerobot.datasets.video_utils.VideoDecoderCache")
    @patch("lerobot.datasets.video_utils.decode_video_frames_torchcodec")
    def test_prefetch_populates_cache(self, mock_decode, _mock_cache_cls):
        """All frames should be cached after a full prefetch."""
        cameras = ["cam_a", "cam_b"]
        ep_length = 5
        app_state, _ = _make_mock_app_state(cameras, ep_length)
        ds_mod = _setup_prefetch_module(app_state)

        mock_decode.side_effect = _mock_decode_fn

        ds_mod._prefetch_generation = 1
        ds_mod._prefetch_episode("test_ds", 0, ep_length, generation=1)

        # All frames for all cameras should be cached
        for frame_idx in range(ep_length):
            for cam in cameras:
                assert app_state.frame_cache.contains("test_ds", 0, frame_idx, cam)

        # With batch_size=30 and 5 frames: 1 batch, decoded once per camera
        assert mock_decode.call_count == len(cameras)

    @patch("lerobot.datasets.video_utils.VideoDecoderCache")
    @patch("lerobot.datasets.video_utils.decode_video_frames_torchcodec")
    def test_prefetch_skips_cached_frames(self, mock_decode, _mock_cache_cls):
        """Frames already in cache should not trigger a decode."""
        cameras = ["cam_a"]
        ep_length = 5
        app_state, _ = _make_mock_app_state(cameras, ep_length)
        ds_mod = _setup_prefetch_module(app_state)

        mock_decode.side_effect = _mock_decode_fn

        # Pre-populate frames 0, 1, 2 in cache
        for i in range(3):
            app_state.frame_cache.put("test_ds", 0, i, "cam_a", b"cached")

        ds_mod._prefetch_generation = 1
        ds_mod._prefetch_episode("test_ds", 0, ep_length, generation=1)

        # 1 batch call with only 2 uncached frames (3 and 4)
        assert mock_decode.call_count == 1
        # Verify only frames 3,4 were in the timestamps arg
        call_args = mock_decode.call_args
        assert len(call_args[0][1]) == 2  # 2 timestamps

    @patch("lerobot.datasets.video_utils.VideoDecoderCache")
    @patch("lerobot.datasets.video_utils.decode_video_frames_torchcodec")
    def test_prefetch_stops_on_generation_change(self, mock_decode, _mock_cache_cls):
        """Prefetch should exit early when the generation counter changes."""
        cameras = ["cam_a"]
        # Use 100 frames so there are multiple batches (batch_size=30: 4 batches)
        ep_length = 100
        app_state, _ = _make_mock_app_state(cameras, ep_length)
        ds_mod = _setup_prefetch_module(app_state)

        # Change generation after the first decode call (first batch)
        call_count = [0]

        def decode_with_cancel(_video_path, timestamps, _tolerance_s, **_kwargs):
            call_count[0] += 1
            if call_count[0] >= 1:
                ds_mod._prefetch_generation = 999  # Simulate episode switch
            return torch.zeros((len(timestamps), 3, 4, 4), dtype=torch.uint8)

        mock_decode.side_effect = decode_with_cancel

        ds_mod._prefetch_generation = 1
        ds_mod._prefetch_episode("test_ds", 0, ep_length, generation=1)

        # Should have decoded only the first batch (30 frames) before stopping
        assert mock_decode.call_count == 1  # 1 batch call for the single camera
        # Only first 30 frames should be cached
        assert app_state.frame_cache.contains("test_ds", 0, 0, "cam_a")
        assert app_state.frame_cache.contains("test_ds", 0, 29, "cam_a")
        assert not app_state.frame_cache.contains("test_ds", 0, 30, "cam_a")

    def test_prefetch_handles_missing_dataset(self):
        """Prefetch should return silently if dataset is not found."""
        cameras = ["cam_a"]
        app_state, _ = _make_mock_app_state(cameras, 5)
        ds_mod = _setup_prefetch_module(app_state)

        # Should not raise
        ds_mod._prefetch_episode("nonexistent_ds", 0, 5, generation=1)

    @patch("lerobot.datasets.video_utils.VideoDecoderCache")
    @patch("lerobot.datasets.video_utils.decode_video_frames_torchcodec")
    def test_prefetch_lookahead_caches_multiple_short_episodes(self, mock_decode, _mock_cache_cls):
        """Short episodes should trigger lookahead prefetching of subsequent episodes."""
        cameras = ["cam_a"]
        # 10 episodes of 50 frames each — well under the 1000-frame lookahead
        ep_length = 50
        num_episodes = 10
        app_state, _ = _make_mock_app_state(cameras, ep_length, num_episodes=num_episodes)
        ds_mod = _setup_prefetch_module(app_state)

        mock_decode.side_effect = _mock_decode_fn

        ds_mod._prefetch_generation = 1
        ds_mod._prefetch_episode("test_ds", 0, ep_length, generation=1)

        # Episode 0 (50 frames) is cached, then lookahead should prefetch
        # episodes 1-9 (450 more frames) until 1000-frame budget is met.
        # 50 * 10 = 500 total, all episodes should be cached since total < 1000 + ep0.
        for ep_idx in range(num_episodes):
            assert app_state.frame_cache.contains("test_ds", ep_idx, 0, "cam_a"), (
                f"Episode {ep_idx} frame 0 should be cached"
            )
            assert app_state.frame_cache.contains("test_ds", ep_idx, ep_length - 1, "cam_a"), (
                f"Episode {ep_idx} last frame should be cached"
            )

    @patch("lerobot.datasets.video_utils.VideoDecoderCache")
    @patch("lerobot.datasets.video_utils.decode_video_frames_torchcodec")
    def test_prefetch_lookahead_stops_after_budget(self, mock_decode, _mock_cache_cls):
        """Lookahead should stop once the frame budget is exhausted."""
        cameras = ["cam_a"]
        # 5 episodes of 400 frames each = 2000 total
        # Lookahead is 1000 frames, so after ep 0 (400), should prefetch
        # ep 1 (400, remaining=600) and ep 2 (400, remaining=200) and
        # ep 3 (400, remaining=-200 -> stop after this one)
        ep_length = 400
        num_episodes = 5
        app_state, _ = _make_mock_app_state(cameras, ep_length, num_episodes=num_episodes)
        ds_mod = _setup_prefetch_module(app_state)

        mock_decode.side_effect = _mock_decode_fn

        ds_mod._prefetch_generation = 1
        ds_mod._prefetch_episode("test_ds", 0, ep_length, generation=1)

        # Episodes 0-3 should be cached (ep0 + 1000 lookahead covers 3 more episodes)
        for ep_idx in range(4):
            assert app_state.frame_cache.contains("test_ds", ep_idx, 0, "cam_a"), (
                f"Episode {ep_idx} should be cached"
            )

        # Episode 4 should NOT be cached (budget exhausted after ep 3)
        assert not app_state.frame_cache.contains("test_ds", 4, 0, "cam_a"), (
            "Episode 4 should not be cached (beyond lookahead budget)"
        )


class TestMaybeStartPrefetch:
    """Tests for _maybe_start_prefetch.

    These tests only verify the deduplication and generation-counter logic.
    We patch the executor to prevent actual background prefetch work (which
    would fail because the mock dataset has no real video files).
    """

    @patch("lerobot.gui.api.datasets._prefetch_executor")
    def test_deduplicates_sequential_playback(self, _mock_executor):
        """Sequential frame advances should not restart prefetch."""
        cameras = ["cam_a"]
        app_state, _ = _make_mock_app_state(cameras, 100)
        ds_mod = _setup_prefetch_module(app_state)

        ds_mod._maybe_start_prefetch("test_ds", 0, 100, start_frame=0)
        gen_after_first = ds_mod._prefetch_generation

        # Simulate sequential playback: frames 1, 2, 3, 4, 5
        for frame in range(1, 6):
            ds_mod._maybe_start_prefetch("test_ds", 0, 100, start_frame=frame)

        # Generation should not have incremented (all within threshold)
        assert ds_mod._prefetch_generation == gen_after_first

    @patch("lerobot.gui.api.datasets._prefetch_executor")
    def test_seek_restarts_prefetch(self, _mock_executor):
        """Seeking far within the same episode should restart prefetch."""
        cameras = ["cam_a"]
        app_state, _ = _make_mock_app_state(cameras, 300)
        ds_mod = _setup_prefetch_module(app_state)

        ds_mod._maybe_start_prefetch("test_ds", 0, 300, start_frame=0)
        gen_after_first = ds_mod._prefetch_generation

        # Simulate a seek to frame 200
        ds_mod._maybe_start_prefetch("test_ds", 0, 300, start_frame=200)
        gen_after_seek = ds_mod._prefetch_generation

        # Generation should have incremented (seek detected)
        assert gen_after_seek > gen_after_first

    @patch("lerobot.gui.api.datasets._prefetch_executor")
    def test_new_episode_increments_generation(self, _mock_executor):
        """Switching episodes should increment the generation counter."""
        cameras = ["cam_a"]
        app_state, _ = _make_mock_app_state(cameras, 5, num_episodes=2)
        ds_mod = _setup_prefetch_module(app_state)

        ds_mod._maybe_start_prefetch("test_ds", 0, 5)
        gen_first = ds_mod._prefetch_generation

        ds_mod._maybe_start_prefetch("test_ds", 1, 5)
        gen_second = ds_mod._prefetch_generation

        assert gen_second > gen_first
        assert ds_mod._prefetch_current == ("test_ds", 1)

    @patch("lerobot.gui.api.datasets._prefetch_executor")
    def test_loop_wrap_around_does_not_restart_prefetch(self, _mock_executor):
        """Playback looping (last frame -> frame 0) should not restart prefetch."""
        cameras = ["cam_a"]
        ep_length = 138
        app_state, _ = _make_mock_app_state(cameras, ep_length)
        ds_mod = _setup_prefetch_module(app_state)

        # Start prefetch and advance to near end of episode
        ds_mod._maybe_start_prefetch("test_ds", 0, ep_length, start_frame=0)
        ds_mod._prefetch_last_frame = ep_length - 1  # Simulate being at last frame
        gen_before_wrap = ds_mod._prefetch_generation

        # Loop back to frame 0 — delta equals ep_length - 1, should be treated as wrap
        ds_mod._maybe_start_prefetch("test_ds", 0, ep_length, start_frame=0)

        # Generation should NOT have incremented (wrap-around detected)
        assert ds_mod._prefetch_generation == gen_before_wrap

    @patch("lerobot.gui.api.datasets._prefetch_executor")
    def test_mid_episode_seek_still_restarts(self, _mock_executor):
        """A real mid-episode seek should still restart prefetch despite wrap-around logic."""
        cameras = ["cam_a"]
        ep_length = 300
        app_state, _ = _make_mock_app_state(cameras, ep_length)
        ds_mod = _setup_prefetch_module(app_state)

        ds_mod._maybe_start_prefetch("test_ds", 0, ep_length, start_frame=10)
        gen_before = ds_mod._prefetch_generation

        # Jump from frame 10 to frame 200 — clearly a seek, not a wrap-around
        ds_mod._maybe_start_prefetch("test_ds", 0, ep_length, start_frame=200)

        assert ds_mod._prefetch_generation > gen_before
