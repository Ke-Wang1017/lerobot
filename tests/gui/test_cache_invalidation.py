"""Tests for lerobot.gui.cache_invalidation.invalidate_caches."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from lerobot.gui.cache_invalidation import invalidate_caches


def _make_app_state(frames_invalidated: int = 0):
    frame_cache = MagicMock()
    frame_cache.invalidate_dataset.return_value = frames_invalidated
    return SimpleNamespace(frame_cache=frame_cache)


def test_invalidates_frame_cache_for_dataset():
    app_state = _make_app_state(frames_invalidated=5)
    with patch("lerobot.datasets.video_utils._default_decoder_cache"):
        invalidate_caches(app_state, "user/ds")
    app_state.frame_cache.invalidate_dataset.assert_called_once_with("user/ds")


def test_clears_video_decoder_cache_when_nonempty():
    app_state = _make_app_state()
    with patch("lerobot.datasets.video_utils._default_decoder_cache") as cache:
        cache.size.return_value = 3
        invalidate_caches(app_state, "user/ds")
    cache.clear.assert_called_once()


def test_skips_video_decoder_clear_when_empty():
    """When the cache has no entries, skip .clear() to avoid noise."""
    app_state = _make_app_state()
    with patch("lerobot.datasets.video_utils._default_decoder_cache") as cache:
        cache.size.return_value = 0
        invalidate_caches(app_state, "user/ds")
    cache.clear.assert_not_called()


def test_calls_episode_index_invalidator_when_provided():
    app_state = _make_app_state()
    invalidator = MagicMock()
    with patch("lerobot.datasets.video_utils._default_decoder_cache"):
        invalidate_caches(app_state, "user/ds", invalidate_episode_indices=invalidator)
    invalidator.assert_called_once_with("user/ds")


def test_skips_episode_index_invalidator_when_none():
    app_state = _make_app_state()
    with patch("lerobot.datasets.video_utils._default_decoder_cache"):
        # Should not raise
        invalidate_caches(app_state, "user/ds", invalidate_episode_indices=None)


def test_frame_cache_error_does_not_prevent_other_invalidations():
    """If frame_cache invalidation raises, video decoder cache and index cache
    must still be cleared."""
    app_state = _make_app_state()
    app_state.frame_cache.invalidate_dataset.side_effect = RuntimeError("broken")
    invalidator = MagicMock()
    with patch("lerobot.datasets.video_utils._default_decoder_cache") as cache:
        cache.size.return_value = 2
        invalidate_caches(app_state, "user/ds", invalidate_episode_indices=invalidator)
    cache.clear.assert_called_once()
    invalidator.assert_called_once_with("user/ds")


def test_video_decoder_error_does_not_prevent_index_invalidation():
    app_state = _make_app_state()
    invalidator = MagicMock()
    with patch("lerobot.datasets.video_utils._default_decoder_cache") as cache:
        cache.size.side_effect = RuntimeError("size failed")
        invalidate_caches(app_state, "user/ds", invalidate_episode_indices=invalidator)
    invalidator.assert_called_once_with("user/ds")


def test_episode_index_invalidator_error_swallowed():
    """A broken index invalidator must not leak out of invalidate_caches."""
    app_state = _make_app_state()
    invalidator = MagicMock(side_effect=RuntimeError("bad"))
    with patch("lerobot.datasets.video_utils._default_decoder_cache"):
        # Should not raise
        invalidate_caches(app_state, "user/ds", invalidate_episode_indices=invalidator)
