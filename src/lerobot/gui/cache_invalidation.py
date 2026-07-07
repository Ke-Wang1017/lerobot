"""Shared helpers for invalidating dataset-scoped caches.

When a dataset's underlying files change (edits, hub download, external
writes), several memoized layers need to be cleared:

  1. The app-wide ``FrameCache`` entries for this dataset.
  2. The global video decoder cache (file-handle caches).
  3. The episode-start-indices cache maintained by ``api.datasets``.

Call sites in edits.py, datasets.py, and merge dialog all repeated this
pattern. This module consolidates it so the order and error handling
stay consistent.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lerobot.gui.state import AppState

logger = logging.getLogger(__name__)


def invalidate_caches(
    app_state: AppState,
    dataset_id: str,
    invalidate_episode_indices: Callable[[str], None] | None = None,
) -> None:
    """Invalidate all caches associated with ``dataset_id``.

    Args:
        app_state: the shared :class:`AppState` instance (has ``.frame_cache``).
        dataset_id: identifier of the dataset whose caches should be cleared.
        invalidate_episode_indices: optional callback to clear the
            ``api.datasets._episode_start_indices`` cache. Passed explicitly
            because the cache is module-private. When omitted, episode
            indices are not touched (call sites that don't care, e.g. the
            merge dialog which operates on a new target dataset, can skip).
    """
    try:
        num = app_state.frame_cache.invalidate_dataset(dataset_id)
        if num > 0:
            logger.info("Invalidated %d cached frames for %s", num, dataset_id)
    except Exception as e:
        logger.warning("Frame cache invalidation failed for %s: %s", dataset_id, e)

    try:
        from lerobot.datasets.video_utils import _default_decoder_cache

        size = _default_decoder_cache.size()
        if size > 0:
            _default_decoder_cache.clear()
            logger.info("Cleared video decoder cache (%d entries)", size)
    except Exception as e:
        logger.warning("Video decoder cache clear failed: %s", e)

    if invalidate_episode_indices is not None:
        try:
            invalidate_episode_indices(dataset_id)
        except Exception as e:
            logger.warning("Episode start index invalidation failed for %s: %s", dataset_id, e)
