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
"""Tests for the FrameCache class."""

from lerobot.gui.frame_cache import FrameCache


class TestFrameCacheInvalidation:
    """Tests for cache invalidation by dataset."""

    def test_invalidate_dataset_removes_correct_entries(self):
        """Verify that invalidate_dataset only removes entries for the specified dataset."""
        cache = FrameCache(max_bytes=1_000_000)

        # Add frames from two different datasets
        cache.put("dataset_a", 0, 0, "cam", b"frame1")
        cache.put("dataset_a", 0, 1, "cam", b"frame2")
        cache.put("dataset_a", 1, 0, "cam", b"frame3")
        cache.put("dataset_b", 0, 0, "cam", b"frame4")
        cache.put("dataset_b", 0, 1, "cam", b"frame5")

        # Invalidate dataset_a
        removed = cache.invalidate_dataset("dataset_a")

        # Verify correct number removed
        assert removed == 3

        # Verify dataset_a entries are gone
        assert cache.get("dataset_a", 0, 0, "cam") is None
        assert cache.get("dataset_a", 0, 1, "cam") is None
        assert cache.get("dataset_a", 1, 0, "cam") is None

        # Verify dataset_b entries remain
        assert cache.get("dataset_b", 0, 0, "cam") == b"frame4"
        assert cache.get("dataset_b", 0, 1, "cam") == b"frame5"

    def test_invalidate_dataset_updates_current_bytes(self):
        """Verify that invalidation correctly updates the byte counter."""
        cache = FrameCache(max_bytes=1_000_000)

        cache.put("dataset_a", 0, 0, "cam", b"A" * 100)
        cache.put("dataset_a", 0, 1, "cam", b"B" * 200)
        cache.put("dataset_b", 0, 0, "cam", b"C" * 150)

        initial_bytes = cache.current_bytes
        assert initial_bytes == 450

        cache.invalidate_dataset("dataset_a")

        assert cache.current_bytes == 150  # Only dataset_b remains

    def test_invalidate_nonexistent_dataset(self):
        """Verify that invalidating a non-existent dataset returns 0."""
        cache = FrameCache(max_bytes=1_000_000)
        cache.put("dataset_a", 0, 0, "cam", b"frame1")

        removed = cache.invalidate_dataset("dataset_nonexistent")

        assert removed == 0
        assert cache.get("dataset_a", 0, 0, "cam") == b"frame1"


class TestFrameCacheLRUEviction:
    """Tests for LRU eviction under memory pressure."""

    def test_lru_eviction_removes_oldest_entries(self):
        """Verify that oldest entries are evicted when memory budget is exceeded."""
        cache = FrameCache(max_bytes=100)

        # Add entries that will exceed budget
        cache.put("ds", 0, 0, "cam", b"A" * 40)  # 40 bytes
        cache.put("ds", 0, 1, "cam", b"B" * 40)  # 40 bytes - total 80
        cache.put("ds", 0, 2, "cam", b"C" * 40)  # 40 bytes - should evict oldest

        # Oldest entry should be evicted
        assert cache.get("ds", 0, 0, "cam") is None
        # Newer entries should remain
        assert cache.get("ds", 0, 1, "cam") is not None
        assert cache.get("ds", 0, 2, "cam") is not None
        # Should be under budget
        assert cache.current_bytes <= 100

    def test_lru_eviction_multiple_entries(self):
        """Verify that multiple entries are evicted if needed."""
        cache = FrameCache(max_bytes=100)

        # Add small entries
        cache.put("ds", 0, 0, "cam", b"A" * 30)
        cache.put("ds", 0, 1, "cam", b"B" * 30)
        cache.put("ds", 0, 2, "cam", b"C" * 30)  # total 90

        # Add large entry that requires evicting multiple
        cache.put("ds", 0, 3, "cam", b"D" * 80)

        # First two should be evicted
        assert cache.get("ds", 0, 0, "cam") is None
        assert cache.get("ds", 0, 1, "cam") is None
        # Last two should remain (30 + 80 > 100, so even C might be evicted)
        assert cache.get("ds", 0, 3, "cam") is not None
        assert cache.current_bytes <= 100

    def test_access_moves_to_end_of_lru(self):
        """Verify that accessing an entry moves it to the end (most recently used)."""
        cache = FrameCache(max_bytes=100)

        cache.put("ds", 0, 0, "cam", b"A" * 30)
        cache.put("ds", 0, 1, "cam", b"B" * 30)
        cache.put("ds", 0, 2, "cam", b"C" * 30)  # total 90

        # Access the oldest entry, moving it to end
        cache.get("ds", 0, 0, "cam")

        # Add new entry - should evict entry 1 (now oldest)
        cache.put("ds", 0, 3, "cam", b"D" * 30)

        # Entry 0 was accessed, so entry 1 should be evicted
        assert cache.get("ds", 0, 0, "cam") is not None  # Accessed, not evicted
        assert cache.get("ds", 0, 1, "cam") is None  # Evicted (was oldest after access)


class TestFrameCacheBasics:
    """Basic cache operation tests."""

    def test_put_and_get(self):
        """Verify basic put and get operations."""
        cache = FrameCache(max_bytes=1_000_000)

        cache.put("ds", 0, 5, "cam", b"test_frame")
        result = cache.get("ds", 0, 5, "cam")

        assert result == b"test_frame"

    def test_cache_miss_returns_none(self):
        """Verify that cache miss returns None."""
        cache = FrameCache(max_bytes=1_000_000)

        result = cache.get("ds", 0, 0, "cam")

        assert result is None

    def test_stats_accuracy(self):
        """Verify that stats are accurate."""
        cache = FrameCache(max_bytes=1_000_000)

        cache.put("ds", 0, 0, "cam", b"A" * 100)
        cache.get("ds", 0, 0, "cam")  # Hit
        cache.get("ds", 0, 1, "cam")  # Miss

        stats = cache.stats()

        assert stats["entries"] == 1
        assert stats["current_bytes"] == 100
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_clear(self):
        """Verify that clear removes all entries."""
        cache = FrameCache(max_bytes=1_000_000)

        cache.put("ds", 0, 0, "cam", b"frame1")
        cache.put("ds", 0, 1, "cam", b"frame2")
        cache.clear()

        assert cache.get("ds", 0, 0, "cam") is None
        assert cache.get("ds", 0, 1, "cam") is None
        assert cache.current_bytes == 0

    def test_get_or_decode_caches_result(self):
        """Verify that get_or_decode caches the decoded result."""
        import torch

        cache = FrameCache(max_bytes=1_000_000)
        decode_count = [0]

        def mock_decode():
            decode_count[0] += 1
            # Return a simple RGB tensor (C, H, W) with uint8 values
            return torch.zeros((3, 10, 10), dtype=torch.uint8)

        # First call should decode
        result1 = cache.get_or_decode("ds", 0, 0, "cam", mock_decode)
        assert decode_count[0] == 1

        # Second call should use cache
        result2 = cache.get_or_decode("ds", 0, 0, "cam", mock_decode)
        assert decode_count[0] == 1  # Not called again

        assert result1 == result2


class TestFrameCacheContains:
    """Tests for the contains() method."""

    def test_contains_returns_true_for_cached_entry(self):
        cache = FrameCache(max_bytes=1_000_000)
        cache.put("ds", 0, 5, "cam", b"test_frame")

        assert cache.contains("ds", 0, 5, "cam") is True

    def test_contains_returns_false_for_missing_entry(self):
        cache = FrameCache(max_bytes=1_000_000)

        assert cache.contains("ds", 0, 0, "cam") is False

    def test_contains_does_not_affect_lru_order(self):
        """Verify that contains() does not move the entry to the end of LRU."""
        cache = FrameCache(max_bytes=100)

        cache.put("ds", 0, 0, "cam", b"A" * 30)
        cache.put("ds", 0, 1, "cam", b"B" * 30)
        cache.put("ds", 0, 2, "cam", b"C" * 30)  # total 90

        # contains() on the oldest entry should NOT protect it from eviction
        assert cache.contains("ds", 0, 0, "cam") is True

        # Add new entry — should evict entry 0 (oldest, not moved by contains)
        cache.put("ds", 0, 3, "cam", b"D" * 30)

        assert cache.get("ds", 0, 0, "cam") is None  # Evicted
        assert cache.get("ds", 0, 1, "cam") is not None
