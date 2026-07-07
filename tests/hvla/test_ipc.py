"""Tests for SharedLatentCache IPC (latent + subtask + confidence)."""

import pytest
import torch

from lerobot.policies.hvla.ipc import SharedLatentCache


@pytest.fixture
def cache_pair():
    """Create a writer/reader SharedLatentCache pair."""
    writer = SharedLatentCache(latent_dim=64, create=True)
    reader = SharedLatentCache(latent_dim=64, create=False)
    yield writer, reader
    reader.cleanup()
    writer.cleanup()


class TestSharedLatentCache:
    def test_latent_round_trip(self, cache_pair):
        writer, reader = cache_pair
        latent = torch.randn(64)
        writer.write(latent)
        got, ts = reader.read()
        assert got.shape == (64,)
        assert torch.allclose(got, latent, atol=1e-6)
        assert ts > 0

    def test_subtask_round_trip(self, cache_pair):
        writer, reader = cache_pair
        writer.write(torch.randn(64), subtask="pick up the cylinder")
        text, ts, conf = reader.read_subtask()
        assert text == "pick up the cylinder"
        assert ts > 0

    def test_confidence_round_trip(self, cache_pair):
        writer, reader = cache_pair
        writer.write(torch.randn(64), subtask="insert ring", confidence=0.87)
        text, ts, conf = reader.read_subtask()
        assert text == "insert ring"
        assert abs(conf - 0.87) < 0.01

    def test_subtask_not_written_without_arg(self, cache_pair):
        writer, reader = cache_pair
        writer.write(torch.randn(64))  # no subtask
        text, ts, conf = reader.read_subtask()
        assert text == ""
        assert conf == 0.0

    def test_subtask_updates(self, cache_pair):
        writer, reader = cache_pair
        writer.write(torch.randn(64), subtask="pick up cylinder", confidence=0.9)
        writer.write(torch.randn(64), subtask="insert into ring", confidence=0.7)
        text, _, conf = reader.read_subtask()
        assert text == "insert into ring"
        assert abs(conf - 0.7) < 0.01

    def test_subtask_persists_across_latent_writes(self, cache_pair):
        """Subtask stays from last write when subsequent writes omit it."""
        writer, reader = cache_pair
        writer.write(torch.randn(64), subtask="pick up", confidence=0.95)
        writer.write(torch.randn(64))  # no subtask
        writer.write(torch.randn(64))  # no subtask
        text, _, conf = reader.read_subtask()
        assert text == "pick up"
        assert abs(conf - 0.95) < 0.01

    def test_count_and_age(self, cache_pair):
        writer, reader = cache_pair
        assert reader.count == 0
        writer.write(torch.randn(64))
        assert reader.count == 1
        assert reader.age_ms < 1000  # should be very recent

    def test_utf8_subtask(self, cache_pair):
        writer, reader = cache_pair
        writer.write(torch.randn(64), subtask="拿起圆柱体", confidence=0.5)
        text, _, conf = reader.read_subtask()
        assert text == "拿起圆柱体"
        assert abs(conf - 0.5) < 0.01

    def test_long_subtask_truncated(self, cache_pair):
        writer, reader = cache_pair
        long_text = "a" * 500  # exceeds 256 byte limit
        writer.write(torch.randn(64), subtask=long_text, confidence=0.1)
        text, _, _ = reader.read_subtask()
        assert len(text) == 256
        assert text == "a" * 256
