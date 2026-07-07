"""Regression guards for the per-camera ``StreamingVideoEncoder`` in
``lerobot.datasets.video_encoder`` (aka ``OurStreamingVideoEncoder``).

Mathematical correctness of the running stats is already covered
end-to-end by ``test_streaming_video_stats_correctness`` in
``test_datasets.py`` (validates mean/std/min/max to ``atol=1e-10`` and
cross-checks quantiles against a PNG round-trip). The tests below
guard properties that are specific to the 2026-05-11 producer/consumer
refactor — they would NOT be caught by the existing math-only test:

1. ``push_frame`` must remain non-blocking. If anyone moves
   ``_reservoir_sample`` back onto the caller thread, the test fails
   loudly (verified: a deliberate regression brings p50 from 0.5 µs
   to ~750 µs).
2. The per-episode RNG seed must make picks reproducible across runs.
3. Episode state must fully reset between consecutive episodes on the
   same encoder instance.
"""

import time
from pathlib import Path

import numpy as np

from lerobot.datasets.video_encoder import StreamingVideoEncoder


def _make_frames(n: int, h: int = 240, w: int = 320, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def _run_episode(frames: list[np.ndarray], tmp_path: Path, **kwargs) -> StreamingVideoEncoder:
    enc = StreamingVideoEncoder(fps=30, **kwargs)
    enc.start_episode(tmp_path / "ep.mp4")
    for f in frames:
        enc.push_frame(f)
    enc.finish()
    return enc


def test_push_frame_is_non_blocking(tmp_path):
    """``push_frame`` must stay a near-instant enqueue.

    Pre-refactor (when ``_reservoir_sample`` ran synchronously on the
    caller) the median was ~1100 µs per call on a 720p uint8 frame —
    100% CPU-bound from a 345 KB float64 allocation + 4 reductions.
    Post-refactor: ~0.5 µs (queue.put). The 100 µs threshold below is
    200× the typical post-refactor value but still 11× under the old
    synchronous cost — so it catches any regression without flaking
    on slow CI runners.
    """
    enc = StreamingVideoEncoder(fps=30)
    enc.start_episode(tmp_path / "ep.mp4")
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    for _ in range(5):  # warmup
        enc.push_frame(frame)

    n = 50
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        enc.push_frame(frame)
        times.append(time.perf_counter() - t0)
    enc.finish()

    times.sort()
    p50_us = times[n // 2] * 1e6
    assert p50_us < 100, f"push_frame p50 regressed: {p50_us:.1f} µs (expected < 100 µs)"


def test_reservoir_reproducible_with_seeded_rng(tmp_path):
    """Per-encoder RNG (seeded fresh in ``start_episode``) must make
    reservoir picks reproducible run-to-run.

    Pre-refactor: ``_reservoir_sample`` used the global ``random`` module,
    so picks depended on whatever else the Python process happened to do
    with random — non-reproducible. The refactor swapped to
    ``random.Random(0)`` per episode; this test locks that contract in.
    """
    frames = _make_frames(200, seed=42)
    enc_a = _run_episode(frames, tmp_path / "a", max_sample_size=20)
    enc_b = _run_episode(frames, tmp_path / "b", max_sample_size=20)
    samples_a = enc_a.get_sampled_frames()
    samples_b = enc_b.get_sampled_frames()
    assert len(samples_a) == len(samples_b) == 20
    for a, b in zip(samples_a, samples_b, strict=True):
        np.testing.assert_array_equal(a, b)


def test_stats_reset_between_episodes(tmp_path):
    """Reusing an encoder across episodes must produce stats reflecting
    only the current episode — no state leakage from prior episodes."""
    enc = StreamingVideoEncoder(fps=30, max_sample_size=10)

    ep1_frames = _make_frames(10, seed=1)
    enc.start_episode(tmp_path / "ep1.mp4")
    for f in ep1_frames:
        enc.push_frame(f)
    enc.finish()
    stats_ep1 = enc.get_running_stats()

    ep2_frames = _make_frames(10, seed=2)
    enc.start_episode(tmp_path / "ep2.mp4")
    for f in ep2_frames:
        enc.push_frame(f)
    enc.finish()
    stats_ep2 = enc.get_running_stats()

    # Episode 2 stats must reflect only episode 2's data — different seed
    # means the per-frame distributions differ, so the means must differ.
    assert not np.allclose(stats_ep1["mean"], stats_ep2["mean"])
    # And the reservoir must contain only episode 2's frames.
    ep2_pool = np.stack([f[:: enc._stats_downsample, :: enc._stats_downsample] for f in ep2_frames])
    for sample in enc.get_sampled_frames():
        match = any(np.array_equal(sample, ep2_pool[i]) for i in range(len(ep2_pool)))
        assert match, "reservoir contains a frame not from episode 2"
