"""Streaming video encoder for real-time frame-level encoding during recording.

Replaces the PNG-then-encode pipeline for video keys. Frames are pushed from the
recording loop and encoded in a background thread via PyAV. A reservoir sample of
frames is kept in memory for stats computation.
"""

import logging
import queue
import random
import threading
import time
from pathlib import Path

import av
import numpy as np
import PIL.Image

# Sentinel object to signal the encoder should discard and abort
_DISCARD = object()


class StreamingVideoEncoder:
    """Encodes video frames incrementally in a background thread during recording.

    Usage:
        encoder = StreamingVideoEncoder(fps=30, vcodec="libsvtav1")
        encoder.start_episode(Path("/tmp/ep0_cam.mp4"))
        for frame in frames:
            encoder.push_frame(frame)  # never blocks
        temp_path = encoder.finish()   # flush encoder, get temp video
        encoder.stop()                 # clean shutdown

    Args:
        fps: Frame rate of the output video.
        vcodec: Video codec to use. One of "libsvtav1", "h264", "hevc".
        pix_fmt: Pixel format for the output video.
        g: GOP size (keyframe interval).
        crf: Constant rate factor (quality).
        fast_decode: Fast decode setting (for AV1/HEVC).
        preset: Encoder preset (speed vs quality). Only used for libsvtav1.
        log_level: PyAV logging level during encoding.
        max_sample_size: Maximum number of frames to keep in the reservoir sample.
    """

    def __init__(
        self,
        fps: int,
        vcodec: str = "libsvtav1",
        pix_fmt: str = "yuv420p",
        g: int = 2,
        crf: int = 30,
        fast_decode: int = 0,
        preset: int = 12,
        log_level: int | None = av.logging.ERROR,
        max_sample_size: int = 300,
    ):
        self.fps = fps
        self.vcodec = vcodec
        self.pix_fmt = pix_fmt
        self.log_level = log_level
        self.max_sample_size = max_sample_size

        # Build codec options (mirrors encode_video_frames in video_utils.py)
        self.codec_options: dict[str, str] = {}
        if g is not None:
            self.codec_options["g"] = str(g)
        if crf is not None:
            self.codec_options["crf"] = str(crf)
        if fast_decode:
            key = "svtav1-params" if vcodec == "libsvtav1" else "tune"
            value = f"fast-decode={fast_decode}" if vcodec == "libsvtav1" else "fastdecode"
            self.codec_options[key] = value
        if vcodec == "libsvtav1":
            self.codec_options["preset"] = str(preset)

        # Encoders/pixel formats incompatibility check
        if (vcodec in ("libsvtav1", "hevc")) and pix_fmt == "yuv444p":
            logging.warning(
                f"Incompatible pixel format 'yuv444p' for codec {vcodec}, auto-selecting format 'yuv420p'"
            )
            self.pix_fmt = "yuv420p"

        self._frame_queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._episode_active = False
        self._tmp_video_path: Path | None = None
        self._error: Exception | None = None

        # Reservoir sample for stats (stores downsampled frames for quantile estimation)
        # All reservoir + running-stats state is read/written ONLY by the
        # encoder thread (post-refactor 2026-05-11), so no locking is needed.
        # ``_frame_count`` is producer-side (push_frame) for the finish-log
        # backlog message; ``_encoded_count`` is consumer-side, used as the
        # reservoir's ``n`` so picks are reproducible across runs given the
        # same encode order.
        self._sampled_frames: list[np.ndarray] = []
        self._frame_count: int = 0  # producer-side: how many frames pushed
        self._stats_downsample: int = 1  # computed on first encoded frame

        # Per-encoder RNG so concurrent per-camera encoder threads don't race
        # on the global random state. Seeded fresh per episode in
        # ``start_episode`` so reservoir picks are reproducible run-to-run.
        self._rng: random.Random = random.Random()

        # Online running stats for exact mean/std/min/max (per-channel)
        self._running_sum: np.ndarray | None = None  # shape (C,), float64
        self._running_sq_sum: np.ndarray | None = None  # shape (C,), float64
        self._running_min: np.ndarray | None = None  # shape (C,), float64
        self._running_max: np.ndarray | None = None  # shape (C,), float64
        self._running_n_pixels: int = 0  # total pixel count across all frames

    def start_episode(self, tmp_video_path: Path) -> None:
        """Begin encoding a new episode to a temporary video file."""
        if self._episode_active:
            raise RuntimeError(
                "Cannot start a new episode while one is active. Call finish() or discard() first."
            )

        tmp_video_path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp_video_path = tmp_video_path
        self._episode_active = True
        self._frame_count = 0
        self._sampled_frames = []
        self._stats_downsample = 1
        self._running_sum = None
        self._running_sq_sum = None
        self._running_min = None
        self._running_max = None
        self._running_n_pixels = 0
        self._error = None
        # Seed deterministically per episode so reservoir picks are
        # reproducible across runs (the encoder consumes frames in the
        # order the producer pushed them).
        self._rng = random.Random(0)

        # Drain any leftover items in the queue (shouldn't happen, but be safe)
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

        self._thread = threading.Thread(
            target=self._encoding_loop,
            args=(tmp_video_path,),
            name=f"video-encoder-{tmp_video_path.stem}",
            daemon=True,
        )
        self._thread.start()

    def push_frame(self, image: np.ndarray | PIL.Image.Image) -> None:
        """Push a frame for encoding. Never blocks the caller.

        Format-normalizes the input and hands it to the encoder thread.
        The reservoir sampling + running-stats accumulation runs in the
        encoder thread (since 2026-05-11), so this method does only the
        bare minimum on the caller — typically well under 100 µs even
        on a 1280×720 uint8 frame.

        Args:
            image: Frame data as numpy array (HWC uint8, CHW uint8, or CHW/HWC float [0,1])
                   or PIL Image.
        """
        if not self._episode_active:
            raise RuntimeError("No active episode. Call start_episode() first.")

        # Normalize to HWC uint8 numpy for consistent handling
        if isinstance(image, PIL.Image.Image):
            image = np.array(image)
        elif isinstance(image, np.ndarray):
            # Handle CHW format (pytorch convention)
            if image.ndim == 3 and image.shape[0] == 3:
                image = image.transpose(1, 2, 0)
            # Handle float [0,1] range
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        self._frame_queue.put(image)
        self._frame_count += 1

    def finish(self) -> Path:
        """Signal end of episode, wait for encoder to flush, return temp video path.

        Typically near-instant if the reset phase gave the encoder time to catch up.

        Returns:
            Path to the encoded temporary video file.

        Raises:
            RuntimeError: If encoding failed in the background thread.
        """
        if not self._episode_active:
            raise RuntimeError("No active episode to finish.")

        backlog = self._frame_queue.qsize()
        t0 = time.monotonic()
        self._frame_queue.put(None)  # sentinel: end of episode
        self._thread.join()
        dt = time.monotonic() - t0
        self._episode_active = False

        if dt > 0.5:
            logging.info(
                f"StreamingVideoEncoder finish: {dt:.1f}s "
                f"(backlog={backlog}, total_frames={self._frame_count})"
            )

        if self._error is not None:
            raise RuntimeError(f"Video encoding failed: {self._error}") from self._error

        return self._tmp_video_path

    def discard(self) -> None:
        """Abort current episode (for re-record). Stops encoder, deletes temp file."""
        if not self._episode_active:
            return

        # Signal discard — encoder thread will exit without flushing
        self._frame_queue.put(_DISCARD)
        self._thread.join()
        self._episode_active = False

        # Clean up temp file
        if self._tmp_video_path is not None and self._tmp_video_path.exists():
            # safe-destruct: our own temp video, controlled path
            self._tmp_video_path.unlink()

    def get_sampled_frames(self) -> list[np.ndarray]:
        """Return the reservoir-sampled frames (HWC uint8 numpy arrays)."""
        return self._sampled_frames

    def get_running_stats(self) -> dict[str, np.ndarray] | None:
        """Return exact per-channel mean/std/min/max computed over all frames.

        Returns dict with keys 'mean', 'std', 'min', 'max' each of shape (C,),
        in raw uint8 scale [0, 255]. Returns None if no frames have been seen.
        """
        if self._running_sum is None or self._running_n_pixels == 0:
            return None
        n = self._running_n_pixels
        mean = self._running_sum / n
        variance = self._running_sq_sum / n - mean**2
        # Clamp tiny negatives from floating-point rounding
        np.maximum(variance, 0, out=variance)
        return {
            "mean": mean,
            "std": np.sqrt(variance),
            "min": self._running_min.copy(),
            "max": self._running_max.copy(),
        }

    def stop(self) -> None:
        """Graceful shutdown. Discards any active episode."""
        if self._episode_active:
            self.discard()

    def _encoding_loop(self, video_path: Path) -> None:
        """Background thread: dequeue frames, run reservoir/stats, encode via PyAV.

        The reservoir sampling and running-stats accumulation live here
        (rather than in ``push_frame``) so the control thread isn't
        paying the ~1 ms/frame CPU cost of those operations. Frames are
        consumed in producer order, so the reservoir sees the same ``n``
        sequence the caller would have seen.
        """
        if self.log_level is not None:
            logging.getLogger("libav").setLevel(self.log_level)

        container = None
        stream = None
        encoded_count = 0
        try:
            container = av.open(str(video_path), "w")
            first_frame = True

            while True:
                frame = self._frame_queue.get()

                if frame is None:
                    # End of episode — flush encoder
                    if stream is not None:
                        for packet in stream.encode():
                            container.mux(packet)
                    break

                if frame is _DISCARD:
                    # Abort without flushing
                    break

                # Reservoir sample + running stats BEFORE encoding so the
                # uncompressed pixel data is what gets summarized.
                self._reservoir_sample(frame, encoded_count)
                encoded_count += 1

                # frame is HWC uint8 numpy array
                if first_frame:
                    height, width = frame.shape[:2]
                    stream = container.add_stream(self.vcodec, self.fps, options=self.codec_options)
                    stream.pix_fmt = self.pix_fmt
                    stream.width = width
                    stream.height = height
                    first_frame = False

                av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                for packet in stream.encode(av_frame):
                    container.mux(packet)

        except Exception as e:
            self._error = e
            logging.error(f"StreamingVideoEncoder error: {e}")
        finally:
            if container is not None:
                container.close()
            if self.log_level is not None:
                av.logging.restore_default_callback()

    def _reservoir_sample(self, image: np.ndarray, n: int) -> None:
        """Reservoir sampling (Algorithm R) + online running stats for every frame.

        Reservoir stores downsampled copies for quantile estimation.
        Running stats accumulate exact mean/std/min/max across all frames.
        A 1280x720 frame downsampled 8x becomes 160x90 (~43KB vs ~2.7MB).

        Args:
            image: The frame to sample. HWC uint8 numpy array.
            n: 0-indexed count of frames seen so far by this encoder.
               Owned by ``_encoding_loop`` since the refactor — the caller's
               ``_frame_count`` is producer-side and would drift ahead.
        """
        # Compute downsample factor on first frame (target ~150px max dimension,
        # matching auto_downsample_height_width in compute_stats.py)
        if n == 0:
            h, w = image.shape[:2]
            max_dim = max(h, w)
            self._stats_downsample = max(1, max_dim // 150) if max_dim > 300 else 1

        ds_frame = self._downsample_copy(image)

        # Reservoir sampling for quantiles
        if n < self.max_sample_size:
            self._sampled_frames.append(ds_frame)
        else:
            j = self._rng.randint(0, n)
            if j < self.max_sample_size:
                self._sampled_frames[j] = ds_frame

        # Online running stats: accumulate per-channel sums over all pixels of all frames.
        # Uses downsampled frame (same resolution as reservoir) for consistency.
        # float64 to avoid precision loss over thousands of frames.
        h, w, c = ds_frame.shape
        pixels = ds_frame.reshape(-1, c).astype(np.float64)  # (H*W, C)
        if self._running_sum is None:
            self._running_sum = pixels.sum(axis=0)
            self._running_sq_sum = (pixels**2).sum(axis=0)
            self._running_min = pixels.min(axis=0)
            self._running_max = pixels.max(axis=0)
        else:
            self._running_sum += pixels.sum(axis=0)
            self._running_sq_sum += (pixels**2).sum(axis=0)
            np.minimum(self._running_min, pixels.min(axis=0), out=self._running_min)
            np.maximum(self._running_max, pixels.max(axis=0), out=self._running_max)
        self._running_n_pixels += h * w

    def _downsample_copy(self, image: np.ndarray) -> np.ndarray:
        """Downsample and copy a frame for the reservoir sample."""
        ds = self._stats_downsample
        if ds > 1:
            return image[::ds, ::ds].copy()
        return image.copy()
