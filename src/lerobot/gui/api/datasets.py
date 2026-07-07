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
"""Dataset API endpoints."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

import pandas as pd
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from lerobot.datasets.dataset_tools import check_episode_video_duration
from lerobot.datasets.utils import DEFAULT_DATA_PATH
from lerobot.utils.constants import HF_LEROBOT_HOME

if TYPE_CHECKING:
    from lerobot.gui.state import AppState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

# Will be set by server.py
_app_state: AppState = None  # type: ignore

# Track metadata file modification times for auto-reload
_dataset_info_mtime: dict[str, float] = {}

# Cache episode start indices (cumulative sum of lengths)
_episode_start_indices: dict[str, list[int]] = {}

# Background prefetch state
# Single worker: sequential frame access is optimal for video decoding.
# The prefetch thread uses its own VideoDecoderCache, so it does NOT
# contend with the main thread's decoder — no lock needed.
_prefetch_executor = ThreadPoolExecutor(max_workers=1)
_prefetch_generation: int = 0
_prefetch_current: tuple[str, int] | None = None  # (dataset_id, episode_idx)
_prefetch_last_frame: int = 0  # Last frame requested by _maybe_start_prefetch
_prefetch_lock = threading.Lock()

# Threshold for detecting seeks vs. normal sequential playback.
# If the frame delta between consecutive _maybe_start_prefetch calls
# exceeds this, we cancel the current prefetch and restart from the new position.
_PREFETCH_SEEK_THRESHOLD = 5


# Dedicated single-worker executor for on-demand frame decode.
#
# Why a dedicated 1-worker pool rather than asyncio's default executor:
# the frame-fetch endpoint receives N parallel HTTP requests per frame
# change (the frontend's <img src=...> grid fires one per camera; for a
# 4-cam dataset that's 4 simultaneous requests targeting the same
# global_idx). Those would all land on different threads in the default
# multi-worker pool, and all of them would call dataset[global_idx]
# concurrently — which goes through the module-level
# ``video_utils._default_decoder_cache``. That cache returns the SAME
# torchcodec VideoDecoder instance to every caller, and libdav1d
# crashes (SIGSEGV at libdav1d.so.7.0.0+0x52c0d) on simultaneous use of
# one decoder. Confirmed in the wild on 2026-05-14 + reproduced
# locally with faulthandler.
#
# Two parts to the fix:
#   1. Use this single-worker executor so only one thread is ever
#      inside the decode work-fn at a time. Eliminates the libdav1d
#      thread-safety problem by construction.
#   2. Re-check the frame_cache inside the work-fn: by the time the
#      2nd, 3rd, ... requests reach the worker, the first one has
#      already decoded ALL cameras for that frame and populated the
#      JPEG cache. The N−1 followers find their cam in cache and skip
#      the redundant decode. Measured: 4 parallel same-frame requests
#      cost ~14 ms total (= 1 decode) instead of ~58 ms (= 4 decodes).
#
# Why not multi-worker with per-thread decoder caches? Benchmarked: the
# per-thread approach actually *loses* across the board because (a)
# libdav1d already uses internal multi-threading and saturates the
# physical cores, so user-level parallelism barely helps (~9 %), and
# (b) without singleflight it duplicates work (4 simultaneous same-frame
# requests do 4 decodes instead of 1). The single-worker + cache
# re-check pattern matches the access profile exactly: dedupe within
# a frame, queue across frames. Throughput ceiling is the same either
# way (limited by libdav1d's own per-decoder rate).
_decode_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gui-decode")


def shutdown_prefetch_executor() -> None:
    """Drop pending prefetch tasks and release the thread on server shutdown.

    The executor's single worker is a daemon thread so it dies on process
    exit anyway, but `shutdown(wait=False, cancel_futures=True)` also
    cancels any queued futures — without it, a long-running prefetch
    (a multi-second `_prefetch_episode` decode pass) would keep logging
    progress after uvicorn has already torn down logging handlers,
    producing the "I/O operation on closed file" stack traces.
    """
    _prefetch_executor.shutdown(wait=False, cancel_futures=True)


def shutdown_decode_executor() -> None:
    """Mirror of :func:`shutdown_prefetch_executor` for the decode pool."""
    _decode_executor.shutdown(wait=False, cancel_futures=True)


def _check_local_dataset_complete(local_path: Path) -> tuple[str, list[str]]:
    """Classify whether a local dataset directory can be opened from disk, and if not, how.

    Used to short-circuit ``LeRobotDataset.__init__``'s implicit Hub download
    when the user opens a local path: the editor is local-only and should not
    silently pull hundreds of MB from the Hub if local files are missing or the
    on-disk metadata is self-inconsistent.

    Two failure modes are distinguished so the caller can offer the right
    recovery (or none) and never show a misleading download prompt:

      * ``"missing_files"`` — the metadata is internally consistent but some
        referenced data/video files are absent on disk (e.g. a partial Hub
        download). Re-downloading from the Hub, when the repo exists, completes
        the cache.
      * ``"metadata_inconsistent"`` — the on-disk metadata contradicts itself:
        ``info.json``'s episode count disagrees with the per-episode metadata
        table, or an episode's path cannot be resolved. A download does not
        address this; the inconsistency is surfaced as-is.

    Returns:
        ``(kind, problems)`` where ``kind`` is ``"complete"``,
        ``"missing_files"``, or ``"metadata_inconsistent"``. ``problems``
        faithfully lists what's wrong; empty only when ``kind == "complete"``.
        Never raises.
    """
    from huggingface_hub.errors import HFValidationError

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    if not (local_path / "meta" / "info.json").exists():
        return "missing_files", ["meta/info.json is missing"]

    # If any meta-side file is missing, `LeRobotDatasetMetadata.__init__` falls
    # through to `_pull_from_repo` → `snapshot_download`, which validates the
    # `repo_id`. Folder names with spaces / non-alphanumeric characters (a
    # legitimate local scenario — users rename / duplicate cached datasets)
    # then surface as a confusing "Repo id must use alphanumeric chars…" error
    # instead of the actual diagnosis (meta files missing). Probe the
    # remaining required meta paths directly so we report the real problem.
    missing_meta: list[str] = []
    if not (local_path / "meta" / "tasks.parquet").exists():
        missing_meta.append("meta/tasks.parquet")
    if not (local_path / "meta" / "episodes").is_dir():
        missing_meta.append("meta/episodes/")
    if missing_meta:
        return "missing_files", [f"{p} is missing" for p in missing_meta]

    try:
        # Passing root= ensures the metadata loader reads from disk; for a local
        # dataset that has meta/, no Hub fetch is triggered.
        meta = LeRobotDatasetMetadata(repo_id=local_path.name, root=local_path)
    except HFValidationError:
        # The constructor only validates `repo_id` against the Hub when local
        # meta is incomplete enough to trigger a Hub fallback. Above we
        # already covered the common "info.json present, episodes missing"
        # case; reaching here means a subtler meta-side problem (a partial
        # episodes/ tree, version mismatch, etc.).
        return "metadata_inconsistent", ["meta/ contents are inconsistent (failed to load on disk)"]
    except Exception as e:
        return "metadata_inconsistent", [f"failed to load metadata: {e}"]

    problems: list[str] = []
    kind = "complete"

    # Metadata self-consistency: `info.json.total_episodes` is the authoritative
    # count, but the per-episode metadata table (`meta/episodes/`) is what
    # resolves data/video paths. When they disagree, path resolution for the
    # "extra" episodes raises IndexError — a self-inconsistency, not a missing
    # file, and not something a download addresses.
    n_info = meta.total_episodes
    n_table = len(meta.episodes) if meta.episodes is not None else 0
    if n_info != n_table:
        kind = "metadata_inconsistent"
        detail = (
            f" (episodes {n_table}–{n_info - 1} have data but no metadata row)" if n_info > n_table else ""
        )
        problems.append(
            f"info.json reports {n_info} episode(s) but the episode metadata table describes {n_table}{detail}"
        )

    # Only probe files for episodes the metadata table can actually resolve;
    # iterating past `n_table` would just re-report the count mismatch above as a
    # pile of "cannot resolve path" noise.
    resolvable = min(n_info, n_table)

    missing_data: set[str] = set()
    for ep in range(resolvable):
        try:
            p = local_path / meta.get_data_file_path(ep)
        except Exception as e:
            # A resolution failure within the resolvable range is a deeper
            # inconsistency, not a missing file.
            kind = "metadata_inconsistent"
            problems.append(f"ep {ep}: cannot resolve data path ({e})")
            continue
        if not p.exists():
            missing_data.add(str(p))
    if missing_data:
        sample = sorted(missing_data)[0]
        problems.append(f"{len(missing_data)} data parquet file(s) missing (e.g. {sample})")

    missing_videos: set[str] = set()
    for vid_key in meta.video_keys:
        for ep in range(resolvable):
            try:
                p = local_path / meta.get_video_file_path(ep, vid_key)
            except Exception as e:
                kind = "metadata_inconsistent"
                problems.append(f"ep {ep} {vid_key}: cannot resolve video path ({e})")
                continue
            if not p.exists():
                missing_videos.add(str(p))
    if missing_videos:
        sample = sorted(missing_videos)[0]
        problems.append(f"{len(missing_videos)} video file(s) missing (e.g. {sample})")

    # Missing files are only the diagnosis when no metadata contradiction was
    # found first.
    if problems and kind == "complete":
        kind = "missing_files"
    return kind, problems


def _get_episode_start_index(dataset_id: str, episode_idx: int) -> int:
    """Get the global HuggingFace dataset index where an episode starts.

    The metadata's dataset_from_index is per-parquet-file, not global.
    This function computes the cumulative sum of episode lengths to get
    the correct global index.
    """
    if dataset_id not in _app_state.datasets:
        return 0

    # Check if we have cached start indices
    if dataset_id not in _episode_start_indices:
        dataset = _app_state.datasets[dataset_id]
        episodes = dataset.meta.episodes

        # Compute cumulative sum of episode lengths
        start_indices = [0]
        for i in range(len(episodes) - 1):
            start_indices.append(start_indices[-1] + episodes[i]["length"])

        _episode_start_indices[dataset_id] = start_indices
        logger.debug(f"Computed episode start indices for {dataset_id}: {len(start_indices)} episodes")

    indices = _episode_start_indices[dataset_id]
    if episode_idx < len(indices):
        return indices[episode_idx]
    return 0


def _invalidate_episode_start_indices(dataset_id: str) -> None:
    """Clear cached episode start indices when metadata changes."""
    _episode_start_indices.pop(dataset_id, None)
    _episode_action_stats.pop(dataset_id, None)
    _per_episode_features_cache.pop(dataset_id, None)
    _per_episode_source_cache.pop(dataset_id, None)
    _per_episode_warnings_cache.pop(dataset_id, None)


# Cache of per-episode action stats keyed by dataset_id. Surfaced via
# EpisodeInfo.action_stats so the GUI (and any other consumer) can apply
# generic quality heuristics — all-zero, static, saturated, jittery — on
# top of the same raw characteristics.
_episode_action_stats: dict[str, dict[int, EpisodeActionStats]] = {}


def _load_episode_action_stats(dataset_root: Path) -> dict[int, EpisodeActionStats]:
    """Load per-episode action stats from the dataset's episode metadata
    parquet files.

    LeRobot stores ``stats/action/{min,max,mean,std,count,q*}`` per episode
    at record time, so we don't need to rescan the data parquet — we just
    read the four summary keys from ``meta/episodes/*.parquet``.
    Falls back to an empty dict if the dataset has no action feature or
    the stats columns aren't present (older datasets, partially-built
    datasets); callers treat ``None`` action_stats as "unknown, render
    no badge".
    """
    out: dict[int, EpisodeActionStats] = {}
    episodes_dir = Path(dataset_root) / "meta" / "episodes"
    if not episodes_dir.exists():
        return out
    parquet_paths = sorted(episodes_dir.rglob("*.parquet"))
    if not parquet_paths:
        return out

    stat_cols = ["stats/action/min", "stats/action/max", "stats/action/mean", "stats/action/std"]
    for path in parquet_paths:
        try:
            df = pd.read_parquet(path, columns=["episode_index", *stat_cols])
        except (KeyError, ValueError):
            # `action` feature absent, or stats not pre-computed — bail
            # silently for the whole dataset rather than per-file, so we
            # don't half-populate the cache.
            return {}
        except Exception as e:
            logger.warning(f"action-stats: skipping unreadable {path.name}: {e}")
            continue
        for _, row in df.iterrows():
            try:
                out[int(row["episode_index"])] = EpisodeActionStats(
                    min=list(row["stats/action/min"]),
                    max=list(row["stats/action/max"]),
                    mean=list(row["stats/action/mean"]),
                    std=list(row["stats/action/std"]),
                )
            except (TypeError, ValueError):
                continue
    return out


def _check_and_reload_metadata(dataset_id: str) -> bool:
    """Check if dataset metadata changed on disk and reload if needed.

    Returns True if metadata was reloaded.
    """
    if dataset_id not in _app_state.datasets:
        return False

    dataset = _app_state.datasets[dataset_id]
    info_file = Path(dataset.root) / "meta" / "info.json"

    if not info_file.exists():
        return False

    current_mtime = info_file.stat().st_mtime
    cached_mtime = _dataset_info_mtime.get(dataset_id)

    if cached_mtime is not None and current_mtime == cached_mtime:
        return False

    # Metadata changed - reload metadata AND HuggingFace dataset
    import datasets as hf_datasets

    from lerobot.datasets.io_utils import (
        load_episodes,
        load_info,
        load_stats,
        load_tasks,
    )

    logger.info(f"Detected metadata change for {dataset_id}, reloading...")

    try:
        root = dataset.root

        # Reload all metadata
        dataset.meta.info = load_info(root)
        dataset.meta.episodes = load_episodes(root)

        # Check and repair episode metadata indices if needed
        from lerobot.datasets.dataset_tools import repair_episode_indices

        try:
            repaired = repair_episode_indices(root)
        except PermissionError as e:
            logger.warning(f"Skipping episode index repair on reload: {e}")
            repaired = 0
        if repaired > 0:
            logger.info(f"Repaired {repaired} episode indices with incorrect dataset_from_index")
            dataset.meta.episodes = load_episodes(root)

        dataset.meta.stats = load_stats(root)
        dataset.meta.tasks = load_tasks(root)
        _dataset_info_mtime[dataset_id] = current_mtime

        # CRITICAL: Also reload the HuggingFace dataset.
        # Post-refactor, hf_dataset is owned by DatasetReader. Direct assignment
        # to dataset.hf_dataset is rejected (read-only property), so route through
        # the reader's load_and_activate() public reload entry.
        if dataset.reader is not None and dataset.reader.hf_dataset is not None:
            try:
                num_cleaned = dataset.reader.hf_dataset.cleanup_cache_files()
                if num_cleaned > 0:
                    logger.info(f"Cleaned up {num_cleaned} HF cache files")
            except Exception as e:
                logger.warning(f"Could not cleanup cache files: {e}")

        # Clear video decoder cache
        try:
            from lerobot.datasets.video_utils import _default_decoder_cache

            cache_size = _default_decoder_cache.size()
            if cache_size > 0:
                _default_decoder_cache.clear()
                logger.info(f"Cleared video decoder cache ({cache_size} entries)")
        except Exception as e:
            logger.warning(f"Could not clear video decoder cache: {e}")

        # Reload HF dataset with caching disabled
        hf_datasets.disable_caching()
        try:
            if dataset.reader is not None:
                dataset.reader.load_and_activate()
        finally:
            hf_datasets.enable_caching()

        from lerobot.gui.cache_invalidation import invalidate_caches

        invalidate_caches(
            _app_state, dataset_id, invalidate_episode_indices=_invalidate_episode_start_indices
        )

        # Verify dataset integrity after reload
        from lerobot.datasets.dataset_tools import verify_dataset

        verification = verify_dataset(root, check_videos=False, verbose=False)
        if not verification.is_valid:
            for err in verification.errors:
                logger.warning(f"Post-reload verification: {err.message}")
        for warn in verification.warnings:
            logger.warning(f"Post-reload verification warning: {warn.message}")

        logger.info(
            f"Reloaded dataset: {dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames"
        )
        return True
    except Exception as e:
        logger.error(f"Failed to reload dataset for {dataset_id}: {e}")
        return False


_PREFETCH_BATCH_SIZE = 30  # Frames per batch (~1 second at 30 fps)

# After caching the current episode, keep prefetching subsequent episodes
# until at least this many frames ahead have been cached. This provides
# a comfortable buffer even at 2x playback speed with short episodes.
# 1000 frames ≈ 33s at 30fps, using ~100-240MB depending on resolution/cameras.
_PREFETCH_LOOKAHEAD_FRAMES = 1000


def _prefetch_episode(
    dataset_id: str, episode_idx: int, ep_length: int, generation: int, start_frame: int = 0
) -> None:
    """Decode and cache all frames of an episode in a background thread.

    Starts from start_frame and wraps around to cover the entire episode.
    Stops early if _prefetch_generation changes (meaning a different episode was selected).

    After completing the current episode, continues prefetching subsequent
    episodes until at least _PREFETCH_LOOKAHEAD_FRAMES have been cached
    ahead, or there are no more episodes.

    Uses batch decoding (multiple timestamps per decode call) for efficiency —
    the video decoder can read sequential frames without re-seeking. Also uses
    its own VideoDecoderCache so it never contends with the main thread.
    """
    from lerobot.datasets.video_utils import VideoDecoderCache

    if dataset_id not in _app_state.datasets:
        return

    dataset = _app_state.datasets[dataset_id]
    video_keys = list(dataset.meta.video_keys)
    first_camera = list(dataset.meta.camera_keys)[0] if dataset.meta.camera_keys else None
    fps = dataset.fps
    tolerance_s = 1 / fps * 0.7

    # Own decoder cache — completely independent from the main thread's decoders
    prefetch_decoder_cache = VideoDecoderCache()

    try:
        _prefetch_single_episode(
            dataset_id,
            dataset,
            episode_idx,
            ep_length,
            generation,
            start_frame,
            video_keys,
            first_camera,
            fps,
            tolerance_s,
            prefetch_decoder_cache,
        )

        # Keep prefetching subsequent episodes until we have enough lookahead
        lookahead_remaining = _PREFETCH_LOOKAHEAD_FRAMES
        next_idx = episode_idx + 1
        while next_idx < dataset.meta.total_episodes and lookahead_remaining > 0:
            if _prefetch_generation != generation:
                return
            next_ep = dataset.meta.episodes[next_idx]
            next_length = next_ep["length"]

            # Skip episodes already fully cached
            if first_camera and _app_state.frame_cache.is_episode_cached(
                dataset_id, next_idx, next_length, first_camera
            ):
                logger.debug(f"Lookahead: episode {next_idx} already cached, skipping")
                lookahead_remaining -= next_length
                next_idx += 1
                continue

            logger.info(
                f"Auto-prefetching episode {next_idx} ({next_length} frames, "
                f"{lookahead_remaining} lookahead remaining)"
            )
            # Clear decoder cache between episodes (different video files)
            prefetch_decoder_cache.clear()
            _prefetch_single_episode(
                dataset_id,
                dataset,
                next_idx,
                next_length,
                generation,
                0,
                video_keys,
                first_camera,
                fps,
                tolerance_s,
                prefetch_decoder_cache,
            )
            lookahead_remaining -= next_length
            next_idx += 1
    finally:
        prefetch_decoder_cache.clear()


def _prefetch_single_episode(
    dataset_id: str,
    dataset,
    episode_idx: int,
    ep_length: int,
    generation: int,
    start_frame: int,
    video_keys: list[str],
    first_camera: str | None,
    fps: float,
    tolerance_s: float,
    prefetch_decoder_cache,
) -> None:
    """Decode and cache all frames of a single episode."""
    import time

    from lerobot.datasets.video_utils import decode_video_frames_torchcodec
    from lerobot.gui.frame_cache import encode_frame_to_jpeg

    ep = dataset.meta.episodes[episode_idx]

    cached_count = 0
    decoded_count = 0
    total_decode_ms = 0.0
    total_encode_ms = 0.0
    prefetch_start = time.perf_counter()

    # Build two contiguous ranges: [start_frame, ep_length) then [0, start_frame)
    # Keeping frame indices sequential within each range lets the decoder
    # read forward without seeking backward.
    contiguous_ranges = [range(start_frame, ep_length)]
    if start_frame > 0:
        contiguous_ranges.append(range(0, start_frame))

    for frame_range in contiguous_ranges:
        for batch_start in range(frame_range.start, frame_range.stop, _PREFETCH_BATCH_SIZE):
            # Check cancellation between batches
            if _prefetch_generation != generation:
                logger.info(
                    f"Prefetch cancelled for episode {episode_idx} at frame {batch_start}/{ep_length} "
                    f"(decoded {decoded_count}, skipped {cached_count} cached)"
                )
                return

            batch_end = min(batch_start + _PREFETCH_BATCH_SIZE, frame_range.stop)

            # Filter out already-cached frames
            uncached_frames = []
            for fi in range(batch_start, batch_end):
                if first_camera and _app_state.frame_cache.contains(
                    dataset_id, episode_idx, fi, first_camera
                ):
                    cached_count += 1
                else:
                    uncached_frames.append(fi)

            if not uncached_frames:
                continue

            # Batch-decode all uncached frames for each camera
            try:
                for vid_key in video_keys:
                    from_timestamp = ep[f"videos/{vid_key}/from_timestamp"]
                    timestamps = [from_timestamp + fi / fps for fi in uncached_frames]
                    video_path = dataset.root / dataset.meta.get_video_file_path(episode_idx, vid_key)

                    t1 = time.perf_counter()
                    frames = decode_video_frames_torchcodec(
                        video_path,
                        timestamps,
                        tolerance_s,
                        decoder_cache=prefetch_decoder_cache,
                    )
                    t2 = time.perf_counter()
                    total_decode_ms += (t2 - t1) * 1000

                    # JPEG-encode each frame and cache it
                    for k, fi in enumerate(uncached_frames):
                        cam_jpeg = encode_frame_to_jpeg(frames[k])
                        _app_state.frame_cache.put(dataset_id, episode_idx, fi, vid_key, cam_jpeg)

                    t3 = time.perf_counter()
                    total_encode_ms += (t3 - t2) * 1000

                decoded_count += len(uncached_frames)
            except Exception:
                logger.warning(
                    f"Prefetch failed for batch {batch_start}-{batch_end} of episode {episode_idx}",
                    exc_info=True,
                )

    elapsed = (time.perf_counter() - prefetch_start) * 1000
    avg_decode = total_decode_ms / decoded_count if decoded_count else 0
    avg_encode = total_encode_ms / decoded_count if decoded_count else 0
    msg = (
        f"Prefetch complete for episode {episode_idx}: "
        f"decoded {decoded_count}, skipped {cached_count} cached, {ep_length} total in {elapsed:.0f}ms "
        f"(avg decode={avg_decode:.1f}ms, encode={avg_encode:.1f}ms)"
    )
    # Use DEBUG for no-op prefetches (everything already cached) to reduce log noise
    if decoded_count == 0:
        logger.debug(msg)
    else:
        logger.info(msg)


def _maybe_start_prefetch(dataset_id: str, episode_idx: int, ep_length: int, start_frame: int = 0) -> None:
    """Start background prefetching for an episode if not already in progress.

    Deduplicates by (dataset_id, episode_idx) for sequential playback.
    Detects seeks (frame jumps > _PREFETCH_SEEK_THRESHOLD) and restarts
    the prefetch from the new position.
    """
    global _prefetch_generation, _prefetch_current, _prefetch_last_frame

    with _prefetch_lock:
        if _prefetch_current == (dataset_id, episode_idx):
            # Same episode — only restart on significant seek
            frame_delta = abs(start_frame - _prefetch_last_frame)
            _prefetch_last_frame = start_frame
            if frame_delta <= _PREFETCH_SEEK_THRESHOLD:
                return  # Normal sequential advance, let current prefetch continue
            # Wrap-around detection: delta ≈ ep_length means playback looped
            if frame_delta >= ep_length - _PREFETCH_SEEK_THRESHOLD:
                return  # Loop wrap-around, not a real seek
            # Big jump detected — cancel old prefetch and restart from new position
            logger.info(
                f"Seek detected (delta={frame_delta}), restarting prefetch "
                f"for episode {episode_idx} from frame {start_frame}"
            )

        _prefetch_generation += 1
        generation = _prefetch_generation
        _prefetch_current = (dataset_id, episode_idx)
        _prefetch_last_frame = start_frame

    logger.info(f"Starting prefetch for episode {episode_idx} from frame {start_frame} ({ep_length} frames)")
    _prefetch_executor.submit(_prefetch_episode, dataset_id, episode_idx, ep_length, generation, start_frame)


def set_app_state(state: AppState) -> None:
    """Set the application state for API handlers."""
    global _app_state
    _app_state = state


# ---------------------------------------------------------------------------
# Dataset sources (folder browser)
# ---------------------------------------------------------------------------

SOURCES_FILE = Path.home() / ".config" / "lerobot" / "dataset_sources.json"
OPENED_FILE = Path.home() / ".config" / "lerobot" / "opened_datasets.json"


def _read_opened() -> list[dict]:
    """Read persisted list of opened datasets."""
    if not OPENED_FILE.exists():
        return []
    try:
        data = json.loads(OPENED_FILE.read_text())
        return data.get("datasets", [])
    except Exception:
        logger.warning("Failed to read opened datasets", exc_info=True)
        return []


def _write_opened(opened: list[dict]) -> None:
    """Persist the list of opened datasets (atomic write).

    Without staging, a crash between `write_text`'s truncate and the
    actual byte write leaves the file empty / partial; the next session
    would lose track of every dataset the user had open. Tiny file but
    not self-recovering — better to be safe.
    """
    import os

    OPENED_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = OPENED_FILE.with_suffix(OPENED_FILE.suffix + ".tmp")
    tmp.write_text(json.dumps({"version": 1, "datasets": opened}, indent=2))
    os.replace(tmp, OPENED_FILE)


def _save_opened_state() -> None:
    """Save the current set of opened datasets from app state."""
    entries = [{"root": str(ds.root)} for ds in _app_state.datasets.values()]
    _write_opened(entries)


def _read_sources() -> list[dict]:
    """Read source folders from config. Returns default source if file missing."""
    default_source = {
        "path": str(HF_LEROBOT_HOME),
        "removable": False,
        "expanded": True,
    }
    if not SOURCES_FILE.exists():
        return [default_source]
    try:
        data = json.loads(SOURCES_FILE.read_text())
        sources = data.get("sources", [])
        # Ensure default source is always present
        default_paths = {str(HF_LEROBOT_HOME)}
        has_default = any(s["path"] in default_paths for s in sources)
        if not has_default:
            sources.insert(0, default_source)
        return sources
    except Exception:
        logger.warning("Failed to read dataset sources, using defaults", exc_info=True)
        return [default_source]


def _write_sources(sources: list[dict]) -> None:
    """Persist source folders to config (atomic write)."""
    import os

    SOURCES_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {"version": 1, "sources": sources}
    tmp = SOURCES_FILE.with_suffix(SOURCES_FILE.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, SOURCES_FILE)


def _scan_source(source_path: str, max_depth: int = 3) -> list[dict]:
    """Scan a directory for datasets (subdirs containing meta/info.json).

    Returns lightweight metadata for each found dataset.
    """
    root = Path(source_path)
    if not root.is_dir():
        return []

    found = []
    _scan_recursive(root, root, found, max_depth, 0)
    # Sort by name
    found.sort(key=lambda d: d["name"])
    return found


def _scan_recursive(base: Path, current: Path, found: list[dict], max_depth: int, depth: int) -> None:
    """Recursively scan for datasets up to max_depth."""
    if depth > max_depth:
        return
    try:
        info_file = current / "meta" / "info.json"
        if info_file.is_file():
            # This directory is a dataset
            try:
                info = json.loads(info_file.read_text())
                rel = current.relative_to(base)
                found.append(
                    {
                        "name": str(rel),
                        "root": str(current),
                        "total_episodes": info.get("total_episodes", 0),
                        "total_frames": info.get("total_frames", 0),
                        "fps": info.get("fps", 0),
                        "robot_type": info.get("robot_type") or "",
                    }
                )
            except Exception:
                logger.debug(f"Failed to read info.json in {current}", exc_info=True)
            return  # Don't recurse into dataset subdirs

        # Not a dataset — recurse into subdirectories
        if depth < max_depth:
            try:
                for child in sorted(current.iterdir()):
                    if child.is_dir() and not child.name.startswith("."):
                        _scan_recursive(base, child, found, max_depth, depth + 1)
            except PermissionError:
                pass
    except Exception:  # nosec B110 - directory scan should never abort enumeration
        pass


class SourceRequest(BaseModel):
    path: str


class SourceInfo(BaseModel):
    path: str
    removable: bool
    expanded: bool


class SourceDatasetInfo(BaseModel):
    name: str
    root: str
    total_episodes: int
    total_frames: int
    fps: int
    robot_type: str = ""


@router.get("/previously-opened")
async def get_previously_opened() -> list[dict]:
    """Return the list of datasets that were open in the previous session."""
    return _read_opened()


@router.get("/sources")
async def list_sources() -> list[SourceInfo]:
    """List dataset source folders."""
    return [SourceInfo(**s) for s in _read_sources()]


@router.post("/sources")
async def add_source(req: SourceRequest) -> SourceInfo:
    """Add a dataset source folder."""
    path = str(Path(req.path).expanduser().resolve())
    if not Path(path).is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {path}")

    sources = _read_sources()
    # Check if already exists
    if any(s["path"] == path for s in sources):
        raise HTTPException(status_code=409, detail="Source already exists")

    new_source = {"path": path, "removable": True, "expanded": True}
    sources.append(new_source)
    _write_sources(sources)
    logger.info(f"Added dataset source: {path}")
    return SourceInfo(**new_source)


@router.delete("/sources/{encoded_path:path}")
async def remove_source(encoded_path: str) -> dict[str, str]:
    """Remove a dataset source folder."""
    path = unquote(encoded_path)
    sources = _read_sources()
    source = next((s for s in sources if s["path"] == path), None)
    if not source:
        raise HTTPException(status_code=404, detail=f"Source not found: {path}")
    if not source.get("removable", True):
        raise HTTPException(status_code=400, detail="Cannot remove default source")

    sources = [s for s in sources if s["path"] != path]
    _write_sources(sources)
    logger.info(f"Removed dataset source: {path}")
    return {"status": "ok"}


@router.put("/sources/{encoded_path:path}/expanded")
async def set_source_expanded(encoded_path: str, expanded: bool = True) -> dict[str, str]:
    """Toggle source folder expansion state."""
    path = unquote(encoded_path)
    sources = _read_sources()
    source = next((s for s in sources if s["path"] == path), None)
    if not source:
        raise HTTPException(status_code=404, detail=f"Source not found: {path}")

    source["expanded"] = expanded
    _write_sources(sources)
    return {"status": "ok"}


@router.post("/open-in-files")
async def open_in_file_manager(body: dict) -> dict:
    """Open a directory in the system file manager.

    Spawns the subprocess in the default executor so a slow fork/exec
    (heavy desktop session, many open FDs) cannot stall the FastAPI
    event loop.
    """
    import asyncio
    import subprocess as _subprocess

    path = body.get("path", "")
    if not path or not Path(path).is_dir():
        raise HTTPException(status_code=400, detail=f"Not a valid directory: {path}")

    def _spawn() -> None:
        _subprocess.Popen(["xdg-open", path])  # nosec B607 - xdg-open is the standard Linux file-opener

    try:
        await asyncio.get_event_loop().run_in_executor(None, _spawn)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail="xdg-open not found") from e

    return {"status": "ok"}


@router.get("/sources/{encoded_path:path}/datasets")
async def scan_source(encoded_path: str) -> list[SourceDatasetInfo]:
    """Scan a source folder for datasets."""
    import asyncio

    path = unquote(encoded_path)
    sources = _read_sources()
    if not any(s["path"] == path for s in sources):
        raise HTTPException(status_code=404, detail=f"Source not found: {path}")

    # Run scan in executor to avoid blocking
    loop = asyncio.get_event_loop()
    datasets = await loop.run_in_executor(None, _scan_source, path)
    return [SourceDatasetInfo(**d) for d in datasets]


class OpenDatasetRequest(BaseModel):
    """Request to open a dataset."""

    repo_id: str | None = None
    local_path: str | None = None
    # When opening a local path with an incomplete cache, the server returns 409
    # with a list of problems. Re-issue with confirm_hub_sync=True to authorize
    # the implicit Hub download (snapshot_download into the existing root).
    confirm_hub_sync: bool = False


class FeatureSchema(BaseModel):
    """Schema info for a single dataset feature.

    Mirrors the per-feature spec from ``info.json`` ``features`` dict —
    just the fields the GUI actually uses for rendering and validation.
    """

    dtype: str  # e.g. "float32", "int64", "bool", "string", "image", "video"
    shape: list[int]  # e.g. [1] for scalar, [14] for vector, [3, 480, 640] for image
    names: list[str] | None = None  # component names for vectors; None for scalars/strings
    is_per_episode: bool = False
    # True if every episode has uniform value for this feature — i.e. it's a logical
    # per-episode field broadcast across the per-frame column. Edits coerce to the
    # whole episode to preserve the broadcast invariant. Detected once per dataset
    # open via _detect_per_episode_features() and cached.
    per_episode_source: str | None = None
    # How was ``is_per_episode`` set? "declared" means the writer wrote
    # ``"per_episode": true`` in ``info.json`` — authoritative, persists across
    # saves. "detected" means the GUI scanned the parquet and inferred it from
    # uniform-within-episode values — empirical, can flip if a single frame
    # changes. ``None`` when ``is_per_episode`` is False. Power users can
    # surface this in tooltips to know which.
    observed_min: float | None = None
    observed_max: float | None = None
    # Dataset-wide observed extrema, sourced from ``meta/stats.json`` (aggregated
    # across episodes by ``compute_stats.py``). Populated only for scalar numeric
    # features (shape == [1] or empty). The GUI shows these next to the feature
    # name and uses them to scale the slider so the range is stable across
    # episodes. Distinct from declared bounds (below): observed, not enforced.
    declared_min: float | None = None
    declared_max: float | None = None
    # Optional declared bounds from the feature spec in ``info.json``. When
    # present, ``validate_feature_dtype_and_shape`` rejects values outside
    # ``[declared_min, declared_max]`` at add_frame and stage time. The GUI
    # uses them to scale the slider (preferred over observed_min/max) and may
    # show them in the card header. Categorical integer features use the
    # ``names`` field instead — the legal range is implicitly ``[0, len(names))``.


class DatasetInfo(BaseModel):
    """Summary info about a dataset."""

    id: str
    repo_id: str
    root: str
    total_episodes: int
    total_frames: int
    fps: int
    robot_type: str = ""
    camera_keys: list[str]
    features: list[str]  # feature names only — preserved for backwards-compat
    features_schema: dict[str, FeatureSchema] = {}
    # Full per-feature schema (dtype, shape, names) keyed by feature name.
    # Populated from ``ds.meta.features``. The GUI renderer registry
    # dispatches on (dtype, ndim) to pick row + Inspector widgets.
    errors: list[str] = []  # Verification errors (metadata mismatches — dataset may be corrupted)
    warnings: list[str] = []  # Non-critical warnings (stale stats, etc.)


_per_episode_features_cache: dict[str, set[str]] = {}
# Per-name source tag: "declared" (info.json said so — authoritative) vs
# "detected" (the GUI inferred it from uniform-within-episode data).
_per_episode_source_cache: dict[str, dict[str, str]] = {}
# Warnings surfaced when a feature was declared per_episode but the data
# isn't actually uniform within every episode. Surfaced via DatasetInfo
# warnings so the user knows their data violates their own declaration.
_per_episode_warnings_cache: dict[str, list[str]] = {}


def _per_episode_warnings_for(dataset_id: str) -> list[str]:
    """Return the (possibly empty) warning list cached by the last
    detector run for ``dataset_id``."""
    return list(_per_episode_warnings_cache.get(dataset_id, []))


def _detect_per_episode_features(dataset_id: str, dataset) -> set[str]:
    """Identify features that are per-episode — declared OR detected.

    Pre: ``dataset`` is fully loaded.
    Post: returns a set of feature names. The set is the union of:

    1. **Declared**: features whose info.json entry has ``per_episode: true``.
       Authoritative — survives saves, doesn't flip if the data drifts.
    2. **Detected**: features without the declaration whose values happen
       to be uniform within every episode (nunique-per-group ≤ 1).
       Heuristic — kept for backward compat with datasets pre-flag.

    Side effects (cached by ``dataset_id``):

    * ``_per_episode_source_cache[dataset_id]`` records the source
      ("declared" or "detected") per feature, so the schema endpoint can
      surface ``per_episode_source`` to the GUI.
    * ``_per_episode_warnings_cache[dataset_id]`` records any feature
      that was declared but whose data isn't actually uniform — the user's
      declaration is being trusted but their data violates it. The
      :class:`DatasetInfo` response surfaces these in ``warnings``.

    Only considers features that are non-image, non-video, non-DEFAULT_FEATURES,
    non-action, non-observation.* — the same gate the staging endpoint uses
    to decide editability.
    """
    if dataset_id in _per_episode_features_cache:
        return _per_episode_features_cache[dataset_id]

    skip_dtypes = {"image", "video"}
    default_features = {"timestamp", "frame_index", "episode_index", "index", "task_index"}

    # Declared `per_episode` hint in the feature spec wins over inference,
    # in BOTH directions. Mirrors the same precedence used in
    # _build_features_schema. Without this, a freshly-added per-frame
    # column (e.g. reward) initialized to a constant fill would look
    # uniform-per-episode and be silently coerced to whole-episode by
    # the staging endpoint — even though the schema says is_per_episode=false.
    declared: set[str] = set()
    declared_not_per_episode: set[str] = set()
    candidate_features: list[str] = []
    for name, ft in dataset.meta.features.items():
        if ft.get("dtype") in skip_dtypes:
            continue
        if name in default_features:
            continue
        if name == "action" or name.startswith("observation."):
            continue
        declared_pe = ft.get("per_episode") if "per_episode" in ft else None
        if declared_pe is False:
            declared_not_per_episode.add(name)
            continue
        # Vectors and matrices can't reasonably be "uniform per episode" — skip.
        shape = ft.get("shape") or [1]
        if len(shape) > 1 or (len(shape) == 1 and shape[0] != 1):
            continue
        if declared_pe is True:
            declared.add(name)
            # Still scan declared features below — to verify the data
            # actually matches the declaration. If it doesn't, we trust
            # the declaration but emit a warning so the user can fix the data.
        candidate_features.append(name)

    source_map: dict[str, str] = dict.fromkeys(declared, "declared")
    warnings: list[str] = []

    if not candidate_features:
        # Still record any declared per-episode features even when there's
        # nothing to infer (e.g. all features are declared one way or the
        # other already).
        _per_episode_features_cache[dataset_id] = set(declared)
        _per_episode_source_cache[dataset_id] = source_map
        _per_episode_warnings_cache[dataset_id] = warnings
        return _per_episode_features_cache[dataset_id]

    data_dir = Path(dataset.root) / "data"
    parquet_files = sorted(data_dir.glob("*/*.parquet"))
    if not parquet_files:
        # No data to scan — trust declared, nothing to detect.
        _per_episode_features_cache[dataset_id] = declared
        _per_episode_source_cache[dataset_id] = source_map
        _per_episode_warnings_cache[dataset_id] = warnings
        return _per_episode_features_cache[dataset_id]

    cols_to_read = ["episode_index", *candidate_features]
    nunique_by_feature: dict[str, int] = dict.fromkeys(candidate_features, 0)

    try:
        for shard in parquet_files:
            df = pd.read_parquet(shard, columns=cols_to_read)
            for name in candidate_features:
                if nunique_by_feature[name] > 1:
                    continue
                # nunique() per episode_index group — max across groups tells us
                # whether ANY episode has variation. >1 disqualifies the feature.
                nunique_max = df.groupby("episode_index")[name].nunique(dropna=False).max()
                if nunique_max is None:
                    continue
                nunique_by_feature[name] = max(nunique_by_feature[name], int(nunique_max))
    except Exception as e:
        logger.warning(f"per-episode-feature detection failed for {dataset_id}: {e}")
        # Detection scan failed — fall back to declared-only.
        _per_episode_features_cache[dataset_id] = declared
        _per_episode_source_cache[dataset_id] = source_map
        _per_episode_warnings_cache[dataset_id] = warnings
        return _per_episode_features_cache[dataset_id]

    per_episode: set[str] = set(declared)
    for name, nu in nunique_by_feature.items():
        is_uniform = 0 < nu <= 1
        if name in declared:
            # Verify the declaration. If the data isn't uniform, trust the
            # declaration (per the design — declared > detected) but warn so
            # the user knows their data has drifted.
            if not is_uniform:
                msg = (
                    f"Feature {name!r} is declared per_episode=true in info.json, "
                    f"but its data is not uniform within every episode "
                    f"(max nunique-per-episode = {nu}). The GUI is treating it "
                    f"as per-episode anyway — fix the data or remove the flag."
                )
                logger.warning(msg)
                warnings.append(msg)
        else:
            if is_uniform:
                per_episode.add(name)
                source_map[name] = "detected"

    _per_episode_features_cache[dataset_id] = per_episode
    _per_episode_source_cache[dataset_id] = source_map
    _per_episode_warnings_cache[dataset_id] = warnings
    logger.info(
        f"per-episode features for {dataset_id}: declared={sorted(declared)} "
        f"detected={sorted(per_episode - declared)} warnings={len(warnings)}"
    )
    return per_episode


# Hardcoded backend representation of the LeRobot 3.0 subtask format. The
# data layer stores ``subtask_index`` (int64[1]) plus ``meta/subtasks.parquet``
# (index → string lookup). The user always thinks in terms of strings, so
# whenever both pieces are present the schema endpoint synthesizes a
# ``subtask`` (string) feature in place of ``subtask_index``. Stage endpoint
# accepts either name and routes to the storage feature; PendingEdit stores
# the storage name so the apply pipeline doesn't need a special case.
SUBTASK_STORAGE_FEATURE = "subtask_index"
SUBTASK_DISPLAY_FEATURE = "subtask"


def _has_subtask_lookup(dataset) -> bool:
    """True if the dataset has a ``meta/subtasks.parquet`` lookup table."""
    return getattr(dataset.meta, "subtasks", None) is not None


def _coerce_optional_float(v: Any) -> float | None:
    """Coerce a value from ``info.json`` to a JSON-friendly Python float.

    Tolerates ``None``, missing, non-numeric, or NaN/inf — returns ``None`` for
    anything we can't safely surface as a slider bound.
    """
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    import math

    if not math.isfinite(f):
        return None
    return f


def _scalar_observed_extrema(
    stats: dict | None, name: str, shape: list[int]
) -> tuple[float | None, float | None]:
    """Pull ``(min, max)`` for a scalar feature out of ``meta/stats.json``.

    ``stats`` is the dict loaded by :func:`load_stats`, structured as
    ``{feature_name: {"min": np.ndarray, "max": np.ndarray, ...}}``. Values are
    aggregated across all episodes by ``compute_stats.py``.

    Returns ``(None, None)`` if stats are missing, the feature isn't in stats,
    the values aren't numeric/finite, or the shape isn't scalar (we don't
    surface component-wise stats for vectors — too messy in the card header).
    """
    if stats is None or name not in stats:
        return None, None
    if shape and not (len(shape) == 1 and shape[0] == 1):
        return None, None  # vectors / matrices: skip
    entry = stats[name]
    if not isinstance(entry, dict):
        return None, None
    raw_min = entry.get("min")
    raw_max = entry.get("max")
    try:
        import numpy as _np

        mn = float(_np.asarray(raw_min).flatten()[0]) if raw_min is not None else None
        mx = float(_np.asarray(raw_max).flatten()[0]) if raw_max is not None else None
    except (TypeError, ValueError, IndexError):
        return None, None
    # Reject NaN / inf — they'd break JSON serialization and confuse the GUI.
    import math

    if mn is not None and not math.isfinite(mn):
        mn = None
    if mx is not None and not math.isfinite(mx):
        mx = None
    return mn, mx


def _build_features_schema(
    features: dict,
    per_episode: set[str] | None = None,
    *,
    subtask_synthesis: bool = False,
    stats: dict | None = None,
    per_episode_source: dict[str, str] | None = None,
) -> dict[str, FeatureSchema]:
    """Convert ``ds.meta.features`` into the JSON-friendly FeatureSchema dict.

    Pre: ``features`` follows the LeRobot ``info.json`` shape — each value
    has ``dtype`` (str), ``shape`` (list/tuple of int), and optional
    ``names`` (list of str or None).
    Post: returned dict has the same keys; values are pydantic-validated
    FeatureSchema instances. Shapes are coerced to ``list[int]`` for
    JSON serialization stability.

    If ``subtask_synthesis=True`` and ``subtask_index`` is in ``features``,
    the storage entry is replaced by a synthetic ``subtask`` (string)
    entry that inherits the per-episode flag from ``subtask_index``. The
    caller passes ``subtask_synthesis=True`` only when the dataset also
    has ``meta/subtasks.parquet``.

    If ``stats`` is provided (the ``ds.meta.stats`` dict from
    ``meta/stats.json``), scalar numeric features get their dataset-wide
    observed ``(min, max)`` populated.
    """
    per_episode = per_episode or set()
    per_episode_source = per_episode_source or {}
    out: dict[str, FeatureSchema] = {}
    for name, ft in features.items():
        if subtask_synthesis and name == SUBTASK_STORAGE_FEATURE:
            # Skip the storage entry; it will be replaced by the synthetic
            # display entry below. We don't include both — the user thinks
            # in strings, so exposing both would just leak the storage name.
            continue
        shape = ft.get("shape", [])
        shape_list = [int(x) for x in shape] if shape is not None else []
        names = ft.get("names")
        if names is not None and not isinstance(names, list):
            if isinstance(names, dict):
                vals = next(iter(names.values()), None)
                names = list(vals) if isinstance(vals, list) else None
            else:
                names = None
        obs_min, obs_max = _scalar_observed_extrema(stats, name, shape_list)
        # Declared bounds: optional ``min`` / ``max`` keys in info.json — coerce
        # to JSON-friendly float and tolerate missing/non-numeric values silently.
        decl_min = _coerce_optional_float(ft.get("min"))
        decl_max = _coerce_optional_float(ft.get("max"))
        # Per-episode: declared hint in the feature spec wins over inference,
        # in BOTH directions (explicit false also overrides inferred-true).
        # This matters for freshly-added per-frame features whose constant
        # initial fill would otherwise look like uniform-per-episode and
        # mis-coerce range edits to the whole episode.
        is_per_ep = bool(ft["per_episode"]) if "per_episode" in ft else name in per_episode
        out[name] = FeatureSchema(
            dtype=str(ft.get("dtype", "")),
            shape=shape_list,
            names=names,
            is_per_episode=is_per_ep or name in per_episode,
            per_episode_source=per_episode_source.get(name),
            observed_min=obs_min,
            observed_max=obs_max,
            declared_min=decl_min,
            declared_max=decl_max,
        )

    if subtask_synthesis and SUBTASK_STORAGE_FEATURE in features:
        # Per-episode flag transfers from the storage feature: if every episode
        # had uniform subtask_index, every episode also has a uniform subtask
        # string, so edits should still coerce to whole-episode.
        out[SUBTASK_DISPLAY_FEATURE] = FeatureSchema(
            dtype="string",
            shape=[1],
            names=None,
            is_per_episode=SUBTASK_STORAGE_FEATURE in per_episode,
            per_episode_source=per_episode_source.get(SUBTASK_STORAGE_FEATURE),
        )
    return out


def _dataset_info_from(
    dataset_id: str,
    dataset,
    *,
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
) -> DatasetInfo:
    """Build a DatasetInfo response from an opened LeRobotDataset.

    Pre: ``dataset.meta`` is fully loaded (call ``ensure_episodes_loaded``
    first if opening). Post: returns a DatasetInfo with both the legacy
    ``features`` name list and the full ``features_schema`` mapping.
    """
    per_episode = _detect_per_episode_features(dataset_id, dataset)
    per_episode_source = dict(_per_episode_source_cache.get(dataset_id, {}))
    pe_warnings = _per_episode_warnings_for(dataset_id)
    # Merge declared-but-inconsistent warnings into the per-call ``warnings``
    # list so the GUI surfaces them alongside any other open-time issues.
    warnings_out = list(warnings or []) + pe_warnings
    # Synthesize the user-facing "subtask" string feature only when the
    # dataset has BOTH the storage column AND the lookup table — an
    # incomplete dataset (one but not the other) would not let us decode.
    subtask_synthesis = SUBTASK_STORAGE_FEATURE in dataset.meta.features and _has_subtask_lookup(dataset)
    feature_names = list(dataset.meta.features.keys())
    if subtask_synthesis:
        # Mirror the schema synthesis in the legacy `features: list[str]` field
        # so frontends that read that list see "subtask" instead of "subtask_index".
        feature_names = [
            SUBTASK_DISPLAY_FEATURE if n == SUBTASK_STORAGE_FEATURE else n for n in feature_names
        ]
    return DatasetInfo(
        id=dataset_id,
        repo_id=dataset.repo_id,
        root=str(dataset.root),
        total_episodes=dataset.meta.total_episodes,
        total_frames=dataset.meta.total_frames,
        fps=dataset.fps,
        robot_type=getattr(dataset.meta, "robot_type", "") or "",
        camera_keys=list(dataset.meta.camera_keys),
        features=feature_names,
        features_schema=_build_features_schema(
            dataset.meta.features,
            per_episode=per_episode,
            subtask_synthesis=subtask_synthesis,
            stats=getattr(dataset.meta, "stats", None),
            per_episode_source=per_episode_source,
        ),
        errors=errors or [],
        warnings=warnings_out,
    )


class EpisodeActionStats(BaseModel):
    """Per-component summary statistics for an episode's recorded action.

    These mirror the per-episode stats LeRobot already stores in
    ``meta/episodes/*.parquet`` under ``stats/action/{min,max,mean,std}``,
    pre-computed at record time — we just expose them in the per-episode
    response so consumers (GUI, tests, future tooling) can apply quality
    heuristics generically rather than each one re-scanning the data
    parquet. Examples of derivable checks:

    * ``all-zero``  → ``max(|min|) == 0 AND max(|max|) == 0``
    * ``static``    → ``max(std) == 0`` (action never changed)
    * ``saturated`` → some component pinned to action-space bounds
    * ``jittery``   → ``mean(std)`` unusually high relative to peers

    None of those checks are baked into the API — the GUI decides which
    visual treatment to apply, based on these raw characteristics.
    """

    min: list[float]
    max: list[float]
    mean: list[float]
    std: list[float]


class EpisodeInfo(BaseModel):
    """Summary info about an episode."""

    episode_index: int
    length: int
    duration_s: float
    task: str | None = None
    video_extra_frames: int = 0  # Frame count difference (positive=extra, negative=missing)
    video_length: int = 0  # Total video frame count (0 = same as length)
    action_stats: EpisodeActionStats | None = None
    # Per-component action stats from dataset metadata (pre-computed at
    # record time). None when the dataset has no action feature, or stats
    # aren't present in the episode metadata (older / partially-built
    # datasets). Consumers derive quality flags from these — see
    # EpisodeActionStats docs.


@router.get("")
async def list_datasets() -> list[DatasetInfo]:
    """List all currently opened datasets."""
    return [_dataset_info_from(dataset_id, ds) for dataset_id, ds in _app_state.datasets.items()]


@router.post("")
async def open_dataset(request: OpenDatasetRequest) -> DatasetInfo:
    """Open a dataset by repo_id or local path."""
    import datasets as hf_datasets

    from lerobot.datasets.io_utils import load_episodes
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    try:
        if request.local_path:
            # Open local dataset
            local_path = Path(request.local_path)
            if not local_path.exists():
                raise HTTPException(status_code=404, detail=f"Path not found: {request.local_path}")

            dataset_id = str(local_path)

            # Check if dataset is already open - return existing instance
            # This is important after edits: the existing instance has fresh data,
            # while creating a new instance might load stale cached Arrow files
            if dataset_id in _app_state.datasets:
                # Check for metadata changes (e.g. new episodes recorded externally)
                _check_and_reload_metadata(dataset_id)
                dataset = _app_state.datasets[dataset_id]
                logger.info(
                    f"Returning existing dataset: {dataset_id} ({dataset.meta.total_episodes} episodes)"
                )
                return _dataset_info_from(dataset_id, dataset)

            # Extract repo_id from path: if under HF_LEROBOT_HOME, use owner/name
            repo_id = request.repo_id
            if not repo_id:
                try:
                    rel = local_path.relative_to(HF_LEROBOT_HOME)
                    parts = rel.parts
                except ValueError:
                    # Not under HF_LEROBOT_HOME — best-effort: use the last two
                    # path components as owner/name. Matches the canonical
                    # "<root>/<owner>/<name>" layout that mirrors HF.
                    parts = local_path.parts[-2:]
                repo_id = f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else local_path.name

            # Editor is local-only by default: surface incomplete on-disk
            # caches as a 409 so the frontend can ask the user to confirm.
            # On confirm (confirm_hub_sync=True), we skip the pre-check and let
            # LeRobotDataset.__init__ pull the missing files via snapshot_download
            # into local_path — same code path the existing /hub/download
            # endpoint uses.
            if not request.confirm_hub_sync:
                kind, problems = _check_local_dataset_complete(local_path)
                if kind != "complete":
                    # Carry `kind` so the frontend can say what's wrong and only
                    # offer a Hub download when it's actually a missing-files
                    # case; a metadata inconsistency isn't a download problem.
                    hub_sync_available = kind == "missing_files"
                    raise HTTPException(
                        status_code=409,
                        detail={
                            "code": "incomplete_local_cache",
                            "kind": kind,
                            "message": (
                                "Local dataset cache is incomplete."
                                if hub_sync_available
                                else "Local dataset metadata is inconsistent."
                            ),
                            "problems": problems,
                            "repo_id": repo_id,
                            "local_path": str(local_path),
                            "hub_sync_available": hub_sync_available,
                        },
                    )

            # Disable HuggingFace caching to ensure fresh data is loaded
            # This is important for datasets that may have been edited
            hf_datasets.disable_caching()
            try:
                dataset = LeRobotDataset(repo_id, root=local_path)
            except Exception as e:
                # LeRobotDataset.__init__ tries to download from Hub when
                # cached data doesn't match info.json (episode count mismatch).
                # For local datasets this is wrong — surface the real issue.
                err_msg = str(e)
                if "Repository Not Found" in err_msg or "doesn't contain all requested episodes" in err_msg:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "Dataset metadata is inconsistent: info.json episode/frame counts "
                            "don't match the actual parquet data. This usually means episodes "
                            "were added or removed without updating info.json. "
                            "Run dataset verification to identify and repair the issue."
                        ),
                    ) from e
                raise
            finally:
                hf_datasets.enable_caching()

        elif request.repo_id:
            dataset_id = request.repo_id

            # Check if dataset is already open
            if dataset_id in _app_state.datasets:
                _check_and_reload_metadata(dataset_id)
                dataset = _app_state.datasets[dataset_id]
                logger.info(
                    f"Returning existing dataset: {dataset_id} ({dataset.meta.total_episodes} episodes)"
                )
                return _dataset_info_from(dataset_id, dataset)

            # Open from HuggingFace Hub
            dataset = LeRobotDataset(request.repo_id)
        else:
            raise HTTPException(status_code=400, detail="Must provide either repo_id or local_path")

        # Ensure episodes are loaded
        from lerobot.gui.dataset_reload import ensure_episodes_loaded

        ensure_episodes_loaded(dataset)

        # Check and repair episode metadata indices if needed
        from lerobot.datasets.dataset_tools import repair_episode_indices, verify_dataset

        errors: list[str] = []
        warnings: list[str] = []

        try:
            repaired = repair_episode_indices(dataset.root)
        except PermissionError as e:
            # Read-only dataset (e.g. backup directory). Skip repair, surface a
            # warning, and let the user open the dataset for inspection.
            logger.warning(f"Skipping episode index repair: {e}")
            warnings.append(f"Episode index repair skipped (dataset is read-only): {e}")
            repaired = 0
        if repaired > 0:
            logger.info(f"Repaired {repaired} episode indices with incorrect dataset_from_index")
            dataset.meta.episodes = load_episodes(dataset.root)
            warnings.append(f"Repaired {repaired} episode indices with incorrect metadata")

        # Verify dataset integrity
        verification = verify_dataset(dataset.root, check_videos=False, verbose=False)
        if not verification.is_valid:
            for err in verification.errors:
                logger.warning(f"Dataset verification ERROR: {err.message}")
                errors.append(err.message)
        for warn in verification.warnings:
            logger.warning(f"Dataset verification warning: {warn.message}")
            warnings.append(warn.message)
        if verification.is_valid and not verification.warnings:
            logger.info("Dataset verification passed with no errors")

        # Crash-recovery: sweep any orphan .tmp files left from an
        # interrupted schema-add or value-edit save before serving the
        # dataset. Fast (one rglob over data/ + meta/) and almost always
        # a no-op on healthy datasets.
        try:
            from lerobot.datasets.dataset_tools import _sweep_orphan_tmp_shards

            removed = _sweep_orphan_tmp_shards(dataset.root)
            if removed:
                logger.info(f"Cleaned {removed} orphan .tmp file(s) for {dataset_id}")
        except Exception as e:
            logger.warning(f"Orphan .tmp sweep failed for {dataset_id}: {e}")

        # Store in app state
        _app_state.datasets[dataset_id] = dataset

        # Track metadata mtime for auto-reload detection
        info_file = Path(dataset.root) / "meta" / "info.json"
        if info_file.exists():
            _dataset_info_mtime[dataset_id] = info_file.stat().st_mtime

        # Load any persisted pending edits from disk
        from lerobot.gui.state import load_edits_from_file

        persisted_edits = load_edits_from_file(dataset.root, dataset_id)
        for edit in persisted_edits:
            _app_state.add_edit(edit)
        if persisted_edits:
            logger.info(f"Restored {len(persisted_edits)} pending edits from disk")

        logger.info(f"Opened dataset: {dataset_id} ({dataset.meta.total_episodes} episodes)")
        _save_opened_state()

        return _dataset_info_from(dataset_id, dataset, errors=errors, warnings=warnings)

    except HTTPException:
        # Preserve intentional HTTP responses (e.g. 400 / 409) — don't wrap
        # them in a 500 with stringified detail.
        raise
    except Exception as e:
        logger.exception(f"Failed to open dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ── Schema-add endpoints (in-place column add) ─────────────────────────


def _migrate_next_success_inplace(dataset) -> None:
    """Collapse per-frame ``next.success`` (bool) into per-episode ``success`` (int8 tri-state).

    ``lerobot-eval`` rollouts on ``main`` write ``next.success`` as a per-frame
    boolean column — typically all-False except the success-transition frame.
    The new ``success`` feature is per-episode int8 tri-state (-1/0/+1). The
    rename machinery only handles same-dtype renames, so this is a small
    bespoke migration.

    Mapping:
      * any frame with ``next.success == True`` → +1 (success)
      * all frames False                        → -1 (failure)

    Then ``next.success`` is dropped. info.json + per-episode stats are
    updated by the underlying ``add_features_inplace`` /
    ``remove_features_inplace`` primitives.

    Pre: ``next.success`` exists in ``dataset.meta.features`` with bool dtype,
    and ``success`` does NOT exist. Caller (``add_default_features``) gates on
    these. No-op for empty datasets — the underlying primitives raise.
    """
    import os

    import numpy as np

    from lerobot.datasets.dataset_tools import (
        _add_new_feature_stats_to_episodes,
        add_features_inplace,
        remove_features_inplace,
    )
    from lerobot.datasets.utils import DATA_DIR

    work_root = Path(dataset.root)
    data_dir = work_root / DATA_DIR
    parquet_files = sorted(data_dir.glob("*/*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    # ── Pass 1: compute per-episode tri-state (any-True → +1, else -1) ──
    # next.success may be stored as scalar bool or shape-[1] list-of-bool.
    # Coerce element extraction so both layouts collapse the same way.
    def _scalarize(v: Any) -> bool:
        if hasattr(v, "__len__") and not isinstance(v, (bytes, str)):
            return bool(v[0]) if len(v) else False
        return bool(v)

    per_episode_value: dict[int, int] = {}
    for shard in parquet_files:
        df = pd.read_parquet(shard, columns=["episode_index", "next.success"])
        for ep_idx, group in df.groupby("episode_index"):
            ep_idx_int = int(ep_idx)
            if per_episode_value.get(ep_idx_int) == 1:
                continue  # already a success — multi-shard episode, no need to recheck
            any_true = any(_scalarize(v) for v in group["next.success"].tolist())
            per_episode_value[ep_idx_int] = 1 if any_true else -1

    # ── Pass 2: add `success` int8[1] column with fill=0 ────────────────
    # Reuses existing primitive so info.json + episode stats get the right
    # entries (per_episode=True hint is preserved into info.json).
    add_features_inplace(
        dataset,
        features={
            "success": (
                0,
                {"dtype": "int8", "shape": [1], "names": None, "per_episode": True},
            )
        },
        recompute_stats=False,  # we'll overwrite the column in pass 3 — stats
        # would be wrong if computed off the all-zeros fill
    )

    # ── Pass 3: rewrite each shard's `success` column from per_episode_value ──
    pending_renames: list[tuple[Path, Path]] = []
    try:
        for shard in parquet_files:
            df = pd.read_parquet(shard)
            mapped = df["episode_index"].map(lambda i: per_episode_value.get(int(i), 0))
            df["success"] = pd.Series(mapped.to_numpy(), dtype=np.dtype("int8"))
            tmp = shard.with_suffix(shard.suffix + ".tmp")
            df.to_parquet(tmp, compression="snappy", index=False)
            pending_renames.append((tmp, shard))
    except Exception:
        for tmp, _ in pending_renames:
            if tmp.exists():
                # safe-destruct: our own .tmp file we just wrote in this function
                tmp.unlink()
        raise
    for tmp, final in pending_renames:
        os.replace(tmp, final)

    # ── Pass 4: recompute stats for `success` from the rewritten column ──
    # add_features_inplace was called with recompute_stats=False, so stats
    # are absent. Computing here (after pass 3 wrote correct values) avoids
    # stale stats from the all-zeros initial fill. The helper overwrites
    # existing stats columns; we wrote none, so it just adds them.
    _add_new_feature_stats_to_episodes(work_root, {"success": dataset.meta.features["success"]})

    # ── Pass 5: drop next.success (drops shard column + stats + info.json) ──
    remove_features_inplace(dataset, ["next.success"])


# Default features the GUI offers to add via the banner. Any keys the user's
# dataset is missing get appended in a single ``add_features_inplace`` call;
# this is the convenience layer on top of the generic POST .../features.
_DEFAULT_FEATURE_SPECS = {
    "reward": {
        "fill_value": 0.0,
        # per_episode=false declared explicitly so that the constant 0.0
        # initial fill doesn't get mis-inferred as a per-episode broadcast
        # column. Otherwise drag-select range edits would coerce to the
        # full episode for reward, which is the opposite of what users want.
        "info": {"dtype": "float32", "shape": [1], "names": None, "per_episode": False},
        # Existing column names that should be preserved (renamed, not
        # discarded) when present in the dataset. Renaming requires the
        # source column to be type-compatible with the destination spec
        # (same dtype + shape after spec_overrides applied).
        "rename_from": ["next.reward"],
    },
    "success": {
        "fill_value": 0,
        "info": {"dtype": "int8", "shape": [1], "names": None, "per_episode": True},
        # next.success (bool, per-frame, written by lerobot-eval rollouts)
        # can't be a plain rename target — dtype + cardinality both differ
        # (per-frame bool → per-episode int8 tri-state). Migration is
        # handled separately by _migrate_next_success_inplace, invoked from
        # add_default_features before the rename/add planning loop.
        "rename_from": [],
    },
}


class AddFeatureRequest(BaseModel):
    """Body for ``POST /api/datasets/{id}/features``.

    The ``fill_value`` is auto-typed by ``add_features_inplace`` based on
    ``dtype``; bool/int/float strings sent from JS get coerced. Only V1
    dtypes are currently editable in the GUI: bool, int8, int64, float32,
    string. Vector / image / video features can be added but won't have
    an editable widget yet.
    """

    name: str
    dtype: str
    shape: list[int] = [1]
    per_episode: bool = False
    fill_value: Any = 0


class AddFeatureResponse(BaseModel):
    added: list[str]
    info: DatasetInfo
    # Optional: renames performed (when /features/defaults reuses an
    # existing column under a different name instead of adding a duplicate).
    # Each entry is "<old_name>→<new_name>". Empty for the generic
    # POST /features endpoint.
    renamed: list[str] = []


def _refresh_dataset_after_schema_change(dataset_id: str) -> None:
    """Drop schema-bound caches after a schema mutation.

    Keeps in-memory state in sync with the new ``info.json`` / parquet
    contents. ``_dataset_info_mtime`` is popped so the next dataset-open
    call re-detects the schema; ``_per_episode_features_cache`` and
    ``_episode_start_indices`` are cleared via the existing helper.
    ``add_features_inplace`` already replaces ``dataset.meta`` in place,
    so future ``_dataset_info_from`` calls will see the new schema.
    """
    from lerobot.gui.cache_invalidation import invalidate_caches

    _dataset_info_mtime.pop(dataset_id, None)
    invalidate_caches(_app_state, dataset_id, invalidate_episode_indices=_invalidate_episode_start_indices)


@router.post("/{dataset_id:path}/features", response_model=AddFeatureResponse)
async def add_dataset_feature(dataset_id: str, body: AddFeatureRequest) -> AddFeatureResponse:
    """Add one new feature column to the dataset in place.

    Refuses (409) if any pending ``feature_set`` edits exist for the
    dataset — the user must Save or Discard them first to avoid mixing
    schema and value mutations on the same parquet shards.
    """
    from lerobot.datasets.dataset_tools import add_features_inplace
    from lerobot.utils.constants import DEFAULT_FEATURES

    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    if body.name in DEFAULT_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"'{body.name}' is a reserved DEFAULT_FEATURE",
        )
    if body.name in _DEFAULT_FEATURE_SPECS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"'{body.name}' is a default feature — use POST /features/defaults "
                "(banner) instead of the generic dialog."
            ),
        )

    pending = _app_state.pending_feature_set_edits_for_dataset(dataset_id)
    if pending:
        raise HTTPException(
            status_code=409,
            detail=f"{len(pending)} pending feature edits — Save or Discard them first",
        )

    async with _app_state.get_lock(dataset_id):
        dataset = _app_state.datasets[dataset_id]
        if body.name in dataset.meta.features:
            raise HTTPException(
                status_code=400,
                detail=f"Feature '{body.name}' already exists in dataset",
            )

        # Always write the declared per_episode value (including False) so
        # the FeatureSchema construction picks it up over the inference
        # fallback. A constant initial fill would otherwise be mis-inferred
        # as per-episode-uniform and silently coerce range edits to whole
        # episodes for non-per-episode columns.
        info = {
            "dtype": body.dtype,
            "shape": list(body.shape),
            "names": None,
            "per_episode": bool(body.per_episode),
        }

        try:
            add_features_inplace(dataset, features={body.name: (body.fill_value, info)})
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.exception(f"add_features_inplace failed for {dataset_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

        _refresh_dataset_after_schema_change(dataset_id)
        return AddFeatureResponse(added=[body.name], info=_dataset_info_from(dataset_id, dataset))


def _compatible_for_rename(existing_spec: dict, target_spec: dict) -> bool:
    """True if `existing_spec` is type-compatible with the target default spec.

    Renaming preserves the on-disk values, so the source column's dtype
    and shape must already match what the target spec declares; otherwise
    we'd silently misadvertise the data type. Optional spec keys (names,
    per_episode) are intentionally ignored — those can be overridden via
    spec_overrides during the rename.
    """
    if existing_spec.get("dtype") != target_spec.get("dtype"):
        return False
    es = list(existing_spec.get("shape") or [])
    ts = list(target_spec.get("shape") or [])
    return es == ts


@router.post("/{dataset_id:path}/features/defaults", response_model=AddFeatureResponse)
async def add_default_features(dataset_id: str) -> AddFeatureResponse:
    """Reconcile the dataset's schema against the default features.

    For each missing default:
      * if a known alternate column exists with a compatible dtype/shape
        (e.g. ``next.reward`` for ``reward``), rename it in place — the
        existing recorded values are preserved instead of being shadowed
        by a fresh all-zeros column.
      * otherwise, add a new column with the default fill value.

    Idempotent: returns ``added=[], renamed=[]`` when nothing's needed.
    Used by the banner shown on dataset open.
    """
    from lerobot.datasets.dataset_tools import add_features_inplace, rename_features_inplace

    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    pending = _app_state.pending_feature_set_edits_for_dataset(dataset_id)
    if pending:
        raise HTTPException(
            status_code=409,
            detail=f"{len(pending)} pending feature edits — Save or Discard them first",
        )

    async with _app_state.get_lock(dataset_id):
        dataset = _app_state.datasets[dataset_id]

        # Special-case migration: next.success (bool, per-frame) → success
        # (int8 tri-state, per-episode). Rename machinery rejects dtype
        # changes, so this is a bespoke transform run before the planner.
        # Gated on next.success being bool to avoid stomping unexpected dtypes.
        migrated_next_success = False
        if (
            "next.success" in dataset.meta.features
            and "success" not in dataset.meta.features
            and dataset.meta.features["next.success"].get("dtype") == "bool"
        ):
            try:
                _migrate_next_success_inplace(dataset)
                migrated_next_success = True
            except Exception as e:
                logger.exception(f"_migrate_next_success_inplace failed for {dataset_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e)) from e

        # Plan: split missing defaults into renames (alternate exists +
        # compatible) vs adds (everything else).
        renames: dict[str, str] = {}
        rename_overrides: dict[str, dict] = {}
        to_add: dict[str, tuple] = {}

        for name, spec in _DEFAULT_FEATURE_SPECS.items():
            if name in dataset.meta.features:
                continue
            target_info = dict(spec["info"])
            picked_alternate = None
            for alt in spec.get("rename_from", []):
                if alt in dataset.meta.features and _compatible_for_rename(
                    dataset.meta.features[alt], target_info
                ):
                    picked_alternate = alt
                    break
            if picked_alternate is not None:
                renames[picked_alternate] = name
                # The alternate may not declare per_episode (e.g. legacy
                # next.reward); merge in the default's hint so the
                # renamed column gets the right is_per_episode behavior.
                rename_overrides[name] = {
                    k: v for k, v in target_info.items() if k in ("per_episode", "names")
                }
            else:
                to_add[name] = (spec["fill_value"], target_info)

        if not renames and not to_add and not migrated_next_success:
            return AddFeatureResponse(added=[], renamed=[], info=_dataset_info_from(dataset_id, dataset))

        try:
            # Renames first so the new names are present before any
            # subsequent add validates against an updated schema.
            if renames:
                rename_features_inplace(dataset, renames=renames, spec_overrides=rename_overrides)
            if to_add:
                add_features_inplace(dataset, features=to_add)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.exception(f"add_default_features failed for {dataset_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

        _refresh_dataset_after_schema_change(dataset_id)
        renamed_pairs = [f"{old}→{new}" for old, new in renames.items()]
        if migrated_next_success:
            renamed_pairs.append("next.success→success")
        return AddFeatureResponse(
            added=sorted(to_add.keys()),
            renamed=sorted(renamed_pairs),
            info=_dataset_info_from(dataset_id, dataset),
        )


class RemoveFeatureResponse(BaseModel):
    removed: list[str]
    info: DatasetInfo


@router.delete(
    "/{dataset_id:path}/features/{feature_name}",
    response_model=RemoveFeatureResponse,
)
async def remove_dataset_feature(dataset_id: str, feature_name: str) -> RemoveFeatureResponse:
    """Drop one feature column from the dataset in place.

    Refuses (400) for ``DEFAULT_FEATURES`` and image/video features
    (their on-disk filenames also encode the feature name — would need
    the forked ``remove_feature`` and a video re-encode).
    Refuses (409) when pending ``feature_set`` edits exist on the
    dataset, same as the schema-add path.
    """
    from lerobot.datasets.dataset_tools import remove_features_inplace
    from lerobot.utils.constants import DEFAULT_FEATURES

    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    if feature_name in DEFAULT_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"'{feature_name}' is a reserved DEFAULT_FEATURE",
        )

    pending = _app_state.pending_feature_set_edits_for_dataset(dataset_id)
    if pending:
        raise HTTPException(
            status_code=409,
            detail=f"{len(pending)} pending feature edits — Save or Discard them first",
        )

    async with _app_state.get_lock(dataset_id):
        dataset = _app_state.datasets[dataset_id]
        if feature_name not in dataset.meta.features:
            raise HTTPException(
                status_code=404,
                detail=f"Feature '{feature_name}' not found in dataset",
            )
        try:
            remove_features_inplace(dataset, names=feature_name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.exception(f"remove_features_inplace failed for {dataset_id}/{feature_name}: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

        _refresh_dataset_after_schema_change(dataset_id)
        return RemoveFeatureResponse(removed=[feature_name], info=_dataset_info_from(dataset_id, dataset))


@router.delete("/{dataset_id:path}")
async def close_dataset(dataset_id: str) -> dict[str, str]:
    """Close a dataset."""
    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    if _app_state.is_locked(dataset_id):
        raise HTTPException(status_code=423, detail="Dataset is busy (operation in progress)")

    del _app_state.datasets[dataset_id]
    _dataset_info_mtime.pop(dataset_id, None)
    # Drop every dataset-scoped cache + per-dataset lock. Without this, each
    # close/re-open cycle left behind cached episode indices, per-episode
    # action stats, the asyncio.Lock, and any frame-cache entries — a slow
    # leak in long-running sessions and a correctness hazard if the same
    # dataset_id is later re-opened against different on-disk content.
    _invalidate_episode_start_indices(dataset_id)
    from lerobot.gui.cache_invalidation import invalidate_caches

    invalidate_caches(_app_state, dataset_id)
    _app_state.clear_edits(dataset_id)
    _app_state.discard_lock(dataset_id)
    _save_opened_state()
    logger.info(f"Closed dataset: {dataset_id}")

    return {"status": "ok", "message": f"Closed dataset: {dataset_id}"}


@router.get("/{dataset_id:path}/episodes")
async def list_episodes(dataset_id: str) -> list[EpisodeInfo]:
    """List all episodes in a dataset."""
    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    # Check if metadata changed on disk (e.g., new episodes recorded)
    _check_and_reload_metadata(dataset_id)

    dataset = _app_state.datasets[dataset_id]
    episodes = dataset.meta.episodes

    if episodes is None:
        from lerobot.datasets.io_utils import load_episodes

        episodes = load_episodes(dataset.root)
        dataset.meta.episodes = episodes

    # Per-episode action stats from dataset metadata (pre-computed at
    # record time). Cached until metadata mtime changes — the cache is
    # invalidated alongside _episode_start_indices by the metadata-reload
    # path. Returns an empty dict for datasets without an action feature
    # or pre-computed stats; consumers treat that as "unknown".
    if dataset_id not in _episode_action_stats:
        _episode_action_stats[dataset_id] = _load_episode_action_stats(Path(dataset.root))
    action_stats_by_ep = _episode_action_stats[dataset_id]

    result = []
    for i in range(dataset.meta.total_episodes):
        ep = episodes[i]
        length = ep["length"]
        duration_s = length / dataset.fps

        # Get task if available (ep["tasks"] contains task name strings directly)
        task = None
        if "tasks" in ep and ep["tasks"]:
            task = ep["tasks"][0] if len(ep["tasks"]) > 0 else None

        # Check for video-data duration mismatch (re-recording artifact or truncation)
        diff_per_cam = check_episode_video_duration(ep, dataset.fps)
        video_extra_frames = max(diff_per_cam.values(), key=abs) if diff_per_cam else 0

        # Total video frame count (matches length when no mismatch)
        video_length = length + max(0, video_extra_frames)

        result.append(
            EpisodeInfo(
                episode_index=i,
                length=length,
                duration_s=duration_s,
                task=task,
                video_extra_frames=video_extra_frames,
                video_length=video_length,
                action_stats=action_stats_by_ep.get(i),
            )
        )

    return result


@router.get("/{dataset_id:path}/episodes/{episode_idx}/frame/{frame_idx}")
async def get_frame(dataset_id: str, episode_idx: int, frame_idx: int, camera: str | None = None) -> Response:
    """Get a single frame as JPEG.

    Args:
        dataset_id: Dataset identifier
        episode_idx: Episode index
        frame_idx: Frame index within the episode
        camera: Camera key (optional, returns first camera if not specified)
    """
    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    dataset = _app_state.datasets[dataset_id]

    # Validate episode index
    if episode_idx < 0 or episode_idx >= dataset.meta.total_episodes:
        raise HTTPException(status_code=404, detail=f"Episode not found: {episode_idx}")

    # Get episode metadata
    episodes = dataset.meta.episodes
    if episodes is None:
        from lerobot.datasets.io_utils import load_episodes

        episodes = load_episodes(dataset.root)
        dataset.meta.episodes = episodes

    ep = episodes[episode_idx]
    ep_length = ep["length"]

    # Compute video length for episodes with extra video frames
    diff_per_cam = check_episode_video_duration(ep, dataset.fps)
    video_extra = max(diff_per_cam.values(), key=abs) if diff_per_cam else 0
    video_length = ep_length + video_extra if video_extra > 0 else ep_length

    # Validate frame index (allow up to video_length for flagged episodes)
    if frame_idx < 0 or frame_idx >= video_length:
        raise HTTPException(
            status_code=404,
            detail=f"Frame not found: {frame_idx} (episode has {ep_length} data frames, {video_length} video frames)",
        )

    # Determine camera key
    camera_keys = list(dataset.meta.camera_keys)
    if not camera_keys:
        raise HTTPException(status_code=400, detail="Dataset has no camera/image keys")

    if camera:
        if camera not in camera_keys:
            raise HTTPException(
                status_code=400, detail=f"Camera not found: {camera}. Available: {camera_keys}"
            )
        camera_key = camera
    else:
        camera_key = camera_keys[0]

    import asyncio
    import time

    # Check if this camera is already cached (cheap lock-protected dict lookup).
    jpeg_bytes = _app_state.frame_cache.get(dataset_id, episode_idx, frame_idx, camera_key)

    if jpeg_bytes is None:
        # Cache miss: do the heavy decode+encode work off the event loop.
        # Otherwise every scrub on a long video stalls FastAPI's loop and
        # cascades into stuck SSE keepalives + delayed concurrent requests.
        from lerobot.gui.frame_cache import encode_frame_to_jpeg

        def _decode_and_cache() -> bytes:
            # Re-check the JPEG cache inside the worker. Multiple browser
            # requests for the same frame (one per camera) all hit cache-
            # miss outside, all submit to this single-worker executor, and
            # the first one to run decodes ALL cameras and caches them.
            # The 2nd .. Nth submissions wake up, find the cache populated,
            # and return immediately without redundant decode work.
            cached = _app_state.frame_cache.get(dataset_id, episode_idx, frame_idx, camera_key)
            if cached is not None:
                return cached

            if frame_idx < ep_length:
                # Normal frame — decode via dataset[global_idx] (all cameras at once)
                episode_start = _get_episode_start_index(dataset_id, episode_idx)
                global_idx = episode_start + frame_idx

                t0 = time.perf_counter()
                item = dataset[global_idx]
                t1 = time.perf_counter()

                # Cache all cameras from this single decode
                primary: bytes | None = None
                for cam in camera_keys:
                    if cam in item:
                        cam_jpeg = encode_frame_to_jpeg(item[cam])
                        _app_state.frame_cache.put(dataset_id, episode_idx, frame_idx, cam, cam_jpeg)
                        if cam == camera_key:
                            primary = cam_jpeg
                t2 = time.perf_counter()

                if primary is None:
                    # Fallback when the requested camera isn't in camera_keys.
                    primary = encode_frame_to_jpeg(item[camera_key])
                    _app_state.frame_cache.put(dataset_id, episode_idx, frame_idx, camera_key, primary)
            else:
                # Extra video frame beyond data length — decode directly from video file
                from lerobot.datasets.video_utils import decode_video_frames_torchcodec

                fps = dataset.fps
                from_ts = ep.get(f"videos/{camera_key}/from_timestamp", 0.0)
                timestamp = from_ts + frame_idx / fps
                tolerance_s = 1 / fps * 0.7

                video_path = dataset.root / dataset.meta.get_video_file_path(episode_idx, camera_key)

                t0 = time.perf_counter()
                frames = decode_video_frames_torchcodec(video_path, [timestamp], tolerance_s)
                t1 = time.perf_counter()

                primary = encode_frame_to_jpeg(frames[0])
                _app_state.frame_cache.put(dataset_id, episode_idx, frame_idx, camera_key, primary)
                t2 = time.perf_counter()

            decode_ms = (t1 - t0) * 1000
            encode_ms = (t2 - t1) * 1000
            logger.info(
                f"get_frame ep={episode_idx} frame={frame_idx} cam={camera_key}: "
                f"decode={decode_ms:.1f}ms encode={encode_ms:.1f}ms"
            )
            return primary

        jpeg_bytes = await asyncio.get_event_loop().run_in_executor(_decode_executor, _decode_and_cache)
    else:
        logger.debug(f"get_frame ep={episode_idx} frame={frame_idx} cam={camera_key}: cache hit")

    # Trigger background prefetching for this episode, starting from the current frame
    _maybe_start_prefetch(dataset_id, episode_idx, ep_length, start_frame=min(frame_idx, ep_length - 1))

    # Prevent browser caching - frames may change after edits
    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


def _coerce_feature_value_to_json(value: Any, dtype: str) -> Any:
    """Convert a single sample's feature value into JSON-serializable form.

    Pre: ``value`` comes from ``dataset[i][name]`` — typically a torch.Tensor,
    np.ndarray, str, bool, int, or float.
    Post: returns a JSON-compatible Python type (bool / int / float / str / list).
    Image / video tensors are NEVER passed in (caller must skip those features).
    """
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            scalar = value.item()
            return bool(scalar) if dtype == "bool" else scalar
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return list(value)
    # Pass-through for str / int / float / bool. Anything else falls back to str()
    # so the response stays JSON-encodable (better than failing the whole frame).
    if isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


@router.get("/{dataset_id:path}/episodes/{episode_idx}/frame/{frame_idx}/features")
async def get_frame_features(dataset_id: str, episode_idx: int, frame_idx: int) -> dict[str, Any]:
    """Return all per-frame feature values at a single frame.

    Skips ``image`` / ``video`` features (those have a dedicated frame
    endpoint). Returns JSON-serializable values: scalars for shape ``[1]``,
    lists for vectors, strings for ``string`` features. ``subtask_index``
    and the decoded ``subtask`` string both appear when the dataset has a
    subtasks lookup table.
    """
    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    dataset = _app_state.datasets[dataset_id]

    if episode_idx < 0 or episode_idx >= dataset.meta.total_episodes:
        raise HTTPException(status_code=404, detail=f"Episode not found: {episode_idx}")

    ep = dataset.meta.episodes[episode_idx]
    ep_length = ep["length"]
    if frame_idx < 0 or frame_idx >= ep_length:
        raise HTTPException(
            status_code=404,
            detail=f"Frame {frame_idx} out of range for episode {episode_idx} (length={ep_length})",
        )

    episode_start = _get_episode_start_index(dataset_id, episode_idx)
    global_idx = episode_start + frame_idx
    item = dataset[global_idx]

    # Filter out image/video features — those have their own frame endpoints.
    skip_dtypes = {"image", "video"}
    out: dict[str, Any] = {}
    for name, ft in dataset.meta.features.items():
        dtype = ft.get("dtype", "")
        if dtype in skip_dtypes:
            continue
        if name not in item:
            # Decoder can drop columns it doesn't know how to load; skip rather than error.
            continue
        try:
            out[name] = _coerce_feature_value_to_json(item[name], dtype)
        except Exception as e:
            logger.warning(f"Failed to coerce feature {name!r} at frame {frame_idx}: {e}")
            # Don't fail the whole response over one bad cell.
            out[name] = None

    # ``dataset[i]`` includes both ``task`` (decoded string) and ``task_index``
    # automatically; same for ``subtask`` / ``subtask_index`` when the lookup
    # table is present. They flow through the loop above.
    return {"frame_index": frame_idx, "episode_index": episode_idx, "values": out}


def _resolve_dataset_urdf_spec(dataset_id: str, episode_idx: int):
    """Validate args + resolve the URDF viz spec from this dataset's schema.

    Returns ``(dataset, episode_row, state_names, action_names_or_None, spec)``
    or ``None`` when no description matches the dataset's motor set.
    Raises 404 ``HTTPException`` for unknown dataset / episode (everything
    else returns ``None`` to surface as ``{"available": false}``).
    """
    from lerobot.gui.urdf_viz import resolve_robot

    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    dataset = _app_state.datasets[dataset_id]
    if episode_idx < 0 or episode_idx >= dataset.meta.total_episodes:
        raise HTTPException(status_code=404, detail=f"Episode not found: {episode_idx}")
    ep = dataset.meta.episodes[episode_idx]

    feature_dict = dataset.meta.features
    state_feat = feature_dict.get("observation.state")
    if state_feat is None or "names" not in state_feat:
        return None
    state_names: list[str] = list(state_feat["names"])
    spec = resolve_robot(state_names)
    if spec is None:
        return None
    action_feat = feature_dict.get("action")
    action_names = list(action_feat["names"]) if action_feat and "names" in action_feat else None
    return dataset, ep, state_names, action_names, spec


@router.get("/{dataset_id:path}/episodes/{episode_idx}/urdf-viz/meta")
async def get_urdf_viz_dataset_meta(dataset_id: str, episode_idx: int) -> dict:
    """One-shot identity + advertised sources for a dataset's URDF viewer.

    Sibling of the live ``/api/run/urdf-viz/meta``. ``sources`` lists what
    the dataset's schema can serve — ``"state"`` whenever ``observation.state``
    is present and matches a vendored description; ``"action"`` when an
    ``action`` feature is also there. The frontend fetches this once when
    the viewer mounts, then polls per-source via the sibling endpoint.
    """
    resolved = _resolve_dataset_urdf_spec(dataset_id, episode_idx)
    if resolved is None:
        return {"available": False}
    _, _, _, action_names, spec = resolved
    sources = ["state"]
    if action_names:
        sources.append("action")
    return {
        "available": True,
        "name": spec.name,
        "urdf": f"/urdf-assets/{spec.urdf_url_path}",
        "bimanual": len(spec.arms) == 2,
        "sources": sources,
        # ee_link is None for descriptions that didn't declare one; the
        # frontend skips polyline rendering in that case.
        "ee_link": spec.ee_link,
    }


@router.get("/{dataset_id:path}/episodes/{episode_idx}/urdf-viz")
async def get_urdf_viz_dataset_source(
    dataset_id: str,
    episode_idx: int,
    frame: int = 0,
    source: str = "state",
    horizon: int = 1,
) -> dict:
    """Per-arm URDF joint angles for one named source at one frame.

    Mirrors the live endpoint's shape (``arms[*].frames[]``). When
    ``horizon`` is 1 (the default) ``frames`` has length 1 — the single
    recorded pose at ``frame``. With ``horizon`` > 1 the endpoint returns
    up to ``horizon`` consecutive frames starting at ``frame`` (clipped at
    the end of the episode), enabling the frontend to draw an EE-trajectory
    polyline through the future poses. ``source`` is one of the names from
    ``/urdf-viz/meta``.
    """
    from lerobot.gui.urdf_viz import compute_joint_angles

    if horizon < 1:
        raise HTTPException(status_code=400, detail=f"horizon must be >= 1 (got {horizon})")

    resolved = _resolve_dataset_urdf_spec(dataset_id, episode_idx)
    if resolved is None:
        return {"available": False}
    dataset, ep, state_names, action_names, spec = resolved

    ep_length = int(ep["length"])
    if frame < 0 or frame >= ep_length:
        raise HTTPException(
            status_code=404,
            detail=f"Frame {frame} out of range for episode {episode_idx} (length={ep_length})",
        )

    if source == "state":
        col, names = "observation.state", state_names
    elif source == "action":
        if not action_names:
            return {"available": False}
        col, names = "action", action_names
    else:
        raise HTTPException(status_code=400, detail=f"unknown source: {source!r}")

    # Pull only the column we need, scoped to this episode. Mirrors the
    # feature-series endpoint: avoids decoding any video features the full
    # ``dataset[i]`` accessor would pull in.
    chunk_idx = int(ep["data/chunk_index"])
    file_idx = int(ep["data/file_index"])
    parquet_path = Path(dataset.root) / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
    if not parquet_path.exists():
        raise HTTPException(status_code=404, detail=f"Data parquet missing: {parquet_path}")
    df = pd.read_parquet(parquet_path, columns=["episode_index", col])
    df = df[df["episode_index"] == episode_idx].reset_index(drop=True)
    if frame >= len(df):
        raise HTTPException(
            status_code=404,
            detail=f"Frame {frame} not present in parquet (have {len(df)} rows)",
        )

    # Clip the horizon at the end of the available rows. The renderer is
    # fine with shorter-than-requested trajectories (just a shorter line).
    end = min(frame + horizon, len(df))
    arms_payload: list[dict] = [{"prefix": a.obs_prefix, "frames": []} for a in spec.arms]
    for i in range(frame, end):
        vec = df[col].iloc[i]
        sample = {n: float(vec[j]) for j, n in enumerate(names) if j < len(vec)}
        angles = compute_joint_angles(spec, sample)
        for k, a in enumerate(spec.arms):
            arms_payload[k]["frames"].append({"joints": angles.get(a.obs_prefix, {})})
    return {"available": True, "arms": arms_payload}


@router.get("/{dataset_id:path}/episodes/{episode_idx}/feature-series")
async def get_episode_feature_series(
    dataset_id: str,
    episode_idx: int,
    features: str = "",
) -> dict[str, Any]:
    """Return the per-frame trajectory of one or more features for an episode.

    The frontend uses this to render line / band / stripe rows under the timeline.
    Image / video features are rejected (they have their own video decode path).

    Pre: ``features`` is a comma-separated list of feature names. Empty / omitted
    means "all non-image, non-video features the dataset declares".
    Post: response shape is ``{episode_index, length, series: {name: [v0..v_{N-1}]}}``.
    For ``task`` / ``subtask``, the decoded string is returned per frame (matching
    what ``dataset[i]`` would yield) — backed by the dataset's lookup table.
    """
    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    dataset = _app_state.datasets[dataset_id]
    if episode_idx < 0 or episode_idx >= dataset.meta.total_episodes:
        raise HTTPException(status_code=404, detail=f"Episode not found: {episode_idx}")

    requested = [name.strip() for name in features.split(",") if name.strip()] if features else None

    # Resolve which raw columns we need to read from parquet, and which
    # decoded views (task / subtask) the caller wants on top of those.
    feature_dict = dataset.meta.features
    skip_dtypes = {"image", "video"}

    # Synthetic columns: not in features dict, but materialized by dataset[i].
    has_subtasks = getattr(dataset.meta, "subtasks", None) is not None and "subtask_index" in feature_dict
    synthetic_decoded = {"task": "task_index"}
    if has_subtasks:
        synthetic_decoded["subtask"] = "subtask_index"

    if requested is None:
        # Default: every per-frame feature except image/video, plus the synthetic
        # decoded views the dataset supports.
        raw_cols: list[str] = [
            name for name, ft in feature_dict.items() if ft.get("dtype") not in skip_dtypes
        ]
        decoded_cols: list[str] = [name for name in synthetic_decoded if synthetic_decoded[name] in raw_cols]
    else:
        raw_cols = []
        decoded_cols = []
        for name in requested:
            if name in synthetic_decoded:
                # caller asked for "task"/"subtask" — read the *_index column and decode.
                if synthetic_decoded[name] in feature_dict:
                    raw_cols.append(synthetic_decoded[name])
                    decoded_cols.append(name)
                continue
            if name not in feature_dict:
                raise HTTPException(status_code=400, detail=f"Unknown feature: {name!r}")
            if feature_dict[name].get("dtype") in skip_dtypes:
                raise HTTPException(
                    status_code=400,
                    detail=f"Feature {name!r} is image/video — fetch via /frame/{{idx}} instead.",
                )
            raw_cols.append(name)

    ep = dataset.meta.episodes[episode_idx]
    chunk_idx = int(ep["data/chunk_index"])
    file_idx = int(ep["data/file_index"])
    parquet_path = Path(dataset.root) / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)

    if not parquet_path.exists():
        raise HTTPException(status_code=404, detail=f"Data parquet missing: {parquet_path}")

    # Always pull episode_index too so we can slice in-memory. Avoids reading the whole
    # shard's worth of columns when we only need a slice — though for 1 episode we read
    # the matching rows only.
    cols_to_read = list(dict.fromkeys(["episode_index", *raw_cols]))
    df = pd.read_parquet(parquet_path, columns=cols_to_read)
    df = df[df["episode_index"] == episode_idx].reset_index(drop=True)

    series: dict[str, list[Any]] = {}
    for name in raw_cols:
        col = df[name].tolist() if name in df.columns else []
        # Pandas keeps numpy arrays for vector cells. Coerce per-row.
        series[name] = [_coerce_feature_value_to_json(v, feature_dict[name].get("dtype", "")) for v in col]

    # Decode synthetic columns ("task", "subtask") from the underlying *_index series.
    for decoded_name in decoded_cols:
        idx_col = synthetic_decoded[decoded_name]
        lookup = dataset.meta.tasks if decoded_name == "task" else dataset.meta.subtasks
        if lookup is None:
            continue
        idx_values = series.get(idx_col, [])
        try:
            series[decoded_name] = [lookup.iloc[int(i)].name for i in idx_values]
        except Exception as e:
            logger.warning(f"Failed to decode {decoded_name} via {idx_col}: {e}")
            series[decoded_name] = [None] * len(idx_values)

    return {
        "episode_index": episode_idx,
        "length": int(ep["length"]),
        "series": series,
    }


@router.get("/{dataset_id:path}/episodes/{episode_idx}/frames")
async def get_frames_batch(
    dataset_id: str,
    episode_idx: int,
    start: int = 0,
    count: int = 10,
    camera: str | None = None,
) -> dict[str, Any]:
    """Get multiple frames as base64-encoded JPEGs.

    Args:
        dataset_id: Dataset identifier
        episode_idx: Episode index
        start: Starting frame index
        count: Number of frames to return (max 100)
        camera: Camera key (optional)

    Returns:
        Dict with frame data and metadata
    """
    import base64

    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    dataset = _app_state.datasets[dataset_id]
    count = min(count, 100)  # Limit batch size

    # Get episode metadata
    episodes = dataset.meta.episodes
    if episodes is None:
        from lerobot.datasets.io_utils import load_episodes

        episodes = load_episodes(dataset.root)
        dataset.meta.episodes = episodes

    ep = episodes[episode_idx]
    ep_length = ep["length"]

    # Determine camera key
    camera_keys = list(dataset.meta.camera_keys)
    if camera:
        if camera not in camera_keys:
            raise HTTPException(status_code=400, detail=f"Camera not found: {camera}")
        camera_key = camera
    else:
        camera_key = camera_keys[0] if camera_keys else None

    if not camera_key:
        raise HTTPException(status_code=400, detail="No camera available")

    # Collect frames
    import asyncio

    from lerobot.gui.frame_cache import encode_frame_to_jpeg

    # Calculate episode start index (cumulative sum, not per-file offset)
    episode_start = _get_episode_start_index(dataset_id, episode_idx)

    def _decode_one_frame(i: int) -> bytes:
        """Decode + encode one frame on cache miss. Caches every camera in
        the decoded item to amortise the multi-camera read cost.

        Runs on the dedicated ``_decode_executor`` (single worker), so
        only one thread is ever inside the underlying torchcodec /
        libdav1d decoder at a time — see the module-level comment on
        ``_decode_executor`` for the libdav1d thread-safety background.
        """
        # Re-check the JPEG cache: a sibling request for the same frame
        # (different camera) may have already populated all cameras via
        # the side-effect cache writes below. If it did, skip the redundant
        # decode entirely.
        cached = _app_state.frame_cache.get(dataset_id, episode_idx, i, camera_key)
        if cached is not None:
            return cached

        global_idx = episode_start + i
        item = dataset[global_idx]
        primary: bytes | None = None
        for cam in camera_keys:
            if cam in item:
                cam_jpeg = encode_frame_to_jpeg(item[cam])
                _app_state.frame_cache.put(dataset_id, episode_idx, i, cam, cam_jpeg)
                if cam == camera_key:
                    primary = cam_jpeg
        if primary is None:
            primary = encode_frame_to_jpeg(item[camera_key])
            _app_state.frame_cache.put(dataset_id, episode_idx, i, camera_key, primary)
        return primary

    loop = asyncio.get_event_loop()
    frames = []
    for i in range(start, min(start + count, ep_length)):
        # Check cache first (cheap)
        jpeg_bytes = _app_state.frame_cache.get(dataset_id, episode_idx, i, camera_key)
        if jpeg_bytes is None:
            # Decode-encode work is multi-ms per miss; push off the event loop.
            jpeg_bytes = await loop.run_in_executor(_decode_executor, _decode_one_frame, i)
        frames.append(
            {
                "frame_idx": i,
                "data": base64.b64encode(jpeg_bytes).decode("ascii"),
            }
        )

    return {
        "episode_idx": episode_idx,
        "camera": camera_key,
        "start": start,
        "count": len(frames),
        "total_frames": ep_length,
        "frames": frames,
    }


@router.get("/{dataset_id:path}/cache/stats")
async def get_cache_stats(dataset_id: str) -> dict[str, Any]:
    """Get frame cache statistics."""
    del dataset_id  # URL-scoped for symmetry with sibling routes; cache is global.
    return _app_state.frame_cache.stats()


def _build_visualize_cmd(repo_id: str, episode_idx: int, root: str) -> list[str]:
    """Build the argv for ``lerobot-dataset-viz`` — extracted so tests can pin
    its shape against the target script's actual argparse (see
    ``tests/gui/test_visualize_episode.py``).

    Two things have bitten us here:

    1. PATH-based ``lerobot-dataset-viz`` resolves to whichever env's bin/
       comes first — typically the *base* conda env rather than the lerobot
       env running the GUI. That env can ship incompatible torch / torchcodec
       versions and the viewer crashes mid-decode with
       ``NotImplementedError: torchcodec_ns::_convert_to_tensor``. Use
       ``sys.executable -m`` so we always run under the same Python as the
       GUI.
    2. ``--display-compressed-images`` is a store_true flag (no value) since
       the upstream refactor; passing ``"False"`` as the next argv makes
       argparse treat ``"False"`` as an unknown positional and crash silently.
       Default (uncompressed) is what we want for the standalone viewer, so
       just omit the flag.
    """
    import sys

    return [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_dataset_viz",
        "--repo-id",
        repo_id,
        "--episode-index",
        str(episode_idx),
        "--root",
        str(root),
    ]


@router.post("/{dataset_id:path}/episodes/{episode_idx}/visualize")
async def visualize_episode(dataset_id: str, episode_idx: int) -> dict[str, str]:
    """Launch Rerun visualization for an episode.

    Starts lerobot-dataset-viz in the background for the specified episode.
    """
    import subprocess

    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    dataset = _app_state.datasets[dataset_id]

    # Validate episode index
    if episode_idx < 0 or episode_idx >= dataset.meta.total_episodes:
        raise HTTPException(status_code=404, detail=f"Episode not found: {episode_idx}")

    cmd = _build_visualize_cmd(dataset.repo_id, episode_idx, str(dataset.root))

    logger.info(f"Launching Rerun viz: {' '.join(cmd)}")

    # Tee stdout/stderr to a per-launch log so silent crashes don't disappear.
    log_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / "gui" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime as _dt

    log_path = log_dir / f"rerun_viz_{_dt.now().strftime('%Y%m%d_%H%M%S')}_ep{episode_idx}.log"
    # Open the log file, hand it to the subprocess as stdout/stderr, then
    # close the parent's copy. The subprocess inherits its own FD via fork,
    # so it can still write to the file. Without the close(), every viz
    # launch leaks one FD in the GUI server process until garbage
    # collection runs — and the server is long-lived.
    log_fh = open(log_path, "w", encoding="utf-8")  # noqa: SIM115 - handed to subprocess; closed below
    try:
        try:
            subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to launch visualizer: {e}") from e
    finally:
        log_fh.close()
    logger.info(f"Rerun viz log: {log_path}")

    return {"status": "ok", "message": f"Launched Rerun for episode {episode_idx}"}


# ---------------------------------------------------------------------------
# HuggingFace Hub operations
# ---------------------------------------------------------------------------


@router.get("/hub/auth-status")
async def hub_auth_status():
    """Check if the user is logged in to HuggingFace Hub."""
    from lerobot.gui.api._hub_core import get_auth_status

    return get_auth_status()


@router.post("/hub/open-job-folder")
async def hub_open_job_folder() -> dict:
    """Open the per-job IPC directory in the GUI host's file manager.

    Same pattern as ``/open-in-files``, but the path is fixed to the
    hub-jobs directory so the frontend doesn't have to know (or be
    trusted with) the absolute path. The directory contains one trio of
    files per job (``<job_id>.json``, ``.log``, ``.pid``); from there
    the user can open the ``.log`` to read the raw HF output that drove
    the milestone string they saw in the Transfers tray.

    Caveat: this opens on the *server*, not the frontend machine — same
    constraint as the existing Open-data-directory affordance. Useful
    when the GUI is running locally; degrades gracefully (xdg-open fails
    with a clean 500) on a headless server.
    """
    import asyncio
    import subprocess as _subprocess

    from lerobot.gui.hub_jobs import JOBS_DIR

    JOBS_DIR.mkdir(parents=True, exist_ok=True)

    def _spawn() -> None:
        _subprocess.Popen(["xdg-open", str(JOBS_DIR)])  # nosec B607 - standard Linux file-opener

    try:
        await asyncio.get_event_loop().run_in_executor(None, _spawn)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail="xdg-open not found") from e

    return {"status": "ok", "path": str(JOBS_DIR)}


@router.get("/hub/repo-info")
async def hub_repo_info(repo_id: str):
    """Get info about a dataset repo on HuggingFace Hub."""
    from lerobot.gui.api._hub_core import get_repo_info

    return get_repo_info(repo_id)


@router.get("/{dataset_id:path}/hub/diff")
async def hub_diff(dataset_id: str, repo_id: str | None = None):
    """Compare local dataset against HuggingFace Hub version by file size.

    Returns lists of modified, local-only, and remote-only files.
    Fast: no downloads, just file size comparison.
    """
    dataset_id = unquote(dataset_id)
    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    dataset = _app_state.datasets[dataset_id]
    target_repo_id = repo_id or dataset.repo_id
    root = dataset.root

    try:
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.dataset_info(target_repo_id, files_metadata=True)
    except Exception:
        return {"status": "error", "message": f"Repo not found: {target_repo_id}"}

    remote_files = {}
    for s in info.siblings or []:
        remote_files[s.rfilename] = {
            "size": s.size,
            "sha": s.lfs.sha256 if s.lfs else s.blob_id,
        }

    local_only = []
    remote_only = []
    modified = []
    unchanged = 0

    for rname, rinfo in remote_files.items():
        local_path = root / rname
        if not local_path.exists():
            remote_only.append(rname)
            continue
        local_size = local_path.stat().st_size
        if rinfo["size"] and local_size != rinfo["size"]:
            modified.append({"file": rname, "local_size": local_size, "remote_size": rinfo["size"]})
        else:
            unchanged += 1

    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root))
        if rel.startswith(".cache/") or rel.startswith(".lerobot"):
            continue
        if rel not in remote_files:
            local_only.append(rel)

    in_sync = len(modified) == 0 and len(local_only) == 0 and len(remote_only) == 0
    return {
        "status": "ok",
        "in_sync": in_sync,
        "unchanged": unchanged,
        "modified": modified,
        "local_only": local_only,
        "remote_only": remote_only,
    }


class HubUploadRequest(BaseModel):
    repo_id: str | None = None  # If None, uses dataset's repo_id
    # If True, bypasses the completeness check that warns about uploading
    # from a local copy missing files present on the remote. Used by the
    # frontend's "Upload anyway" follow-up after the user sees the warning.
    confirm_force: bool = False


class HubDownloadRequest(BaseModel):
    repo_id: str | None = None  # If None, uses dataset's repo_id


def _verify_hub_auth() -> None:
    """Raise HTTPException 401 if not logged in to the Hub.

    Cheap (single GET to whoami); we check before kicking off a background
    job so the frontend can surface "not logged in" synchronously rather
    than via a delayed job-failed status in the Transfers tray.
    """
    try:
        from huggingface_hub import HfApi

        HfApi().whoami()
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail="Not logged in to HuggingFace Hub. Run `huggingface-cli login` in terminal.",
        ) from e


# Async lock keyed by dataset_id to serialise the active-job check and
# the job-registration step for the same dataset. Two rapid Upload clicks
# do NOT both run: the first registers a job and spawns a worker; the
# second acquires the lock, finds the just-registered job via
# active_hub_job_for(), and is rejected with 409 carrying the first's
# job_id (so both tabs/clicks end up watching the same Transfers card).
# Lock is held only across check + registration; the actual transfer
# runs in a subprocess outside the lock so the GUI server stays responsive.
_hub_spawn_locks: dict[str, Any] = {}  # values are asyncio.Lock


def _hub_spawn_lock_for(dataset_id: str):
    """Get-or-create the spawn lock for a dataset."""
    import asyncio

    lock = _hub_spawn_locks.get(dataset_id)
    if lock is None:
        lock = asyncio.Lock()
        _hub_spawn_locks[dataset_id] = lock
    return lock


def _refresh_progress_from_file(job) -> None:
    """Merge the worker's progress JSON into the in-memory HubJobState.

    Called on every /hub/jobs and /hub/progress poll. If the worker's
    progress file is missing or unreadable, leaves the in-memory state
    as-is — the server-side fallback (PID liveness check + sweep) handles
    the "worker is dead" case separately.
    """
    import json

    from lerobot.gui.hub_jobs import JOBS_DIR, JobPaths

    paths = JobPaths.for_job(job.job_id, JOBS_DIR)
    try:
        snap = json.loads(paths.progress.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return
    job.merge_progress(snap)


def _send_signal_with_identity_check(job, sig) -> bool:
    """Send ``sig`` to ``job``'s worker, verifying (pid, start_time) first.

    Returns True if the signal was sent; False if the worker isn't alive
    or the PID has been recycled (in which case we mark the job failed).
    """
    import os

    from lerobot.gui.hub_jobs import JOBS_DIR, JobPaths, is_worker_alive, read_pid_file

    paths = JobPaths.for_job(job.job_id, JOBS_DIR)
    payload = read_pid_file(paths.pid)
    if payload is None or not is_worker_alive(payload):
        # Worker is already gone — synthesize a failure state and drop the
        # stale PID file so we don't keep re-checking it on every cancel
        # attempt (the startup sweep would catch it eventually, but only on
        # next server restart).
        if job.status not in ("complete", "failed", "cancelled"):
            job.status = "failed"
            job.error = "Worker exited without finalizing"
            job.error_class = "other"
            job.finished_at = time.time()
        # safe-destruct: stale PID file from a dead worker we owned
        paths.pid.unlink(missing_ok=True)
        return False
    try:
        os.kill(payload["pid"], sig)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _sweep_orphan_pid_files() -> int:
    """Inspect every <job_id>.pid in the jobs dir; reap dead workers.

    For each PID file:
      * If the worker is alive AND we have a registry entry → adopt
        (no-op; the entry's progress JSON catches up on next poll).
      * If the worker is alive AND we have no registry entry → leave
        the file alone; future restart-sweep iterations will pick it up
        once the registry knows about the job. (This case is rare —
        the registry is the spawn-time source of truth.)
      * If the worker is dead → mark any matching registry entry
        ``failed`` and delete the PID file.

    Called once on server startup. Returns the number of orphan files
    that were reaped.
    """
    from lerobot.gui.hub_jobs import JOBS_DIR, is_worker_alive, read_pid_file

    if not JOBS_DIR.exists():
        return 0

    reaped = 0
    for pid_path in JOBS_DIR.glob("*.pid"):
        job_id = pid_path.stem
        payload = read_pid_file(pid_path)
        if payload is None:
            # safe-destruct: malformed PID file we wrote; cleaning up our own debris
            pid_path.unlink(missing_ok=True)
            reaped += 1
            continue
        if is_worker_alive(payload):
            continue
        # Worker is dead.
        entry = _app_state.hub_jobs.get(job_id)
        if entry is not None and entry.status not in ("complete", "failed", "cancelled"):
            entry.status = "failed"
            entry.error = "Worker exited without finalizing (detected on server startup)"
            entry.error_class = "other"
            entry.finished_at = time.time()
        # safe-destruct: stale PID file from a dead worker we owned
        pid_path.unlink(missing_ok=True)
        reaped += 1
    if reaped:
        logger.info("Reaped %d orphan Hub worker PID file(s) on startup", reaped)
    return reaped


def _find_existing_pr_for_retry(dataset_id: str, repo_id: str, repo_type: str = "dataset") -> int | None:
    """For an upload retry, find a draft PR we can reuse.

    Looks back through ``_app_state.hub_jobs`` for the most recent
    terminal (failed/cancelled) entry matching this (dataset, repo). If
    that entry has a ``pr_num`` AND the PR is still in draft state on
    HF, return it for the new worker to resume into.

    Returns ``None`` if no resumable PR exists; the new worker will
    create a fresh one.

    Side effect: when a usable PR is found, **transfers ownership of
    that pr_num to the new caller** by clearing ``pr_num`` on every
    terminal source-job entry that pointed at the same PR. Callers
    that subsequently invoke ``hub_progress_dismiss`` on those source
    entries will then skip the ``change_discussion_status(closed)``
    branch — without this transfer, a Retry click would close the very
    PR the new worker is trying to resume into.

    ``repo_type`` flows through so the lookup works for ``"model"`` repos
    when the future Model Tab adds model-upload endpoints — the
    underlying mechanism is the same, only the repo namespace differs.
    """
    candidates = sorted(
        (
            j
            for j in _app_state.hub_jobs.values()
            if j.dataset_id == dataset_id
            and j.repo_id == repo_id
            and j.repo_type == repo_type
            and j.status in ("failed", "cancelled")
            and j.pr_num is not None
        ),
        key=lambda j: j.started_at,
        reverse=True,
    )
    if not candidates:
        return None
    pr_num = candidates[0].pr_num
    try:
        from huggingface_hub import HfApi

        details = HfApi().get_discussion_details(
            repo_id=repo_id,
            repo_type=repo_type,
            discussion_num=pr_num,
        )
        if details.status == "draft":
            # Transfer ownership: clear pr_num on every source-entry pointing
            # at this PR so a follow-up dismiss does not close it.
            for src in candidates:
                if src.pr_num == pr_num:
                    src.pr_num = None
            return pr_num
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Could not check PR #%d on %s (%s) for resume: %s. Will create a fresh PR.",
            pr_num,
            repo_id,
            repo_type,
            e,
        )
    return None


def _spawn_hub_worker(
    *,
    job,
    local_path: Path,
    reuse_pr_num: int | None = None,
    private: bool = True,
    commit_message: str | None = None,
) -> None:
    """Spawn the worker subprocess for ``job``.

    Pre: ``job`` already in ``_app_state.hub_jobs``. Caller holds the
    per-dataset spawn lock.
    Post: ``job.pid`` is set; the worker is running detached
    (``start_new_session=True``) so killing the GUI server doesn't
    immediately reap the worker until PR_SET_PDEATHSIG kicks in inside
    the worker.
    """
    import subprocess
    import sys

    from lerobot.gui.hub_jobs import (
        DEFAULT_UPLOAD_IGNORES,
        JOBS_DIR,
        JobConfig,
        JobPaths,
        atomic_write_json,
    )

    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    paths = JobPaths.for_job(job.job_id, JOBS_DIR)

    # Build the config. For uploads we forward the default ignore set;
    # downloads pass them through too (snapshot_download will skip these
    # patterns from the remote sibling list if they exist).
    cfg = JobConfig(
        job_id=job.job_id,
        dataset_id=job.dataset_id,
        direction=job.direction,
        repo_id=job.repo_id,
        repo_type=job.repo_type,
        local_path=str(local_path),
        jobs_dir=str(JOBS_DIR),
        private=private,
        commit_message=commit_message,
        allow_patterns=None,
        ignore_patterns=DEFAULT_UPLOAD_IGNORES if job.direction == "upload" else None,
        reuse_pr_num=reuse_pr_num,
    )

    # Stub the progress file so a /hub/jobs poll right after spawn has
    # something to read. The worker rewrites it on its own schedule.
    atomic_write_json(
        paths.progress,
        {
            "job_id": job.job_id,
            "status": "pending",
            "milestone": f"Starting {job.direction}",
            "milestone_at": job.started_at,
        },
    )

    env = os.environ.copy()
    env["LEROBOT_HUB_WORKER_CONFIG"] = cfg.to_json()

    proc = subprocess.Popen(  # noqa: S603 — args are well-controlled
        [sys.executable, "-m", "lerobot.gui.hub_worker"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    job.pid = proc.pid
    logger.info(
        "Spawned Hub worker pid=%d job=%s dir=%s repo=%s reuse_pr=%s",
        proc.pid,
        job.job_id,
        job.direction,
        job.repo_id,
        reuse_pr_num,
    )


@router.post("/{dataset_id:path}/hub/upload")
async def hub_upload(dataset_id: str, request: HubUploadRequest | None = None):
    """Start a Hub upload. Returns ``{job_id}`` immediately.

    Spawns a subprocess worker that runs the full PR pipeline
    (create_pull_request → upload_large_folder → super_squash_history →
    merge_pull_request). The Transfers tray polls /hub/jobs for progress.

    On retry of a failed/cancelled job for the same (dataset, repo), the
    worker is told to resume into the previous PR branch rather than
    creating a new one.
    """
    from lerobot.gui.hub_jobs import check_upload_completeness, make_job

    dataset_id = unquote(dataset_id)
    if dataset_id not in _app_state.datasets:
        # Auto-open if path exists on disk (handles GUI restart with stale frontend)
        p = Path(dataset_id)
        if p.exists() and (p / "meta" / "info.json").exists():
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            _app_state.datasets[dataset_id] = LeRobotDataset(str(p), local_files_only=True)
            logger.info("Auto-opened dataset for upload: %s", dataset_id)
        else:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    dataset = _app_state.datasets[dataset_id]
    repo_id = (request.repo_id if request and request.repo_id else None) or dataset.repo_id

    # Spawn lock: serialises concurrent spawn attempts for the same dataset.
    async with _hub_spawn_lock_for(dataset_id):
        active = _app_state.active_hub_job_for(dataset_id)
        if active is not None:
            raise HTTPException(
                status_code=409,
                detail={"message": "A Hub transfer is already in progress", "job_id": active.job_id},
            )

        _verify_hub_auth()

        # Upload-time completeness check: defends against download-fail-then-upload
        # corruption. If local is missing files that exist on the remote, warn the
        # caller. The frontend can re-issue with confirm_force=true to override.
        confirm_force = request.confirm_force if request else False
        if not confirm_force:
            try:
                missing = check_upload_completeness(Path(dataset.root), repo_id)
            except Exception as e:  # noqa: BLE001 — completeness check is best-effort
                logger.warning("Completeness check failed for %s vs %s: %s", dataset_id, repo_id, e)
                missing = {"missing_locally": [], "incomplete_locally": []}
            if missing["missing_locally"] or missing["incomplete_locally"]:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "incomplete_local_state",
                        "message": (
                            "Local copy is missing files that exist on the remote. "
                            "Re-download first, or confirm to upload anyway."
                        ),
                        "missing_locally": missing["missing_locally"][:20],
                        "incomplete_locally": missing["incomplete_locally"][:20],
                    },
                )

        # Note: model-repo uploads (when the Model Tab adds the endpoint)
        # will need to thread the repo's true repo_type here instead of the
        # default "dataset".
        reuse_pr = _find_existing_pr_for_retry(dataset_id, repo_id, repo_type="dataset")
        job = make_job(dataset_id=dataset_id, direction="upload", repo_id=repo_id)
        _app_state.hub_jobs[job.job_id] = job
        if reuse_pr is not None:
            job.pr_num = reuse_pr

        logger.info(
            "Hub upload start: dataset=%s repo=%s job=%s reuse_pr=%s",
            dataset_id,
            repo_id,
            job.job_id,
            reuse_pr,
        )
        _spawn_hub_worker(
            job=job,
            local_path=Path(dataset.root),
            reuse_pr_num=reuse_pr,
            private=True,
        )

    return {"job_id": job.job_id, "status": "started"}


@router.post("/{dataset_id:path}/hub/download")
async def hub_download(dataset_id: str, request: HubDownloadRequest | None = None):
    """Start a Hub download. Returns ``{job_id}`` immediately.

    snapshot_download writes directly into dataset.root; HF's etag-skip
    and .incomplete-resume primitives handle resumability without a temp
    staging directory.
    """
    from lerobot.gui.hub_jobs import make_job

    dataset_id = unquote(dataset_id)
    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    dataset = _app_state.datasets[dataset_id]
    repo_id = (request.repo_id if request and request.repo_id else None) or dataset.repo_id

    async with _hub_spawn_lock_for(dataset_id):
        active = _app_state.active_hub_job_for(dataset_id)
        if active is not None:
            raise HTTPException(
                status_code=409,
                detail={"message": "A Hub transfer is already in progress", "job_id": active.job_id},
            )

        _verify_hub_auth()

        job = make_job(dataset_id=dataset_id, direction="download", repo_id=repo_id)
        _app_state.hub_jobs[job.job_id] = job
        logger.info("Hub download start: dataset=%s repo=%s job=%s", dataset_id, repo_id, job.job_id)
        _spawn_hub_worker(job=job, local_path=Path(dataset.root))

    return {"job_id": job.job_id, "status": "started"}


@router.get("/hub/jobs")
async def hub_jobs():
    """Return all Hub transfers, sorted newest-first.

    Each entry is the merged view of the server's in-memory HubJobState
    + the worker's latest progress JSON. Calling this opportunistically
    garbage-collects terminal jobs older than 30 minutes.
    """
    from lerobot.gui.api._hub_core import list_hub_jobs

    result = list_hub_jobs(_app_state)
    # Preserve the legacy FastAPI response shape (the GUI's Transfers
    # tray only reads the "jobs" key).
    return {"jobs": result["jobs"]}


@router.get("/hub/progress/{job_id}")
async def hub_progress(job_id: str):
    """Single-job snapshot for clients that want to attach to one specific job."""
    from lerobot.gui.api._hub_core import HubJobNotFoundError, get_job_progress

    try:
        return get_job_progress(_app_state, job_id)
    except HubJobNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.post("/hub/progress/{job_id}/cancel")
async def hub_progress_cancel(job_id: str):
    """SIGTERM the worker for an active job. Idempotent."""
    import signal as _signal

    job = _app_state.hub_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if job.status in ("pending", "running"):
        sent = _send_signal_with_identity_check(job, _signal.SIGTERM)
        if not sent:
            # Worker already gone — status was synthesized to failed.
            pass
    return {"status": "cancel_requested", "job_id": job_id}


@router.post("/hub/progress/{job_id}/dismiss")
async def hub_progress_dismiss(job_id: str):
    """Remove a terminal job from the registry + clean up its IPC files.

    For cancelled/failed uploads whose ``pr_num`` is still set, also close
    the draft PR on HF to prevent it from cluttering the repo's discussion
    list. PR-ownership transfer happens earlier in the retry path: when a
    new worker inherits a source job's PR via ``_find_existing_pr_for_retry``,
    that function clears the source's ``pr_num`` so a subsequent dismiss
    on the source skips the close branch. A Retry-then-Discard sequence
    therefore does not close the PR the retry is resuming into.
    """
    from lerobot.gui.hub_jobs import JOBS_DIR, JobPaths

    job = _app_state.hub_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if job.status in ("pending", "running"):
        raise HTTPException(
            status_code=409,
            detail="Job is still running; cancel it before dismissing.",
        )

    # Close the draft PR if we created one and it's still open. Only on
    # cancelled/failed paths — a completed upload's PR was already merged
    # (and HF auto-cleans it).
    if job.status in ("cancelled", "failed") and job.pr_num is not None:
        try:
            from huggingface_hub import HfApi

            details = HfApi().get_discussion_details(
                repo_id=job.repo_id,
                repo_type=job.repo_type,
                discussion_num=job.pr_num,
            )
            if details.status == "draft":
                HfApi().change_discussion_status(
                    repo_id=job.repo_id,
                    repo_type=job.repo_type,
                    discussion_num=job.pr_num,
                    new_status="closed",
                    comment="Discarded from LeRobot GUI.",
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not close PR #%d for discarded job %s: %s", job.pr_num, job_id, e)

    # Clean up the per-job IPC files.
    paths = JobPaths.for_job(job_id, JOBS_DIR)
    import contextlib as _contextlib

    for p in (paths.progress, paths.log, paths.pid):
        with _contextlib.suppress(OSError):
            # safe-destruct: per-job IPC files we created, user-confirmed dismiss
            p.unlink(missing_ok=True)

    del _app_state.hub_jobs[job_id]
    return {"status": "dismissed", "job_id": job_id}
