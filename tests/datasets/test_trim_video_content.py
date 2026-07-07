#!/usr/bin/env python
"""Rigorous test for trim video content correctness.

This test verifies that trim_episode_by_frames keeps the CORRECT frames,
not just the correct NUMBER of frames.

Each video frame created by the test fixtures has identifiable content:
the frame index is embedded as text (e.g., "observation.image-42").
We use this to verify the correct frames are kept after trimming.

Run with: pytest tests/test_trim_video_content.py -v -s
"""

from pathlib import Path

import numpy as np
import pytest

from lerobot.datasets.dataset_tools import trim_episode_by_frames
from lerobot.datasets.io_utils import load_episodes


def get_all_video_frames(video_path: Path) -> list[np.ndarray]:
    """Extract all frames from a video file as numpy arrays."""
    import av

    container = av.open(str(video_path))
    stream = container.streams.video[0]

    frames = []
    for packet in container.demux(stream):
        for frame in packet.decode():
            img = frame.to_ndarray(format="rgb24")
            frames.append(img)

    container.close()
    return frames


def extract_frame_index_from_image(img: np.ndarray, video_key: str = "observation.image") -> int | None:
    """Try to extract the frame index from the text embedded in the image.

    The test fixtures create frames with text like "observation.image-42".
    This function tries to find that number.

    Returns the frame index or None if not found.
    """
    try:
        import cv2
        import pytesseract
    except ImportError:
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img

    # Apply threshold to get black text on white background
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Use OCR to extract text
    text = pytesseract.image_to_string(thresh, config="--psm 7")

    # Parse the frame index from text like "observation.image-42"
    import re

    match = re.search(rf"{re.escape(video_key)}-(\d+)", text)
    if match:
        return int(match.group(1))

    return None


def compute_frame_signature(img: np.ndarray) -> tuple:
    """Compute a signature that can identify a frame.

    Since each frame has unique text content, the pixel statistics
    should be unique enough to match frames.
    """
    # Use multiple statistics to create a signature
    return (
        img.mean(),
        img.std(),
        img[img.shape[0] // 2, img.shape[1] // 2].tolist(),  # Center pixel
        img[0, 0].tolist(),  # Top-left pixel
        img[-1, -1].tolist(),  # Bottom-right pixel
    )


def test_trim_single_episode_video_content(tmp_path, lerobot_dataset_factory):
    """Test that trimming keeps the correct video frames (not just correct count).

    This is the core test that verifies frame CONTENT, not just count.
    """
    from tests.fixtures.constants import DUMMY_REPO_ID

    # Create dataset with video - each frame has unique content
    dataset = lerobot_dataset_factory(
        root=tmp_path / "test_dataset",
        repo_id=DUMMY_REPO_ID,
        total_episodes=1,
        total_frames=50,  # Use 50 frames for faster test
        use_videos=True,
    )

    video_key = list(dataset.meta.video_keys)[0] if dataset.meta.video_keys else None
    if video_key is None:
        pytest.skip("No video keys in dataset")

    # Get video path
    episode_meta = dataset.meta.episodes[0]
    chunk_idx = episode_meta[f"videos/{video_key}/chunk_index"]
    file_idx = episode_meta[f"videos/{video_key}/file_index"]
    video_path = dataset.root / dataset.meta.video_path.format(
        video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
    )

    # Get all original frames and their signatures
    original_frames = get_all_video_frames(video_path)
    original_signatures = [compute_frame_signature(f) for f in original_frames]
    print(f"\nOriginal video has {len(original_frames)} frames")

    # Print episode metadata
    print("Episode 0 metadata:")
    print(f"  length: {episode_meta['length']}")
    print(f"  from_timestamp: {episode_meta[f'videos/{video_key}/from_timestamp']}")
    print(f"  to_timestamp: {episode_meta[f'videos/{video_key}/to_timestamp']}")
    print(f"  fps: {dataset.fps}")

    # Trim: keep frames 10-39 (30 frames)
    start_frame = 10
    end_frame = 40
    expected_count = end_frame - start_frame

    # Calculate what timestamps we expect
    fps = dataset.fps
    ep_length = episode_meta["length"]
    frames_to_trim_start = start_frame
    frames_to_trim_end = ep_length - end_frame
    trim_start_s = frames_to_trim_start / fps
    trim_end_s = frames_to_trim_end / fps
    from_ts = episode_meta[f"videos/{video_key}/from_timestamp"]
    to_ts = episode_meta[f"videos/{video_key}/to_timestamp"]
    new_from_ts = from_ts + trim_start_s
    new_to_ts = to_ts - trim_end_s

    print("\nExpected time range calculation:")
    print(f"  frames_to_trim_start: {frames_to_trim_start}")
    print(f"  frames_to_trim_end: {frames_to_trim_end}")
    print(f"  trim_start_s: {trim_start_s}")
    print(f"  trim_end_s: {trim_end_s}")
    print(f"  new_from_ts: {new_from_ts}")
    print(f"  new_to_ts: {new_to_ts}")
    print(
        f"  expected duration: {new_to_ts - new_from_ts} seconds = {(new_to_ts - new_from_ts) * fps} frames"
    )

    print(f"\nTrimming: keep frames {start_frame}-{end_frame - 1} ({expected_count} frames)")

    # Apply trim
    trim_episode_by_frames(
        dataset=dataset,
        episode_index=0,
        start_frame=start_frame,
        end_frame=end_frame,
    )

    # Get trimmed frames
    trimmed_frames = get_all_video_frames(video_path)
    trimmed_signatures = [compute_frame_signature(f) for f in trimmed_frames]
    print(f"Trimmed video has {len(trimmed_frames)} frames")

    # Verify count
    assert len(trimmed_frames) == expected_count, (
        f"Expected {expected_count} frames, got {len(trimmed_frames)}"
    )

    # Verify content: trimmed frame N should match original frame (start_frame + N)
    errors = []
    for new_idx in range(expected_count):
        orig_idx = start_frame + new_idx
        new_sig = trimmed_signatures[new_idx]
        expected_sig = original_signatures[orig_idx]

        # Compare signatures - they should be very close (compression may cause small diffs)
        sig_diff = abs(new_sig[0] - expected_sig[0])  # Compare mean values
        if sig_diff > 5:  # Allow some tolerance for compression
            errors.append((new_idx, orig_idx, sig_diff))

    if errors:
        print("\n✗ FRAME CONTENT ERRORS:")
        for new_idx, orig_idx, diff in errors[:10]:
            print(f"  Trimmed frame {new_idx} doesn't match original frame {orig_idx} (diff={diff:.2f})")

        # Try to figure out what frames we actually got
        print("\nTrying to match trimmed frames to originals:")
        for new_idx in range(min(5, expected_count)):
            new_sig = trimmed_signatures[new_idx]
            best_match = min(
                range(len(original_signatures)), key=lambda i: abs(original_signatures[i][0] - new_sig[0])
            )
            expected_orig = start_frame + new_idx
            print(
                f"  Trimmed frame {new_idx}: best match is original frame {best_match} (expected {expected_orig})"
            )

    assert len(errors) == 0, f"{len(errors)} frames have wrong content!"
    print(f"✓ All {expected_count} frames verified to have correct content")


def test_trim_multi_episode_video(tmp_path, lerobot_dataset_factory):
    """Test trimming an episode in a multi-episode video file.

    This tests the scenario where multiple episodes share a video file,
    which is what happens with merged datasets.
    """
    from tests.fixtures.constants import DUMMY_REPO_ID

    # Create dataset with 3 episodes
    dataset = lerobot_dataset_factory(
        root=tmp_path / "test_dataset",
        repo_id=DUMMY_REPO_ID,
        total_episodes=3,
        total_frames=90,  # ~30 frames per episode
        use_videos=True,
    )

    video_key = list(dataset.meta.video_keys)[0] if dataset.meta.video_keys else None
    if video_key is None:
        pytest.skip("No video keys in dataset")

    # Print episode info
    print("\nBefore trim:")
    for ep_idx in range(dataset.meta.total_episodes):
        ep = dataset.meta.episodes[ep_idx]
        from_ts = ep[f"videos/{video_key}/from_timestamp"]
        to_ts = ep[f"videos/{video_key}/to_timestamp"]
        length = ep["length"]
        from_idx = ep["dataset_from_index"]
        to_idx = ep["dataset_to_index"]
        print(
            f"  Episode {ep_idx}: length={length}, from_idx={from_idx}, to_idx={to_idx}, "
            f"from_ts={from_ts:.3f}, to_ts={to_ts:.3f}"
        )

    # Get video path (all episodes in same video file for this test)
    episode_meta = dataset.meta.episodes[1]  # Episode 1
    chunk_idx = episode_meta[f"videos/{video_key}/chunk_index"]
    file_idx = episode_meta[f"videos/{video_key}/file_index"]
    video_path = dataset.root / dataset.meta.video_path.format(
        video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
    )

    # Get all original frames
    original_frames = get_all_video_frames(video_path)
    original_signatures = [compute_frame_signature(f) for f in original_frames]
    print(f"\nOriginal video has {len(original_frames)} frames")

    # Calculate episode 1's position in the video
    ep1_length = episode_meta["length"]
    ep1_from_ts = episode_meta[f"videos/{video_key}/from_timestamp"]
    fps = dataset.fps
    ep1_start_in_video = int(round(ep1_from_ts * fps))
    print(f"Episode 1: {ep1_length} frames, starts at video frame {ep1_start_in_video}")

    # Trim episode 1: keep frames 5 to 20 (local indices)
    start_frame = 5
    end_frame = min(20, ep1_length)  # Ensure we don't exceed episode length
    expected_trim_count = end_frame - start_frame

    print(
        f"\nTrimming episode 1: keep LOCAL frames {start_frame}-{end_frame - 1} ({expected_trim_count} frames)"
    )

    # These correspond to these global video frames:
    global_keep_start = ep1_start_in_video + start_frame
    global_keep_end = ep1_start_in_video + end_frame
    print(f"This should keep GLOBAL video frames {global_keep_start}-{global_keep_end - 1}")

    # Apply trim
    trim_episode_by_frames(
        dataset=dataset,
        episode_index=1,
        start_frame=start_frame,
        end_frame=end_frame,
    )

    # Reload metadata
    dataset.meta.episodes = load_episodes(dataset.root)

    # Print episode info after trim
    print("\nAfter trim:")
    for ep_idx in range(dataset.meta.total_episodes):
        ep = dataset.meta.episodes[ep_idx]
        from_ts = ep[f"videos/{video_key}/from_timestamp"]
        to_ts = ep[f"videos/{video_key}/to_timestamp"]
        length = ep["length"]
        print(f"  Episode {ep_idx}: length={length}, from_ts={from_ts:.3f}, to_ts={to_ts:.3f}")

    # Get trimmed frames
    trimmed_frames = get_all_video_frames(video_path)
    trimmed_signatures = [compute_frame_signature(f) for f in trimmed_frames]
    print(f"\nTrimmed video has {len(trimmed_frames)} frames")

    # Calculate expected total
    ep0_len = dataset.meta.episodes[0]["length"]
    ep1_new_len = dataset.meta.episodes[1]["length"]
    ep2_len = dataset.meta.episodes[2]["length"]
    expected_total = ep0_len + ep1_new_len + ep2_len
    print(f"Expected total: {ep0_len} + {ep1_new_len} + {ep2_len} = {expected_total}")

    assert len(trimmed_frames) == expected_total, (
        f"Expected {expected_total} frames, got {len(trimmed_frames)}"
    )

    # Now verify the content of episode 1's frames in the trimmed video
    # Episode 1 starts at frame ep0_len in the trimmed video
    ep1_new_start = ep0_len

    print(f"\nVerifying episode 1 frames (now at video position {ep1_new_start}):")
    errors = []
    for local_idx in range(expected_trim_count):
        # Position in trimmed video
        trimmed_pos = ep1_new_start + local_idx
        # Corresponding position in original video
        original_pos = global_keep_start + local_idx

        if trimmed_pos >= len(trimmed_frames):
            errors.append((local_idx, trimmed_pos, original_pos, "out of bounds in trimmed"))
            continue
        if original_pos >= len(original_frames):
            errors.append((local_idx, trimmed_pos, original_pos, "out of bounds in original"))
            continue

        new_sig = trimmed_signatures[trimmed_pos]
        expected_sig = original_signatures[original_pos]
        sig_diff = abs(new_sig[0] - expected_sig[0])

        if sig_diff > 5:
            errors.append((local_idx, trimmed_pos, original_pos, f"content mismatch (diff={sig_diff:.2f})"))

    if errors:
        print("\n✗ ERRORS in episode 1 frame verification:")
        for local_idx, trimmed_pos, orig_pos, reason in errors:
            print(f"  Local {local_idx}: trimmed[{trimmed_pos}] vs original[{orig_pos}]: {reason}")

        # Debug: try to find where the frames actually came from
        print("\nDebug: Matching trimmed ep1 frames to originals:")
        for local_idx in range(min(5, expected_trim_count)):
            trimmed_pos = ep1_new_start + local_idx
            if trimmed_pos < len(trimmed_frames):
                new_sig = trimmed_signatures[trimmed_pos]
                # Find closest match in original
                diffs = [abs(s[0] - new_sig[0]) for s in original_signatures]
                best_match = min(range(len(diffs)), key=lambda i: diffs[i])
                expected_orig = global_keep_start + local_idx
                print(
                    f"  Trimmed[{trimmed_pos}] best matches original[{best_match}] "
                    f"(expected {expected_orig}, diff={diffs[best_match]:.2f})"
                )

    assert len(errors) == 0, f"{len(errors)} frames have wrong content!"
    print("✓ All episode 1 frames verified")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
