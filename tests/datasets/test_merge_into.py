#!/usr/bin/env python
"""Tests for merge_into() — in-place dataset merging.

Tests the workflow: merge source episodes into an existing target dataset,
keeping target's existing files untouched.
"""

import hashlib
from pathlib import Path
from unittest.mock import patch

import torch

from lerobot.datasets.dataset_tools import delete_episodes, merge_into, trim_episode_by_frames
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_EPISODES_PATH
from tests.fixtures.constants import DUMMY_REPO_ID


def load_dataset(repo_id: str, root):
    """Load a dataset from disk, mocking Hub calls."""
    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mv,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as ms,
    ):
        mv.return_value = "v3.0"
        ms.return_value = str(root)
        return LeRobotDataset(repo_id, root=root)


def assert_meta_files_exist(ds):
    for ep_idx in range(ds.num_episodes):
        ep = ds.meta.episodes[ep_idx]
        path = ds.root / DEFAULT_EPISODES_PATH.format(
            chunk_index=int(ep["meta/episodes/chunk_index"]),
            file_index=int(ep["meta/episodes/file_index"]),
        )
        assert path.exists(), f"Episode {ep_idx} references missing meta file: {path}"


def assert_data_files_exist(ds):
    for ep_idx in range(ds.num_episodes):
        path = ds.root / ds.meta.get_data_file_path(ep_idx)
        assert path.exists(), f"Episode {ep_idx} references missing data file: {path}"


def assert_video_files_exist(ds):
    for ep_idx in range(ds.num_episodes):
        for vid_key in ds.meta.video_keys:
            path = ds.root / ds.meta.get_video_file_path(ep_idx, vid_key)
            assert path.exists(), f"Episode {ep_idx}/{vid_key} references missing video: {path}"


def assert_episode_indices_contiguous(ds):
    indices = [int(ds.meta.episodes[i]["episode_index"]) for i in range(ds.num_episodes)]
    assert indices == list(range(ds.num_episodes)), f"Episode indices not contiguous: {indices}"


def assert_frame_counts_match(ds):
    for ep_idx in range(ds.num_episodes):
        ep = ds.meta.episodes[ep_idx]
        length = int(ep["length"])
        from_idx = int(ep["dataset_from_index"])
        to_idx = int(ep["dataset_to_index"])
        assert length == to_idx - from_idx, (
            f"Episode {ep_idx}: length {length} != range {to_idx}-{from_idx}={to_idx - from_idx}"
        )


def assert_video_timestamps_valid(ds):
    for ep_idx in range(ds.num_episodes):
        ep = ds.meta.episodes[ep_idx]
        for vid_key in ds.meta.video_keys:
            from_ts = ep[f"videos/{vid_key}/from_timestamp"]
            to_ts = ep[f"videos/{vid_key}/to_timestamp"]
            assert from_ts >= 0, f"Episode {ep_idx}, {vid_key}: from_ts ({from_ts}) < 0"
            assert from_ts < to_ts, f"Episode {ep_idx}, {vid_key}: from_ts ({from_ts}) >= to_ts ({to_ts})"


def assert_full_iteration(ds, expected_frames):
    count = sum(1 for _ in ds)
    assert count == expected_frames, f"Expected {expected_frames} frames, got {count}"


def run_integrity_checks(ds, expected_episodes, expected_frames, has_videos):
    assert ds.num_episodes == expected_episodes, (
        f"Expected {expected_episodes} episodes, got {ds.num_episodes}"
    )
    assert ds.num_frames == expected_frames, f"Expected {expected_frames} frames, got {ds.num_frames}"
    assert_meta_files_exist(ds)
    assert_data_files_exist(ds)
    assert_episode_indices_contiguous(ds)
    assert_frame_counts_match(ds)
    if has_videos:
        assert_video_files_exist(ds)
        assert_video_timestamps_valid(ds)
    assert_full_iteration(ds, expected_frames)


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_file_hashes(root: Path) -> dict[str, str]:
    """Hash all data and video files (not metadata) under root."""
    hashes = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root))
        # Only hash data and video files — metadata is expected to change
        if rel.startswith("data/") or rel.startswith("videos/"):
            hashes[rel] = hash_file(p)
    return hashes


class TestMergeInto:
    def test_basic_merge_into(self, tmp_path, lerobot_dataset_factory):
        """Basic in-place merge with image-only datasets."""
        ds_a = lerobot_dataset_factory(
            root=tmp_path / "ds_a",
            repo_id=f"{DUMMY_REPO_ID}_a",
            total_episodes=5,
            total_frames=250,
            use_videos=False,
        )
        ds_b = lerobot_dataset_factory(
            root=tmp_path / "ds_b",
            repo_id=f"{DUMMY_REPO_ID}_b",
            total_episodes=4,
            total_frames=200,
            use_videos=False,
        )

        expected_eps = ds_a.num_episodes + ds_b.num_episodes
        expected_frames = ds_a.num_frames + ds_b.num_frames

        result = merge_into(ds_a, ds_b)
        run_integrity_checks(result, expected_eps, expected_frames, has_videos=False)

    def test_merge_into_with_video(self, tmp_path, lerobot_dataset_factory):
        """In-place merge with video datasets."""
        ds_a = lerobot_dataset_factory(
            root=tmp_path / "ds_a",
            repo_id=f"{DUMMY_REPO_ID}_a",
            total_episodes=5,
            total_frames=250,
            use_videos=True,
        )
        ds_b = lerobot_dataset_factory(
            root=tmp_path / "ds_b",
            repo_id=f"{DUMMY_REPO_ID}_b",
            total_episodes=4,
            total_frames=200,
            use_videos=True,
        )

        expected_eps = ds_a.num_episodes + ds_b.num_episodes
        expected_frames = ds_a.num_frames + ds_b.num_frames

        result = merge_into(ds_a, ds_b)
        run_integrity_checks(result, expected_eps, expected_frames, has_videos=True)

    def test_merge_into_preserves_existing_files(self, tmp_path, lerobot_dataset_factory):
        """Verify target's existing data and video files are untouched after merge."""
        ds_a = lerobot_dataset_factory(
            root=tmp_path / "ds_a",
            repo_id=f"{DUMMY_REPO_ID}_a",
            total_episodes=5,
            total_frames=250,
            use_videos=True,
        )
        ds_b = lerobot_dataset_factory(
            root=tmp_path / "ds_b",
            repo_id=f"{DUMMY_REPO_ID}_b",
            total_episodes=3,
            total_frames=150,
            use_videos=True,
        )

        # Hash all data/video files before merge
        hashes_before = collect_file_hashes(tmp_path / "ds_a")
        assert len(hashes_before) > 0, "Target should have files to hash"

        merge_into(ds_a, ds_b)

        # Verify all original files still have the same hash
        hashes_after = collect_file_hashes(tmp_path / "ds_a")
        for rel_path, original_hash in hashes_before.items():
            assert rel_path in hashes_after, f"Original file disappeared: {rel_path}"
            assert hashes_after[rel_path] == original_hash, f"Original file was modified: {rel_path}"

        # Verify new files were added
        new_files = set(hashes_after.keys()) - set(hashes_before.keys())
        assert len(new_files) > 0, "Merge should have added new files"

    def test_merge_into_chained(self, tmp_path, lerobot_dataset_factory):
        """Merge B into A, then C into A — the core multi-source workflow."""
        ds_a = lerobot_dataset_factory(
            root=tmp_path / "ds_a",
            repo_id=f"{DUMMY_REPO_ID}_a",
            total_episodes=4,
            total_frames=200,
            use_videos=True,
        )
        ds_b = lerobot_dataset_factory(
            root=tmp_path / "ds_b",
            repo_id=f"{DUMMY_REPO_ID}_b",
            total_episodes=3,
            total_frames=150,
            use_videos=True,
        )
        ds_c = lerobot_dataset_factory(
            root=tmp_path / "ds_c",
            repo_id=f"{DUMMY_REPO_ID}_c",
            total_episodes=5,
            total_frames=250,
            use_videos=True,
        )

        total_eps = ds_a.num_episodes + ds_b.num_episodes + ds_c.num_episodes
        total_frames = ds_a.num_frames + ds_b.num_frames + ds_c.num_frames

        # Merge B into A
        ds_a = merge_into(ds_a, ds_b)
        run_integrity_checks(ds_a, ds_a.num_episodes, ds_a.num_frames, has_videos=True)

        # Merge C into A
        ds_a = merge_into(ds_a, ds_c)
        run_integrity_checks(ds_a, total_eps, total_frames, has_videos=True)

    def test_merge_into_different_tasks(self, tmp_path, lerobot_dataset_factory):
        """Source has tasks not in target — verify task merging preserves indices."""
        ds_a = lerobot_dataset_factory(
            root=tmp_path / "ds_a",
            repo_id=f"{DUMMY_REPO_ID}_a",
            total_episodes=4,
            total_frames=200,
            use_videos=False,
            multi_task=True,
        )
        ds_b = lerobot_dataset_factory(
            root=tmp_path / "ds_b",
            repo_id=f"{DUMMY_REPO_ID}_b",
            total_episodes=4,
            total_frames=200,
            use_videos=False,
            multi_task=True,
        )

        a_tasks_before = set(ds_a.meta.tasks.index)
        b_tasks = set(ds_b.meta.tasks.index)
        expected_eps = ds_a.num_episodes + ds_b.num_episodes
        expected_frames = ds_a.num_frames + ds_b.num_frames

        result = merge_into(ds_a, ds_b)

        # All tasks from both datasets should be present
        result_tasks = set(result.meta.tasks.index)
        assert a_tasks_before | b_tasks == result_tasks, f"Expected union of tasks, got {result_tasks}"
        run_integrity_checks(result, expected_eps, expected_frames, has_videos=False)

    def test_merge_into_content_integrity(self, tmp_path, lerobot_dataset_factory):
        """Frame-by-frame comparison: target data unchanged, source data appended correctly."""
        ds_a = lerobot_dataset_factory(
            root=tmp_path / "ds_a",
            repo_id=f"{DUMMY_REPO_ID}_a",
            total_episodes=3,
            total_frames=150,
            use_videos=True,
        )
        ds_b = lerobot_dataset_factory(
            root=tmp_path / "ds_b",
            repo_id=f"{DUMMY_REPO_ID}_b",
            total_episodes=2,
            total_frames=100,
            use_videos=True,
        )

        # Snapshot target's first and last frames before merge
        a_first = {k: v.clone() if torch.is_tensor(v) else v for k, v in ds_a[0].items()}
        a_last = {k: v.clone() if torch.is_tensor(v) else v for k, v in ds_a[-1].items()}
        b_first = {k: v.clone() if torch.is_tensor(v) else v for k, v in ds_b[0].items()}
        b_last = {k: v.clone() if torch.is_tensor(v) else v for k, v in ds_b[-1].items()}

        a_len = len(ds_a)

        result = merge_into(ds_a, ds_b)

        keys_to_ignore = {"episode_index", "index", "timestamp", "frame_index", "task_index"}

        # Target's first frame should be unchanged
        result_first = result[0]
        for key in a_first:
            if key in keys_to_ignore:
                continue
            if torch.is_tensor(a_first[key]) and torch.is_tensor(result_first[key]):
                assert torch.allclose(a_first[key], result_first[key], atol=1e-6), (
                    f"Target first frame key '{key}' changed after merge"
                )

        # Target's last frame should be unchanged
        result_a_last = result[a_len - 1]
        for key in a_last:
            if key in keys_to_ignore:
                continue
            if torch.is_tensor(a_last[key]) and torch.is_tensor(result_a_last[key]):
                assert torch.allclose(a_last[key], result_a_last[key], atol=1e-6), (
                    f"Target last frame key '{key}' changed after merge"
                )

        # Source's first frame should appear at index a_len
        result_b_first = result[a_len]
        for key in b_first:
            if key in keys_to_ignore:
                continue
            if torch.is_tensor(b_first[key]) and torch.is_tensor(result_b_first[key]):
                assert torch.allclose(b_first[key], result_b_first[key], atol=1e-6), (
                    f"Source first frame key '{key}' doesn't match at index {a_len}"
                )

        # Source's last frame should be at the end
        result_b_last = result[-1]
        for key in b_last:
            if key in keys_to_ignore:
                continue
            if torch.is_tensor(b_last[key]) and torch.is_tensor(result_b_last[key]):
                assert torch.allclose(b_last[key], result_b_last[key], atol=1e-6), (
                    f"Source last frame key '{key}' doesn't match at end"
                )

    def test_merge_into_after_trim_and_delete(self, tmp_path, lerobot_dataset_factory):
        """Curate source (trim + delete) before merging into target."""
        ds_a = lerobot_dataset_factory(
            root=tmp_path / "ds_a",
            repo_id=f"{DUMMY_REPO_ID}_a",
            total_episodes=5,
            total_frames=250,
            use_videos=True,
        )
        ds_b = lerobot_dataset_factory(
            root=tmp_path / "ds_b",
            repo_id=f"{DUMMY_REPO_ID}_b",
            total_episodes=6,
            total_frames=300,
            use_videos=True,
        )

        a_frames = ds_a.num_frames
        a_eps = ds_a.num_episodes

        # Trim source ep 0
        b_ep0_len = int(ds_b.meta.episodes[0]["length"])
        trim_episode_by_frames(ds_b, episode_index=0, start_frame=3, end_frame=b_ep0_len - 3)

        # Reload and delete source ep 2
        ds_b = load_dataset(ds_b.repo_id, ds_b.root)
        with (
            patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mv,
            patch("lerobot.datasets.lerobot_dataset.snapshot_download") as ms,
        ):
            mv.return_value = "v3.0"
            ms.return_value = str(tmp_path / "ds_b_edited")
            ds_b_edited = delete_episodes(
                ds_b,
                episode_indices=[2],
                output_dir=tmp_path / "ds_b_edited",
                repo_id=f"{DUMMY_REPO_ID}_b_edited",
            )

        b_edited_frames = ds_b_edited.num_frames
        b_edited_eps = ds_b_edited.num_episodes

        # Merge curated source into target — no Hub mocks needed
        result = merge_into(ds_a, ds_b_edited)

        run_integrity_checks(
            result,
            a_eps + b_edited_eps,
            a_frames + b_edited_frames,
            has_videos=True,
        )

    def test_merge_into_with_file_rotation(self, tmp_path, lerobot_dataset_factory):
        """Small file limits force multiple new files during merge — stress test index tracking."""
        ds_a = lerobot_dataset_factory(
            root=tmp_path / "ds_a",
            repo_id=f"{DUMMY_REPO_ID}_a",
            total_episodes=8,
            total_frames=400,
            use_videos=True,
        )
        ds_b = lerobot_dataset_factory(
            root=tmp_path / "ds_b",
            repo_id=f"{DUMMY_REPO_ID}_b",
            total_episodes=6,
            total_frames=300,
            use_videos=True,
        )

        expected_eps = ds_a.num_episodes + ds_b.num_episodes
        expected_frames = ds_a.num_frames + ds_b.num_frames

        a_data_files_before = list((tmp_path / "ds_a" / "data").rglob("*.parquet"))

        # Set tiny file limits on the target so merge creates many small files
        ds_a.meta.update_chunk_settings(
            data_files_size_in_mb=0.01,
            video_files_size_in_mb=0.1,
        )

        result = merge_into(ds_a, ds_b)
        run_integrity_checks(result, expected_eps, expected_frames, has_videos=True)

        # Verify new data files were added
        a_data_files_after = list((tmp_path / "ds_a" / "data").rglob("*.parquet"))
        assert len(a_data_files_after) > len(a_data_files_before), "Merge should have added new data files"
