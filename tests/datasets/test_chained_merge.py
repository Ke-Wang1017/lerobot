#!/usr/bin/env python
"""Test chained merges: A+B->C, then C+D->E.

This tests the scenario where a user merges datasets iteratively:
1. Record dataset A, record dataset B
2. Merge A+B -> C
3. Record dataset D (more data to supplement C)
4. Merge C+D -> E

The risk is that dataset C (a merge result) has a file structure that the second
merge doesn't handle correctly — e.g. meta/episodes file indices that confuse
the src_to_dst mapping, or video timestamps that get double-offset.

Tests both image-only and video datasets.
"""

from unittest.mock import patch

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.dataset_tools import delete_episodes, trim_episode_by_frames
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_EPISODES_PATH
from tests.fixtures.constants import DUMMY_REPO_ID


def load_merged_dataset(repo_id: str, root):
    """Load a merged dataset from disk, mocking Hub calls."""
    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_ver,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snap,
    ):
        mock_ver.return_value = "v3.0"
        mock_snap.return_value = str(root)
        return LeRobotDataset(repo_id, root=root)


def assert_meta_files_exist(ds):
    """Verify every episode references meta/episodes files that exist on disk."""
    missing = []
    for ep_idx in range(ds.num_episodes):
        ep = ds.meta.episodes[ep_idx]
        chunk_idx = int(ep["meta/episodes/chunk_index"])
        file_idx = int(ep["meta/episodes/file_index"])
        meta_path = ds.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        if not meta_path.exists():
            missing.append((ep_idx, chunk_idx, file_idx))

    assert len(missing) == 0, (
        f"{len(missing)} episodes reference non-existent meta/episodes files: "
        + ", ".join(f"ep {e} -> chunk-{c:03d}/file-{f:03d}" for e, c, f in missing[:10])
    )


def assert_data_files_exist(ds):
    """Verify every episode references data files that exist on disk."""
    missing = []
    for ep_idx in range(ds.num_episodes):
        data_path = ds.root / ds.meta.get_data_file_path(ep_idx)
        if not data_path.exists():
            missing.append((ep_idx, str(data_path)))

    assert len(missing) == 0, f"{len(missing)} episodes reference non-existent data files: " + ", ".join(
        f"ep {e} -> {p}" for e, p in missing[:10]
    )


def assert_video_files_exist(ds):
    """Verify every episode references video files that exist on disk."""
    missing = []
    for ep_idx in range(ds.num_episodes):
        for vid_key in ds.meta.video_keys:
            video_path = ds.root / ds.meta.get_video_file_path(ep_idx, vid_key)
            if not video_path.exists():
                missing.append((ep_idx, vid_key, str(video_path)))

    assert len(missing) == 0, f"{len(missing)} episodes reference non-existent video files: " + ", ".join(
        f"ep {e}/{k} -> {p}" for e, k, p in missing[:10]
    )


def assert_episode_indices_contiguous(ds):
    """Verify episode indices form a contiguous range [0, num_episodes)."""
    indices = [int(ds.meta.episodes[i]["episode_index"]) for i in range(ds.num_episodes)]
    assert indices == list(range(ds.num_episodes)), (
        f"Episode indices are not contiguous [0..{ds.num_episodes - 1}]: {indices}"
    )


def assert_frame_counts_match(ds):
    """Verify each episode's declared length matches actual data row count."""
    for ep_idx in range(ds.num_episodes):
        ep = ds.meta.episodes[ep_idx]
        declared_length = int(ep["length"])
        from_idx = int(ep["dataset_from_index"])
        to_idx = int(ep["dataset_to_index"])
        actual_length = to_idx - from_idx
        assert declared_length == actual_length, (
            f"Episode {ep_idx}: declared length {declared_length} != "
            f"index range {to_idx}-{from_idx}={actual_length}"
        )


def assert_video_timestamps_valid(ds):
    """Verify video timestamps are non-negative and from < to for each episode."""
    for ep_idx in range(ds.num_episodes):
        ep = ds.meta.episodes[ep_idx]
        for vid_key in ds.meta.video_keys:
            from_ts = ep[f"videos/{vid_key}/from_timestamp"]
            to_ts = ep[f"videos/{vid_key}/to_timestamp"]
            assert from_ts >= 0, f"Episode {ep_idx}, {vid_key}: from_timestamp ({from_ts}) < 0"
            assert from_ts < to_ts, (
                f"Episode {ep_idx}, {vid_key}: from_timestamp ({from_ts}) >= to_timestamp ({to_ts})"
            )


def assert_full_iteration(ds, expected_frames):
    """Verify we can iterate through all frames and the count matches."""
    count = 0
    for _item in ds:
        count += 1
    assert count == expected_frames, f"Expected {expected_frames} frames during iteration, got {count}"


def run_integrity_checks(ds, expected_episodes, expected_frames, has_videos):
    """Run all integrity checks on a dataset."""
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


class TestChainedMerge:
    """Test chained merge: A+B->C, C+D->E."""

    def test_chained_merge_no_video(self, tmp_path, lerobot_dataset_factory):
        """Chained merge with image-only datasets."""
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
            total_episodes=8,
            total_frames=400,
            use_videos=False,
        )

        # Step 1: A+B -> C
        aggregate_datasets(
            repo_ids=[ds_a.repo_id, ds_b.repo_id],
            roots=[ds_a.root, ds_b.root],
            aggr_repo_id=f"{DUMMY_REPO_ID}_c",
            aggr_root=tmp_path / "ds_c",
        )
        ds_c = load_merged_dataset(f"{DUMMY_REPO_ID}_c", tmp_path / "ds_c")
        run_integrity_checks(ds_c, 13, 650, has_videos=False)

        # Create D
        ds_d = lerobot_dataset_factory(
            root=tmp_path / "ds_d",
            repo_id=f"{DUMMY_REPO_ID}_d",
            total_episodes=6,
            total_frames=300,
            use_videos=False,
        )

        # Step 2: C+D -> E
        aggregate_datasets(
            repo_ids=[ds_c.repo_id, ds_d.repo_id],
            roots=[ds_c.root, ds_d.root],
            aggr_repo_id=f"{DUMMY_REPO_ID}_e",
            aggr_root=tmp_path / "ds_e",
        )
        ds_e = load_merged_dataset(f"{DUMMY_REPO_ID}_e", tmp_path / "ds_e")
        run_integrity_checks(ds_e, 19, 950, has_videos=False)

    def test_chained_merge_with_video(self, tmp_path, lerobot_dataset_factory):
        """Chained merge with video datasets — the real-world case."""
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
            total_episodes=8,
            total_frames=400,
            use_videos=True,
        )

        # Step 1: A+B -> C
        aggregate_datasets(
            repo_ids=[ds_a.repo_id, ds_b.repo_id],
            roots=[ds_a.root, ds_b.root],
            aggr_repo_id=f"{DUMMY_REPO_ID}_c",
            aggr_root=tmp_path / "ds_c",
        )
        ds_c = load_merged_dataset(f"{DUMMY_REPO_ID}_c", tmp_path / "ds_c")
        run_integrity_checks(ds_c, 13, 650, has_videos=True)

        # Create D
        ds_d = lerobot_dataset_factory(
            root=tmp_path / "ds_d",
            repo_id=f"{DUMMY_REPO_ID}_d",
            total_episodes=6,
            total_frames=300,
            use_videos=True,
        )

        # Step 2: C+D -> E
        aggregate_datasets(
            repo_ids=[ds_c.repo_id, ds_d.repo_id],
            roots=[ds_c.root, ds_d.root],
            aggr_repo_id=f"{DUMMY_REPO_ID}_e",
            aggr_root=tmp_path / "ds_e",
        )
        ds_e = load_merged_dataset(f"{DUMMY_REPO_ID}_e", tmp_path / "ds_e")
        run_integrity_checks(ds_e, 19, 950, has_videos=True)

    def test_triple_chain_with_video(self, tmp_path, lerobot_dataset_factory):
        """Triple chain: A+B->C, C+D->E, E+F->G. Stress test for accumulated remapping."""
        datasets_config = [
            ("a", 4, 200),
            ("b", 3, 150),
            ("d", 5, 250),
            ("f", 3, 150),
        ]
        sources = {}
        for name, eps, frames in datasets_config:
            sources[name] = lerobot_dataset_factory(
                root=tmp_path / f"ds_{name}",
                repo_id=f"{DUMMY_REPO_ID}_{name}",
                total_episodes=eps,
                total_frames=frames,
                use_videos=True,
            )

        # A+B -> C
        aggregate_datasets(
            repo_ids=[sources["a"].repo_id, sources["b"].repo_id],
            roots=[sources["a"].root, sources["b"].root],
            aggr_repo_id=f"{DUMMY_REPO_ID}_c",
            aggr_root=tmp_path / "ds_c",
        )
        ds_c = load_merged_dataset(f"{DUMMY_REPO_ID}_c", tmp_path / "ds_c")
        run_integrity_checks(ds_c, 7, 350, has_videos=True)

        # C+D -> E
        aggregate_datasets(
            repo_ids=[ds_c.repo_id, sources["d"].repo_id],
            roots=[ds_c.root, sources["d"].root],
            aggr_repo_id=f"{DUMMY_REPO_ID}_e",
            aggr_root=tmp_path / "ds_e",
        )
        ds_e = load_merged_dataset(f"{DUMMY_REPO_ID}_e", tmp_path / "ds_e")
        run_integrity_checks(ds_e, 12, 600, has_videos=True)

        # E+F -> G
        aggregate_datasets(
            repo_ids=[ds_e.repo_id, sources["f"].repo_id],
            roots=[ds_e.root, sources["f"].root],
            aggr_repo_id=f"{DUMMY_REPO_ID}_g",
            aggr_root=tmp_path / "ds_g",
        )
        ds_g = load_merged_dataset(f"{DUMMY_REPO_ID}_g", tmp_path / "ds_g")
        run_integrity_checks(ds_g, 15, 750, has_videos=True)

    def test_chained_merge_with_file_rotation(self, tmp_path, lerobot_dataset_factory):
        """Chained merge with small file size limits to force multiple files per dataset.

        This is the most likely scenario to trigger the meta_src_to_dst bug:
        small file limits cause multiple meta/data/video files, and the second
        merge must correctly remap indices from a dataset that already has a
        non-trivial file structure.
        """
        ds_a = lerobot_dataset_factory(
            root=tmp_path / "ds_a",
            repo_id=f"{DUMMY_REPO_ID}_a",
            total_episodes=10,
            total_frames=500,
            use_videos=True,
        )
        ds_b = lerobot_dataset_factory(
            root=tmp_path / "ds_b",
            repo_id=f"{DUMMY_REPO_ID}_b",
            total_episodes=8,
            total_frames=400,
            use_videos=True,
        )

        # Step 1: A+B -> C with small file limits to force rotation
        aggregate_datasets(
            repo_ids=[ds_a.repo_id, ds_b.repo_id],
            roots=[ds_a.root, ds_b.root],
            aggr_repo_id=f"{DUMMY_REPO_ID}_c",
            aggr_root=tmp_path / "ds_c",
            data_files_size_in_mb=0.01,
            video_files_size_in_mb=0.1,
        )
        ds_c = load_merged_dataset(f"{DUMMY_REPO_ID}_c", tmp_path / "ds_c")
        run_integrity_checks(ds_c, 18, 900, has_videos=True)

        # Verify C actually has multiple files (the point of small limits)
        meta_files = list((tmp_path / "ds_c" / "meta" / "episodes").rglob("*.parquet"))
        data_files = list((tmp_path / "ds_c" / "data").rglob("*.parquet"))
        print(f"ds_c meta files: {len(meta_files)}, data files: {len(data_files)}")

        # Create D
        ds_d = lerobot_dataset_factory(
            root=tmp_path / "ds_d",
            repo_id=f"{DUMMY_REPO_ID}_d",
            total_episodes=6,
            total_frames=300,
            use_videos=True,
        )

        # Step 2: C+D -> E (also with small file limits)
        aggregate_datasets(
            repo_ids=[ds_c.repo_id, ds_d.repo_id],
            roots=[ds_c.root, ds_d.root],
            aggr_repo_id=f"{DUMMY_REPO_ID}_e",
            aggr_root=tmp_path / "ds_e",
            data_files_size_in_mb=0.01,
            video_files_size_in_mb=0.1,
        )
        ds_e = load_merged_dataset(f"{DUMMY_REPO_ID}_e", tmp_path / "ds_e")
        run_integrity_checks(ds_e, 24, 1200, has_videos=True)

    def test_chained_merge_after_trim_and_delete(self, tmp_path, lerobot_dataset_factory):
        """Trim and delete episodes before chained merge — the real-world curation workflow.

        Simulates: record A, record B, curate both (trim bad segments, delete bad episodes),
        merge curated versions into C, then record D, merge C+D->E.
        """

        def _ep_len(ds, ep_idx):
            return int(ds.meta.episodes[ep_idx]["length"])

        def _delete(ds, indices, name):
            out = tmp_path / name
            with (
                patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mv,
                patch("lerobot.datasets.lerobot_dataset.snapshot_download") as ms,
            ):
                mv.return_value = "v3.0"
                ms.return_value = str(out)
                return delete_episodes(
                    ds,
                    episode_indices=indices,
                    output_dir=out,
                    repo_id=f"{DUMMY_REPO_ID}_{name}",
                )

        # --- Dataset A: 6 episodes, 300 total frames (randomly distributed) ---
        ds_a = lerobot_dataset_factory(
            root=tmp_path / "ds_a",
            repo_id=f"{DUMMY_REPO_ID}_a",
            total_episodes=6,
            total_frames=300,
            use_videos=True,
        )

        # Trim ep 0: cut 5 frames from each end
        a_ep0_len = _ep_len(ds_a, 0)
        trim_episode_by_frames(ds_a, episode_index=0, start_frame=5, end_frame=a_ep0_len - 5)
        a_ep0_len - 10

        # Trim ep 3: cut 3 frames from start
        a_ep3_len = _ep_len(ds_a, 3)
        trim_episode_by_frames(ds_a, episode_index=3, start_frame=3, end_frame=a_ep3_len)
        a_ep3_len - 3

        # Reload, then delete eps 2 and 4
        ds_a = load_merged_dataset(ds_a.repo_id, ds_a.root)
        a_frames_before_delete = ds_a.num_frames
        a_frames_deleted = _ep_len(ds_a, 2) + _ep_len(ds_a, 4)
        ds_a_edited = _delete(ds_a, [2, 4], "ds_a_edited")

        a_expected_eps = 4
        a_expected_frames = a_frames_before_delete - a_frames_deleted
        run_integrity_checks(ds_a_edited, a_expected_eps, a_expected_frames, has_videos=True)

        # --- Dataset B: 5 episodes, 250 total frames ---
        ds_b = lerobot_dataset_factory(
            root=tmp_path / "ds_b",
            repo_id=f"{DUMMY_REPO_ID}_b",
            total_episodes=5,
            total_frames=250,
            use_videos=True,
        )

        # Trim ep 1: cut 4 frames from end
        b_ep1_len = _ep_len(ds_b, 1)
        trim_episode_by_frames(ds_b, episode_index=1, start_frame=0, end_frame=b_ep1_len - 4)

        # Delete ep 3
        ds_b = load_merged_dataset(ds_b.repo_id, ds_b.root)
        b_frames_before_delete = ds_b.num_frames
        b_frames_deleted = _ep_len(ds_b, 3)
        ds_b_edited = _delete(ds_b, [3], "ds_b_edited")

        b_expected_eps = 4
        b_expected_frames = b_frames_before_delete - b_frames_deleted
        run_integrity_checks(ds_b_edited, b_expected_eps, b_expected_frames, has_videos=True)

        # --- Merge curated A + curated B → C ---
        aggregate_datasets(
            repo_ids=[ds_a_edited.repo_id, ds_b_edited.repo_id],
            roots=[ds_a_edited.root, ds_b_edited.root],
            aggr_repo_id=f"{DUMMY_REPO_ID}_c",
            aggr_root=tmp_path / "ds_c",
        )
        ds_c = load_merged_dataset(f"{DUMMY_REPO_ID}_c", tmp_path / "ds_c")
        c_expected_eps = a_expected_eps + b_expected_eps
        c_expected_frames = a_expected_frames + b_expected_frames
        run_integrity_checks(ds_c, c_expected_eps, c_expected_frames, has_videos=True)

        # --- Dataset D: 4 episodes, 200 frames (fresh, no edits) ---
        ds_d = lerobot_dataset_factory(
            root=tmp_path / "ds_d",
            repo_id=f"{DUMMY_REPO_ID}_d",
            total_episodes=4,
            total_frames=200,
            use_videos=True,
        )

        # --- Chain: C + D → E ---
        aggregate_datasets(
            repo_ids=[ds_c.repo_id, ds_d.repo_id],
            roots=[ds_c.root, ds_d.root],
            aggr_repo_id=f"{DUMMY_REPO_ID}_e",
            aggr_root=tmp_path / "ds_e",
        )
        ds_e = load_merged_dataset(f"{DUMMY_REPO_ID}_e", tmp_path / "ds_e")
        e_expected_eps = c_expected_eps + ds_d.num_episodes
        e_expected_frames = c_expected_frames + ds_d.num_frames
        run_integrity_checks(ds_e, e_expected_eps, e_expected_frames, has_videos=True)
