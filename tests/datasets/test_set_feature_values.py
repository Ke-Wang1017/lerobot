# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for ``dataset_tools.set_feature_values`` — the in-place
per-frame value editing primitive.

This is the LeRobot-level peer of :func:`modify_features`. The GUI's
"apply edits" path translates staged edits into a single call here;
notebooks / CLI tools can use it directly. These tests pin the
contract: half-open ``[from, to)`` ranges, atomic shard rewrites,
schema unchanged, episode stats recomputed, image/video features
rejected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lerobot.datasets.dataset_tools import (
    FeatureValueEdit,
    set_feature_values,
)


def _read_data_df(root: Path) -> pd.DataFrame:
    """Load all data parquet shards merged into a single DataFrame, sorted by global ``index``."""
    parts = []
    for shard in sorted((root / "data").glob("*/*.parquet")):
        parts.append(pd.read_parquet(shard))
    df = pd.concat(parts, ignore_index=True).sort_values("index").reset_index(drop=True)
    return df


def _read_episode_stats_min_max(root: Path, feature: str) -> dict[int, tuple[float, float]]:
    """Return ``{episode_index: (min, max)}`` for ``feature`` from the episodes parquet."""
    out: dict[int, tuple[float, float]] = {}
    for shard in sorted((root / "meta" / "episodes").glob("*/*.parquet")):
        df = pd.read_parquet(shard)
        for _, row in df.iterrows():
            ep = int(row["episode_index"])
            mn = row.get(f"stats/{feature}/min")
            mx = row.get(f"stats/{feature}/max")
            mn = float(mn[0]) if hasattr(mn, "__len__") else float(mn)
            mx = float(mx[0]) if hasattr(mx, "__len__") else float(mx)
            out[ep] = (mn, mx)
    return out


# ── Validation ──────────────────────────────────────────────────────────────


class TestValidation:
    def test_unknown_feature_raises(self, tmp_path, lerobot_dataset_factory):
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        with pytest.raises(ValueError, match="Unknown feature"):
            set_feature_values(ds, edits=[{"feature": "no_such", "from_index": 0, "to_index": 1, "value": 0}])

    def test_image_feature_rejected(self, tmp_path, lerobot_dataset_factory):
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        img = next((n for n, ft in ds.meta.features.items() if ft["dtype"] in ("image", "video")), None)
        if img is None:
            pytest.skip("factory built dataset without image/video features")
        with pytest.raises(ValueError, match="not editable"):
            set_feature_values(ds, edits=[{"feature": img, "from_index": 0, "to_index": 1, "value": None}])

    def test_out_of_range_rejected(self, tmp_path, lerobot_dataset_factory):
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        # action is reliably present.
        with pytest.raises(ValueError, match="Invalid range"):
            set_feature_values(
                ds,
                edits=[{"feature": "action", "from_index": -1, "to_index": 5, "value": 0.0}],
            )

    def test_inverted_range_rejected(self, tmp_path, lerobot_dataset_factory):
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        with pytest.raises(ValueError, match="Invalid range"):
            set_feature_values(
                ds,
                edits=[{"feature": "action", "from_index": 10, "to_index": 5, "value": 0.0}],
            )

    def test_empty_edit_list_is_noop(self, tmp_path, lerobot_dataset_factory):
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        # Should not raise; just returns.
        set_feature_values(ds, edits=[])

    def test_in_place_with_output_dir_rejected(self, tmp_path, lerobot_dataset_factory):
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        with pytest.raises(ValueError, match="output_dir is not used"):
            set_feature_values(
                ds,
                edits=[{"feature": "action", "from_index": 0, "to_index": 1, "value": 0.0}],
                in_place=True,
                output_dir=tmp_path / "fork",
            )

    def test_fork_without_output_dir_rejected(self, tmp_path, lerobot_dataset_factory):
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        with pytest.raises(ValueError, match="output_dir is required"):
            set_feature_values(
                ds,
                edits=[{"feature": "action", "from_index": 0, "to_index": 1, "value": 0.0}],
                in_place=False,
            )


# ── Core overwrite semantics ────────────────────────────────────────────────


class TestOverwrite:
    def test_vector_feature_overwrite_writes_to_target_range_only(self, tmp_path, lerobot_dataset_factory):
        """Action (vector) overwrite — confirm only the listed rows change."""
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=3, total_frames=30)
        action_feature = ds.meta.features["action"]
        action_dim = action_feature["shape"][0]

        before = _read_data_df(ds.root).copy()
        new_value = [0.42] * action_dim

        set_feature_values(
            ds,
            edits=[FeatureValueEdit(feature="action", from_index=5, to_index=12, value=new_value)],
        )

        after = _read_data_df(ds.root)

        # In range: every cell equals the new value (per-component).
        in_range = after[(after["index"] >= 5) & (after["index"] < 12)]
        for _, row in in_range.iterrows():
            cell = np.asarray(row["action"])
            assert cell.shape == (action_dim,)
            assert np.allclose(cell, 0.42)

        # Out of range: completely unchanged from the before snapshot.
        for idx in range(0, 5):
            row_before = before[before["index"] == idx].iloc[0]
            row_after = after[after["index"] == idx].iloc[0]
            assert np.allclose(np.asarray(row_after["action"]), np.asarray(row_before["action"]))
        for idx in range(12, len(before)):
            row_before = before[before["index"] == idx].iloc[0]
            row_after = after[after["index"] == idx].iloc[0]
            assert np.allclose(np.asarray(row_after["action"]), np.asarray(row_before["action"]))

    def test_half_open_boundary_excludes_to_index(self, tmp_path, lerobot_dataset_factory):
        """``[5, 10)`` writes frames 5..9, leaves frame 10 untouched."""
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        action_dim = ds.meta.features["action"]["shape"][0]

        before = _read_data_df(ds.root)
        marker = [-99.0] * action_dim
        original_at_10 = np.asarray(before[before["index"] == 10].iloc[0]["action"])

        set_feature_values(
            ds, edits=[{"feature": "action", "from_index": 5, "to_index": 10, "value": marker}]
        )

        after = _read_data_df(ds.root)

        # Frame 9 IS changed (last frame in half-open range).
        cell9 = np.asarray(after[after["index"] == 9].iloc[0]["action"])
        assert np.allclose(cell9, -99.0)
        # Frame 10 is NOT changed.
        cell10 = np.asarray(after[after["index"] == 10].iloc[0]["action"])
        assert np.allclose(cell10, original_at_10)

    def test_single_frame_edit(self, tmp_path, lerobot_dataset_factory):
        """``[N, N+1)`` is the canonical single-frame edit."""
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        action_dim = ds.meta.features["action"]["shape"][0]

        marker = [7.0] * action_dim
        set_feature_values(ds, edits=[{"feature": "action", "from_index": 7, "to_index": 8, "value": marker}])

        after = _read_data_df(ds.root)
        cell7 = np.asarray(after[after["index"] == 7].iloc[0]["action"])
        cell6 = np.asarray(after[after["index"] == 6].iloc[0]["action"])
        cell8 = np.asarray(after[after["index"] == 8].iloc[0]["action"])
        assert np.allclose(cell7, 7.0)
        assert not np.allclose(cell6, 7.0)
        assert not np.allclose(cell8, 7.0)

    def test_multiple_features_in_one_call(self, tmp_path, lerobot_dataset_factory):
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        action_dim = ds.meta.features["action"]["shape"][0]

        # Pick a non-image, non-action numeric feature to also edit.
        secondary = next(
            (
                n
                for n, ft in ds.meta.features.items()
                if n != "action"
                and ft["dtype"] not in ("image", "video", "string")
                and not n.startswith("observation.images")
            ),
            None,
        )
        if secondary is None:
            pytest.skip("no secondary editable feature in factory dataset")

        sec_shape = ds.meta.features[secondary]["shape"]
        sec_value = (
            0.5
            if (not sec_shape or sec_shape == [1])
            else [0.5] * (sec_shape[0] if isinstance(sec_shape[0], int) else 1)
        )

        set_feature_values(
            ds,
            edits=[
                {"feature": "action", "from_index": 3, "to_index": 6, "value": [1.5] * action_dim},
                {"feature": secondary, "from_index": 10, "to_index": 12, "value": sec_value},
            ],
        )

        after = _read_data_df(ds.root)
        # Both ranges took effect.
        for idx in range(3, 6):
            assert np.allclose(np.asarray(after[after["index"] == idx].iloc[0]["action"]), 1.5)


# ── Schema / videos must NOT change ──────────────────────────────────────────


class TestSchemaUnchanged:
    def test_info_json_untouched(self, tmp_path, lerobot_dataset_factory):
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        info_path = ds.root / "meta" / "info.json"
        before = info_path.read_bytes()

        set_feature_values(
            ds,
            edits=[
                {
                    "feature": "action",
                    "from_index": 0,
                    "to_index": 5,
                    "value": [0.0] * ds.meta.features["action"]["shape"][0],
                }
            ],
        )

        after = info_path.read_bytes()
        assert before == after, "info.json should never be rewritten by set_feature_values"

    def test_video_files_untouched(self, tmp_path, lerobot_dataset_factory):
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20, use_videos=True)
        videos = list((ds.root / "videos").rglob("*.mp4")) if (ds.root / "videos").exists() else []
        if not videos:
            pytest.skip("factory built dataset without video files")
        before_mtimes = {p: p.stat().st_mtime_ns for p in videos}

        set_feature_values(
            ds,
            edits=[
                {
                    "feature": "action",
                    "from_index": 0,
                    "to_index": 5,
                    "value": [0.0] * ds.meta.features["action"]["shape"][0],
                }
            ],
        )

        for p, mtime in before_mtimes.items():
            assert p.stat().st_mtime_ns == mtime, f"video {p} was modified"


# ── Stats recomputation ─────────────────────────────────────────────────────


class TestStatsRecomputed:
    def test_action_stats_reflect_new_min_max(self, tmp_path, lerobot_dataset_factory):
        """Overwriting frames in episode 0 with a marker value pulls the
        per-episode action-min and action-max in that direction."""
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=3, total_frames=30)
        action_dim = ds.meta.features["action"]["shape"][0]

        # Plant a wildly out-of-bound marker value in episode 0 (frames 0..3).
        marker_min = -123.456
        set_feature_values(
            ds,
            edits=[
                {
                    "feature": "action",
                    "from_index": 0,
                    "to_index": 3,
                    "value": [marker_min] * action_dim,
                }
            ],
        )

        stats = _read_episode_stats_min_max(ds.root, "action")
        # Episode 0's new min must be at most the marker value.
        ep0_min, _ = stats[0]
        assert ep0_min <= marker_min + 1e-6, f"expected ep0 stats/action/min ≤ {marker_min}, got {ep0_min}"


# ── Fork-mode (in_place=False) ───────────────────────────────────────────────


class TestForkMode:
    def test_fork_writes_to_output_and_leaves_original_intact(self, tmp_path, lerobot_dataset_factory):
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        action_dim = ds.meta.features["action"]["shape"][0]

        before_original = _read_data_df(ds.root).copy()
        fork_root = tmp_path / "fork"

        set_feature_values(
            ds,
            edits=[
                {
                    "feature": "action",
                    "from_index": 0,
                    "to_index": 5,
                    "value": [9.0] * action_dim,
                }
            ],
            in_place=False,
            output_dir=fork_root,
        )

        # Original is untouched.
        after_original = _read_data_df(ds.root)
        for i in range(len(before_original)):
            row_b = before_original.iloc[i]
            row_a = after_original.iloc[i]
            assert np.allclose(np.asarray(row_a["action"]), np.asarray(row_b["action"]))

        # Fork has the edits.
        forked = _read_data_df(fork_root)
        for idx in range(0, 5):
            assert np.allclose(np.asarray(forked[forked["index"] == idx].iloc[0]["action"]), 9.0)


# ── Subtask string resolution at Save time ──────────────────────────────────


class TestSubtaskStringResolution:
    """``subtask_index`` edits may stage strings; Save resolves them to ints
    against ``meta/subtasks.parquet`` (creating the file if missing)."""

    def test_string_value_resolved_to_index(self, tmp_path, lerobot_dataset_factory):
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        # Inject a subtask_index column into the schema. Factory doesn't include
        # it by default, so we splice it in.
        ds.meta.features["subtask_index"] = {"dtype": "int64", "shape": [1], "names": None}
        # Add a subtask_index column to every shard so the rewrite has something to
        # update. Initialize all rows to 0.
        for shard in (ds.root / "data").glob("*/*.parquet"):
            df = pd.read_parquet(shard)
            df["subtask_index"] = np.zeros(len(df), dtype=np.int64)
            df.to_parquet(shard, compression="snappy", index=False)

        # Stage a string edit for frames [0, 5).
        set_feature_values(
            ds,
            edits=[
                {
                    "feature": "subtask_index",
                    "from_index": 0,
                    "to_index": 5,
                    "value": "approach",
                }
            ],
        )

        # subtasks.parquet now exists with "approach" → index 0.
        subtasks_path = ds.root / "meta" / "subtasks.parquet"
        assert subtasks_path.exists(), "subtasks.parquet was not written"
        subtasks_df = pd.read_parquet(subtasks_path)
        names = list(subtasks_df.index)
        assert "approach" in names

        # And the data shard cells in [0, 5) hold that index.
        from pathlib import Path as _Path  # noqa: F401  (avoid Path shadowing in test)

        merged = pd.concat([pd.read_parquet(s) for s in (ds.root / "data").glob("*/*.parquet")])
        merged = merged.sort_values("index").reset_index(drop=True)
        approach_idx = int(subtasks_df.loc["approach", "subtask_index"])
        in_range = merged[(merged["index"] >= 0) & (merged["index"] < 5)]
        assert all(int(v) == approach_idx for v in in_range["subtask_index"])

    def test_concurrent_new_strings_assign_distinct_indices(self, tmp_path, lerobot_dataset_factory):
        """Two distinct new strings staged in one Save get distinct indices."""
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        ds.meta.features["subtask_index"] = {"dtype": "int64", "shape": [1], "names": None}
        for shard in (ds.root / "data").glob("*/*.parquet"):
            df = pd.read_parquet(shard)
            df["subtask_index"] = np.zeros(len(df), dtype=np.int64)
            df.to_parquet(shard, compression="snappy", index=False)

        set_feature_values(
            ds,
            edits=[
                {"feature": "subtask_index", "from_index": 0, "to_index": 3, "value": "approach"},
                {"feature": "subtask_index", "from_index": 5, "to_index": 8, "value": "grasp"},
            ],
        )

        subtasks = pd.read_parquet(ds.root / "meta" / "subtasks.parquet")
        names_to_idx = {str(name): int(row["subtask_index"]) for name, row in subtasks.iterrows()}
        assert "approach" in names_to_idx and "grasp" in names_to_idx
        assert names_to_idx["approach"] != names_to_idx["grasp"]


# ── Cross-file ordering: pass-1 writes all .tmp before any rename ───────────


class TestCrossFileOrdering:
    def test_no_orphan_tmp_after_successful_save(self, tmp_path, lerobot_dataset_factory):
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=3, total_frames=30)
        action_dim = ds.meta.features["action"]["shape"][0]

        set_feature_values(
            ds,
            edits=[{"feature": "action", "from_index": 5, "to_index": 12, "value": [0.5] * action_dim}],
        )

        orphans = list((ds.root / "data").rglob("*.tmp"))
        assert orphans == [], f"unexpected orphan .tmp files: {orphans}"

    def test_pass1_failure_leaves_no_committed_renames(self, tmp_path, lerobot_dataset_factory, monkeypatch):
        """Force a tmp-write error and confirm:

        * original shard files are byte-identical (no rename happened),
        * no orphan ``.tmp`` files are left behind.

        Patches ``pd.DataFrame.to_parquet`` to fail only on writes targeting a
        ``.tmp`` path inside the dataset's ``data/`` dir, isolating from the
        stats-recompute path which writes parquet elsewhere.
        """
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=3, total_frames=30)
        data_dir = ds.root / "data"
        before_bytes = {p: p.read_bytes() for p in data_dir.rglob("*.parquet")}

        action_dim = ds.meta.features["action"]["shape"][0]
        original = pd.DataFrame.to_parquet

        def fail_on_data_tmp_write(self, path, *args, **kwargs):
            target = Path(path)
            if str(target).endswith(".tmp") and data_dir in target.parents:
                raise OSError("simulated tmp-write failure")
            return original(self, path, *args, **kwargs)

        monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_on_data_tmp_write)

        with pytest.raises(OSError):
            set_feature_values(
                ds,
                edits=[{"feature": "action", "from_index": 0, "to_index": 5, "value": [9.0] * action_dim}],
            )

        monkeypatch.undo()

        # Original shard files must be byte-identical — pass-1 failed before any rename.
        for p, expected in before_bytes.items():
            assert p.read_bytes() == expected, f"{p} was modified despite pass-1 failure"

        # And no orphan .tmp left lying around.
        orphans = list((ds.root / "data").rglob("*.tmp"))
        assert orphans == [], f"orphan .tmp files: {orphans}"


class TestStatsRecomputationFailure:
    """``set_feature_values`` writes data shards in pass-2, then runs
    ``_recompute_episode_stats_from_data`` on every affected episode. A
    failure there is surfaced as ``StatsRecomputationError`` rather than
    silently logged — stale stats break the schema endpoint's
    ``observed_min``/``max`` chip and break normalization at training
    time, which are real silent-corruption paths.

    Contract: data is committed first; the exception fires AFTER. So
    callers always see fresh values on disk even if stats are stale.
    """

    def test_failure_raises_stats_recomputation_error(self, tmp_path, lerobot_dataset_factory, monkeypatch):
        from lerobot.datasets import feature_value_edits as fve

        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        action_dim = ds.meta.features["action"]["shape"][0]

        # Patch on the feature_value_edits module — that's where the import
        # lands in the function-local scope.
        def _boom(*args, **kwargs):
            raise RuntimeError("simulated stats failure")

        monkeypatch.setattr(
            "lerobot.datasets.dataset_tools._recompute_episode_stats_from_data",
            _boom,
        )

        # Re-import inside the function to test that the late-import path
        # picks up the monkey-patch. (set_feature_values does ``from
        # lerobot.datasets.dataset_tools import _recompute_episode_stats_from_data``
        # at call time.)
        with pytest.raises(fve.StatsRecomputationError) as exc_info:
            set_feature_values(
                ds,
                edits=[{"feature": "action", "from_index": 0, "to_index": 5, "value": [9.0] * action_dim}],
            )

        # The exception carries the list of affected episodes and the
        # underlying cause, so callers can decide what to do.
        err = exc_info.value
        assert isinstance(err.episodes, list) and err.episodes, err.episodes
        assert isinstance(err.cause, RuntimeError), type(err.cause)
        assert "simulated stats failure" in str(err.cause)

    def test_failure_still_commits_data_to_disk(self, tmp_path, lerobot_dataset_factory, monkeypatch):
        """Durability invariant: even when stats fail, the data shards
        already carry the new values. This is what makes the failure
        recoverable — the user can re-run stats computation manually
        rather than re-do their edits."""
        from lerobot.datasets import feature_value_edits as fve

        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        action_dim = ds.meta.features["action"]["shape"][0]
        new_value = [-7.5] * action_dim

        monkeypatch.setattr(
            "lerobot.datasets.dataset_tools._recompute_episode_stats_from_data",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("nope")),
        )

        with pytest.raises(fve.StatsRecomputationError):
            set_feature_values(
                ds,
                edits=[{"feature": "action", "from_index": 0, "to_index": 5, "value": new_value}],
            )

        # Data on disk: rows 0..4 have the new value.
        df = (
            pd.concat(
                [pd.read_parquet(p) for p in sorted((ds.root / "data").rglob("*.parquet"))],
                ignore_index=True,
            )
            .sort_values("index")
            .reset_index(drop=True)
        )
        for i in range(5):
            cell = np.asarray(df.iloc[i]["action"])
            assert np.allclose(cell, -7.5), f"frame {i} should have new value, got {cell}"

        # And no orphan .tmp left over (pass-2 rename completed before stats ran).
        orphans = list((ds.root / "data").rglob("*.tmp"))
        assert orphans == [], f"orphan .tmp files: {orphans}"

    def test_no_failure_returns_normally(self, tmp_path, lerobot_dataset_factory):
        """Sanity: in the normal happy-path, ``set_feature_values`` returns
        ``None`` and doesn't raise StatsRecomputationError."""
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        action_dim = ds.meta.features["action"]["shape"][0]
        # Should not raise.
        ret = set_feature_values(
            ds,
            edits=[{"feature": "action", "from_index": 0, "to_index": 3, "value": [0.1] * action_dim}],
        )
        assert ret is None


# ── Edge cases for subtask string resolution ────────────────────────────────


def _setup_dataset_with_subtask_index(tmp_path, lerobot_dataset_factory, *, with_lookup=True):
    """Build a factory dataset and inject ``subtask_index`` + (optionally) a lookup table.

    Returns the ``LeRobotDataset`` ready for ``set_feature_values`` to mutate.
    """
    ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
    ds.meta.features["subtask_index"] = {"dtype": "int64", "shape": [1], "names": None}
    for shard in (ds.root / "data").glob("*/*.parquet"):
        df = pd.read_parquet(shard)
        df["subtask_index"] = np.zeros(len(df), dtype=np.int64)
        df.to_parquet(shard, compression="snappy", index=False)
    if with_lookup:
        from lerobot.datasets.io_utils import load_subtasks

        subtasks = pd.DataFrame(
            {"subtask_index": [0, 1, 2]},
            index=pd.Index(["approach", "grasp", "release"], name="subtask"),
        )
        (ds.root / "meta").mkdir(exist_ok=True)
        subtasks.to_parquet(ds.root / "meta" / "subtasks.parquet")
        ds.meta.subtasks = load_subtasks(ds.root)
    return ds


class TestSubtaskResolutionEdgeCases:
    """Pin the contract for resolving string ``subtask_index`` edits at Save:

    * existing strings reuse their index (no renumbering, no churn),
    * new strings get appended with ``max(existing) + 1`` so prior data
      that references existing indices is never invalidated,
    * the same new string used in multiple staged edits collapses to a
      single appended row with one index,
    * a missing lookup table is created from scratch on first Save.
    """

    def test_existing_strings_reuse_indices_no_renumbering(self, tmp_path, lerobot_dataset_factory):
        ds = _setup_dataset_with_subtask_index(tmp_path, lerobot_dataset_factory)
        before = pd.read_parquet(ds.root / "meta" / "subtasks.parquet").copy()
        before_indices = {str(name): int(row["subtask_index"]) for name, row in before.iterrows()}

        # Stage edits that all use EXISTING strings.
        set_feature_values(
            ds,
            edits=[
                {"feature": "subtask_index", "from_index": 0, "to_index": 5, "value": "grasp"},
                {"feature": "subtask_index", "from_index": 5, "to_index": 10, "value": "release"},
            ],
        )

        # Lookup table is unchanged — existing indices are stable, no renumbering.
        after = pd.read_parquet(ds.root / "meta" / "subtasks.parquet")
        after_indices = {str(name): int(row["subtask_index"]) for name, row in after.iterrows()}
        assert after_indices == before_indices, (
            f"existing subtask indices must not be renumbered. before={before_indices} after={after_indices}"
        )

        # Data shards now have the resolved ints — the same ints existing referenced.
        merged = pd.concat([pd.read_parquet(s) for s in (ds.root / "data").glob("*/*.parquet")])
        merged = merged.sort_values("index").reset_index(drop=True)
        for fi in range(0, 5):
            assert int(merged.iloc[fi]["subtask_index"]) == before_indices["grasp"]
        for fi in range(5, 10):
            assert int(merged.iloc[fi]["subtask_index"]) == before_indices["release"]

    def test_new_string_appended_with_max_plus_one(self, tmp_path, lerobot_dataset_factory):
        """A new string gets ``max(existing_indices) + 1``, regardless of gaps.
        Critical: appending must never collide with existing rows."""
        ds = _setup_dataset_with_subtask_index(tmp_path, lerobot_dataset_factory)

        set_feature_values(
            ds,
            edits=[{"feature": "subtask_index", "from_index": 0, "to_index": 3, "value": "inspect knot"}],
        )

        after = pd.read_parquet(ds.root / "meta" / "subtasks.parquet")
        names_to_idx = {str(name): int(row["subtask_index"]) for name, row in after.iterrows()}
        assert names_to_idx["inspect knot"] == 3, (
            f"new string should land at max(existing)+1 = 3, got {names_to_idx['inspect knot']}"
        )
        # Original entries are byte-identical.
        assert names_to_idx["approach"] == 0
        assert names_to_idx["grasp"] == 1
        assert names_to_idx["release"] == 2

    def test_lookup_with_gaps_appends_after_max_not_in_gap(self, tmp_path, lerobot_dataset_factory):
        """If a user manually edited subtasks.parquet to skip indices, our
        next-index policy is ``max + 1`` (stable), NOT 'fill the gap' —
        gap-filling could collide with stale references in the data."""
        ds = _setup_dataset_with_subtask_index(tmp_path, lerobot_dataset_factory)
        # Manually rewrite the lookup with gaps: indices {0, 5, 7}.
        sparse = pd.DataFrame(
            {"subtask_index": [0, 5, 7]},
            index=pd.Index(["approach", "lift", "place"], name="subtask"),
        )
        sparse.to_parquet(ds.root / "meta" / "subtasks.parquet")

        set_feature_values(
            ds,
            edits=[{"feature": "subtask_index", "from_index": 0, "to_index": 2, "value": "wave"}],
        )

        after = pd.read_parquet(ds.root / "meta" / "subtasks.parquet")
        names_to_idx = {str(name): int(row["subtask_index"]) for name, row in after.iterrows()}
        assert names_to_idx["wave"] == 8, (
            f"new string should be max(existing 0,5,7)+1 = 8, got {names_to_idx['wave']}"
        )

    def test_same_new_string_in_multiple_edits_appends_once(self, tmp_path, lerobot_dataset_factory):
        """Multiple staged edits using the SAME new string in one Save must
        appear as a SINGLE row in the lookup with ONE index — no duplicates."""
        ds = _setup_dataset_with_subtask_index(tmp_path, lerobot_dataset_factory)

        set_feature_values(
            ds,
            edits=[
                {"feature": "subtask_index", "from_index": 0, "to_index": 3, "value": "wave"},
                {"feature": "subtask_index", "from_index": 5, "to_index": 8, "value": "wave"},
                {"feature": "subtask_index", "from_index": 10, "to_index": 12, "value": "wave"},
            ],
        )

        after = pd.read_parquet(ds.root / "meta" / "subtasks.parquet")
        rows = [str(name) for name in after.index]
        assert rows.count("wave") == 1, f"wave should appear exactly once in lookup; got rows={rows}"

        wave_idx = int(after.loc["wave", "subtask_index"])
        merged = pd.concat([pd.read_parquet(s) for s in (ds.root / "data").glob("*/*.parquet")])
        merged = merged.sort_values("index").reset_index(drop=True)
        # All three ranges resolved to the same int.
        for fi in (0, 1, 2, 5, 6, 7, 10, 11):
            assert int(merged.iloc[fi]["subtask_index"]) == wave_idx, (
                f"frame {fi} should hold the wave index {wave_idx}, got {merged.iloc[fi]['subtask_index']}"
            )

    def test_lookup_table_created_when_missing(self, tmp_path, lerobot_dataset_factory):
        """Dataset has subtask_index column but no lookup file yet — first Save
        with a string value must create the lookup with index 0."""
        ds = _setup_dataset_with_subtask_index(tmp_path, lerobot_dataset_factory, with_lookup=False)
        assert not (ds.root / "meta" / "subtasks.parquet").exists()

        set_feature_values(
            ds,
            edits=[{"feature": "subtask_index", "from_index": 0, "to_index": 5, "value": "first"}],
        )

        path = ds.root / "meta" / "subtasks.parquet"
        assert path.exists()
        df = pd.read_parquet(path)
        assert list(df.index) == ["first"]
        assert int(df.loc["first", "subtask_index"]) == 0

    def test_two_distinct_new_strings_in_one_save_get_distinct_indices(
        self, tmp_path, lerobot_dataset_factory
    ):
        """Two new strings staged together must resolve to TWO different indices
        appended in registration order, both ≥ ``max(existing) + 1``."""
        ds = _setup_dataset_with_subtask_index(tmp_path, lerobot_dataset_factory)

        set_feature_values(
            ds,
            edits=[
                {"feature": "subtask_index", "from_index": 0, "to_index": 3, "value": "alpha"},
                {"feature": "subtask_index", "from_index": 3, "to_index": 6, "value": "beta"},
            ],
        )

        after = pd.read_parquet(ds.root / "meta" / "subtasks.parquet")
        names_to_idx = {str(name): int(row["subtask_index"]) for name, row in after.iterrows()}
        assert names_to_idx["alpha"] == 3, names_to_idx
        assert names_to_idx["beta"] == 4, names_to_idx
        # And the original entries are intact.
        assert names_to_idx["approach"] == 0
        assert names_to_idx["release"] == 2

    def test_two_consecutive_saves_each_append_one(self, tmp_path, lerobot_dataset_factory):
        """Save 1 adds "alpha" → index 3. Save 2 adds "beta" → index 4. The
        second Save sees the post-Save-1 lookup and continues from there."""
        ds = _setup_dataset_with_subtask_index(tmp_path, lerobot_dataset_factory)

        set_feature_values(
            ds,
            edits=[{"feature": "subtask_index", "from_index": 0, "to_index": 2, "value": "alpha"}],
        )
        first = pd.read_parquet(ds.root / "meta" / "subtasks.parquet")
        alpha_idx = int(first.loc["alpha", "subtask_index"])

        set_feature_values(
            ds,
            edits=[{"feature": "subtask_index", "from_index": 5, "to_index": 7, "value": "beta"}],
        )
        second = pd.read_parquet(ds.root / "meta" / "subtasks.parquet")
        beta_idx = int(second.loc["beta", "subtask_index"])

        assert alpha_idx == 3
        assert beta_idx == 4
        # alpha row is unchanged after the second Save (no renumbering).
        assert int(second.loc["alpha", "subtask_index"]) == alpha_idx

    def test_same_string_across_two_saves_keeps_one_index(self, tmp_path, lerobot_dataset_factory):
        """The same new string staged in two separate Saves must end up as ONE
        row — the second Save sees the first's append and reuses it."""
        ds = _setup_dataset_with_subtask_index(tmp_path, lerobot_dataset_factory)

        set_feature_values(
            ds,
            edits=[{"feature": "subtask_index", "from_index": 0, "to_index": 3, "value": "wave"}],
        )
        after_first = pd.read_parquet(ds.root / "meta" / "subtasks.parquet")
        wave_idx_1 = int(after_first.loc["wave", "subtask_index"])

        set_feature_values(
            ds,
            edits=[{"feature": "subtask_index", "from_index": 5, "to_index": 8, "value": "wave"}],
        )
        after_second = pd.read_parquet(ds.root / "meta" / "subtasks.parquet")

        wave_rows = [n for n in after_second.index if n == "wave"]
        assert len(wave_rows) == 1, "wave must appear exactly once across two Saves"
        assert int(after_second.loc["wave", "subtask_index"]) == wave_idx_1

    def test_int_value_for_subtask_index_passes_through_unchanged(self, tmp_path, lerobot_dataset_factory):
        """If the caller bypasses the synthesis layer and writes an int directly,
        we don't try to look it up — the int is treated as the resolved index.
        This is the raw-API path; the lookup table is left alone."""
        ds = _setup_dataset_with_subtask_index(tmp_path, lerobot_dataset_factory)
        before_size = len(pd.read_parquet(ds.root / "meta" / "subtasks.parquet"))

        set_feature_values(
            ds,
            edits=[
                {"feature": "subtask_index", "from_index": 0, "to_index": 3, "value": 1}  # int
            ],
        )

        after = pd.read_parquet(ds.root / "meta" / "subtasks.parquet")
        assert len(after) == before_size, "lookup must not grow when value is an int"

        merged = pd.concat([pd.read_parquet(s) for s in (ds.root / "data").glob("*/*.parquet")])
        merged = merged.sort_values("index").reset_index(drop=True)
        for fi in (0, 1, 2):
            assert int(merged.iloc[fi]["subtask_index"]) == 1

    def test_mixed_int_and_string_values_in_one_save(self, tmp_path, lerobot_dataset_factory):
        """One staged edit uses int 1, another uses a new string — resolution
        only applies to the string edit. Both land correctly."""
        ds = _setup_dataset_with_subtask_index(tmp_path, lerobot_dataset_factory)

        set_feature_values(
            ds,
            edits=[
                {"feature": "subtask_index", "from_index": 0, "to_index": 3, "value": 1},  # int
                {"feature": "subtask_index", "from_index": 5, "to_index": 8, "value": "newish"},  # str
            ],
        )

        after = pd.read_parquet(ds.root / "meta" / "subtasks.parquet")
        names = list(after.index)
        assert "newish" in names
        assert int(after.loc["newish", "subtask_index"]) == 3

        merged = pd.concat([pd.read_parquet(s) for s in (ds.root / "data").glob("*/*.parquet")])
        merged = merged.sort_values("index").reset_index(drop=True)
        # First range got the int 1.
        for fi in (0, 1, 2):
            assert int(merged.iloc[fi]["subtask_index"]) == 1
        # Second range got the resolved newish index 3.
        for fi in (5, 6, 7):
            assert int(merged.iloc[fi]["subtask_index"]) == 3

    def test_string_value_when_dataset_has_no_subtask_index_feature_raises(
        self, tmp_path, lerobot_dataset_factory
    ):
        """The set_feature_values resolver assumes the dataset declares
        ``subtask_index`` in features. If it doesn't, attempting to stage
        a string value for it must raise — there's no place to write."""
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        # Don't add subtask_index to features.
        with pytest.raises(ValueError, match="Unknown feature"):
            set_feature_values(
                ds,
                edits=[{"feature": "subtask_index", "from_index": 0, "to_index": 3, "value": "x"}],
            )


# ── Regression: in-memory meta.subtasks must refresh after appending ────────


class TestSubtaskMetaInMemoryRefresh:
    def test_meta_subtasks_updated_after_append(self, tmp_path, lerobot_dataset_factory):
        """``dataset[i]`` decodes ``subtask_index → subtask`` via
        ``self._meta.subtasks.iloc[idx]``. If we write a new index to the data
        but forget to update the in-memory ``meta.subtasks``, the next read
        of any frame with the new index hits an out-of-bounds IndexError.
        Regression test for that path — discovered via the headless GUI flow
        when the auto-prefetcher hit it after a Save."""
        from lerobot.datasets.io_utils import load_subtasks

        ds = _setup_dataset_with_subtask_index(tmp_path, lerobot_dataset_factory)
        before_size = len(ds.meta.subtasks)

        set_feature_values(
            ds,
            edits=[{"feature": "subtask_index", "from_index": 0, "to_index": 3, "value": "wave"}],
        )

        # In-memory meta must reflect the new row immediately — not just the
        # disk file. Without the dataset.meta.subtasks = updated line in
        # _resolve_subtask_string_edits, this assertion fails and the next
        # dataset[i] would IndexError.
        assert len(ds.meta.subtasks) == before_size + 1, (
            f"in-memory meta.subtasks did not pick up appended row: "
            f"size={len(ds.meta.subtasks)} expected={before_size + 1}"
        )
        assert "wave" in ds.meta.subtasks.index

        # Sanity: disk and memory agree.
        on_disk = load_subtasks(ds.root)
        assert list(on_disk.index) == list(ds.meta.subtasks.index)

        # Functional check: decoding a frame with the new index must succeed.
        new_idx = int(ds.meta.subtasks.loc["wave", "subtask_index"])
        decoded = ds.meta.subtasks.iloc[new_idx].name
        assert decoded == "wave"
