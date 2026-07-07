# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for ``_migrate_next_success_inplace``.

Pins the contract of the bespoke ``next.success`` (per-frame bool, written by
``lerobot-eval`` rollouts) → ``success`` (per-episode int8 tri-state) migration.

The migration is lossy by design (frame index of the success transition is
discarded); these tests fix the data-layer behavior so any future "we made it
reversible" claim has a regression guard. The UX layer (banner copy / explicit
confirmation) is tracked separately as I1 in add_feature_review.md.
"""

from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.gui.api.datasets import _migrate_next_success_inplace

# ── Helpers ──────────────────────────────────────────────────────────


def _make_dataset_with_next_success(
    factory,
    root,
    *,
    success_pattern_per_episode: list[list[bool]],
    next_success_shape: tuple[int, ...] = (1,),
):
    """Build a tiny dataset with an action / observation.state / next.success schema.

    ``success_pattern_per_episode[ep][frame]`` controls the value of
    ``next.success`` at each frame. The dataset is finalized before return.
    """
    features = {
        "action": {"dtype": "float32", "shape": (2,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
        "next.success": {"dtype": "bool", "shape": next_success_shape, "names": None},
    }
    ds = factory(root=root, features=features)
    for ep_pattern in success_pattern_per_episode:
        for v in ep_pattern:
            cell = np.array([v], dtype=bool) if next_success_shape == (1,) else np.array(v, dtype=bool)
            ds.add_frame(
                {
                    "action": np.zeros(2, dtype=np.float32),
                    "observation.state": np.zeros(2, dtype=np.float32),
                    "next.success": cell,
                    "task": "t",
                }
            )
        ds.save_episode()
    ds.finalize()
    return ds


def _read_success_per_episode(ds) -> dict[int, set[int]]:
    """Return ``{episode_index: {success_value, ...}}`` across all data shards.

    A correctly-migrated per-episode success has a singleton set per episode.
    """
    out: dict[int, set[int]] = {}
    for f in (ds.root / "data").rglob("*.parquet"):
        t = pq.read_table(f, columns=["episode_index", "success"]).to_pandas()
        for ep, group in t.groupby("episode_index"):
            out.setdefault(int(ep), set()).update(int(v) for v in group["success"].tolist())
    return out


# ── Tests ────────────────────────────────────────────────────────────


class TestMigrationCorrectness:
    def test_episode_with_any_true_frame_becomes_success(self, tmp_path, empty_lerobot_dataset_factory):
        """One True frame anywhere in the episode → success=+1."""
        ep0 = [False] * 5 + [True] + [False] * 4  # True at frame 5
        ds = _make_dataset_with_next_success(
            empty_lerobot_dataset_factory,
            tmp_path / "ds",
            success_pattern_per_episode=[ep0],
        )
        _migrate_next_success_inplace(ds)
        assert _read_success_per_episode(ds) == {0: {1}}

    def test_all_false_episode_becomes_failure(self, tmp_path, empty_lerobot_dataset_factory):
        """All-False episode → success=-1."""
        ds = _make_dataset_with_next_success(
            empty_lerobot_dataset_factory,
            tmp_path / "ds",
            success_pattern_per_episode=[[False] * 8],
        )
        _migrate_next_success_inplace(ds)
        assert _read_success_per_episode(ds) == {0: {-1}}

    def test_mixed_episodes_collapse_independently(self, tmp_path, empty_lerobot_dataset_factory):
        """Two episodes: one success, one failure — collapse independently."""
        ds = _make_dataset_with_next_success(
            empty_lerobot_dataset_factory,
            tmp_path / "ds",
            success_pattern_per_episode=[
                [False, False, True, False],  # ep 0: success
                [False] * 4,  # ep 1: failure
            ],
        )
        _migrate_next_success_inplace(ds)
        assert _read_success_per_episode(ds) == {0: {1}, 1: {-1}}

    def test_all_true_episode_becomes_success(self, tmp_path, empty_lerobot_dataset_factory):
        """All-True is degenerate but valid: any-True dominates → +1."""
        ds = _make_dataset_with_next_success(
            empty_lerobot_dataset_factory,
            tmp_path / "ds",
            success_pattern_per_episode=[[True] * 5],
        )
        _migrate_next_success_inplace(ds)
        assert _read_success_per_episode(ds) == {0: {1}}


class TestMigrationLossiness:
    """Pins the lossy contract: the frame index of the success transition is
    discarded. Reversibility would be a regression — surface it via the UI,
    don't try to rebuild it from disk."""

    def test_next_success_column_is_dropped(self, tmp_path, empty_lerobot_dataset_factory):
        """Post-migration: next.success is gone from data shards and info.json."""
        ds = _make_dataset_with_next_success(
            empty_lerobot_dataset_factory,
            tmp_path / "ds",
            success_pattern_per_episode=[[False, True, False]],
        )
        _migrate_next_success_inplace(ds)

        for f in (ds.root / "data").rglob("*.parquet"):
            cols = pq.read_table(f).column_names
            assert "next.success" not in cols, f"next.success still in {f}"

        with open(ds.root / "meta" / "info.json") as f:
            info = json.load(f)
        assert "next.success" not in info["features"]
        assert "success" in info["features"]

    def test_frame_index_of_success_is_unrecoverable(self, tmp_path, empty_lerobot_dataset_factory):
        """Once migrated, two episodes that differed only in WHEN they
        succeeded look identical. Documents the lossy property explicitly:
        if this assertion ever flips (post-migration data carries the timing),
        someone made the migration reversible — that's a deliberate change,
        not a regression to fix."""
        # Episode A: success at frame 1
        ds_a = _make_dataset_with_next_success(
            empty_lerobot_dataset_factory,
            tmp_path / "ds_a",
            success_pattern_per_episode=[[False, True, False, False]],
        )
        _migrate_next_success_inplace(ds_a)
        # Episode B: success at frame 3
        ds_b = _make_dataset_with_next_success(
            empty_lerobot_dataset_factory,
            tmp_path / "ds_b",
            success_pattern_per_episode=[[False, False, False, True]],
        )
        _migrate_next_success_inplace(ds_b)
        # Both collapse to the same single value: +1.
        assert _read_success_per_episode(ds_a) == _read_success_per_episode(ds_b) == {0: {1}}


class TestMigrationSchema:
    def test_per_episode_hint_lands_in_info_json(self, tmp_path, empty_lerobot_dataset_factory):
        """The new success column is declared per_episode=True so range edits
        coerce correctly downstream (per the declared-wins-over-inference fix)."""
        ds = _make_dataset_with_next_success(
            empty_lerobot_dataset_factory,
            tmp_path / "ds",
            success_pattern_per_episode=[[True, False]],
        )
        _migrate_next_success_inplace(ds)
        with open(ds.root / "meta" / "info.json") as f:
            info = json.load(f)
        assert info["features"]["success"]["dtype"] == "int8"
        assert info["features"]["success"]["per_episode"] is True
        assert list(info["features"]["success"]["shape"]) == [1]

    def test_success_column_dtype_is_int8_on_disk(self, tmp_path, empty_lerobot_dataset_factory):
        """Disk dtype matches the declared int8 — no float drift through pandas."""
        ds = _make_dataset_with_next_success(
            empty_lerobot_dataset_factory,
            tmp_path / "ds",
            success_pattern_per_episode=[[False, True]],
        )
        _migrate_next_success_inplace(ds)
        for f in (ds.root / "data").rglob("*.parquet"):
            t = pq.read_table(f)
            assert "success" in t.column_names
            assert str(t.schema.field("success").type) == "int8"

    def test_stats_for_success_present_in_episodes_parquet(self, tmp_path, empty_lerobot_dataset_factory):
        """Per-episode stats columns exist for the new success feature
        (proves Pass 4 of the migration ran after the column was rewritten,
        not before — otherwise stats would reflect the all-zeros fill)."""
        ds = _make_dataset_with_next_success(
            empty_lerobot_dataset_factory,
            tmp_path / "ds",
            success_pattern_per_episode=[[False, True]],
        )
        _migrate_next_success_inplace(ds)
        eps_dir = ds.root / "meta" / "episodes"
        for f in eps_dir.rglob("*.parquet"):
            cols = pq.read_table(f).column_names
            assert any(c.startswith("stats/success/") for c in cols), (
                f"no stats/success/* columns in {f}: {cols}"
            )


class TestMigrationBoolLayouts:
    """next.success may be stored as scalar bool or shape-[1] list-of-bool
    depending on how it was recorded. _scalarize is supposed to handle both."""

    def test_shape1_bool_layout_collapses_correctly(self, tmp_path, empty_lerobot_dataset_factory):
        """shape=(1,) bool — typical lerobot-eval output — collapses correctly."""
        ds = _make_dataset_with_next_success(
            empty_lerobot_dataset_factory,
            tmp_path / "ds",
            success_pattern_per_episode=[[False, True, False]],
            next_success_shape=(1,),
        )
        _migrate_next_success_inplace(ds)
        assert _read_success_per_episode(ds) == {0: {1}}
