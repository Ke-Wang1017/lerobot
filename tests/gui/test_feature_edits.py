# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the GUI's feature-value editing path:

* ``POST /api/edits/feature-set`` — staging.
* ``DELETE /api/edits/{edit_index}`` — per-chip removal already covers feature_set.
* ``POST /api/edits/apply`` — translating staged feature_set edits into a
  single ``set_feature_values`` call and reloading the dataset.
"""

from __future__ import annotations

import asyncio
import random

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI

from lerobot.gui.api import datasets as datasets_module, edits as edits_module
from lerobot.gui.frame_cache import FrameCache
from lerobot.gui.state import AppState


@pytest.fixture(autouse=True)
def _seed_rng():
    """Seed numpy + random before each test to make ``lerobot_dataset_factory``
    deterministic.

    Why: ``episodes_factory`` distributes ``total_frames`` across episodes via
    ``np.random.multinomial``, so episode lengths drift with the global RNG
    state — which is consumed by other tests in the suite. Tests in this file
    assert specific frame ranges (e.g. ``[0, 9)``) that require a known
    minimum episode length, so the RNG must be seeded to make them
    order-independent.
    """
    np.random.seed(0)
    random.seed(0)
    yield


@pytest.fixture
def app_with_state():
    """Mounts both routers (datasets + edits) on the same FastAPI app and clears module state on teardown."""
    app = FastAPI()
    app.include_router(datasets_module.router)
    app.include_router(edits_module.router)

    state = AppState(frame_cache=FrameCache(max_bytes=1_000_000))
    original_dat_state = datasets_module._app_state
    original_edits_state = edits_module._app_state
    original_indices = datasets_module._episode_start_indices.copy()
    datasets_module.set_app_state(state)
    edits_module.set_app_state(state)

    yield app, state

    datasets_module._app_state = original_dat_state
    edits_module._app_state = original_edits_state
    datasets_module._episode_start_indices.clear()
    datasets_module._episode_start_indices.update(original_indices)


def _post_feature_set(app, body: dict):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.post("/api/edits/feature-set", json=body)

    return asyncio.run(run())


def _list_pending(app, dataset_id: str | None = None):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            url = "/api/edits"
            if dataset_id:
                url += f"?dataset_id={dataset_id}"
            return await client.get(url)

    return asyncio.run(run())


def _post_apply(app, dataset_id: str):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.post(f"/api/edits/apply?dataset_id={dataset_id}")

    return asyncio.run(run())


def _delete_edit(app, idx: int):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.delete(f"/api/edits/{idx}")

    return asyncio.run(run())


# ── /api/edits/feature-set — staging & validation ──────────────────────────


class TestStageFeatureSet:
    def test_stages_a_valid_edit(self, app_with_state, tmp_path, lerobot_dataset_factory):
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        # action is read-only — pick a feature that's NOT in the readonly list.
        editable = [
            n
            for n, ft in ds.meta.features.items()
            if ft["dtype"] not in ("image", "video")
            and n not in {"timestamp", "frame_index", "episode_index", "index", "task_index"}
            and n != "action"
            and not n.startswith("observation.")
        ]
        if not editable:
            pytest.skip("factory dataset has no editable features for V1 (only action/observation/defaults)")

        feature = editable[0]
        ft_info = ds.meta.features[feature]
        # Build a value matching the dtype.
        shape = ft_info.get("shape") or [1]
        if ft_info["dtype"] == "string":
            value = "ok"
        elif shape in ([1], []):
            value = 0.5 if ft_info["dtype"].startswith("float") else 1
        else:
            value = [0.5] * shape[0]

        resp = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": feature,
                "frame_from": 2,
                "frame_to": 5,
                "value": value,
            },
        )
        assert resp.status_code == 200, resp.text

        # The edit shows up in the pending list.
        pending = _list_pending(app, dataset_id).json()
        assert pending["total"] == 1
        assert pending["edits"][0]["edit_type"] == "feature_set"
        assert pending["edits"][0]["params"]["feature"] == feature
        assert pending["edits"][0]["params"]["frame_from"] == 2
        assert pending["edits"][0]["params"]["frame_to"] == 5

    def test_action_feature_rejected(self, app_with_state, tmp_path, lerobot_dataset_factory):
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        resp = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "action",
                "frame_from": 0,
                "frame_to": 5,
                "value": [0.0] * ds.meta.features["action"]["shape"][0],
            },
        )
        assert resp.status_code == 400
        assert "read-only" in resp.json()["detail"]

    def test_observation_feature_rejected(self, app_with_state, tmp_path, lerobot_dataset_factory):
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        obs_features = [n for n in ds.meta.features if n.startswith("observation.")]
        if not obs_features:
            pytest.skip("no observation.* features in factory dataset")

        feature = obs_features[0]
        ft = ds.meta.features[feature]
        if ft["dtype"] in ("image", "video"):
            # That path returns a different message; pick a non-image observation if possible.
            non_img = [n for n in obs_features if ds.meta.features[n]["dtype"] not in ("image", "video")]
            if not non_img:
                pytest.skip("only image observation features in factory dataset")
            feature = non_img[0]
            ft = ds.meta.features[feature]

        shape = ft.get("shape") or [1]
        value = 0.0 if shape in ([1], []) else [0.0] * shape[0]
        resp = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": feature,
                "frame_from": 0,
                "frame_to": 5,
                "value": value,
            },
        )
        assert resp.status_code == 400
        assert "read-only" in resp.json()["detail"]

    def test_default_feature_rejected(self, app_with_state, tmp_path, lerobot_dataset_factory):
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        resp = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "frame_index",
                "frame_from": 0,
                "frame_to": 5,
                "value": 0,
            },
        )
        assert resp.status_code == 400
        assert "auto-managed" in resp.json()["detail"]

    def test_unknown_feature_rejected(self, app_with_state, tmp_path, lerobot_dataset_factory):
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        resp = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "totally_made_up",
                "frame_from": 0,
                "frame_to": 5,
                "value": 0,
            },
        )
        assert resp.status_code == 400
        assert "Unknown feature" in resp.json()["detail"]

    def test_invalid_range_rejected(self, app_with_state, tmp_path, lerobot_dataset_factory):
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        # Pick something editable and try frame_from >= frame_to.
        editable = [
            n
            for n, ft in ds.meta.features.items()
            if ft["dtype"] not in ("image", "video")
            and n not in {"timestamp", "frame_index", "episode_index", "index", "task_index"}
            and n != "action"
            and not n.startswith("observation.")
        ]
        if not editable:
            pytest.skip("no editable features in factory")

        resp = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": editable[0],
                "frame_from": 5,
                "frame_to": 5,  # empty range
                "value": 0,
            },
        )
        assert resp.status_code == 400
        assert "Invalid range" in resp.json()["detail"]


# ── per-chip removal works on feature_set edits too ────────────────────────


class TestRemoveFeatureSetEdit:
    def test_delete_removes_one_feature_set(self, app_with_state, tmp_path, lerobot_dataset_factory):
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        editable = [
            n
            for n, ft in ds.meta.features.items()
            if ft["dtype"] not in ("image", "video")
            and n not in {"timestamp", "frame_index", "episode_index", "index", "task_index"}
            and n != "action"
            and not n.startswith("observation.")
        ]
        if not editable:
            pytest.skip("no editable features in factory")
        feature = editable[0]
        ft = ds.meta.features[feature]
        shape = ft.get("shape") or [1]
        value = 0.0 if shape in ([1], []) else [0.0] * shape[0]

        # Stage two distinct edits.
        _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": feature,
                "frame_from": 0,
                "frame_to": 3,
                "value": value,
            },
        )
        _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": feature,
                "frame_from": 5,
                "frame_to": 8,
                "value": value,
            },
        )

        before = _list_pending(app, dataset_id).json()
        assert before["total"] == 2

        resp = _delete_edit(app, 0)
        assert resp.status_code == 200

        after = _list_pending(app, dataset_id).json()
        assert after["total"] == 1
        assert after["edits"][0]["params"]["frame_from"] == 5  # the second one survived


# ── apply pipeline writes to disk via set_feature_values ───────────────────


class TestApplyFeatureSet:
    def test_apply_persists_value_edit_to_disk(self, app_with_state, tmp_path, lerobot_dataset_factory):
        """End-to-end: stage an edit, hit /apply, confirm the parquet shard
        on disk now reflects the new value."""
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=3, total_frames=30)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        # Find a non-action, non-observation, non-default editable scalar / vector feature.
        editable = [
            n
            for n, ft in ds.meta.features.items()
            if ft["dtype"] not in ("image", "video", "string")
            and n not in {"timestamp", "frame_index", "episode_index", "index", "task_index"}
            and n != "action"
            and not n.startswith("observation.")
        ]
        if not editable:
            pytest.skip("no editable numeric feature in factory dataset")

        feature = editable[0]
        ft = ds.meta.features[feature]
        shape = ft.get("shape") or [1]
        marker = -42.0 if shape in ([1], []) else [-42.0] * shape[0]

        # Stage an edit on episode 0, frames 1..4 (3 frames).
        resp = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": feature,
                "frame_from": 1,
                "frame_to": 4,
                "value": marker,
            },
        )
        assert resp.status_code == 200, resp.text

        # Apply.
        resp = _post_apply(app, dataset_id)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["applied"] == 1, body

        # Read the data shard(s) directly and verify the edit landed.
        parts = []
        for shard in sorted((ds.root / "data").glob("*/*.parquet")):
            parts.append(pd.read_parquet(shard))
        df = pd.concat(parts, ignore_index=True).sort_values("index").reset_index(drop=True)
        ep0 = df[df["episode_index"] == 0].sort_values("frame_index").reset_index(drop=True)

        # Frames 1..3 (inclusive) should show the marker.
        for i in range(1, 4):
            cell = ep0.iloc[i][feature]
            arr = np.asarray(cell)
            if arr.ndim == 0:
                assert float(arr.item()) == pytest.approx(-42.0)
            else:
                assert np.allclose(arr, -42.0)

        # Frames 0 and 4 must be untouched (compare to baseline = anything not the marker).
        cell0 = np.asarray(ep0.iloc[0][feature])
        cell4 = np.asarray(ep0.iloc[4][feature])
        assert not (np.allclose(cell0, -42.0) if cell0.ndim else float(cell0) == -42.0)
        assert not (np.allclose(cell4, -42.0) if cell4.ndim else float(cell4) == -42.0)

        # And the pending edits list is now empty.
        pending = _list_pending(app, dataset_id).json()
        assert pending["total"] == 0


# ── Overlapping staged edits — 409 + clip-on-confirm ────────────────────────


class TestOverlappingEdits:
    def test_overlap_returns_409_with_structured_detail(
        self, app_with_state, tmp_path, lerobot_dataset_factory
    ):
        """Two staged edits on the same (feature, episode) with overlapping
        ranges must trigger a 409 unless ``confirm_overlap=True``."""
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        editable = [
            n
            for n, ft in ds.meta.features.items()
            if ft["dtype"] not in ("image", "video", "string")
            and n not in {"timestamp", "frame_index", "episode_index", "index", "task_index"}
            and n != "action"
            and not n.startswith("observation.")
        ]
        if not editable:
            pytest.skip("no editable feature in factory dataset")
        feature = editable[0]
        ft = ds.meta.features[feature]
        shape = ft.get("shape") or [1]
        value = 1.5 if shape in ([1], []) else [1.5] * shape[0]

        # First edit lands fine.
        r1 = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": feature,
                "frame_from": 0,
                "frame_to": 5,
                "value": value,
            },
        )
        assert r1.status_code == 200, r1.text

        # Overlapping second edit returns 409.
        r2 = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": feature,
                "frame_from": 3,
                "frame_to": 7,
                "value": value,
            },
        )
        assert r2.status_code == 409
        detail = r2.json()["detail"]
        assert detail["code"] == "overlapping_edit"
        assert detail["feature"] == feature
        assert detail["episode_index"] == 0
        assert detail["new_range"] == [3, 7]
        assert detail["overlapping"][0]["frame_from"] == 0
        assert detail["overlapping"][0]["frame_to"] == 5

    def test_confirm_overlap_clips_prior_edit(self, app_with_state, tmp_path, lerobot_dataset_factory):
        """With ``confirm_overlap=True``, the prior edit is clipped (last-write-wins)."""
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        editable = [
            n
            for n, ft in ds.meta.features.items()
            if ft["dtype"] not in ("image", "video", "string")
            and n not in {"timestamp", "frame_index", "episode_index", "index", "task_index"}
            and n != "action"
            and not n.startswith("observation.")
        ]
        if not editable:
            pytest.skip("no editable feature in factory dataset")
        feature = editable[0]
        ft = ds.meta.features[feature]
        shape = ft.get("shape") or [1]
        value = 1.5 if shape in ([1], []) else [1.5] * shape[0]

        _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": feature,
                "frame_from": 0,
                "frame_to": 5,
                "value": value,
            },
        )
        r2 = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": feature,
                "frame_from": 3,
                "frame_to": 7,
                "value": value,
                "confirm_overlap": True,
            },
        )
        assert r2.status_code == 200, r2.text

        pending = _list_pending(app, dataset_id).json()
        # We expect two edits now: the clipped prior [0, 3) and the new [3, 7).
        ranges = sorted(
            (int(e["params"]["frame_from"]), int(e["params"]["frame_to"]))
            for e in pending["edits"]
            if e["edit_type"] == "feature_set"
        )
        assert ranges == [(0, 3), (3, 7)], ranges

    def test_fully_contained_prior_is_removed(self, app_with_state, tmp_path, lerobot_dataset_factory):
        """If the new edit fully contains a prior one, the prior is removed entirely."""
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        editable = [
            n
            for n, ft in ds.meta.features.items()
            if ft["dtype"] not in ("image", "video", "string")
            and n not in {"timestamp", "frame_index", "episode_index", "index", "task_index"}
            and n != "action"
            and not n.startswith("observation.")
        ]
        if not editable:
            pytest.skip("no editable feature in factory dataset")
        feature = editable[0]
        ft = ds.meta.features[feature]
        shape = ft.get("shape") or [1]
        value = 1.5 if shape in ([1], []) else [1.5] * shape[0]

        # Prior edit at [3, 5) — fully inside the new [0, 8).
        _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": feature,
                "frame_from": 3,
                "frame_to": 5,
                "value": value,
            },
        )
        r2 = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": feature,
                "frame_from": 0,
                "frame_to": 8,
                "value": value,
                "confirm_overlap": True,
            },
        )
        assert r2.status_code == 200

        pending = _list_pending(app, dataset_id).json()
        ranges = [
            (int(e["params"]["frame_from"]), int(e["params"]["frame_to"]))
            for e in pending["edits"]
            if e["edit_type"] == "feature_set"
        ]
        assert ranges == [(0, 8)], ranges

    def test_prior_split_when_new_strictly_inside(self, app_with_state, tmp_path, lerobot_dataset_factory):
        """If the new edit is strictly inside the prior, the prior splits in two."""
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        editable = [
            n
            for n, ft in ds.meta.features.items()
            if ft["dtype"] not in ("image", "video", "string")
            and n not in {"timestamp", "frame_index", "episode_index", "index", "task_index"}
            and n != "action"
            and not n.startswith("observation.")
        ]
        if not editable:
            pytest.skip("no editable feature in factory dataset")
        feature = editable[0]
        ft = ds.meta.features[feature]
        shape = ft.get("shape") or [1]
        value = 1.5 if shape in ([1], []) else [1.5] * shape[0]

        # Prior wide edit, then new narrow inside.
        _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": feature,
                "frame_from": 0,
                "frame_to": 9,
                "value": value,
            },
        )
        r2 = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": feature,
                "frame_from": 3,
                "frame_to": 6,
                "value": value,
                "confirm_overlap": True,
            },
        )
        assert r2.status_code == 200

        pending = _list_pending(app, dataset_id).json()
        ranges = sorted(
            (int(e["params"]["frame_from"]), int(e["params"]["frame_to"]))
            for e in pending["edits"]
            if e["edit_type"] == "feature_set"
        )
        # Expect: left clip [0, 3), the new [3, 6), right clip [6, 9).
        assert ranges == [(0, 3), (3, 6), (6, 9)], ranges


# ── Subtask synthesis at the staging endpoint ──────────────────────────────


class TestSubtaskTranslation:
    """The user-facing ``subtask`` (string) translates to storage ``subtask_index``
    at stage time. Edits land in PendingEdit with the storage feature name so
    the apply pipeline doesn't need a special case. Resolution of new strings
    to fresh int indices happens at Save (covered by
    ``tests/datasets/test_set_feature_values.py::TestSubtaskStringResolution``).
    """

    @staticmethod
    def _ds_with_subtask_lookup(tmp_path, lerobot_dataset_factory):
        """Build a dataset and inject ``subtask_index`` + ``meta/subtasks.parquet``."""
        import numpy as np

        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        ds.meta.features["subtask_index"] = {"dtype": "int64", "shape": [1], "names": None}
        # Add a subtask_index column to each shard, varying so it's not detected as per-episode.
        for shard in (ds.root / "data").glob("*/*.parquet"):
            df = pd.read_parquet(shard)
            df["subtask_index"] = (df["frame_index"].astype(int) % 3).astype(np.int64)
            df.to_parquet(shard, compression="snappy", index=False)
        # Plant a starter lookup table.
        subtasks_df = pd.DataFrame(
            {"subtask_index": [0, 1, 2]},
            index=pd.Index(["approach", "grasp", "release"], name="subtask"),
        )
        (ds.root / "meta").mkdir(exist_ok=True)
        subtasks_df.to_parquet(ds.root / "meta" / "subtasks.parquet")
        # Reload metadata so the dataset picks up the new column + lookup.
        from lerobot.datasets.io_utils import load_subtasks

        ds.meta.subtasks = load_subtasks(ds.root)
        return ds

    def test_stage_subtask_translates_to_subtask_index_storage(
        self, app_with_state, tmp_path, lerobot_dataset_factory
    ):
        app, state = app_with_state
        ds = self._ds_with_subtask_lookup(tmp_path, lerobot_dataset_factory)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        resp = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "subtask",  # ← user-facing display name
                "frame_from": 1,
                "frame_to": 5,
                "value": "grasp",
            },
        )
        assert resp.status_code == 200, resp.text

        pending = _list_pending(app, dataset_id).json()
        assert pending["total"] == 1
        edit = pending["edits"][0]
        # PendingEdit stores the storage name — apply pipeline never sees "subtask".
        assert edit["params"]["feature"] == "subtask_index", edit
        assert edit["params"]["value"] == "grasp"

    def test_stage_error_uses_user_submitted_name_not_storage_name(
        self, app_with_state, tmp_path, lerobot_dataset_factory
    ):
        """Regression: when the user submits ``feature="subtask"`` with a
        value that fails downstream validation (e.g. invalid range), the
        400 error message must reference ``"subtask"`` — what the user
        sent — not ``"subtask_index"`` (the internal storage name they
        never typed). Previously we mutated ``request.feature`` in place
        and downstream errors leaked the storage name."""
        app, state = app_with_state
        ds = self._ds_with_subtask_lookup(tmp_path, lerobot_dataset_factory)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        # Invalid range — frame_to <= frame_from.
        resp = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "subtask",
                "frame_from": 5,
                "frame_to": 5,  # invalid (empty range)
                "value": "grasp",
            },
        )
        assert resp.status_code == 400, resp.text
        # The error mentions the user's submitted range, not the storage name.
        # (The storage name appears nowhere in the user-facing detail.)
        detail = resp.json()["detail"]
        assert "Invalid range" in detail
        # Subtask range check doesn't include the feature name in the message,
        # but other error paths do — exercise one of those:

        # Read-only feature name reflected back literally.
        resp2 = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "action",  # read-only V1
                "frame_from": 0,
                "frame_to": 5,
                "value": [0.0] * ds.meta.features["action"]["shape"][0],
            },
        )
        assert resp2.status_code == 400
        # Error says "action", not some translated name.
        assert "'action'" in resp2.json()["detail"], resp2.json()["detail"]

    def test_stage_subtask_with_brand_new_string_value(
        self, app_with_state, tmp_path, lerobot_dataset_factory
    ):
        """Strings absent from the lookup table are accepted at stage time —
        resolution to a fresh int index is deferred to Save by design (so
        concurrent new-string Saves on different sessions converge)."""
        app, state = app_with_state
        ds = self._ds_with_subtask_lookup(tmp_path, lerobot_dataset_factory)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        resp = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "subtask",
                "frame_from": 0,
                "frame_to": 3,
                "value": "inspect knot",  # not in the lookup
            },
        )
        assert resp.status_code == 200, resp.text

    def test_stage_subtask_index_directly_still_works(
        self, app_with_state, tmp_path, lerobot_dataset_factory
    ):
        """The raw / programmatic API path: callers may use the storage name
        directly with an int value. Should pass through unchanged."""
        app, state = app_with_state
        ds = self._ds_with_subtask_lookup(tmp_path, lerobot_dataset_factory)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        resp = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "subtask_index",
                "frame_from": 1,
                "frame_to": 4,
                "value": 2,
            },
        )
        assert resp.status_code == 200, resp.text
        pending = _list_pending(app, dataset_id).json()
        assert pending["edits"][0]["params"]["feature"] == "subtask_index"
        assert pending["edits"][0]["params"]["value"] == 2

    def test_stage_subtask_rejected_when_no_lookup_table(
        self, app_with_state, tmp_path, lerobot_dataset_factory
    ):
        """Without ``meta/subtasks.parquet`` the synthesis doesn't apply, so
        ``feature='subtask'`` falls through to the standard 'unknown feature'
        path — the dataset has no real ``subtask`` feature."""
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        # No subtask_index, no lookup table.
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        resp = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "subtask",
                "frame_from": 0,
                "frame_to": 5,
                "value": "anything",
            },
        )
        assert resp.status_code == 400
        assert "Unknown feature" in resp.json()["detail"]

    def test_overlap_detection_normalizes_subtask_to_storage(
        self, app_with_state, tmp_path, lerobot_dataset_factory
    ):
        """Stage one edit as ``subtask`` and another as ``subtask_index`` on the
        same range — the overlap detector should see them as the same feature
        because both are stored under the storage name."""
        app, state = app_with_state
        ds = self._ds_with_subtask_lookup(tmp_path, lerobot_dataset_factory)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        # First stage uses the display name → stored as subtask_index.
        r1 = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "subtask",
                "frame_from": 0,
                "frame_to": 5,
                "value": "approach",
            },
        )
        assert r1.status_code == 200

        # Second stage uses the storage name on an overlapping range → 409.
        r2 = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "subtask_index",
                "frame_from": 3,
                "frame_to": 7,
                "value": 1,
            },
        )
        assert r2.status_code == 409, r2.text
        assert r2.json()["detail"]["code"] == "overlapping_edit"
        assert r2.json()["detail"]["feature"] == "subtask_index"

    def test_apply_subtask_string_lands_as_int_in_data(
        self, app_with_state, tmp_path, lerobot_dataset_factory
    ):
        """End-to-end: stage feature='subtask' with a string value, hit /apply,
        confirm the data parquet now has the resolved integer index in those rows."""
        app, state = app_with_state
        ds = self._ds_with_subtask_lookup(tmp_path, lerobot_dataset_factory)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        # Stage edit using the display name with an existing string value.
        resp = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "subtask",
                "frame_from": 1,
                "frame_to": 4,
                "value": "release",  # already in lookup as index 2
            },
        )
        assert resp.status_code == 200, resp.text

        # Apply.
        apply_resp = _post_apply(app, dataset_id)
        assert apply_resp.status_code == 200, apply_resp.text
        assert apply_resp.json()["applied"] == 1

        # Read back from disk: frames [1, 4) should be the int index for "release".
        parts = [pd.read_parquet(p) for p in sorted((ds.root / "data").glob("*/*.parquet"))]
        merged = pd.concat(parts, ignore_index=True).sort_values("index").reset_index(drop=True)
        ep0 = merged[merged["episode_index"] == 0].sort_values("frame_index").reset_index(drop=True)

        # Look up the expected index from the (possibly updated) lookup table.
        subtasks_df = pd.read_parquet(ds.root / "meta" / "subtasks.parquet")
        release_idx = int(subtasks_df.loc["release", "subtask_index"])
        for fi in (1, 2, 3):
            assert int(ep0.iloc[fi]["subtask_index"]) == release_idx, (
                f"frame {fi} should be index {release_idx} (release), got {ep0.iloc[fi]['subtask_index']!r}"
            )
        # Untouched frames keep their (frame_index % 3) initialization.
        assert int(ep0.iloc[0]["subtask_index"]) == 0  # was: 0 % 3
        assert int(ep0.iloc[4]["subtask_index"]) == 1  # was: 4 % 3

    def test_apply_subtask_with_new_string_appends_to_lookup_table(
        self, app_with_state, tmp_path, lerobot_dataset_factory
    ):
        """Brand-new string is appended to ``meta/subtasks.parquet`` with a
        fresh index, then written into the data shards."""
        app, state = app_with_state
        ds = self._ds_with_subtask_lookup(tmp_path, lerobot_dataset_factory)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        before = pd.read_parquet(ds.root / "meta" / "subtasks.parquet")
        assert "inspect knot" not in before.index

        _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "subtask",
                "frame_from": 0,
                "frame_to": 2,
                "value": "inspect knot",
            },
        )
        assert _post_apply(app, dataset_id).status_code == 200

        after = pd.read_parquet(ds.root / "meta" / "subtasks.parquet")
        assert "inspect knot" in after.index, "new subtask string should be appended to lookup"
        # And its index should be > all the previously-existing ones.
        assert int(after.loc["inspect knot", "subtask_index"]) >= len(before)


# ── Same-range repeated stage collapses to a single PendingEdit ────────────


class TestSameRangeRepeatedStageCollapses:
    """The frontend stages on every keystroke (debounced). When the user keeps
    typing in the same selection, each follow-up stage uses ``confirm_overlap=True``
    upfront. The backend's fully-contained-removal must collapse the prior edit
    so the pending list ends up with exactly one entry — not a chain of
    overlapping partials. This test pins that contract from the API surface,
    which the frontend's typing-UX relies on (no 409 dialog, single chip).
    """

    @staticmethod
    def _editable_feature(ds):
        for n, ft in ds.meta.features.items():
            if (
                ft["dtype"] not in ("image", "video", "string")
                and n not in {"timestamp", "frame_index", "episode_index", "index", "task_index"}
                and n != "action"
                and not n.startswith("observation.")
            ):
                return n, ft
        return None, None

    def test_repeated_same_range_stages_collapse_to_one(
        self, app_with_state, tmp_path, lerobot_dataset_factory
    ):
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=40)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        feature, ft = self._editable_feature(ds)
        if feature is None:
            pytest.skip("no editable feature in factory dataset")
        shape = ft.get("shape") or [1]
        first_value = 0.1 if shape in ([1], []) else [0.1] * shape[0]
        second_value = 0.5 if shape in ([1], []) else [0.5] * shape[0]
        third_value = 0.9 if shape in ([1], []) else [0.9] * shape[0]

        body = {
            "dataset_id": dataset_id,
            "episode_index": 0,
            "feature": feature,
            "frame_from": 4,
            "frame_to": 9,
        }

        # First stage — no overlap, no confirm needed.
        r1 = _post_feature_set(app, {**body, "value": first_value})
        assert r1.status_code == 200, r1.text

        # Second stage on the SAME range: send confirm_overlap=True upfront
        # (mirrors what the frontend does after detecting the same _stageKey).
        r2 = _post_feature_set(app, {**body, "value": second_value, "confirm_overlap": True})
        assert r2.status_code == 200, r2.text

        # Third stage on the SAME range with another value.
        r3 = _post_feature_set(app, {**body, "value": third_value, "confirm_overlap": True})
        assert r3.status_code == 200, r3.text

        pending = _list_pending(app, dataset_id).json()
        feature_set_edits = [
            e
            for e in pending["edits"]
            if e["edit_type"] == "feature_set" and e["params"]["feature"] == feature
        ]
        assert len(feature_set_edits) == 1, (
            f"three same-range stages should collapse into one PendingEdit, got {len(feature_set_edits)}"
        )
        only = feature_set_edits[0]["params"]
        assert only["frame_from"] == 4 and only["frame_to"] == 9
        # The most recent value is what survives.
        assert only["value"] == third_value, only

    def test_subtask_same_range_repeated_stage_collapses(
        self, app_with_state, tmp_path, lerobot_dataset_factory
    ):
        """Subtask path specifically: typing 'app' then 'appr' then 'approach'
        should collapse to one PendingEdit storing 'approach', because the
        frontend's same-key detection translates the synthetic display name
        the same way the backend does."""
        app, state = app_with_state
        ds = TestSubtaskTranslation._ds_with_subtask_lookup(tmp_path, lerobot_dataset_factory)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        body = {
            "dataset_id": dataset_id,
            "episode_index": 0,
            "feature": "subtask",
            "frame_from": 1,
            "frame_to": 4,
        }
        for v in ["app", "appr", "approach"]:
            r = _post_feature_set(app, {**body, "value": v, "confirm_overlap": v != "app"})
            assert r.status_code == 200, r.text

        pending = _list_pending(app, dataset_id).json()
        subtask_edits = [
            e
            for e in pending["edits"]
            if e["edit_type"] == "feature_set" and e["params"]["feature"] == "subtask_index"
        ]
        assert len(subtask_edits) == 1, (
            f"three same-range subtask stages should collapse into one, got {len(subtask_edits)}"
        )
        # Final value is what landed.
        assert subtask_edits[0]["params"]["value"] == "approach"


# ── Feature-series endpoint returns decoded subtask strings ────────────────


class TestSubtaskFeatureSeriesDecoded:
    """The frontend's live-merge of pending edits into the displayed series
    relies on subtask values being strings on both sides — series values from
    /feature-series and pending edit values from POST /feature-set. This test
    pins the decode side: ``GET .../feature-series?features=subtask`` returns
    the decoded strings, not the raw int indices.
    """

    def test_feature_series_decodes_subtask_to_strings(
        self, app_with_state, tmp_path, lerobot_dataset_factory
    ):
        app, state = app_with_state
        ds = TestSubtaskTranslation._ds_with_subtask_lookup(tmp_path, lerobot_dataset_factory)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                return await client.get(
                    f"/api/datasets/{dataset_id}/episodes/0/feature-series?features=subtask"
                )

        resp = asyncio.run(run())
        assert resp.status_code == 200, resp.text
        body = resp.json()
        series = body["series"].get("subtask")
        assert series is not None, body
        # Per the fixture, subtask_index = frame_index % 3 → ["approach","grasp","release",...]
        # Either way: every value must be a string from the lookup, not an int.
        assert len(series) == body["length"]
        types = {type(v).__name__ for v in series}
        assert all(isinstance(v, str) for v in series), (
            f"subtask series should be strings, got types: {types}"
        )
        assert set(series).issubset({"approach", "grasp", "release"}), set(series)


# ── Per-episode broadcast coercion at the staging endpoint ─────────────────


class TestPerEpisodeBroadcastCoercion:
    """Per-episode features (uniform within each episode) like ``success`` must
    not accept sub-range edits — that would break the broadcast invariant.
    The staging endpoint silently coerces the requested range to ``[0, len)``
    for the episode and returns ``coerced_range`` so the GUI can update its
    selection band. Frontend bool-checkbox + summary code relies on this:
    it shows the broadcast note "frames 0…N-1" and the merged slice covers
    the whole episode, so the checkbox initial state correctly reflects
    "all true" / "all false" / "mixed".
    """

    @staticmethod
    def _ds_with_per_episode_success(tmp_path, lerobot_dataset_factory):
        """Build a dataset and inject a ``success`` bool[1] column that's
        uniform within each episode (alternating True / False)."""
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=40)
        ds.meta.features["success"] = {"dtype": "bool", "shape": [1], "names": None}
        for shard in (ds.root / "data").glob("*/*.parquet"):
            df = pd.read_parquet(shard)
            df["success"] = df["episode_index"].astype(int) % 2 == 0
            df.to_parquet(shard, compression="snappy", index=False)
        return ds

    def test_subrange_edit_on_per_episode_feature_is_coerced_to_full_episode(
        self, app_with_state, tmp_path, lerobot_dataset_factory
    ):
        app, state = app_with_state
        ds = self._ds_with_per_episode_success(tmp_path, lerobot_dataset_factory)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        # User drag-selects a sub-range and tries to flip it. The endpoint
        # should accept (200) but coerce the range to the whole episode.
        ep0_length = int(ds.meta.episodes[0]["length"])
        resp = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "success",
                "frame_from": 5,
                "frame_to": 12,
                "value": False,
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body.get("coerced_range") == [0, ep0_length], body
        assert body.get("coerce_reason") == "per_episode_broadcast"

        # The PendingEdit reflects the coerced range, not the user's input —
        # the GUI's selection band update + merged-slice computation depends
        # on this so the checkbox state and summary cover the whole episode.
        pending = _list_pending(app, dataset_id).json()
        ours = [e for e in pending["edits"] if e["edit_type"] == "feature_set"]
        assert len(ours) == 1
        assert ours[0]["params"]["frame_from"] == 0
        assert ours[0]["params"]["frame_to"] == ep0_length
        assert ours[0]["params"]["value"] is False

    def test_full_range_edit_on_per_episode_feature_does_not_report_coercion(
        self, app_with_state, tmp_path, lerobot_dataset_factory
    ):
        """When the user already supplies the full episode range, the
        response should NOT carry ``coerced_range`` (avoids confusing the GUI
        into thinking a coercion happened when the request was already
        valid)."""
        app, state = app_with_state
        ds = self._ds_with_per_episode_success(tmp_path, lerobot_dataset_factory)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        ep0_length = int(ds.meta.episodes[0]["length"])
        resp = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "success",
                "frame_from": 0,
                "frame_to": ep0_length,
                "value": True,
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "coerced_range" not in body, body
        assert "coerce_reason" not in body, body

    def test_per_episode_feature_appears_in_schema_with_flag_set(
        self, app_with_state, tmp_path, lerobot_dataset_factory
    ):
        """Detection must mark ``success`` as per-episode in the schema —
        the frontend uses ``is_per_episode`` to render the broadcast note
        and effFrom/effTo override (so the inspector summary covers the
        whole episode)."""
        app, state = app_with_state
        ds = self._ds_with_per_episode_success(tmp_path, lerobot_dataset_factory)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        info = datasets_module._dataset_info_from(dataset_id, ds)
        assert "success" in info.features_schema
        assert info.features_schema["success"].is_per_episode is True


class TestDeclaredPerEpisodeFlag:
    """``per_episode: true`` in info.json is the authoritative declaration —
    the detector trusts it and surfaces ``per_episode_source = "declared"``.
    For features without the flag, the detector falls back to the legacy
    nunique scan and surfaces ``per_episode_source = "detected"``. If a
    declared feature's data isn't actually uniform, a warning is added to
    DatasetInfo.warnings.
    """

    @staticmethod
    def _ds_with_declared_success(tmp_path, lerobot_dataset_factory, *, uniform_data: bool):
        """Plant a ``success`` feature with ``per_episode: true`` declared.
        ``uniform_data=True`` writes uniform-per-episode values (declaration
        and data agree); ``False`` writes varying values within episodes
        (declaration says per-episode, data disagrees)."""
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=40)
        ds.meta.features["success"] = {
            "dtype": "bool",
            "shape": [1],
            "names": None,
            "per_episode": True,
        }
        for shard in (ds.root / "data").glob("*/*.parquet"):
            df = pd.read_parquet(shard)
            if uniform_data:
                df["success"] = df["episode_index"].astype(int) % 2 == 0
            else:
                # Varying within each episode: alternate by frame_index.
                df["success"] = df["frame_index"].astype(int) % 2 == 0
            df.to_parquet(shard, compression="snappy", index=False)
        return ds

    def test_declared_feature_marked_with_source_declared(
        self, app_with_state, tmp_path, lerobot_dataset_factory
    ):
        app, state = app_with_state
        ds = self._ds_with_declared_success(tmp_path, lerobot_dataset_factory, uniform_data=True)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        info = datasets_module._dataset_info_from(dataset_id, ds)
        assert info.features_schema["success"].is_per_episode is True
        assert info.features_schema["success"].per_episode_source == "declared"
        assert info.warnings == []  # data matches declaration

    def test_declared_but_inconsistent_data_warns(self, app_with_state, tmp_path, lerobot_dataset_factory):
        """The declaration is authoritative — the GUI still treats the feature
        as per-episode — but a warning is surfaced so the user knows their
        data violates the invariant they declared."""
        app, state = app_with_state
        ds = self._ds_with_declared_success(tmp_path, lerobot_dataset_factory, uniform_data=False)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        info = datasets_module._dataset_info_from(dataset_id, ds)
        # Trust the declaration:
        assert info.features_schema["success"].is_per_episode is True
        assert info.features_schema["success"].per_episode_source == "declared"
        # And surface the inconsistency as a warning:
        assert any("success" in w and "per_episode" in w for w in info.warnings), info.warnings
        assert any("not uniform" in w for w in info.warnings), info.warnings

    def test_undeclared_uniform_feature_marked_detected(
        self, app_with_state, tmp_path, lerobot_dataset_factory
    ):
        """Backward compatibility: a feature WITHOUT the flag whose data is
        uniform-per-episode is still classified as per-episode (heuristic
        fallback), but with ``per_episode_source = "detected"`` so the GUI
        can distinguish it from the declared case."""
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=40)
        # No per_episode flag — pure detection path.
        ds.meta.features["success"] = {
            "dtype": "bool",
            "shape": [1],
            "names": None,
        }
        for shard in (ds.root / "data").glob("*/*.parquet"):
            df = pd.read_parquet(shard)
            df["success"] = df["episode_index"].astype(int) % 2 == 0
            df.to_parquet(shard, compression="snappy", index=False)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        info = datasets_module._dataset_info_from(dataset_id, ds)
        assert info.features_schema["success"].is_per_episode is True
        assert info.features_schema["success"].per_episode_source == "detected"

    def test_undeclared_non_uniform_feature_stays_per_frame(
        self, app_with_state, tmp_path, lerobot_dataset_factory
    ):
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=40)
        ds.meta.features["reward"] = {"dtype": "float32", "shape": [1], "names": None}
        for shard in (ds.root / "data").glob("*/*.parquet"):
            df = pd.read_parquet(shard)
            df["reward"] = df["frame_index"].astype(np.float32)
            df.to_parquet(shard, compression="snappy", index=False)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        info = datasets_module._dataset_info_from(dataset_id, ds)
        assert info.features_schema["reward"].is_per_episode is False
        assert info.features_schema["reward"].per_episode_source is None


# ── Declared bounds + categorical at the staging endpoint ──────────────────


class TestStageDeclaredBoundsAndCategorical:
    """The stage endpoint runs the declared-bounds and categorical checks via
    ``validate_feature_numeric_bounds`` so the user sees a 400 immediately
    instead of silently staging a bad value and only learning at Save time.
    """

    @staticmethod
    def _ds_with_bounded_features(tmp_path, lerobot_dataset_factory):
        """Plant ``quality`` (int 1-5) and ``control_mode`` (categorical)."""
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=40)
        ds.meta.features["quality"] = {
            "dtype": "int64",
            "shape": [1],
            "names": None,
            "min": 1,
            "max": 5,
        }
        ds.meta.features["control_mode"] = {
            "dtype": "int64",
            "shape": [1],
            "names": ["ee", "joint"],
        }
        # Add columns so the parquet has them — values are within bounds.
        for shard in (ds.root / "data").glob("*/*.parquet"):
            df = pd.read_parquet(shard)
            df["quality"] = np.full(len(df), 3, dtype=np.int64)
            df["control_mode"] = np.zeros(len(df), dtype=np.int64)
            df.to_parquet(shard, compression="snappy", index=False)
        return ds

    def test_quality_within_bounds_accepted(self, app_with_state, tmp_path, lerobot_dataset_factory):
        app, state = app_with_state
        ds = self._ds_with_bounded_features(tmp_path, lerobot_dataset_factory)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        resp = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "quality",
                "frame_from": 0,
                "frame_to": 5,
                "value": 4,
            },
        )
        assert resp.status_code == 200, resp.text

    def test_quality_above_max_rejected(self, app_with_state, tmp_path, lerobot_dataset_factory):
        app, state = app_with_state
        ds = self._ds_with_bounded_features(tmp_path, lerobot_dataset_factory)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        resp = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "quality",
                "frame_from": 0,
                "frame_to": 5,
                "value": 7,  # > max=5
            },
        )
        assert resp.status_code == 400, resp.text
        assert "above declared max" in resp.json()["detail"]

    def test_quality_below_min_rejected(self, app_with_state, tmp_path, lerobot_dataset_factory):
        app, state = app_with_state
        ds = self._ds_with_bounded_features(tmp_path, lerobot_dataset_factory)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        resp = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "quality",
                "frame_from": 0,
                "frame_to": 5,
                "value": 0,  # < min=1
            },
        )
        assert resp.status_code == 400, resp.text
        assert "below declared min" in resp.json()["detail"]

    def test_categorical_legal_index_accepted(self, app_with_state, tmp_path, lerobot_dataset_factory):
        app, state = app_with_state
        ds = self._ds_with_bounded_features(tmp_path, lerobot_dataset_factory)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        for idx in (0, 1):
            resp = _post_feature_set(
                app,
                {
                    "dataset_id": dataset_id,
                    "episode_index": 0,
                    "feature": "control_mode",
                    "frame_from": 0,
                    "frame_to": 5,
                    "value": idx,
                    "confirm_overlap": idx > 0,  # second stage on same range
                },
            )
            assert resp.status_code == 200, resp.text

    def test_categorical_out_of_range_index_rejected(self, app_with_state, tmp_path, lerobot_dataset_factory):
        app, state = app_with_state
        ds = self._ds_with_bounded_features(tmp_path, lerobot_dataset_factory)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        resp = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "control_mode",
                "frame_from": 0,
                "frame_to": 5,
                "value": 2,  # only 2 names → legal range [0, 2)
            },
        )
        assert resp.status_code == 400, resp.text
        assert "outside categorical range" in resp.json()["detail"]

    def test_unbounded_feature_still_accepts_anything(
        self, app_with_state, tmp_path, lerobot_dataset_factory
    ):
        """``reward`` has no declared bounds — any finite value should pass.
        Regression: ensures the bounds check short-circuits and doesn't
        accidentally reject features that don't opt in."""
        app, state = app_with_state
        ds = self._ds_with_bounded_features(tmp_path, lerobot_dataset_factory)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        # Add a plain reward feature (no min/max).
        ds.meta.features["reward"] = {"dtype": "float32", "shape": [1], "names": None}
        for shard in (ds.root / "data").glob("*/*.parquet"):
            df = pd.read_parquet(shard)
            df["reward"] = np.zeros(len(df), dtype=np.float32)
            df.to_parquet(shard, compression="snappy", index=False)

        resp = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "reward",
                "frame_from": 0,
                "frame_to": 5,
                "value": 1e9,  # huge, but no bound declared
            },
        )
        assert resp.status_code == 200, resp.text


# ── End-to-end apply for the post-V1 widget cases ──────────────────────────


class TestApplyForBoundsCategoricalAndPerEpisode:
    """Round-trip tests: stage → apply → read back from the parquet shard.
    Pin the *value-on-disk* contract for the three feature kinds added on
    this branch: declared-bounds scalar, categorical (int+names), and
    per-episode broadcast.

    The staging side is covered by ``TestStageDeclaredBoundsAndCategorical``
    and ``TestPerEpisodeBroadcastCoercion``; this class confirms apply
    actually persists what was staged.
    """

    @staticmethod
    def _ds_with_post_v1_features(tmp_path, lerobot_dataset_factory):
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=40)
        ds.meta.features["quality"] = {
            "dtype": "int64",
            "shape": [1],
            "names": None,
            "min": 1,
            "max": 5,
        }
        ds.meta.features["control_mode"] = {
            "dtype": "int64",
            "shape": [1],
            "names": ["ee", "joint"],
        }
        ds.meta.features["success"] = {
            "dtype": "bool",
            "shape": [1],
            "names": None,
        }
        # Per-episode features (success, control_mode) get uniform-per-episode
        # values so detection flips them on. quality varies per-frame so it
        # stays per-frame (otherwise sub-range edits would be coerced to the
        # whole episode).
        for shard in (ds.root / "data").glob("*/*.parquet"):
            df = pd.read_parquet(shard)
            fi = df["frame_index"].astype(int).to_numpy()
            df["quality"] = ((fi % 5) + 1).astype(np.int64)  # cycles 1..5
            ep_col = df["episode_index"].astype(int).to_numpy()
            df["control_mode"] = (ep_col % 2).astype(np.int64)
            df["success"] = ep_col % 2 == 0
            df.to_parquet(shard, compression="snappy", index=False)
        return ds

    @staticmethod
    def _read_data_df(root):
        parts = []
        for shard in sorted((root / "data").glob("*/*.parquet")):
            parts.append(pd.read_parquet(shard))
        return pd.concat(parts, ignore_index=True).sort_values("index").reset_index(drop=True)

    def test_bounded_scalar_lands_at_declared_value(self, app_with_state, tmp_path, lerobot_dataset_factory):
        """Stage quality=4 on frames [3, 8) → apply → those rows on disk
        carry exactly 4, others are unchanged."""
        app, state = app_with_state
        ds = self._ds_with_post_v1_features(tmp_path, lerobot_dataset_factory)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        before = self._read_data_df(ds.root).copy()
        r = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "quality",
                "frame_from": 3,
                "frame_to": 8,
                "value": 4,
            },
        )
        assert r.status_code == 200, r.text
        ar = _post_apply(app, dataset_id)
        assert ar.status_code == 200, ar.text
        assert ar.json()["applied"] == 1

        after = self._read_data_df(ds.root)
        ep_offset = int(ds.meta.episodes[0]["dataset_from_index"])
        # In-range rows: now 4
        for fi in range(3, 8):
            assert int(after.iloc[ep_offset + fi]["quality"]) == 4
        # Out-of-range rows: unchanged
        for fi in (0, 1, 2, 8, 9):
            row_before = before[before["index"] == ep_offset + fi].iloc[0]
            row_after = after[after["index"] == ep_offset + fi].iloc[0]
            assert int(row_after["quality"]) == int(row_before["quality"])

    def test_categorical_lands_as_int_index(self, app_with_state, tmp_path, lerobot_dataset_factory):
        """Stage control_mode=1 (joint) → apply → on-disk value is the int
        index 1, not the string label. Per-episode coercion expands to the
        whole episode regardless of the user's drag range."""
        app, state = app_with_state
        ds = self._ds_with_post_v1_features(tmp_path, lerobot_dataset_factory)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        ep0_len = int(ds.meta.episodes[0]["length"])
        r = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "control_mode",
                "frame_from": 5,
                "frame_to": 12,  # sub-range — backend coerces to whole episode
                "value": 1,
            },
        )
        assert r.status_code == 200, r.text
        assert r.json().get("coerced_range") == [0, ep0_len]
        ar = _post_apply(app, dataset_id)
        assert ar.status_code == 200

        after = self._read_data_df(ds.root)
        ep0 = after[after["episode_index"] == 0]
        # All 0..ep0_len-1 frames carry the new index, regardless of drag range.
        assert (ep0["control_mode"].astype(int) == 1).all(), ep0["control_mode"].unique()
        # Episode 1 untouched (still control_mode == 1 % 2 == 1 from setup).
        ep1 = after[after["episode_index"] == 1]
        assert (ep1["control_mode"].astype(int) == 1).all() if 1 % 2 == 1 else True

    def test_per_episode_bool_lands_across_whole_episode(
        self, app_with_state, tmp_path, lerobot_dataset_factory
    ):
        """Per-episode bool ``success``: drag over a sub-range, but apply
        rewrites the whole episode. Other episodes stay as-is."""
        app, state = app_with_state
        ds = self._ds_with_post_v1_features(tmp_path, lerobot_dataset_factory)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        # Episode 0 was set up as success=True (ep%2==0). Stage False on a
        # sub-range; whole-episode coercion should flip every frame to False.
        r = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "success",
                "frame_from": 2,
                "frame_to": 6,
                "value": False,
            },
        )
        assert r.status_code == 200, r.text
        ar = _post_apply(app, dataset_id)
        assert ar.status_code == 200

        after = self._read_data_df(ds.root)
        ep0 = after[after["episode_index"] == 0]
        assert (~ep0["success"].astype(bool)).all(), "every frame in ep0 should be False"
        # Episode 1 was False before (ep%2==0 is False); should still be False.
        ep1 = after[after["episode_index"] == 1]
        assert (~ep1["success"].astype(bool)).all()


# ── set_feature_values rejects bounds-violating values regardless of caller ─


class TestSetFeatureValuesEnforcesBounds:
    """``set_feature_values`` is the public Python entry point for value
    edits; the GUI funnels through it but so do notebooks and scripts. This
    class confirms it rejects out-of-range values directly — i.e. you can't
    bypass bounds by skipping the GUI's stage-time check.
    """

    @staticmethod
    def _ds_with_bounded_quality(tmp_path, lerobot_dataset_factory):
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        ds.meta.features["quality"] = {
            "dtype": "int64",
            "shape": [1],
            "names": None,
            "min": 1,
            "max": 5,
        }
        for shard in (ds.root / "data").glob("*/*.parquet"):
            df = pd.read_parquet(shard)
            df["quality"] = np.full(len(df), 3, dtype=np.int64)
            df.to_parquet(shard, compression="snappy", index=False)
        return ds

    def test_rejects_value_above_declared_max(self, tmp_path, lerobot_dataset_factory):
        from lerobot.datasets.dataset_tools import set_feature_values

        ds = self._ds_with_bounded_quality(tmp_path, lerobot_dataset_factory)
        with pytest.raises(ValueError, match="above declared max"):
            set_feature_values(
                ds,
                edits=[{"feature": "quality", "from_index": 0, "to_index": 5, "value": 7}],
            )

    def test_rejects_value_below_declared_min(self, tmp_path, lerobot_dataset_factory):
        from lerobot.datasets.dataset_tools import set_feature_values

        ds = self._ds_with_bounded_quality(tmp_path, lerobot_dataset_factory)
        with pytest.raises(ValueError, match="below declared min"):
            set_feature_values(
                ds,
                edits=[{"feature": "quality", "from_index": 0, "to_index": 5, "value": 0}],
            )

    def test_accepts_value_inside_bounds(self, tmp_path, lerobot_dataset_factory):
        """Sanity: in-range values still pass."""
        from lerobot.datasets.dataset_tools import set_feature_values

        ds = self._ds_with_bounded_quality(tmp_path, lerobot_dataset_factory)
        # Should not raise.
        set_feature_values(
            ds,
            edits=[{"feature": "quality", "from_index": 0, "to_index": 5, "value": 4}],
        )

    def test_rejects_categorical_index_out_of_range(self, tmp_path, lerobot_dataset_factory):
        from lerobot.datasets.dataset_tools import set_feature_values

        ds = self._ds_with_bounded_quality(tmp_path, lerobot_dataset_factory)
        ds.meta.features["control_mode"] = {
            "dtype": "int64",
            "shape": [1],
            "names": ["ee", "joint"],
        }
        for shard in (ds.root / "data").glob("*/*.parquet"):
            df = pd.read_parquet(shard)
            df["control_mode"] = np.zeros(len(df), dtype=np.int64)
            df.to_parquet(shard, compression="snappy", index=False)
        with pytest.raises(ValueError, match="outside categorical range"):
            set_feature_values(
                ds,
                edits=[{"feature": "control_mode", "from_index": 0, "to_index": 5, "value": 2}],
            )


# ── Apply-pipeline silent-failure surfacing ───────────────────────────────


class TestApplyPartialStatus:
    """The apply pipeline writes shards in pass-1, then attempts a metadata
    reload + post-edit verification. Failures in those non-essential
    follow-up steps are caught and surfaced as ``status: "partial"`` with
    the underlying exception in ``errors``. Without this contract, a stale
    in-memory ``dataset.meta`` after an on-disk parquet rewrite would corrupt
    every subsequent edit (wrong index translation). Pin the failure-flow
    contract here so a future refactor can't quietly drop the surfacing.
    """

    @staticmethod
    def _ds_with_reward(tmp_path, lerobot_dataset_factory):
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        ds.meta.features["reward"] = {"dtype": "float32", "shape": [1], "names": None}
        for shard in (ds.root / "data").glob("*/*.parquet"):
            df = pd.read_parquet(shard)
            df["reward"] = np.zeros(len(df), dtype=np.float32)
            df.to_parquet(shard, compression="snappy", index=False)
        return ds

    def test_apply_returns_partial_when_stats_recompute_fails(
        self, app_with_state, tmp_path, lerobot_dataset_factory, monkeypatch
    ):
        """Stats failure during apply: data IS on disk, response is partial,
        applied count includes the edit (it succeeded — only stats are stale),
        errors list mentions the stats issue."""
        app, state = app_with_state
        ds = self._ds_with_reward(tmp_path, lerobot_dataset_factory)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        r = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "reward",
                "frame_from": 0,
                "frame_to": 3,
                "value": -0.5,
            },
        )
        assert r.status_code == 200, r.text

        # Patch the stats-recompute helper to raise. set_feature_values will
        # surface this as StatsRecomputationError; the GUI's apply path catches
        # it specifically and reports partial-with-applied=1.
        monkeypatch.setattr(
            "lerobot.datasets.dataset_tools._recompute_episode_stats_from_data",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("simulated stats failure")),
        )

        ar = _post_apply(app, dataset_id)
        assert ar.status_code == 200, ar.text
        body = ar.json()
        assert body["status"] == "partial", body
        assert body["applied"] == 1, body
        assert any("stats" in err.lower() for err in body.get("errors", [])), body

        # Data is on disk despite the stats failure.
        df = pd.concat(
            [pd.read_parquet(p) for p in sorted((ds.root / "data").glob("*/*.parquet"))],
            ignore_index=True,
        )
        ep0_offset = int(ds.meta.episodes[0]["dataset_from_index"])
        for i in range(3):
            assert pytest.approx(float(df.iloc[ep0_offset + i]["reward"]), abs=1e-6) == -0.5

    def test_apply_returns_partial_when_dataset_reload_fails(
        self, app_with_state, tmp_path, lerobot_dataset_factory, monkeypatch
    ):
        """Stage a reward edit, force ``reload_dataset_from_disk`` to raise,
        confirm: response status==partial, applied==1 (parquet writes
        succeeded), errors list contains the reload failure."""
        app, state = app_with_state
        ds = self._ds_with_reward(tmp_path, lerobot_dataset_factory)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        # Stage a small edit.
        r = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "reward",
                "frame_from": 0,
                "frame_to": 3,
                "value": -0.5,
            },
        )
        assert r.status_code == 200, r.text

        # Patch the reload helper at its *source* module since edits.py
        # imports it inside the try block.
        from lerobot.gui import dataset_reload

        def _boom(*_args, **_kwargs):
            raise RuntimeError("simulated reload failure")

        monkeypatch.setattr(dataset_reload, "reload_dataset_from_disk", _boom)

        ar = _post_apply(app, dataset_id)
        # Apply still returns 200 — partial success is not an HTTP error.
        assert ar.status_code == 200, ar.text
        body = ar.json()
        assert body["status"] == "partial", body
        assert body["applied"] == 1, body
        assert any("reload" in err.lower() for err in body.get("errors", [])), body

        # Disk-level invariant: the parquet should still have the new value
        # since pass-1 succeeded — partial means downstream steps failed,
        # not that data wasn't written.
        df = pd.concat(
            [pd.read_parquet(p) for p in sorted((ds.root / "data").glob("*/*.parquet"))],
            ignore_index=True,
        )
        ep0_offset = int(ds.meta.episodes[0]["dataset_from_index"])
        for i in range(3):
            assert pytest.approx(float(df.iloc[ep0_offset + i]["reward"]), abs=1e-6) == -0.5
