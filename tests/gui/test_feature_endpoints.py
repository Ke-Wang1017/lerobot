# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the per-frame feature-value endpoint and its JSON coercion helper.

The endpoint exists so the front-end Inspector can show "what is this feature
at the current frame" without round-tripping through video decode. Schema is
served via the dataset-open response (Phase A1); these tests cover the
per-frame *values* (Phase A2).
"""

from __future__ import annotations

import asyncio

import httpx
import numpy as np
import pytest
import torch
from fastapi import FastAPI

from lerobot.gui.api import datasets as datasets_module
from lerobot.gui.api.datasets import _coerce_feature_value_to_json
from lerobot.gui.frame_cache import FrameCache
from lerobot.gui.state import AppState

# ── _coerce_feature_value_to_json (pure helper) ────────────────────────────


class TestCoerceFeatureValueToJson:
    """Schema-driven Inspector cards rely on these conversions."""

    def test_torch_scalar_tensor_returns_python_scalar(self) -> None:
        # shape [1] tensors come out of dataset[i] for scalar features.
        assert _coerce_feature_value_to_json(torch.tensor([0.5]), "float32") == pytest.approx(0.5)
        assert _coerce_feature_value_to_json(torch.tensor([3]), "int64") == 3

    def test_torch_bool_tensor_returns_python_bool(self) -> None:
        # The dtype hint is what flips an int 0/1 into a JSON true/false.
        assert _coerce_feature_value_to_json(torch.tensor([1]), "bool") is True
        assert _coerce_feature_value_to_json(torch.tensor([0]), "bool") is False

    def test_torch_vector_returns_list(self) -> None:
        out = _coerce_feature_value_to_json(torch.tensor([1.0, 2.0, 3.0]), "float32")
        assert out == [1.0, 2.0, 3.0]

    def test_numpy_scalar_array_returns_python_scalar(self) -> None:
        assert _coerce_feature_value_to_json(np.array([0.25], dtype=np.float32), "float32") == pytest.approx(
            0.25
        )

    def test_numpy_vector_returns_list(self) -> None:
        out = _coerce_feature_value_to_json(np.array([1, 2, 3], dtype=np.int64), "int64")
        assert out == [1, 2, 3]

    def test_string_passes_through(self) -> None:
        assert _coerce_feature_value_to_json("approach", "string") == "approach"

    def test_bool_passes_through(self) -> None:
        assert _coerce_feature_value_to_json(True, "bool") is True

    def test_unknown_type_falls_back_to_str(self) -> None:
        class Weird:
            def __str__(self) -> str:
                return "weird-repr"

        assert _coerce_feature_value_to_json(Weird(), "float32") == "weird-repr"

    def test_numpy_generic_scalar(self) -> None:
        # np.float32(0.5) is a 0-dim "generic" type, not a 1-d ndarray.
        assert _coerce_feature_value_to_json(np.float32(0.5), "float32") == pytest.approx(0.5)


# ── GET /api/datasets/{id}/episodes/{ep}/frame/{frame}/features ─────────────


@pytest.fixture
def app_with_state():
    """FastAPI app with the datasets router and a clean module-level state."""
    app = FastAPI()
    app.include_router(datasets_module.router)

    state = AppState(frame_cache=FrameCache(max_bytes=1_000_000))
    original_state = datasets_module._app_state
    original_indices = datasets_module._episode_start_indices.copy()
    datasets_module.set_app_state(state)

    yield app, state

    datasets_module._app_state = original_state
    datasets_module._episode_start_indices.clear()
    datasets_module._episode_start_indices.update(original_indices)


def _post_for_frame_features(app, dataset_id: str, ep: int, frame: int):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.get(f"/api/datasets/{dataset_id}/episodes/{ep}/frame/{frame}/features")

    return asyncio.run(run())


class TestGetFrameFeatures:
    def test_unknown_dataset_returns_404(self, app_with_state):
        app, _state = app_with_state
        resp = _post_for_frame_features(app, "no-such-dataset", 0, 0)
        assert resp.status_code == 404
        assert "Dataset not found" in resp.json()["detail"]

    def test_returns_values_dict_with_episode_and_frame_index(
        self, app_with_state, tmp_path, lerobot_dataset_factory
    ):
        """Smoke test: open a tiny dataset, fetch features for frame 0 of episode 0,
        and confirm the response shape + that image features are skipped."""
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        resp = _post_for_frame_features(app, dataset_id, 0, 0)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["episode_index"] == 0
        assert body["frame_index"] == 0
        assert isinstance(body["values"], dict)

        # Image / video features must NOT be included — they have their own
        # frame endpoint and are never JSON-serialized.
        for name, ft in ds.meta.features.items():
            if ft["dtype"] in ("image", "video"):
                assert name not in body["values"], f"{name} ({ft['dtype']}) leaked into values"

        # Non-image features that DO exist in the sample should be present.
        # The factory always produces an action feature.
        if "action" in ds.meta.features and ds.meta.features["action"]["dtype"] not in (
            "image",
            "video",
        ):
            assert "action" in body["values"]

    def test_invalid_frame_returns_404(self, app_with_state, tmp_path, lerobot_dataset_factory):
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        # Way past the end of episode 0.
        resp = _post_for_frame_features(app, dataset_id, 0, 999_999)
        assert resp.status_code == 404
        assert "out of range" in resp.json()["detail"]

    def test_invalid_episode_returns_404(self, app_with_state, tmp_path, lerobot_dataset_factory):
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        resp = _post_for_frame_features(app, dataset_id, 999, 0)
        assert resp.status_code == 404
        assert "Episode not found" in resp.json()["detail"]


# ── GET /api/datasets/{id}/episodes/{ep}/feature-series ────────────────────


def _get_feature_series(app, dataset_id: str, ep: int, features: str = ""):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            url = f"/api/datasets/{dataset_id}/episodes/{ep}/feature-series"
            if features:
                url += f"?features={features}"
            return await client.get(url)

    return asyncio.run(run())


class TestGetEpisodeFeatureSeries:
    def test_unknown_dataset_returns_404(self, app_with_state):
        app, _state = app_with_state
        resp = _get_feature_series(app, "no-such-dataset", 0)
        assert resp.status_code == 404

    def test_default_returns_non_image_features(self, app_with_state, tmp_path, lerobot_dataset_factory):
        """Without ``?features=…``, we return every non-image, non-video feature."""
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        resp = _get_feature_series(app, dataset_id, 0)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["episode_index"] == 0
        assert body["length"] > 0
        series = body["series"]
        assert isinstance(series, dict)

        # Every series should be a list with length == episode length.
        for name, vals in series.items():
            assert isinstance(vals, list), f"{name!r} not a list"
            assert len(vals) == body["length"], f"{name!r}: expected {body['length']} values, got {len(vals)}"

        # Image / video features must be excluded.
        for name, ft in ds.meta.features.items():
            if ft["dtype"] in ("image", "video"):
                assert name not in series, f"{name} ({ft['dtype']}) leaked into series"

    def test_image_feature_explicitly_requested_returns_400(
        self, app_with_state, tmp_path, lerobot_dataset_factory
    ):
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        # Find an image / video feature and ask for it explicitly.
        img_feature = next(
            (n for n, ft in ds.meta.features.items() if ft["dtype"] in ("image", "video")), None
        )
        if img_feature is None:
            pytest.skip("factory built dataset without image/video features")

        resp = _get_feature_series(app, dataset_id, 0, features=img_feature)
        assert resp.status_code == 400
        assert "image/video" in resp.json()["detail"]

    def test_unknown_feature_returns_400(self, app_with_state, tmp_path, lerobot_dataset_factory):
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        resp = _get_feature_series(app, dataset_id, 0, features="totally_made_up")
        assert resp.status_code == 400
        assert "Unknown feature" in resp.json()["detail"]

    def test_subset_of_features_returns_only_those(self, app_with_state, tmp_path, lerobot_dataset_factory):
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        # action is reliably present in the factory output.
        resp = _get_feature_series(app, dataset_id, 0, features="action")
        assert resp.status_code == 200, resp.text
        series = resp.json()["series"]
        assert set(series.keys()) == {"action"}

    def test_task_decoded_as_string_via_lookup_table(self, app_with_state, tmp_path, lerobot_dataset_factory):
        """``task`` is a synthetic decoded view of ``task_index`` — request both
        and confirm the strings come from ``meta.tasks``."""
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        resp = _get_feature_series(app, dataset_id, 0, features="task,task_index")
        assert resp.status_code == 200, resp.text
        series = resp.json()["series"]
        assert "task" in series
        assert "task_index" in series
        # Tasks must be strings; indices ints. Length matches.
        assert all(isinstance(t, str) for t in series["task"])
        assert all(isinstance(i, int) for i in series["task_index"])
        # And the strings come from the dataset's tasks lookup table.
        valid = set(ds.meta.tasks.index)
        assert set(series["task"]) <= valid
