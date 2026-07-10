# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for dataset-level task (language description) exposure + rename.

The Inspector's dataset summary shows the natural-language task(s) from
``meta/tasks.parquet`` and lets the user rename them. The rename is
metadata-only (data shards store ``task_index``): it rewrites
``meta/tasks.parquet`` and the per-episode ``tasks`` lists in
``meta/episodes/*.parquet``, then refreshes the in-memory metadata so the
already-open dataset serves the new string.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import FastAPI

from lerobot.datasets.io_utils import load_tasks
from lerobot.gui.api import datasets as datasets_module
from lerobot.gui.frame_cache import FrameCache
from lerobot.gui.state import AppState


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


def _patch_tasks(app, dataset_id: str, old_task: str, new_task: str):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.patch(
                f"/api/datasets/{dataset_id}/tasks",
                json={"old_task": old_task, "new_task": new_task},
            )

    return asyncio.run(run())


def _get(app, url: str):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.get(url)

    return asyncio.run(run())


class TestDatasetInfoTasks:
    def test_dataset_info_includes_tasks(self, app_with_state, tmp_path, lerobot_dataset_factory):
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        resp = _get(app, "/api/datasets")
        assert resp.status_code == 200, resp.text
        (info,) = resp.json()
        expected = list(ds.meta.tasks.sort_values("task_index").index)
        assert info["tasks"] == expected
        assert len(info["tasks"]) >= 1


class TestRenameTask:
    def test_unknown_dataset_returns_404(self, app_with_state):
        app, _state = app_with_state
        resp = _patch_tasks(app, "no-such-dataset", "a", "b")
        assert resp.status_code == 404

    def test_rename_updates_disk_memory_and_episodes(
        self, app_with_state, tmp_path, lerobot_dataset_factory
    ):
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        old_task = ds.meta.tasks.sort_values("task_index").index[0]
        old_index = int(ds.meta.tasks.loc[old_task, "task_index"])
        new_task = "Pick up the red cube and hand it over."

        resp = _patch_tasks(app, dataset_id, old_task, new_task)
        assert resp.status_code == 200, resp.text
        info = resp.json()
        assert new_task in info["tasks"]
        assert old_task not in info["tasks"]

        # In-memory metadata refreshed, task_index preserved.
        assert new_task in ds.meta.tasks.index
        assert old_task not in ds.meta.tasks.index
        assert int(ds.meta.tasks.loc[new_task, "task_index"]) == old_index

        # tasks.parquet rewritten on disk.
        on_disk = load_tasks(ds.root)
        assert new_task in on_disk.index
        assert old_task not in on_disk.index

        # Per-episode metadata (tasks lists) rewritten — both in memory and
        # via the episodes endpoint the GUI refreshes after a rename.
        for ep in ds.meta.episodes:
            assert old_task not in ep["tasks"]
        ep_resp = _get(app, f"/api/datasets/{dataset_id}/episodes")
        assert ep_resp.status_code == 200
        listed_tasks = {e["task"] for e in ep_resp.json() if e["task"] is not None}
        assert old_task not in listed_tasks

    def test_unknown_old_task_returns_400(self, app_with_state, tmp_path, lerobot_dataset_factory):
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        resp = _patch_tasks(app, dataset_id, "no such task", "whatever")
        assert resp.status_code == 400
        assert "Unknown task" in resp.json()["detail"]

    def test_empty_new_task_returns_400(self, app_with_state, tmp_path, lerobot_dataset_factory):
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        old_task = ds.meta.tasks.index[0]
        resp = _patch_tasks(app, dataset_id, old_task, "   ")
        assert resp.status_code == 400
        assert "non-empty" in resp.json()["detail"]

    def test_rename_onto_existing_task_returns_400(
        self, app_with_state, tmp_path, lerobot_dataset_factory
    ):
        app, state = app_with_state
        # Factory datasets can have a single task; synthesize a second one so
        # the collision path is always exercised.
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        tasks = sorted(ds.meta.tasks.index)
        if len(tasks) < 2:
            from lerobot.datasets.io_utils import write_tasks

            ds.meta.tasks.loc["Another existing task."] = len(ds.meta.tasks)
            write_tasks(ds.meta.tasks, ds.root)
            tasks = sorted(ds.meta.tasks.index)

        resp = _patch_tasks(app, dataset_id, tasks[0], tasks[1])
        assert resp.status_code == 400
        assert "already exists" in resp.json()["detail"]

    def test_noop_rename_returns_200(self, app_with_state, tmp_path, lerobot_dataset_factory):
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        old_task = ds.meta.tasks.index[0]
        resp = _patch_tasks(app, dataset_id, old_task, old_task)
        assert resp.status_code == 200
        assert old_task in resp.json()["tasks"]
