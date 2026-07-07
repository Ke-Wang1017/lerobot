# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Tests for the edit-tier Hub MCP tool surface.

Tests mock the worker-spawning + HfApi at their respective boundaries
so they don't actually hit Hugging Face Hub. The transcript proof
exercises the real upload pipeline against a throwaway repo.
"""

from __future__ import annotations

import asyncio
import random
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lerobot.gui.api import datasets as datasets_module, edits as edits_module
from lerobot.gui.frame_cache import FrameCache
from lerobot.gui.state import AppState
from lerobot.mcp.server import build_server

pytest_plugins = ["tests.fixtures.dataset_factories"]


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)
    random.seed(0)
    yield


@pytest.fixture
def dataset_and_state(tmp_path, lerobot_dataset_factory):
    """A real on-disk dataset + AppState wired into both module globals."""
    ds = lerobot_dataset_factory(
        root=tmp_path / "ds", repo_id="test_org/sample", total_episodes=2, total_frames=20
    )
    state = AppState(frame_cache=FrameCache(max_bytes=1_000_000))
    state.datasets[ds.repo_id] = ds

    orig_dat = datasets_module._app_state
    orig_edits = edits_module._app_state
    orig_idx = datasets_module._episode_start_indices.copy()
    datasets_module.set_app_state(state)
    edits_module.set_app_state(state)

    mcp = build_server(app_state=state, dataset_root=tmp_path / "_unused_root")
    try:
        yield mcp, state, ds
    finally:
        datasets_module._app_state = orig_dat
        edits_module._app_state = orig_edits
        datasets_module._episode_start_indices.clear()
        datasets_module._episode_start_indices.update(orig_idx)


def _call(mcp, name, args):
    _, structured = asyncio.run(mcp.call_tool(name, args))
    return structured


# ── hub_start_upload ──────────────────────────────────────────────────────


class TestHubStartUpload:
    def test_kicks_off_worker_returns_job_id(self, dataset_and_state):
        mcp, state, ds = dataset_and_state
        fake_api = MagicMock()
        fake_api.whoami.return_value = {"name": "alice"}
        with (
            patch("huggingface_hub.HfApi", return_value=fake_api),
            patch.object(datasets_module, "_spawn_hub_worker") as spawn,
            patch(
                "lerobot.gui.hub_jobs.check_upload_completeness",
                return_value={
                    "missing_locally": [],
                    "incomplete_locally": [],
                },
            ),
        ):
            result = _call(
                mcp,
                "hub_start_upload",
                {"dataset_id": ds.repo_id},
            )
        assert "job_id" in result
        assert result["status"] == "started"
        # The job landed in AppState.hub_jobs so a later hub_job_progress
        # call can find it.
        assert result["job_id"] in state.hub_jobs
        assert spawn.called

    def test_incomplete_local_returns_structured_conflict(self, dataset_and_state):
        mcp, _, ds = dataset_and_state
        fake_api = MagicMock()
        fake_api.whoami.return_value = {"name": "alice"}
        with (
            patch("huggingface_hub.HfApi", return_value=fake_api),
            patch.object(datasets_module, "_spawn_hub_worker"),
            patch(
                "lerobot.gui.hub_jobs.check_upload_completeness",
                return_value={
                    "missing_locally": ["meta/info.json"],
                    "incomplete_locally": [],
                },
            ),
        ):
            result = _call(mcp, "hub_start_upload", {"dataset_id": ds.repo_id})
        # The AI sees a structured conflict it can retry with confirm_force=true
        assert result["status"] == "conflict"
        assert result["detail"]["code"] == "incomplete_local_state"
        assert "missing_locally" in result["detail"]

    def test_confirm_force_bypasses_completeness_check(self, dataset_and_state):
        mcp, _, ds = dataset_and_state
        fake_api = MagicMock()
        fake_api.whoami.return_value = {"name": "alice"}
        with (
            patch("huggingface_hub.HfApi", return_value=fake_api),
            patch.object(datasets_module, "_spawn_hub_worker") as spawn,
            patch("lerobot.gui.hub_jobs.check_upload_completeness") as check,
        ):
            result = _call(
                mcp,
                "hub_start_upload",
                {"dataset_id": ds.repo_id, "confirm_force": True},
            )
        # When confirm_force=True the completeness check is skipped entirely.
        assert check.call_count == 0
        assert result["status"] == "started"
        assert spawn.called

    def test_unknown_dataset_raises(self, dataset_and_state):
        mcp, _, _ = dataset_and_state
        # Auto-open code path tries Path("no/such").exists() (False), then 404.
        with pytest.raises(Exception, match="Dataset not found"):
            _call(mcp, "hub_start_upload", {"dataset_id": "no/such_dataset"})

    def test_active_transfer_collision_returns_conflict(self, dataset_and_state):
        mcp, state, ds = dataset_and_state
        # Seed an active job for the same dataset
        from lerobot.gui.hub_jobs import make_job

        active = make_job(dataset_id=ds.repo_id, direction="upload", repo_id=ds.repo_id)
        active.status = "running"
        state.hub_jobs[active.job_id] = active

        fake_api = MagicMock()
        fake_api.whoami.return_value = {"name": "alice"}
        with patch("huggingface_hub.HfApi", return_value=fake_api):
            result = _call(mcp, "hub_start_upload", {"dataset_id": ds.repo_id})
        assert result["status"] == "conflict"
        assert "in progress" in result["detail"]["message"]
        assert result["detail"]["job_id"] == active.job_id


# ── hub_start_download ────────────────────────────────────────────────────


class TestHubStartDownload:
    def test_kicks_off_worker_returns_job_id(self, dataset_and_state):
        mcp, state, ds = dataset_and_state
        fake_api = MagicMock()
        fake_api.whoami.return_value = {"name": "alice"}
        with (
            patch("huggingface_hub.HfApi", return_value=fake_api),
            patch.object(datasets_module, "_spawn_hub_worker") as spawn,
        ):
            result = _call(mcp, "hub_start_download", {"dataset_id": ds.repo_id})
        assert "job_id" in result
        assert result["status"] == "started"
        assert result["job_id"] in state.hub_jobs
        assert state.hub_jobs[result["job_id"]].direction == "download"
        assert spawn.called

    def test_unknown_dataset_raises(self, dataset_and_state):
        mcp, _, _ = dataset_and_state
        with pytest.raises(Exception, match="Dataset not found"):
            _call(mcp, "hub_start_download", {"dataset_id": "no/such"})


# ── hub_cancel_job ────────────────────────────────────────────────────────


class TestHubCancelJob:
    def test_cancels_known_job(self, dataset_and_state):
        mcp, state, ds = dataset_and_state
        from lerobot.gui.hub_jobs import make_job

        job = make_job(dataset_id=ds.repo_id, direction="upload", repo_id=ds.repo_id)
        job.status = "running"
        job.pid = 1
        job.process_start_time = 0.0
        state.hub_jobs[job.job_id] = job

        with patch.object(datasets_module, "_send_signal_with_identity_check", return_value=True):
            result = _call(mcp, "hub_cancel_job", {"job_id": job.job_id})
        assert result["status"] == "cancel_requested"
        assert result["job_id"] == job.job_id

    def test_unknown_job_raises(self, dataset_and_state):
        mcp, _, _ = dataset_and_state
        with pytest.raises(Exception, match="Job not found"):
            _call(mcp, "hub_cancel_job", {"job_id": "does-not-exist"})


# ── cross-surface shared state ────────────────────────────────────────────


class TestCrossSurfaceSharedState:
    """A job kicked off via MCP appears via the read-tier `hub_list_jobs`
    + `hub_job_progress` MCP tools, which both read from the same
    AppState.hub_jobs registry the GUI's Transfers tray reads.
    """

    def test_mcp_upload_job_visible_via_hub_list_jobs(self, dataset_and_state):
        mcp, _, ds = dataset_and_state
        fake_api = MagicMock()
        fake_api.whoami.return_value = {"name": "alice"}
        with (
            patch("huggingface_hub.HfApi", return_value=fake_api),
            patch.object(datasets_module, "_spawn_hub_worker"),
            patch(
                "lerobot.gui.hub_jobs.check_upload_completeness",
                return_value={
                    "missing_locally": [],
                    "incomplete_locally": [],
                },
            ),
        ):
            start = _call(mcp, "hub_start_upload", {"dataset_id": ds.repo_id})
            listed = _call(mcp, "hub_list_jobs", {})
        job_ids = [j["job_id"] for j in listed["jobs"]]
        assert start["job_id"] in job_ids
        # The active count includes the just-kicked-off pending job
        assert listed["active"] >= 1
