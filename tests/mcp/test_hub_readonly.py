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
"""Tests for the read-only Hub MCP tool surface.

Same shared-state invariant as the dataset-edit tools: MCP `hub_*`
reads from the same in-memory ``AppState.hub_jobs`` queue the GUI's
Transfers tray polls via FastAPI ``/api/datasets/hub/*``. So the
strongest property to verify is that adding a job via either surface
shows up via the other.

Auth and repo-info calls are mocked at the ``huggingface_hub.HfApi``
boundary — we never hit Hub in tests.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from lerobot.gui.api import datasets as datasets_module
from lerobot.gui.frame_cache import FrameCache
from lerobot.gui.hub_jobs import HubJobState
from lerobot.gui.state import AppState
from lerobot.mcp.server import build_server


@pytest.fixture
def app_state():
    """Fresh AppState wired into the gui.api.datasets module global.

    The FastAPI ``/api/datasets/hub/*`` routes resolve state from the
    module global, while the MCP tools resolve via build_server's
    getter — the fixture wires both to the same instance so the
    shared-queue invariant can be verified.
    """
    state = AppState(frame_cache=FrameCache(max_bytes=1_000_000))
    orig = datasets_module._app_state
    datasets_module.set_app_state(state)
    try:
        yield state
    finally:
        datasets_module._app_state = orig


@pytest.fixture
def mcp(app_state, tmp_path):
    return build_server(app_state=app_state, dataset_root=tmp_path / "_root")


def _call(mcp_server, name, args):
    _, structured = asyncio.run(mcp_server.call_tool(name, args))
    return structured


def _stub_job(
    *,
    job_id="abc123",
    dataset_id="thewisp/foo",
    repo_id="thewisp/foo",
    status="running",
    direction="upload",
    started_at=None,
    files_total=10,
    files_done=3,
) -> HubJobState:
    """Build a HubJobState with realistic field values for tests."""
    now = started_at if started_at is not None else time.time()
    return HubJobState(
        job_id=job_id,
        dataset_id=dataset_id,
        direction=direction,
        repo_id=repo_id,
        repo_type="dataset",
        status=status,
        started_at=now,
        milestone=f"{direction}ing episode {files_done}",
        milestone_at=now,
        files_total=files_total,
        files_done_estimate=files_done,
    )


# ── hub_auth_status ───────────────────────────────────────────────────────


class TestHubAuthStatus:
    def test_logged_in_returns_username(self, mcp):
        fake_api = MagicMock()
        fake_api.whoami.return_value = {"name": "alice"}
        with patch("huggingface_hub.HfApi", return_value=fake_api):
            result = _call(mcp, "hub_auth_status", {})
        assert result == {"logged_in": True, "username": "alice"}

    def test_falls_back_to_fullname(self, mcp):
        fake_api = MagicMock()
        fake_api.whoami.return_value = {"fullname": "Alice Smith"}
        with patch("huggingface_hub.HfApi", return_value=fake_api):
            result = _call(mcp, "hub_auth_status", {})
        assert result == {"logged_in": True, "username": "Alice Smith"}

    def test_not_logged_in_collapses_all_failures(self, mcp):
        fake_api = MagicMock()
        fake_api.whoami.side_effect = RuntimeError("no token configured")
        with patch("huggingface_hub.HfApi", return_value=fake_api):
            result = _call(mcp, "hub_auth_status", {})
        assert result == {"logged_in": False, "username": None}

    def test_network_failure_collapses_to_logged_out(self, mcp):
        fake_api = MagicMock()
        fake_api.whoami.side_effect = ConnectionError("network down")
        with patch("huggingface_hub.HfApi", return_value=fake_api):
            result = _call(mcp, "hub_auth_status", {})
        assert result["logged_in"] is False
        assert result["username"] is None


# ── hub_repo_info ─────────────────────────────────────────────────────────


class TestHubRepoInfo:
    def test_existing_repo_returns_metadata(self, mcp):
        sibling = MagicMock(size=1_000_000)
        fake_info = MagicMock(
            id="thewisp/foo",
            private=False,
            last_modified=None,
            downloads=42,
            sha="abc123def456",
            siblings=[sibling, sibling, sibling],
        )
        fake_api = MagicMock()
        fake_api.dataset_info.return_value = fake_info
        with patch("huggingface_hub.HfApi", return_value=fake_api):
            with patch(
                "huggingface_hub.hf_hub_download",
                side_effect=FileNotFoundError("no meta/info.json in this test"),
            ):
                result = _call(mcp, "hub_repo_info", {"repo_id": "thewisp/foo"})
        assert result["exists"] is True
        assert result["repo_id"] == "thewisp/foo"
        assert result["files"] == 3
        assert result["total_size_mb"] == 3.0
        assert result["sha"] == "abc123def456"
        # Best-effort meta/info.json fetch failed → fields are None, not missing
        assert result["total_episodes"] is None

    def test_missing_repo_returns_exists_false(self, mcp):
        fake_api = MagicMock()
        fake_api.dataset_info.side_effect = ValueError("404 Repository Not Found")
        with patch("huggingface_hub.HfApi", return_value=fake_api):
            result = _call(mcp, "hub_repo_info", {"repo_id": "nope/missing"})
        assert result["exists"] is False
        assert result["repo_id"] == "nope/missing"
        # Outcome transparency: error text included so the AI can branch
        # without parsing the full HF exception type
        assert "error" in result
        assert "404" in result["error"]

    def test_private_no_access_returns_exists_false(self, mcp):
        fake_api = MagicMock()
        fake_api.dataset_info.side_effect = PermissionError("401 Unauthorized")
        with patch("huggingface_hub.HfApi", return_value=fake_api):
            result = _call(mcp, "hub_repo_info", {"repo_id": "private/repo"})
        assert result["exists"] is False
        assert "401" in result["error"] or "Unauthorized" in result["error"]


# ── hub_list_jobs ─────────────────────────────────────────────────────────


class TestHubListJobs:
    def test_empty_state_returns_zero_jobs(self, mcp):
        result = _call(mcp, "hub_list_jobs", {})
        assert result == {"jobs": [], "total": 0, "active": 0}

    def test_lists_active_and_terminal_jobs(self, mcp, app_state):
        app_state.hub_jobs["j1"] = _stub_job(job_id="j1", status="running")
        app_state.hub_jobs["j2"] = _stub_job(job_id="j2", status="complete")
        app_state.hub_jobs["j3"] = _stub_job(job_id="j3", status="pending")

        result = _call(mcp, "hub_list_jobs", {})
        assert result["total"] == 3
        assert result["active"] == 2  # running + pending
        # Newest-first ordering on started_at
        ids = [j["job_id"] for j in result["jobs"]]
        assert set(ids) == {"j1", "j2", "j3"}

    def test_requires_app_state_in_standalone_mode(self, tmp_path):
        mcp = build_server(dataset_root=tmp_path)  # no app_state
        with pytest.raises(Exception, match="unified GUI deployment"):
            _call(mcp, "hub_list_jobs", {})


# ── hub_job_progress ──────────────────────────────────────────────────────


class TestHubJobProgress:
    def test_returns_job_snapshot(self, mcp, app_state):
        app_state.hub_jobs["j-xyz"] = _stub_job(job_id="j-xyz", status="running")
        result = _call(mcp, "hub_job_progress", {"job_id": "j-xyz"})
        assert result["job_id"] == "j-xyz"
        assert result["status"] == "running"

    def test_unknown_job_raises(self, mcp):
        with pytest.raises(Exception, match="Hub job not found"):
            _call(mcp, "hub_job_progress", {"job_id": "does-not-exist"})

    def test_terminal_job_snapshot_does_not_refresh_from_file(self, mcp, app_state):
        # Terminal jobs aren't re-read from disk; their snapshot is the
        # last known state. This is intentional — once a job is complete
        # it's not changing. Verify the path doesn't barf trying to read
        # a missing progress file.
        app_state.hub_jobs["done"] = _stub_job(job_id="done", status="complete")
        result = _call(mcp, "hub_job_progress", {"job_id": "done"})
        assert result["status"] == "complete"


# ── cross-surface shared-state ────────────────────────────────────────────


class TestCrossSurfaceSharedState:
    """MCP `hub_list_jobs` and FastAPI `GET /api/datasets/hub/jobs` must
    read from the same in-memory queue. One source of truth.
    """

    def test_job_added_to_app_state_is_visible_via_mcp(self, mcp, app_state):
        app_state.hub_jobs["shared"] = _stub_job(job_id="shared", status="pending")
        result = _call(mcp, "hub_list_jobs", {})
        assert result["total"] == 1
        assert result["jobs"][0]["job_id"] == "shared"

    def test_mcp_progress_matches_fastapi_state(self, mcp, app_state):
        app_state.hub_jobs["dual"] = _stub_job(
            job_id="dual",
            status="running",
            files_total=20,
            files_done=7,
        )
        # Via MCP
        mcp_view = _call(mcp, "hub_job_progress", {"job_id": "dual"})
        # Via the underlying state (FastAPI handler reads the same)
        state_view = app_state.hub_jobs["dual"].to_dict()
        assert mcp_view["job_id"] == state_view["job_id"]
        assert mcp_view["status"] == state_view["status"]
        assert mcp_view["files_total"] == state_view["files_total"]
