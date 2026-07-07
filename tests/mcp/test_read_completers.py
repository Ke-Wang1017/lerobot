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
"""Tests for the read-tier completers PR.

Five tools, all pure reads. Mocks live at the boundaries (sidecar
store, HfApi). No real Hub access in tests.
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
def state_and_dataset(tmp_path, lerobot_dataset_factory):
    ds = lerobot_dataset_factory(
        root=tmp_path / "ds", repo_id="test_org/sample", total_episodes=3, total_frames=30
    )
    state = AppState(frame_cache=FrameCache(max_bytes=1_000_000))
    state.datasets[ds.repo_id] = ds

    orig_dat = datasets_module._app_state
    orig_edits = edits_module._app_state
    orig_idx = datasets_module._episode_start_indices.copy()
    datasets_module.set_app_state(state)
    edits_module.set_app_state(state)

    # Stage the dataset under the discovery layout so get_meta() resolves.
    # Isolated sidecar DB per test so tag writes from one test don't bleed
    # into another's list_tagged_episodes assertions.
    import shutil

    (tmp_path / "test_org").mkdir(parents=True, exist_ok=True)
    src = ds.root
    target = tmp_path / "test_org" / "sample"
    if not target.exists():
        shutil.copytree(src, target)
    mcp = build_server(app_state=state, dataset_root=tmp_path, db_path=tmp_path / "_mcp_annotations.sqlite")

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


# ── list_tagged_episodes ──────────────────────────────────────────────────


class TestListTaggedEpisodes:
    def test_no_tags_returns_empty(self, state_and_dataset):
        mcp, _, ds = state_and_dataset
        result = _call(mcp, "list_tagged_episodes", {"repo_id": ds.repo_id})
        assert result == {
            "repo_id": ds.repo_id,
            "key": None,
            "value": None,
            "episodes": [],
            "total": 0,
        }

    def test_returns_all_tagged_episodes_when_key_is_none(self, state_and_dataset):
        mcp, _, ds = state_and_dataset
        # Seed tags via the existing tag_episode MCP tool
        _call(
            mcp, "tag_episode", {"repo_id": ds.repo_id, "episode_id": 0, "key": "outcome", "value": "success"}
        )
        _call(mcp, "tag_episode", {"repo_id": ds.repo_id, "episode_id": 2, "key": "outcome", "value": "fail"})
        result = _call(mcp, "list_tagged_episodes", {"repo_id": ds.repo_id})
        assert result["total"] == 2
        ep_ids = sorted(e["episode_id"] for e in result["episodes"])
        assert ep_ids == [0, 2]

    def test_filter_by_key_returns_value_and_set_at(self, state_and_dataset):
        mcp, _, ds = state_and_dataset
        _call(
            mcp, "tag_episode", {"repo_id": ds.repo_id, "episode_id": 0, "key": "outcome", "value": "success"}
        )
        _call(
            mcp, "tag_episode", {"repo_id": ds.repo_id, "episode_id": 1, "key": "gripper_miss", "value": True}
        )
        result = _call(mcp, "list_tagged_episodes", {"repo_id": ds.repo_id, "key": "outcome"})
        assert result["total"] == 1
        assert result["episodes"][0]["episode_id"] == 0
        assert result["episodes"][0]["value"] == "success"
        assert "set_at" in result["episodes"][0]

    def test_filter_by_key_and_value(self, state_and_dataset):
        mcp, _, ds = state_and_dataset
        _call(
            mcp, "tag_episode", {"repo_id": ds.repo_id, "episode_id": 0, "key": "outcome", "value": "success"}
        )
        _call(mcp, "tag_episode", {"repo_id": ds.repo_id, "episode_id": 1, "key": "outcome", "value": "fail"})
        _call(mcp, "tag_episode", {"repo_id": ds.repo_id, "episode_id": 2, "key": "outcome", "value": "fail"})
        result = _call(
            mcp,
            "list_tagged_episodes",
            {"repo_id": ds.repo_id, "key": "outcome", "value": "fail"},
        )
        assert result["total"] == 2
        assert sorted(e["episode_id"] for e in result["episodes"]) == [1, 2]

    def test_unknown_dataset_raises(self, state_and_dataset):
        mcp, _, _ = state_and_dataset
        with pytest.raises(Exception, match="not found"):
            _call(mcp, "list_tagged_episodes", {"repo_id": "no/such"})


# ── get_feature_series ────────────────────────────────────────────────────


class TestGetFeatureSeries:
    def test_returns_series_for_all_features_when_omitted(self, state_and_dataset):
        mcp, _, ds = state_and_dataset
        result = _call(mcp, "get_feature_series", {"repo_id": ds.repo_id, "episode_id": 0})
        assert result["repo_id"] == ds.repo_id
        assert result["episode_index"] == 0
        # length should match what's in meta
        assert result["length"] == int(ds.meta.episodes[0]["length"])
        # series should be a dict mapping feature name -> list per frame
        assert isinstance(result["series"], dict)
        assert len(result["series"]) > 0

    def test_returns_only_requested_features(self, state_and_dataset):
        mcp, _, ds = state_and_dataset
        # Pick any non-image feature
        non_image = [
            name for name, ft in ds.meta.features.items() if ft.get("dtype") not in {"image", "video"}
        ]
        assert non_image, "test dataset has no per-frame non-image features"
        chosen = non_image[0]
        result = _call(
            mcp,
            "get_feature_series",
            {"repo_id": ds.repo_id, "episode_id": 0, "features": [chosen]},
        )
        assert set(result["series"].keys()) == {chosen}

    def test_unknown_dataset_raises(self, state_and_dataset):
        mcp, _, _ = state_and_dataset
        with pytest.raises(Exception, match="not opened in GUI"):
            _call(mcp, "get_feature_series", {"repo_id": "no/such", "episode_id": 0})

    def test_unknown_episode_raises(self, state_and_dataset):
        mcp, _, ds = state_and_dataset
        with pytest.raises(Exception, match="Episode not found"):
            _call(mcp, "get_feature_series", {"repo_id": ds.repo_id, "episode_id": 999})

    def test_unknown_feature_raises(self, state_and_dataset):
        mcp, _, ds = state_and_dataset
        with pytest.raises(Exception, match="Unknown feature"):
            _call(
                mcp,
                "get_feature_series",
                {"repo_id": ds.repo_id, "episode_id": 0, "features": ["definitely_not_a_feature"]},
            )


# ── hub_diff_local_vs_remote ──────────────────────────────────────────────


class TestHubDiffLocalVsRemote:
    def test_in_sync_when_remote_files_match_local(self, state_and_dataset):
        mcp, _, ds = state_and_dataset
        # Mock HfApi to return the actual on-disk files as the "remote" set
        fake_siblings = []
        for p in ds.root.rglob("*"):
            if not p.is_file():
                continue
            rel = str(p.relative_to(ds.root))
            if rel.startswith(".cache/") or rel.startswith(".lerobot"):
                continue
            mock_s = MagicMock()
            mock_s.rfilename = rel
            mock_s.size = p.stat().st_size
            mock_s.lfs = None
            mock_s.blob_id = "abc"
            fake_siblings.append(mock_s)
        info = MagicMock(siblings=fake_siblings)
        fake_api = MagicMock()
        fake_api.dataset_info.return_value = info
        with patch("huggingface_hub.HfApi", return_value=fake_api):
            result = _call(mcp, "hub_diff_local_vs_remote", {"dataset_id": ds.repo_id})
        assert result["status"] == "ok"
        assert result["in_sync"] is True
        assert result["modified"] == []
        assert result["local_only"] == []
        assert result["remote_only"] == []

    def test_size_mismatch_reported_as_modified(self, state_and_dataset):
        mcp, _, ds = state_and_dataset
        # Build a "remote" with the same files but a wrong size on one of them
        files = list(ds.root.rglob("*"))
        first_file = next(p for p in files if p.is_file())
        fake_siblings = []
        for p in files:
            if not p.is_file():
                continue
            rel = str(p.relative_to(ds.root))
            if rel.startswith(".cache/") or rel.startswith(".lerobot"):
                continue
            mock_s = MagicMock()
            mock_s.rfilename = rel
            mock_s.size = (p.stat().st_size + 1000) if p == first_file else p.stat().st_size
            mock_s.lfs = None
            mock_s.blob_id = "abc"
            fake_siblings.append(mock_s)
        info = MagicMock(siblings=fake_siblings)
        fake_api = MagicMock()
        fake_api.dataset_info.return_value = info
        with patch("huggingface_hub.HfApi", return_value=fake_api):
            result = _call(mcp, "hub_diff_local_vs_remote", {"dataset_id": ds.repo_id})
        assert result["in_sync"] is False
        assert len(result["modified"]) == 1

    def test_remote_unreachable_returns_error_status(self, state_and_dataset):
        mcp, _, ds = state_and_dataset
        fake_api = MagicMock()
        fake_api.dataset_info.side_effect = RuntimeError("Network down")
        with patch("huggingface_hub.HfApi", return_value=fake_api):
            result = _call(mcp, "hub_diff_local_vs_remote", {"dataset_id": ds.repo_id})
        assert result["status"] == "error"
        assert "Repo not found" in result["message"]

    def test_unknown_dataset_raises(self, state_and_dataset):
        mcp, _, _ = state_and_dataset
        with pytest.raises(Exception, match="not opened in GUI"):
            _call(mcp, "hub_diff_local_vs_remote", {"dataset_id": "no/such"})


# ── lerobot_list_tools ────────────────────────────────────────────────────


class TestLerobotListTools:
    def test_returns_every_registered_tool(self, state_and_dataset):
        mcp, _, _ = state_and_dataset
        result = _call(mcp, "lerobot_list_tools", {})
        names = {t["name"] for t in result["tools"]}
        # Spot-check tools from various tiers + the introspection pair itself
        for expected in (
            "list_datasets",
            "tag_episode",
            "propose_delete_episode",
            "hub_start_upload",
            "lerobot_list_tools",
            "lerobot_whoami",
            "list_tagged_episodes",
            "hub_diff_local_vs_remote",
            "get_feature_series",
        ):
            assert expected in names, f"missing: {expected}"
        assert result["total"] == len(result["tools"])

    def test_scope_field_reflects_decorator(self, state_and_dataset):
        mcp, _, _ = state_and_dataset
        result = _call(mcp, "lerobot_list_tools", {})
        by_name = {t["name"]: t for t in result["tools"]}
        assert by_name["list_datasets"]["scope"] == "read"
        assert by_name["tag_episode"]["scope"] == "comment"
        assert by_name["propose_delete_episode"]["scope"] == "edit"
        assert by_name["hub_start_upload"]["scope"] == "edit"


# ── lerobot_whoami ────────────────────────────────────────────────────────


class TestLerobotWhoami:
    def test_stdio_mode_returns_all_scopes(self, state_and_dataset):
        """Without a TokenStore the server runs unauthenticated; the
        introspection tool falls back to reporting every scope.
        """
        mcp, _, _ = state_and_dataset
        result = _call(mcp, "lerobot_whoami", {})
        assert "read" in result["scopes"]
        assert result["logged_in"] is False
        assert result["client_id"] is None
        # all_scopes is always the canonical 4-tier list
        assert set(result["all_scopes"]) == {"read", "comment", "edit", "operate"}
