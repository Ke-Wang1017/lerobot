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
"""Tests for the dataset-edit MCP tool surface.

These tools share the GUI's in-memory ``AppState`` — the same queue the
FastAPI ``/api/edits`` routes drive. So the strongest property to
verify is that MCP-side proposals show up via the GUI-side list, and
vice versa: one source of truth for both surfaces.
"""

from __future__ import annotations

import asyncio
import random

import numpy as np
import pytest

from lerobot.gui.api import datasets as datasets_module, edits as edits_module
from lerobot.gui.frame_cache import FrameCache
from lerobot.gui.state import AppState
from lerobot.mcp.server import build_server

# Re-use the GUI test suite's synthetic dataset factory.
pytest_plugins = ["tests.fixtures.dataset_factories"]


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)
    random.seed(0)
    yield


@pytest.fixture
def state_and_dataset(tmp_path, lerobot_dataset_factory):
    """Throwaway in-tmp dataset + fresh AppState + MCP server built against it.

    Wires the AppState into both gui.api.datasets and gui.api.edits
    module globals so the FastAPI handlers and the MCP tools share the
    same queue (the unified-process invariant we depend on).
    """
    ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=3, total_frames=30)
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
    """Run a tool call and return its structured response."""
    _, structured = asyncio.run(mcp.call_tool(name, args))
    return structured


class TestProposeDelete:
    def test_proposes_a_valid_delete(self, state_and_dataset):
        mcp, state, ds = state_and_dataset
        result = _call(mcp, "propose_delete_episode", {"repo_id": ds.repo_id, "episode_id": 1})
        assert result["status"] == "ok"
        assert result["episode_index"] == 1
        assert state.is_episode_deleted(ds.repo_id, 1)

    def test_rejects_duplicate_delete(self, state_and_dataset):
        mcp, _, ds = state_and_dataset
        _call(mcp, "propose_delete_episode", {"repo_id": ds.repo_id, "episode_id": 0})
        with pytest.raises(Exception, match="already marked"):
            _call(mcp, "propose_delete_episode", {"repo_id": ds.repo_id, "episode_id": 0})

    def test_rejects_out_of_range_episode(self, state_and_dataset):
        mcp, _, ds = state_and_dataset
        with pytest.raises(Exception, match="Invalid episode index"):
            _call(mcp, "propose_delete_episode", {"repo_id": ds.repo_id, "episode_id": 999})

    def test_rejects_unknown_dataset(self, state_and_dataset):
        mcp, _, _ = state_and_dataset
        with pytest.raises(Exception, match="Dataset not found"):
            _call(mcp, "propose_delete_episode", {"repo_id": "nope/missing", "episode_id": 0})


class TestProposeTrim:
    def test_proposes_a_valid_trim(self, state_and_dataset):
        mcp, state, ds = state_and_dataset
        result = _call(
            mcp,
            "propose_trim_episode",
            {"repo_id": ds.repo_id, "episode_id": 0, "keep_from": 2, "keep_to": 8},
        )
        assert result["status"] == "ok"
        # MCP-layer naming: keep_from / keep_to (FastAPI keeps the legacy names)
        assert result["keep_from"] == 2
        assert result["keep_to"] == 8
        # Outcome transparency: response also echoes how many frames are kept vs dropped
        assert result["kept_frames"] == 6
        assert result["dropped_frames"] == result["episode_length_before"] - 6
        # Internal state is still keyed by start_frame/end_frame (FastAPI compat)
        assert state.get_episode_trim(ds.repo_id, 0) == (2, 8)

    def test_replaces_prior_trim(self, state_and_dataset):
        mcp, state, ds = state_and_dataset
        _call(
            mcp,
            "propose_trim_episode",
            {"repo_id": ds.repo_id, "episode_id": 0, "keep_from": 1, "keep_to": 5},
        )
        _call(
            mcp,
            "propose_trim_episode",
            {"repo_id": ds.repo_id, "episode_id": 0, "keep_from": 3, "keep_to": 9},
        )
        assert state.get_episode_trim(ds.repo_id, 0) == (3, 9)

    def test_full_range_clears_the_trim(self, state_and_dataset):
        mcp, state, ds = state_and_dataset
        ep_length = int(ds.meta.episodes[0]["length"])
        _call(
            mcp,
            "propose_trim_episode",
            {"repo_id": ds.repo_id, "episode_id": 0, "keep_from": 2, "keep_to": 5},
        )
        _call(
            mcp,
            "propose_trim_episode",
            {"repo_id": ds.repo_id, "episode_id": 0, "keep_from": 0, "keep_to": ep_length},
        )
        assert state.get_episode_trim(ds.repo_id, 0) is None

    def test_rejects_bad_range(self, state_and_dataset):
        mcp, _, ds = state_and_dataset
        with pytest.raises(Exception, match="keep window must have positive length"):
            _call(
                mcp,
                "propose_trim_episode",
                {"repo_id": ds.repo_id, "episode_id": 0, "keep_from": 5, "keep_to": 5},
            )


class TestListAndDiscard:
    def test_list_reflects_proposed_edits(self, state_and_dataset):
        mcp, _, ds = state_and_dataset
        _call(mcp, "propose_delete_episode", {"repo_id": ds.repo_id, "episode_id": 0})
        _call(
            mcp,
            "propose_trim_episode",
            {"repo_id": ds.repo_id, "episode_id": 1, "keep_from": 1, "keep_to": 5},
        )
        result = _call(mcp, "list_pending_edits", {})
        assert result["total"] == 2
        types = {e["edit_type"] for e in result["edits"]}
        assert types == {"delete", "trim"}

    def test_list_filters_by_dataset(self, state_and_dataset):
        mcp, _, ds = state_and_dataset
        _call(mcp, "propose_delete_episode", {"repo_id": ds.repo_id, "episode_id": 0})
        result = _call(mcp, "list_pending_edits", {"repo_id": "other/dataset"})
        assert result["total"] == 0
        result2 = _call(mcp, "list_pending_edits", {"repo_id": ds.repo_id})
        assert result2["total"] == 1

    def test_discard_clears_the_queue(self, state_and_dataset):
        mcp, state, ds = state_and_dataset
        _call(mcp, "propose_delete_episode", {"repo_id": ds.repo_id, "episode_id": 0})
        _call(mcp, "propose_delete_episode", {"repo_id": ds.repo_id, "episode_id": 1})
        result = _call(mcp, "discard_pending_edits", {"repo_id": ds.repo_id})
        assert result["discarded"] == 2
        assert state.pending_edits == []


class TestProposeSetFeature:
    """Per-frame feature edit. Schema-validated; overlap and large-edit
    conflicts surface as ``status='conflict'`` dicts so the AI can retry
    with the right confirm flag.
    """

    def _ds_with_editable_feature(self, tmp_path, lerobot_dataset_factory, feature_name="reward"):
        """Synthesize a dataset with a float32 ``reward`` feature the
        edit tools will accept. The factory's default schema has no
        editable feature (action + observation.* are read-only), so we
        layer one in.
        """
        info_extra = {
            feature_name: {
                "dtype": "float32",
                "shape": [1],
                "names": [feature_name],
            },
        }
        ds = lerobot_dataset_factory(
            root=tmp_path / "ds",
            total_episodes=2,
            total_frames=20,
            info=None,  # let the factory build default + we'll patch
        )
        # The factory doesn't take feature-add directly; instead we extend
        # meta.features in-memory which is what these tools read.
        ds.meta.features[feature_name] = info_extra[feature_name]
        return ds

    def test_proposes_a_valid_feature_set(self, tmp_path, lerobot_dataset_factory):
        ds = self._ds_with_editable_feature(tmp_path, lerobot_dataset_factory)
        state = AppState(frame_cache=FrameCache(max_bytes=1_000_000))
        state.datasets[ds.repo_id] = ds
        datasets_module.set_app_state(state)
        edits_module.set_app_state(state)
        mcp = build_server(app_state=state, dataset_root=tmp_path / "_root")

        result = _call(
            mcp,
            "propose_set_feature",
            {
                "repo_id": ds.repo_id,
                "episode_id": 0,
                "feature": "reward",
                "frame_from": 0,
                "frame_to": 5,
                "value": 0.5,
            },
        )
        assert result["status"] == "ok"
        assert result["feature"] == "reward"
        assert len(state.pending_edits) == 1

    def test_overlap_returns_structured_conflict(self, tmp_path, lerobot_dataset_factory):
        ds = self._ds_with_editable_feature(tmp_path, lerobot_dataset_factory)
        state = AppState(frame_cache=FrameCache(max_bytes=1_000_000))
        state.datasets[ds.repo_id] = ds
        datasets_module.set_app_state(state)
        edits_module.set_app_state(state)
        mcp = build_server(app_state=state, dataset_root=tmp_path / "_root")

        # First edit lands ok.
        _call(
            mcp,
            "propose_set_feature",
            {
                "repo_id": ds.repo_id,
                "episode_id": 0,
                "feature": "reward",
                "frame_from": 0,
                "frame_to": 5,
                "value": 0.5,
            },
        )
        # Overlapping second edit returns a structured conflict (NOT an exception).
        result = _call(
            mcp,
            "propose_set_feature",
            {
                "repo_id": ds.repo_id,
                "episode_id": 0,
                "feature": "reward",
                "frame_from": 3,
                "frame_to": 7,
                "value": 0.9,
            },
        )
        assert result["status"] == "conflict"
        assert result["detail"]["code"] == "overlapping_edit"
        # The AI is told how to retry.
        assert "confirm_overlap=true" in result["detail"]["message"]

    def test_overlap_clipped_on_confirm(self, tmp_path, lerobot_dataset_factory):
        ds = self._ds_with_editable_feature(tmp_path, lerobot_dataset_factory)
        state = AppState(frame_cache=FrameCache(max_bytes=1_000_000))
        state.datasets[ds.repo_id] = ds
        datasets_module.set_app_state(state)
        edits_module.set_app_state(state)
        mcp = build_server(app_state=state, dataset_root=tmp_path / "_root")

        _call(
            mcp,
            "propose_set_feature",
            {
                "repo_id": ds.repo_id,
                "episode_id": 0,
                "feature": "reward",
                "frame_from": 0,
                "frame_to": 5,
                "value": 0.5,
            },
        )
        result = _call(
            mcp,
            "propose_set_feature",
            {
                "repo_id": ds.repo_id,
                "episode_id": 0,
                "feature": "reward",
                "frame_from": 3,
                "frame_to": 7,
                "value": 0.9,
                "confirm_overlap": True,
            },
        )
        assert result["status"] == "ok"
        # The first edit was clipped to [0, 3); the new edit [3, 7) is added.
        ranges = [
            (e.params["frame_from"], e.params["frame_to"], e.params["value"])
            for e in state.pending_edits
            if e.edit_type == "feature_set"
        ]
        assert (0, 3, 0.5) in ranges
        assert (3, 7, 0.9) in ranges

    def test_rejects_readonly_action_feature(self, state_and_dataset):
        mcp, _, ds = state_and_dataset
        with pytest.raises(Exception, match="read-only"):
            _call(
                mcp,
                "propose_set_feature",
                {
                    "repo_id": ds.repo_id,
                    "episode_id": 0,
                    "feature": "action",
                    "frame_from": 0,
                    "frame_to": 5,
                    "value": [0.0, 0.0],
                },
            )


class TestRequiresAppState:
    """Edit-tier tools must raise a descriptive error when MCP is built
    without the GUI's AppState (standalone ``lerobot-mcp serve`` mode).
    """

    def test_standalone_mode_raises_descriptive_error(self, tmp_path):
        mcp = build_server(dataset_root=tmp_path)  # no app_state passed
        with pytest.raises(Exception, match="unified GUI deployment"):
            _call(mcp, "propose_delete_episode", {"repo_id": "x/y", "episode_id": 0})

    def test_standalone_list_also_blocks(self, tmp_path):
        mcp = build_server(dataset_root=tmp_path)
        with pytest.raises(Exception, match="unified GUI deployment"):
            _call(mcp, "list_pending_edits", {})


class TestCrossSurfaceSharedState:
    """The MCP edit tools share the in-memory queue with FastAPI's
    /api/edits routes. Proving this is the strongest single property:
    one source of truth, surfaceable from either side.
    """

    def test_mcp_proposal_visible_via_fastapi_state(self, state_and_dataset):
        mcp, state, ds = state_and_dataset
        # MCP-side propose.
        _call(mcp, "propose_delete_episode", {"repo_id": ds.repo_id, "episode_id": 0})
        # Read it back via the AppState directly (FastAPI's GET /api/edits
        # reads the same state).
        assert state.is_episode_deleted(ds.repo_id, 0)
        # And via the MCP list tool.
        result = _call(mcp, "list_pending_edits", {"repo_id": ds.repo_id})
        assert result["total"] == 1
        assert result["edits"][0]["edit_type"] == "delete"


class TestApply:
    """Propose → apply → verify the dataset on disk actually changed.

    The synthetic dataset_factory builds a real on-disk dataset under
    tmp_path; apply_pending_edits writes through to its parquet files.
    Cross-checks both the in-memory ``meta.total_episodes`` and the
    on-disk total_frames via meta reload.
    """

    def test_apply_on_empty_queue_returns_zero(self, state_and_dataset):
        mcp, _, ds = state_and_dataset
        result = _call(mcp, "apply_pending_edits", {"repo_id": ds.repo_id})
        assert result["applied"] == 0

    def test_propose_then_apply_deletes_episode_from_disk(self, state_and_dataset):
        """The destructive end-to-end. AI proposes delete; AI applies;
        the dataset's on-disk metadata reflects the deletion.
        """
        mcp, state, ds = state_and_dataset
        episodes_before = ds.meta.total_episodes
        frames_before = ds.meta.total_frames
        ep_to_delete_length = int(ds.meta.episodes[1]["length"])

        # 1) AI stages a delete via MCP.
        _call(mcp, "propose_delete_episode", {"repo_id": ds.repo_id, "episode_id": 1})
        assert state.is_episode_deleted(ds.repo_id, 1)
        assert _call(mcp, "list_pending_edits", {})["total"] == 1

        # 2) AI applies via MCP. This writes through to parquet on disk.
        result = _call(mcp, "apply_pending_edits", {"repo_id": ds.repo_id})
        assert result["status"] == "ok", result
        assert result["applied"] == 1

        # 3) State is cleared (in-memory queue + on-disk edit-state file).
        assert state.pending_edits == []
        edits_file = ds.root / ".lerobot_gui_edits.json"
        assert not edits_file.exists() or edits_file.read_text().strip() in ("", "[]", "{}")

        # 4) The dataset's metadata reflects the deletion. Reload from disk
        # to bypass any in-memory caching — same path the GUI takes after
        # apply, ensuring future opens see the same state.
        from lerobot.gui.dataset_reload import reload_dataset_from_disk

        reload_dataset_from_disk(ds)
        assert ds.meta.total_episodes == episodes_before - 1
        # Total frames dropped by exactly the deleted episode's length.
        assert ds.meta.total_frames == frames_before - ep_to_delete_length

    def test_apply_with_trim_changes_episode_length(self, tmp_path, lerobot_dataset_factory):
        """Propose a trim and apply it; the trimmed episode's length
        drops to the kept window.
        """
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        state = AppState(frame_cache=FrameCache(max_bytes=1_000_000))
        state.datasets[ds.repo_id] = ds
        orig_dat = datasets_module._app_state
        orig_edits = edits_module._app_state
        orig_idx = datasets_module._episode_start_indices.copy()
        datasets_module.set_app_state(state)
        edits_module.set_app_state(state)
        try:
            mcp = build_server(app_state=state, dataset_root=tmp_path / "_root")
            ep_length_before = int(ds.meta.episodes[0]["length"])
            keep_start, keep_end = 1, min(5, ep_length_before - 1)

            _call(
                mcp,
                "propose_trim_episode",
                {"repo_id": ds.repo_id, "episode_id": 0, "keep_from": keep_start, "keep_to": keep_end},
            )
            result = _call(mcp, "apply_pending_edits", {"repo_id": ds.repo_id})
            assert result["status"] == "ok"
            assert result["applied"] == 1

            from lerobot.gui.dataset_reload import reload_dataset_from_disk

            reload_dataset_from_disk(ds)
            ep_length_after = int(ds.meta.episodes[0]["length"])
            assert ep_length_after == keep_end - keep_start, (
                f"trim window was [{keep_start}, {keep_end}) "
                f"(width {keep_end - keep_start}), got length {ep_length_after}"
            )
        finally:
            datasets_module._app_state = orig_dat
            edits_module._app_state = orig_edits
            datasets_module._episode_start_indices.clear()
            datasets_module._episode_start_indices.update(orig_idx)
