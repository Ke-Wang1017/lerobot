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
"""Tests for the dataset-merge MCP tool surface.

Shared-core invariant: MCP `merge_into_dataset` and FastAPI
`/api/edits/merge-into` both call the same `_edits_core.merge_dataset_into`
helper. Tests use the synthetic `lerobot_dataset_factory` to build two
real on-disk datasets under tmp_path, merge one into the other, then
verify both the response shape and the resulting metadata.
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

pytest_plugins = ["tests.fixtures.dataset_factories"]


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)
    random.seed(0)
    yield


@pytest.fixture
def two_datasets(tmp_path, lerobot_dataset_factory):
    """Two synthetic datasets opened against a fresh AppState.

    Source: 3 episodes / 30 frames. Target: 2 episodes / 20 frames.
    Both share the same default schema so a clean merge succeeds.
    """
    source = lerobot_dataset_factory(
        root=tmp_path / "source",
        repo_id="test_org/source",
        total_episodes=3,
        total_frames=30,
    )
    target = lerobot_dataset_factory(
        root=tmp_path / "target",
        repo_id="test_org/target",
        total_episodes=2,
        total_frames=20,
    )
    state = AppState(frame_cache=FrameCache(max_bytes=1_000_000))
    state.datasets[source.repo_id] = source
    state.datasets[target.repo_id] = target

    orig_dat = datasets_module._app_state
    orig_edits = edits_module._app_state
    orig_idx = datasets_module._episode_start_indices.copy()
    datasets_module.set_app_state(state)
    edits_module.set_app_state(state)

    mcp = build_server(app_state=state, dataset_root=tmp_path / "_unused_root")
    try:
        yield mcp, state, source, target
    finally:
        datasets_module._app_state = orig_dat
        edits_module._app_state = orig_edits
        datasets_module._episode_start_indices.clear()
        datasets_module._episode_start_indices.update(orig_idx)


def _call(mcp, name, args):
    _, structured = asyncio.run(mcp.call_tool(name, args))
    return structured


# ── validate_dataset_merge ────────────────────────────────────────────────


class TestValidateMerge:
    def test_compatible_datasets_return_empty_mismatches(self, two_datasets):
        mcp, _, source, target = two_datasets
        result = _call(
            mcp,
            "validate_dataset_merge",
            {"source_repo_id": source.repo_id, "target_repo_id": target.repo_id},
        )
        assert result["compatible"] is True
        assert result["mismatches"] == []

    def test_fps_mismatch_surfaces_in_mismatches(self, two_datasets):
        mcp, state, source, target = two_datasets
        # Tweak target's fps to force a mismatch (in-memory only)
        target.meta.info["fps"] = source.meta.info["fps"] + 5
        # The meta property reads from info so the change is visible
        result = _call(
            mcp,
            "validate_dataset_merge",
            {"source_repo_id": source.repo_id, "target_repo_id": target.repo_id},
        )
        assert result["compatible"] is False
        assert any(m.get("field") == "fps" for m in result["mismatches"])

    def test_unknown_source_raises(self, two_datasets):
        mcp, _, _, target = two_datasets
        with pytest.raises(Exception, match="Source dataset not found"):
            _call(
                mcp,
                "validate_dataset_merge",
                {"source_repo_id": "no/such", "target_repo_id": target.repo_id},
            )

    def test_unknown_target_raises(self, two_datasets):
        mcp, _, source, _ = two_datasets
        with pytest.raises(Exception, match="Target dataset not found"):
            _call(
                mcp,
                "validate_dataset_merge",
                {"source_repo_id": source.repo_id, "target_repo_id": "no/such"},
            )


# ── merge_into_dataset ────────────────────────────────────────────────────


class TestMergeIntoDataset:
    def test_merge_grows_target(self, two_datasets):
        mcp, _, source, target = two_datasets
        source_eps = source.num_episodes
        source_frames = source.num_frames
        target_eps_before = target.num_episodes
        target_frames_before = target.num_frames

        result = _call(
            mcp,
            "merge_into_dataset",
            {"source_repo_id": source.repo_id, "target_repo_id": target.repo_id},
        )
        assert result["status"] == "ok"
        assert result["source_episodes_merged"] == source_eps
        assert result["source_frames_merged"] == source_frames
        assert result["target_episodes_before"] == target_eps_before
        assert result["target_episodes_after"] == target_eps_before + source_eps
        assert result["target_frames_before"] == target_frames_before
        assert result["target_frames_after"] == target_frames_before + source_frames
        assert result["force_used"] is False
        # Live target reflects the new size
        assert target.num_episodes == target_eps_before + source_eps
        assert target.num_frames == target_frames_before + source_frames

    def test_self_merge_rejected(self, two_datasets):
        mcp, _, source, _ = two_datasets
        with pytest.raises(Exception, match="Cannot merge a dataset into itself"):
            _call(
                mcp,
                "merge_into_dataset",
                {"source_repo_id": source.repo_id, "target_repo_id": source.repo_id},
            )

    def test_unknown_source_raises(self, two_datasets):
        mcp, _, _, target = two_datasets
        with pytest.raises(Exception, match="Source dataset not found"):
            _call(
                mcp,
                "merge_into_dataset",
                {"source_repo_id": "no/such", "target_repo_id": target.repo_id},
            )

    def test_busy_target_raises(self, two_datasets):
        mcp, state, source, target = two_datasets
        lock = state.get_lock(target.repo_id)
        # Acquire then release — but we need it held for the check. Use an
        # event-loop-aware acquire instead.
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(lock.acquire())
            with pytest.raises(Exception, match="is busy"):
                _call(
                    mcp,
                    "merge_into_dataset",
                    {"source_repo_id": source.repo_id, "target_repo_id": target.repo_id},
                )
        finally:
            lock.release()
            loop.close()

    def test_force_skips_schema_validation(self, two_datasets):
        """When force=True, schema mismatches detected by merge_into are
        bypassed at the dataset_tools level. We don't trigger a real
        mismatch here (the factory builds identical schemas) but verify
        that force=True is reflected in the response.
        """
        mcp, _, source, target = two_datasets
        result = _call(
            mcp,
            "merge_into_dataset",
            {
                "source_repo_id": source.repo_id,
                "target_repo_id": target.repo_id,
                "force": True,
            },
        )
        assert result["force_used"] is True


class TestCrossSurfaceSharedState:
    """The MCP merge tool and the FastAPI /api/edits/merge-into route
    both delegate to `_edits_core.merge_dataset_into`. A merge via
    either surface mutates the same on-disk dataset.
    """

    def test_mcp_merge_visible_via_appstate(self, two_datasets):
        mcp, state, source, target = two_datasets
        eps_before = target.num_episodes
        _call(
            mcp,
            "merge_into_dataset",
            {"source_repo_id": source.repo_id, "target_repo_id": target.repo_id},
        )
        # state.datasets[target] holds the same in-memory dataset that
        # the FastAPI handlers read from — confirm growth.
        assert state.datasets[target.repo_id].num_episodes > eps_before
