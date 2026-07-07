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
"""Tests for the read-only Run MCP tool surface.

These tools observe the GUI's subprocess (teleop / record / replay /
training) but cannot start, signal, or stop it. The lifecycle state
lives in module globals on ``lerobot.gui.api.run``; tests patch those
globals directly — same pattern as the existing GUI run tests.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from lerobot.gui.api import run as run_module
from lerobot.mcp.server import build_server


@pytest.fixture
def mcp(tmp_path):
    return build_server(dataset_root=tmp_path / "_root")


def _call(mcp_server, name, args):
    _, structured = asyncio.run(mcp_server.call_tool(name, args))
    return structured


# ── get_run_status ────────────────────────────────────────────────────────


class TestGetRunStatus:
    def test_no_active_process(self, mcp):
        with (
            patch.object(run_module, "_active_process", None),
            patch.object(run_module, "_active_command", None),
        ):
            result = _call(mcp, "get_run_status", {})
        assert result == {"running": False, "command": None}

    def test_active_process(self, mcp):
        proc = MagicMock(returncode=None, pid=12345)
        with (
            patch.object(run_module, "_active_process", proc),
            patch.object(run_module, "_active_command", "teleoperate"),
        ):
            result = _call(mcp, "get_run_status", {})
        assert result["running"] is True
        assert result["command"] == "teleoperate"
        assert result["pid"] == 12345
        # ``returncode`` not present when running
        assert "returncode" not in result

    def test_exited_process_carries_returncode(self, mcp):
        proc = MagicMock(returncode=0, pid=12345)
        with (
            patch.object(run_module, "_active_process", proc),
            patch.object(run_module, "_active_command", "record"),
        ):
            result = _call(mcp, "get_run_status", {})
        assert result["running"] is False
        assert result["command"] == "record"
        assert result["returncode"] == 0

    def test_exited_with_nonzero_returncode(self, mcp):
        # Non-zero returncode = error or killed by signal; the AI should
        # surface this rather than treat "not running" as "succeeded".
        proc = MagicMock(returncode=-15, pid=12345)  # -15 = SIGTERM
        with (
            patch.object(run_module, "_active_process", proc),
            patch.object(run_module, "_active_command", "hvla"),
        ):
            result = _call(mcp, "get_run_status", {})
        assert result == {"running": False, "command": "hvla", "returncode": -15}


# ── get_run_output ────────────────────────────────────────────────────────


class TestGetRunOutput:
    def test_empty_buffer(self, mcp):
        with patch.object(run_module, "_output_lines", []):
            result = _call(mcp, "get_run_output", {})
        assert result == {"lines": [], "total_buffered": 0, "truncated": False}

    def test_returns_tail(self, mcp):
        buffer = [f"line {i}" for i in range(50)]
        with patch.object(run_module, "_output_lines", buffer):
            result = _call(mcp, "get_run_output", {"last_n": 10})
        assert result["lines"] == [f"line {i}" for i in range(40, 50)]
        assert result["total_buffered"] == 50
        assert result["truncated"] is True

    def test_last_n_zero_returns_no_lines_but_buffered_count(self, mcp):
        buffer = ["a", "b", "c"]
        with patch.object(run_module, "_output_lines", buffer):
            result = _call(mcp, "get_run_output", {"last_n": 0})
        assert result == {"lines": [], "total_buffered": 3, "truncated": True}

    def test_asking_for_more_than_buffered_returns_all(self, mcp):
        buffer = ["a", "b", "c"]
        with patch.object(run_module, "_output_lines", buffer):
            result = _call(mcp, "get_run_output", {"last_n": 100})
        assert result == {"lines": ["a", "b", "c"], "total_buffered": 3, "truncated": False}

    def test_negative_last_n_raises(self, mcp):
        with pytest.raises(Exception, match="last_n must be >= 0"):
            _call(mcp, "get_run_output", {"last_n": -1})


# ── get_latency_metrics ───────────────────────────────────────────────────


class TestGetLatencyMetrics:
    def test_no_snapshot_file_returns_empty_stub(self, mcp, tmp_path):
        with patch.object(run_module, "LATENCY_SOURCES", {"teleop": tmp_path}):
            result = _call(mcp, "get_latency_metrics", {"source": "teleop"})
        assert result == {
            "n_records": 0,
            "dropped_records": 0,
            "overrun_ratio": 0.0,
            "stages": {},
            "series": {},
        }

    def test_unknown_source_returns_empty_stub(self, mcp):
        with patch.object(run_module, "LATENCY_SOURCES", {"teleop": "/tmp/teleop"}):
            result = _call(mcp, "get_latency_metrics", {"source": "made_up"})
        assert result["n_records"] == 0

    def test_reads_existing_snapshot_file(self, mcp, tmp_path):
        snapshot = {
            "n_records": 1234,
            "dropped_records": 3,
            "overrun_ratio": 0.012,
            "stages": {"capture": {"mean_ms": 4.2}},
            "series": {},
        }
        (tmp_path / "latency_snapshot.json").write_text(json.dumps(snapshot))
        with patch.object(run_module, "LATENCY_SOURCES", {"teleop": tmp_path}):
            result = _call(mcp, "get_latency_metrics", {"source": "teleop"})
        assert result["n_records"] == 1234
        assert result["stages"]["capture"]["mean_ms"] == 4.2

    def test_default_source_is_teleop(self, mcp, tmp_path):
        snapshot = {"n_records": 7, "stages": {}, "series": {}}
        (tmp_path / "latency_snapshot.json").write_text(json.dumps(snapshot))
        with patch.object(run_module, "LATENCY_SOURCES", {"teleop": tmp_path}):
            result = _call(mcp, "get_latency_metrics", {})
        assert result["n_records"] == 7


# ── get_rlt_metrics ───────────────────────────────────────────────────────


class TestGetRltMetrics:
    def test_no_active_run_returns_idle_stub(self, mcp):
        with patch.object(run_module, "_active_config", None):
            result = _call(mcp, "get_rlt_metrics", {})
        assert result["mode"] == "IDLE"
        assert result["episode"] == 0
        assert result["total_episodes"] == 0

    def test_reads_metrics_file_when_active(self, mcp, tmp_path):
        rlt_dir = tmp_path / "rlt_run"
        rlt_dir.mkdir()
        metrics = {
            "episode": 42,
            "step_count": 1500,
            "buffer_size": 5000,
            "total_updates": 800,
            "mode": "TRAINING",
            "success_rate": 0.65,
            "total_successes": 27,
            "total_episodes": 42,
            "series": {},
        }
        (rlt_dir / "metrics.json").write_text(json.dumps(metrics))
        with patch.object(run_module, "_active_config", {"rlt_output_dir": str(rlt_dir)}):
            result = _call(mcp, "get_rlt_metrics", {})
        assert result["mode"] == "TRAINING"
        assert result["episode"] == 42
        assert result["success_rate"] == 0.65


# ── cross-surface shared-state ────────────────────────────────────────────


class TestCrossSurfaceSharedState:
    """The MCP tools and FastAPI handlers read from the same module
    globals on ``lerobot.gui.api.run``. Verify that patching the global
    surfaces through both layers.
    """

    def test_status_visible_via_both_surfaces(self, mcp):
        proc = MagicMock(returncode=None, pid=99999)
        with (
            patch.object(run_module, "_active_process", proc),
            patch.object(run_module, "_active_command", "replay"),
        ):
            # MCP side
            mcp_result = _call(mcp, "get_run_status", {})
            # FastAPI handler reads the same globals
            from lerobot.gui.api._run_core import get_run_status

            fastapi_result = get_run_status()
        assert mcp_result == fastapi_result
        assert mcp_result["pid"] == 99999
