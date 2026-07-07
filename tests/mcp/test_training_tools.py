# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the training-orchestrator MCP tools (``training_*``).

Thin wrappers over the run orchestrator the GUI uses. We patch ``get_state`` to
inject a fake orchestrator that returns real ``Run`` / ``RunSnapshot`` objects,
so serialization + arg-mapping are exercised without running real training.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from lerobot.gui.api import training as training_api
from lerobot.gui.training.orchestrator import (
    HostBusyError,
    RunSnapshot,
    StartRequest,
    UnknownHostError,
    UnknownRunError,
)
from lerobot.gui.training.runs import Run, RunState
from lerobot.mcp.server import build_server


@pytest.fixture
def mcp(tmp_path):
    return build_server(dataset_root=tmp_path / "_root")


def _call(mcp_server, name, args):
    _, structured = asyncio.run(mcp_server.call_tool(name, args))
    return structured


def _run(run_id="abc", state=RunState.RUNNING):
    return Run(
        run_id=run_id,
        host_id="lab-L40s",
        recipe_name="my-run",
        dataset_id="lerobot/pusht",
        args={},
        state=state,
        created_at=1.0,
    )


def _orch_returning(**methods):
    orch = MagicMock()
    for name, value in methods.items():
        getattr(orch, name).return_value = value
    return orch


def test_training_tools_registered_with_scopes(mcp):
    by_name = {t["name"]: t for t in _call(mcp, "lerobot_list_tools", {})["tools"]}
    assert by_name["training_list_runs"]["scope"] == "read"
    assert by_name["training_get_run"]["scope"] == "read"
    assert by_name["training_list_hosts"]["scope"] == "read"
    assert by_name["training_start_run"]["scope"] == "operate"  # spawns a billable VM
    assert by_name["training_stop_run"]["scope"] == "operate"


def test_unavailable_when_orchestrator_not_initialized(mcp):
    training_api.reset_state_for_testing()
    out = _call(mcp, "training_list_runs", {})
    assert out["error"] == "training_unavailable"


def test_list_runs_serializes(mcp):
    orch = _orch_returning(list_runs=[_run("r1"), _run("r2")])
    with patch.object(training_api, "get_state", return_value=(orch, MagicMock())):
        out = _call(mcp, "training_list_runs", {})
    assert [r["run_id"] for r in out["runs"]] == ["r1", "r2"]
    assert out["runs"][0]["recipe_name"] == "my-run"


def test_get_run_serializes_snapshot(mcp):
    snap = RunSnapshot(
        run=_run("r5"), progress={"step": 30}, checkpoints=[], stderr_tail="", events=[], metrics=[]
    )
    orch = _orch_returning(poll=snap)
    with patch.object(training_api, "get_state", return_value=(orch, MagicMock())):
        out = _call(mcp, "training_get_run", {"run_id": "r5"})
    assert out["run"]["run_id"] == "r5"
    assert out["progress"] == {"step": 30}


def test_get_run_unknown(mcp):
    orch = MagicMock()
    orch.poll.side_effect = UnknownRunError("nope")
    with patch.object(training_api, "get_state", return_value=(orch, MagicMock())):
        out = _call(mcp, "training_get_run", {"run_id": "zzz"})
    assert out["error"] == "unknown_run"


def test_start_run_maps_args_and_fills_dataset(mcp):
    orch = _orch_returning(start=_run("new", state=RunState.PENDING))
    with patch.object(training_api, "get_state", return_value=(orch, MagicMock())):
        out = _call(
            mcp,
            "training_start_run",
            {
                "host_id": "lab-L40s",
                "dataset_id": "lerobot/pusht",
                "recipe_name": "smoke",
                "args": {"steps": 30},
                "idempotency_key": "k1",
            },
        )
    assert out["run_id"] == "new"
    req = orch.start.call_args.args[0]
    assert isinstance(req, StartRequest)
    assert (req.host_id, req.dataset_id, req.recipe_name, req.idempotency_key) == (
        "lab-L40s",
        "lerobot/pusht",
        "smoke",
        "k1",
    )
    # dataset_id is mirrored into the default recipe's dataset flag so the
    # caller doesn't have to know the dotted-key convention.
    assert req.args == {"steps": 30, "dataset.repo_id": "lerobot/pusht"}


def test_start_run_respects_explicit_dataset_repo_id(mcp):
    orch = _orch_returning(start=_run("new", state=RunState.PENDING))
    with patch.object(training_api, "get_state", return_value=(orch, MagicMock())):
        _call(
            mcp,
            "training_start_run",
            {
                "host_id": "h",
                "dataset_id": "lerobot/pusht",
                "recipe_name": "x",
                "args": {"dataset.repo_id": "someone/other"},
            },
        )
    req = orch.start.call_args.args[0]
    assert req.args["dataset.repo_id"] == "someone/other"  # caller wins over auto-fill


def test_start_run_hvla_recipe_fills_bare_key(mcp):
    orch = _orch_returning(start=_run("new", state=RunState.PENDING))
    with patch.object(training_api, "get_state", return_value=(orch, MagicMock())):
        _call(
            mcp,
            "training_start_run",
            {
                "host_id": "h",
                "dataset_id": "lerobot/pusht",
                "recipe_name": "x",
                "args": {"__recipe__": "hvla_flow_s1"},
            },
        )
    req = orch.start.call_args.args[0]
    assert req.args["dataset_repo_id"] == "lerobot/pusht"  # HVLA's bare key
    assert "dataset.repo_id" not in req.args


def test_start_run_unknown_host(mcp):
    orch = MagicMock()
    orch.start.side_effect = UnknownHostError("unknown host id: 'nope'")
    with patch.object(training_api, "get_state", return_value=(orch, MagicMock())):
        out = _call(mcp, "training_start_run", {"host_id": "nope", "dataset_id": "d", "recipe_name": "x"})
    assert out["error"] == "unknown_host"
    assert out["host_id"] == "nope"


def test_list_hosts_serializes(mcp):
    host = MagicMock()
    host.model_dump.return_value = {"id": "lab-L40s", "transport_kind": "ephemeral"}
    with (
        patch.object(training_api, "get_state", return_value=(MagicMock(), MagicMock())),
        patch.object(training_api, "list_hosts", return_value=[host]),
    ):
        out = _call(mcp, "training_list_hosts", {})
    assert out["hosts"] == [{"id": "lab-L40s", "transport_kind": "ephemeral"}]


def test_start_run_reports_host_busy(mcp):
    orch = MagicMock()
    orch.start.side_effect = HostBusyError("host lab-L40s already has an active run")
    with patch.object(training_api, "get_state", return_value=(orch, MagicMock())):
        out = _call(mcp, "training_start_run", {"host_id": "lab-L40s", "dataset_id": "d", "recipe_name": "x"})
    assert out["error"] == "host_busy"


def test_stop_run_delegates(mcp):
    orch = _orch_returning(stop=_run("r9", state=RunState.ABORTED))
    with patch.object(training_api, "get_state", return_value=(orch, MagicMock())):
        out = _call(mcp, "training_stop_run", {"run_id": "r9"})
    assert out["run_id"] == "r9"
    orch.stop.assert_called_once_with("r9")
