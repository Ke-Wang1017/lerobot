# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the training API endpoints.

Uses a FastAPI TestClient and wires the orchestrator against a tmp_path
runs dir + a single test host. Verifies the routes' behavior, status codes,
and DTO shapes — separate from the orchestrator unit tests.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lerobot.gui.api import training as training_api
from lerobot.gui.training.hosts import HostRegistry, TrainingHost
from lerobot.gui.training.orchestrator import Orchestrator
from lerobot.gui.training.runs import RunRegistry
from lerobot.gui.training.transport import SubprocessTransport


@pytest.fixture
def app(tmp_path: Path):
    """Fresh app with the training router and a tmp orchestrator wired in."""
    training_api.reset_state_for_testing()
    host = TrainingHost(
        id="test-host",
        display_name="Test Host",
        transport=SubprocessTransport(workdir=tmp_path / "workdir"),
        capabilities={"gpu_name": "Test", "vram_mb": 16384, "gpu_count_detected": 1},
    )
    hosts = HostRegistry(hosts=[host])
    runs = RunRegistry(runs_dir=tmp_path / "runs")
    orch = Orchestrator(host_registry=hosts, run_registry=runs)
    training_api.init_state(orch=orch, host_registry=hosts)
    app = FastAPI()
    app.include_router(training_api.router)
    yield app
    training_api.reset_state_for_testing()


@pytest.fixture
def client(app: FastAPI):
    return TestClient(app)


def _start_run_payload(**over) -> dict:
    base = {
        "host_id": "test-host",
        "recipe_name": "fake",
        "dataset_id": "fake/ds",
        "args": {"__recipe__": "__fake__", "num_steps": 5, "save_every": 10, "step_seconds": 0.05},
    }
    base.update(over)
    return base


def _wait_until_state(client: TestClient, run_id: str, want: str, timeout: float = 30.0):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        resp = client.get(f"/api/training/runs/{run_id}")
        assert resp.status_code == 200
        body = resp.json()
        last = body
        if body["run"]["state"] == want or body["run"]["state"] in {
            "completed",
            "failed",
            "aborted",
        }:
            return body
        time.sleep(0.05)
    raise AssertionError(f"timed out; last state={last['run']['state'] if last else None}")


# ── /hosts ────────────────────────────────────────────────────────────────────


def test_list_hosts(client: TestClient) -> None:
    resp = client.get("/api/training/hosts")
    assert resp.status_code == 200
    hosts = resp.json()
    assert len(hosts) == 1
    h = hosts[0]
    assert h["id"] == "test-host"
    assert h["display_name"] == "Test Host"
    assert h["transport_kind"] == "subprocess"
    assert h["capabilities"]["gpu_name"] == "Test"


# ── /runs (list + create) ─────────────────────────────────────────────────────


def test_list_runs_empty(client: TestClient) -> None:
    resp = client.get("/api/training/runs")
    assert resp.status_code == 200
    assert resp.json() == []


def test_start_run_201(client: TestClient) -> None:
    """POST returns immediately with state=pending (C5 background prep
    thread does image pull + worker launch; advances to running on
    completion). The state machine + session_id show up on subsequent
    polls — we wait until completed to assert the full lifecycle ran."""
    resp = client.post("/api/training/runs", json=_start_run_payload())
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["state"] == "pending"
    assert body["host_id"] == "test-host"
    assert body["recipe_name"] == "fake"
    # session_id is set by the prep thread once it spawns the worker
    assert body["session_id"] is None
    # Drive through to completion so subsequent fixtures see a clean slate.
    final = _wait_until_state(client, body["run_id"], "completed")
    assert final["run"]["session_id"] is not None


def test_start_run_unknown_host_404(client: TestClient) -> None:
    resp = client.post("/api/training/runs", json=_start_run_payload(host_id="nope"))
    assert resp.status_code == 404
    assert "unknown host" in resp.json()["detail"].lower()


def test_start_run_validation_400(client: TestClient) -> None:
    resp = client.post(
        "/api/training/runs",
        json={"host_id": "test-host", "recipe_name": "", "dataset_id": "x"},
    )
    assert resp.status_code == 422  # FastAPI validation error for min_length=1


def test_start_run_host_busy_409(client: TestClient) -> None:
    """Two start requests targeting the same host while the first is running →
    second gets 409."""
    long_payload = _start_run_payload(
        args={"__recipe__": "__fake__", "num_steps": 1000, "save_every": 100, "step_seconds": 0.05}
    )
    r1 = client.post("/api/training/runs", json=long_payload)
    assert r1.status_code == 201
    try:
        r2 = client.post("/api/training/runs", json=_start_run_payload())
        assert r2.status_code == 409
        assert "busy" in r2.json()["detail"].lower()
    finally:
        client.post(f"/api/training/runs/{r1.json()['run_id']}/stop")
        _wait_until_state(client, r1.json()["run_id"], "aborted")


def test_start_run_idempotency_returns_same_id(client: TestClient) -> None:
    p = _start_run_payload(idempotency_key="abc")
    r1 = client.post("/api/training/runs", json=p)
    assert r1.status_code == 201
    _wait_until_state(client, r1.json()["run_id"], "completed")
    # Resubmit with same key → same run_id (even though first is done)
    r2 = client.post("/api/training/runs", json=p)
    assert r2.status_code == 201
    assert r2.json()["run_id"] == r1.json()["run_id"]


# ── /runs/{id} ────────────────────────────────────────────────────────────────


def test_get_run_404(client: TestClient) -> None:
    resp = client.get("/api/training/runs/missing")
    assert resp.status_code == 404


def test_get_run_snapshot_after_completion(client: TestClient) -> None:
    r = client.post("/api/training/runs", json=_start_run_payload()).json()
    body = _wait_until_state(client, r["run_id"], "completed")
    assert body["run"]["state"] == "completed"
    assert body["progress"]["step"] == 5
    assert body["progress"]["loss"] > 0
    # save_every=10, num_steps=5 — no checkpoint expected
    assert body["checkpoints"] == []
    assert "[runner]" in body["stderr_tail"]


def test_get_run_snapshot_with_checkpoints(client: TestClient) -> None:
    r = client.post(
        "/api/training/runs",
        json=_start_run_payload(
            args={"__recipe__": "__fake__", "num_steps": 10, "save_every": 5, "step_seconds": 0.05}
        ),
    ).json()
    body = _wait_until_state(client, r["run_id"], "completed")
    assert len(body["checkpoints"]) == 2
    assert body["checkpoints"][0]["step"] == 5
    assert body["checkpoints"][1]["step"] == 10
    assert all(c["sha256"] for c in body["checkpoints"])


# ── /runs/{id}/stop ───────────────────────────────────────────────────────────


def test_stop_run_404(client: TestClient) -> None:
    resp = client.post("/api/training/runs/missing/stop")
    assert resp.status_code == 404


def test_stop_run_aborts(client: TestClient) -> None:
    r = client.post(
        "/api/training/runs",
        json=_start_run_payload(
            args={"__recipe__": "__fake__", "num_steps": 1000, "save_every": 100, "step_seconds": 0.05}
        ),
    ).json()
    time.sleep(0.3)
    resp = client.post(f"/api/training/runs/{r['run_id']}/stop")
    assert resp.status_code == 200
    body = _wait_until_state(client, r["run_id"], "aborted")
    assert body["run"]["state"] == "aborted"


def test_stop_run_idempotent_on_completed(client: TestClient) -> None:
    r = client.post("/api/training/runs", json=_start_run_payload()).json()
    _wait_until_state(client, r["run_id"], "completed")
    resp = client.post(f"/api/training/runs/{r['run_id']}/stop")
    assert resp.status_code == 200
    assert resp.json()["state"] == "completed"


# ── /runs list reflects activity ──────────────────────────────────────────────


def test_list_runs_after_some_activity(client: TestClient) -> None:
    a = client.post("/api/training/runs", json=_start_run_payload()).json()
    _wait_until_state(client, a["run_id"], "completed")
    b = client.post("/api/training/runs", json=_start_run_payload()).json()
    _wait_until_state(client, b["run_id"], "completed")
    listed = client.get("/api/training/runs").json()
    ids = {r["run_id"] for r in listed}
    assert {a["run_id"], b["run_id"]}.issubset(ids)


# ── State guard ───────────────────────────────────────────────────────────────


def test_uninitialized_state_raises(tmp_path: Path) -> None:
    """If init_state was never called, endpoints should fail loudly."""
    training_api.reset_state_for_testing()
    app = FastAPI()
    app.include_router(training_api.router)
    cli = TestClient(app, raise_server_exceptions=True)
    # Calling an endpoint without init_state() → 500 from the get_state() guard
    with pytest.raises(RuntimeError, match="not initialized"):
        cli.get("/api/training/hosts")


# ── DTO consistency ───────────────────────────────────────────────────────────


def test_run_dto_does_not_leak_idempotency_key(client: TestClient) -> None:
    r = client.post(
        "/api/training/runs",
        json=_start_run_payload(idempotency_key="my-secret"),
    ).json()
    assert "idempotency_key" not in r
    _wait_until_state(client, r["run_id"], "completed")


# ── DELETE + clear endpoints (housekeeping) ───────────────────────────────────


def test_delete_run_404_on_unknown(client: TestClient) -> None:
    resp = client.delete("/api/training/runs/does-not-exist")
    assert resp.status_code == 404


def test_delete_completed_run_returns_kept_model_true(client: TestClient) -> None:
    """Delete a completed run with a model: response says
    ``kept_model: true``, subsequent GET 404s."""
    payload = _start_run_payload(
        args={"__recipe__": "__fake__", "num_steps": 10, "save_every": 5, "step_seconds": 0.05}
    )
    r = client.post("/api/training/runs", json=payload).json()
    _wait_until_state(client, r["run_id"], "completed")
    resp = client.delete(f"/api/training/runs/{r['run_id']}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["run_id"] == r["run_id"]
    assert body["metadata_bytes_freed"] >= 0
    assert body["kept_model"] is True  # fake-runner produced 2 checkpoints
    # Subsequent GET returns 404 (gone from history)
    assert client.get(f"/api/training/runs/{r['run_id']}").status_code == 404


def test_delete_active_run_409(client: TestClient) -> None:
    """Can't delete a running run; must Stop first."""
    payload = _start_run_payload()
    payload["args"] = {
        "__recipe__": "__fake__",
        "num_steps": 1000,
        "save_every": 100,
        "step_seconds": 1.0,
    }
    r = client.post("/api/training/runs", json=payload).json()
    _wait_until_state(client, r["run_id"], "running")
    resp = client.delete(f"/api/training/runs/{r['run_id']}")
    assert resp.status_code == 409
    assert "stop it first" in resp.json()["detail"]
    # Cleanup
    client.post(f"/api/training/runs/{r['run_id']}/stop")
    _wait_until_state(client, r["run_id"], "aborted")


def test_clear_terminal_endpoint(client: TestClient) -> None:
    """POST /api/training/runs/clear returns deleted list + metadata_bytes
    + models_kept."""
    payload = _start_run_payload(
        args={"__recipe__": "__fake__", "num_steps": 10, "save_every": 5, "step_seconds": 0.05}
    )
    r = client.post("/api/training/runs", json=payload).json()
    _wait_until_state(client, r["run_id"], "completed")
    resp = client.post("/api/training/runs/clear")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert r["run_id"] in body["deleted"]
    assert body["models_kept"] == 1  # the fake run produced a checkpoint
    # Idempotent: second call has nothing to delete
    resp2 = client.post("/api/training/runs/clear")
    body2 = resp2.json()
    assert body2["deleted"] == []
    assert body2["models_kept"] == 0


# ── /api/training/policies (catalog) ──────────────────────────────────────────


def test_list_policies_includes_act_and_hvla(client: TestClient) -> None:
    """Auto-discovery should pick up every PreTrainedConfig subclass + the
    manually-registered HVLA recipe."""
    resp = client.get("/api/training/policies")
    assert resp.status_code == 200
    catalog = resp.json()
    types = [p["type_name"] for p in catalog]
    # Auto-discovered draccus policies (sampling — full set depends on extras)
    assert "act" in types
    assert "diffusion" in types
    # Manually-registered non-draccus recipe
    assert "hvla_flow_s1" in types


def test_list_policies_act_entry_has_renderable_fields(client: TestClient) -> None:
    """ACT's draccus config should expose its scalar fields with defaults
    + recognisable form types."""
    catalog = client.get("/api/training/policies").json()
    act = next(p for p in catalog if p["type_name"] == "act")
    assert act["recipe"] is None  # default lerobot-train
    assert act["arg_key_prefix"] == "policy."
    assert act["fields"], "act should have at least one renderable field"
    field_names = {f["name"] for f in act["fields"]}
    # Spot-check that the headline ACT fields are present
    assert "chunk_size" in field_names
    assert "n_action_steps" in field_names
    assert "dim_model" in field_names
    # Every field has a usable form type
    for f in act["fields"]:
        assert f["type"] in {"int", "float", "bool", "string", "select"}
        assert "default" in f


def test_list_policies_hvla_entry_uses_recipe_marker(client: TestClient) -> None:
    """HVLA's manual entry should declare its recipe marker + bare
    (no-prefix) arg keys."""
    catalog = client.get("/api/training/policies").json()
    hvla = next(p for p in catalog if p["type_name"] == "hvla_flow_s1")
    assert hvla["recipe"] == "hvla_flow_s1"
    assert hvla["arg_key_prefix"] == ""
    field_names = {f["name"] for f in hvla["fields"]}
    assert {"chunk_size", "num_inference_steps", "hidden_dim"} <= field_names


def test_list_policies_skips_complex_fields(client: TestClient) -> None:
    """Fields the form can't usefully render (list/dict/nested-dataclass)
    must be dropped from the catalog — otherwise the form would silently
    fall back to a free-text input that the user can't fill correctly."""
    catalog = client.get("/api/training/policies").json()
    act = next(p for p in catalog if p["type_name"] == "act")
    # ACT's config defines complex-typed fields like
    # optimizer_lr_backbone_scale, image_features, etc. — none of those
    # should be in the catalog.
    for f in act["fields"]:
        assert f["type"] in {"int", "float", "bool", "string", "select"}
