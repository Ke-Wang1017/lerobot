# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for POST/GET/DELETE /api/training/hosts + POST .../probe.

The probe endpoint is exercised by mocking ``probe_ssh`` in the API
module so the tests don't shell out to real SSH. The lifecycle endpoints
write to a tmp HOSTS_DIR via monkeypatch.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lerobot.gui.api import training as training_api
from lerobot.gui.training.hosts import WORKSTATION_HOST_ID, HostRegistry, TrainingHost
from lerobot.gui.training.jobs import HostProfile
from lerobot.gui.training.orchestrator import Orchestrator
from lerobot.gui.training.probe import CheckItem, ProbeResult
from lerobot.gui.training.runs import RunRegistry
from lerobot.gui.training.transport import SubprocessTransport


@pytest.fixture
def hosts_dir(tmp_path, monkeypatch):
    """Redirect HOSTS_DIR to a tmp dir for the duration of the test."""
    d = tmp_path / "training_hosts"
    monkeypatch.setattr(training_api, "HOSTS_DIR", d)
    return d


@pytest.fixture
def app(tmp_path: Path, hosts_dir: Path):
    training_api.reset_state_for_testing()
    # Workstation stub so we can test the "reserved name" rejection.
    workstation = TrainingHost(
        id=WORKSTATION_HOST_ID,
        display_name="This server",
        transport=SubprocessTransport(workdir=tmp_path / "workdir"),
        capabilities={"gpu_name": "Test", "vram_mb": 16384, "gpu_count_detected": 1},
    )
    hosts = HostRegistry(hosts=[workstation])
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


# ── POST /hosts ───────────────────────────────────────────────────────────────


def test_post_host_creates_and_persists(client: TestClient, hosts_dir: Path):
    body = {"name": "lab-h100", "host": "user@lab-h100.example.com"}
    resp = client.post("/api/training/hosts", json=body)
    assert resp.status_code == 201, resp.text
    info = resp.json()
    assert info["id"] == "lab-h100"
    assert info["display_name"] == "lab-h100"
    assert info["transport_kind"] == "ssh"
    # File written
    profile = HostProfile.load(hosts_dir / "lab-h100.json")
    assert profile.ssh_user == "user"
    assert profile.ssh_host == "lab-h100.example.com"
    assert profile.ssh_port == 22


def test_post_host_parses_alias_with_port(client: TestClient, hosts_dir: Path):
    resp = client.post(
        "/api/training/hosts",
        json={"name": "remote", "host": "deploy@10.0.0.5:2222", "display_name": "Lab"},
    )
    assert resp.status_code == 201
    profile = HostProfile.load(hosts_dir / "remote.json")
    assert profile.ssh_user == "deploy"
    assert profile.ssh_host == "10.0.0.5"
    assert profile.ssh_port == 2222
    assert profile.display_name == "Lab"


def test_post_host_collision_returns_409(client: TestClient):
    body = {"name": "dup", "host": "user@host"}
    assert client.post("/api/training/hosts", json=body).status_code == 201
    resp = client.post("/api/training/hosts", json=body)
    assert resp.status_code == 409


def test_post_host_reserved_name_rejected(client: TestClient):
    resp = client.post(
        "/api/training/hosts",
        json={"name": WORKSTATION_HOST_ID, "host": "user@host"},
    )
    assert resp.status_code == 400
    assert "reserved" in resp.json()["detail"]


def test_post_host_invalid_name_rejected_by_pydantic(client: TestClient):
    resp = client.post("/api/training/hosts", json={"name": "bad name!", "host": "u@h"})
    assert resp.status_code == 422


def test_post_host_bare_ipv6_passes_through_unsplit(client: TestClient, hosts_dir: Path):
    """Regression: bare IPv6 has multiple colons — must not misparse
    '…:1' as host + port. The address goes through verbatim (ssh accepts
    bare IPv6 destinations)."""
    resp = client.post("/api/training/hosts", json={"name": "v6", "host": "feit@2001:db8::1"})
    assert resp.status_code == 201
    profile = HostProfile.load(hosts_dir / "v6.json")
    assert profile.ssh_host == "2001:db8::1"
    assert profile.ssh_port == 22


def test_post_host_failure_leaves_no_orphan_file(client: TestClient, hosts_dir: Path, monkeypatch):
    """Regression: registry.add runs BEFORE profile.save, so a collision
    that slips past the early 409 check (e.g. concurrent POSTs) fails
    before any file is written — no orphan for a GUI restart to load."""
    from lerobot.gui.api import training as api_mod

    def boom(profile):
        raise RuntimeError("simulated registry failure")

    monkeypatch.setattr(api_mod, "profile_to_training_host", boom)
    with pytest.raises(RuntimeError, match="simulated registry failure"):
        client.post("/api/training/hosts", json={"name": "orphan", "host": "u@h"})
    assert not (hosts_dir / "orphan.json").exists()


# ── GET /hosts (now lists workstation + saved) ────────────────────────────────


def test_get_hosts_lists_new_host_after_post(client: TestClient):
    client.post("/api/training/hosts", json={"name": "lab", "host": "u@h"})
    resp = client.get("/api/training/hosts")
    assert resp.status_code == 200
    listed = resp.json()
    ids = {h["id"] for h in listed}
    assert WORKSTATION_HOST_ID in ids
    assert "lab" in ids


# ── DELETE /hosts/{id} ────────────────────────────────────────────────────────


def test_delete_host_removes_and_returns_204(client: TestClient, hosts_dir: Path):
    client.post("/api/training/hosts", json={"name": "ephem", "host": "u@h"})
    assert (hosts_dir / "ephem.json").exists()
    resp = client.delete("/api/training/hosts/ephem")
    assert resp.status_code == 204
    assert not (hosts_dir / "ephem.json").exists()
    # GET no longer includes it
    listed = client.get("/api/training/hosts").json()
    assert "ephem" not in {h["id"] for h in listed}


def test_delete_workstation_rejected_400(client: TestClient):
    resp = client.delete(f"/api/training/hosts/{WORKSTATION_HOST_ID}")
    assert resp.status_code == 400


def test_delete_unknown_host_404(client: TestClient):
    resp = client.delete("/api/training/hosts/nope")
    assert resp.status_code == 404


# ── POST /hosts/probe ─────────────────────────────────────────────────────────


def test_probe_endpoint_returns_dto(client: TestClient, monkeypatch):
    async def fake_probe_ssh(host_spec, **kw):
        return ProbeResult(
            ok=True,
            latency_ms=123,
            checks=[
                CheckItem(name="ssh", ok=True, detail="connected"),
                CheckItem(name="docker", ok=True, detail="/usr/bin/docker"),
                CheckItem(name="tmux", ok=True, detail="/usr/bin/tmux"),
                CheckItem(name="nvidia", ok=True, detail="GPU 0: NVIDIA L40S"),
            ],
            error_class=None,
            message="All checks passed",
        )

    monkeypatch.setattr(training_api, "probe_ssh", fake_probe_ssh)
    resp = client.post("/api/training/hosts/probe", json={"host": "user@host"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["latency_ms"] == 123
    assert [c["name"] for c in body["checks"]] == ["ssh", "docker", "tmux", "nvidia"]


def test_probe_endpoint_passes_host_string_verbatim(client: TestClient, monkeypatch):
    captured: dict[str, str] = {}

    async def fake_probe_ssh(host_spec, **kw):
        captured["host"] = host_spec
        return ProbeResult(
            ok=False,
            latency_ms=10,
            checks=[
                CheckItem(name=n, ok=False, detail="not reached") for n in ("ssh", "docker", "tmux", "nvidia")
            ],
            error_class="dns",
            message="hostname unresolvable",
        )

    monkeypatch.setattr(training_api, "probe_ssh", fake_probe_ssh)
    client.post("/api/training/hosts/probe", json={"host": "alias-with-config-block"})
    assert captured["host"] == "alias-with-config-block"


def test_probe_endpoint_does_not_write_anything(client: TestClient, hosts_dir: Path, monkeypatch):
    async def fake_probe_ssh(host_spec, **kw):
        return ProbeResult(
            ok=True,
            latency_ms=1,
            checks=[CheckItem(name=n, ok=True, detail="ok") for n in ("ssh", "docker", "tmux", "nvidia")],
        )

    monkeypatch.setattr(training_api, "probe_ssh", fake_probe_ssh)
    client.post("/api/training/hosts/probe", json={"host": "u@h"})
    # Probe is stateless — no profile file is created.
    assert not hosts_dir.exists() or list(hosts_dir.iterdir()) == []


# ── POST /hosts — Ephemeral cloud host (provider_id branch) ──────────────────


def test_post_ephemeral_host_persists_spawn_spec(client: TestClient, hosts_dir: Path):
    resp = client.post(
        "/api/training/hosts",
        json={
            "name": "nebius-l40s",
            "provider_id": "nebius",
            "gpu": "L40S",
            "disk_gib": 100,
            "ttl_hours": 12,
            "display_name": "Nebius L40S",
        },
    )
    assert resp.status_code == 201, resp.text
    info = resp.json()
    assert info["id"] == "nebius-l40s"
    assert info["transport_kind"] == "ephemeral"
    assert info["capabilities"]["provider_id"] == "nebius"
    assert info["capabilities"]["gpu_name"] == "L40S"
    assert info["capabilities"]["ttl_hours"] == 12
    # Persisted profile carries the spawn spec, no SSH endpoint.
    profile = HostProfile.load(hosts_dir / "nebius-l40s.json")
    assert profile.is_ephemeral
    assert profile.provider_id == "nebius"
    assert profile.gpu == "L40S"
    assert profile.ttl_hours == 12
    assert profile.ssh_host == ""


def test_post_ephemeral_host_materializes_ephemeral_training_host(client: TestClient):
    client.post(
        "/api/training/hosts",
        json={"name": "neb", "provider_id": "nebius", "gpu": "L40S"},
    )
    _, registry = training_api.get_state()
    th = registry.get("neb")
    assert th is not None
    assert th.is_ephemeral
    assert th.transport is None
    assert th.provider_id == "nebius"
    assert th.spawn_spec.gpu == "L40S"
    assert th.spawn_spec.ttl_seconds == 24 * 3600  # default 24h


def test_post_persistent_host_still_requires_host(client: TestClient):
    resp = client.post("/api/training/hosts", json={"name": "no-endpoint"})
    assert resp.status_code == 422
    assert "host is required" in resp.json()["detail"]
