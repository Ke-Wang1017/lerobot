# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for GET/PUT/DELETE /api/training/nebius/connection.

The store's directory is redirected to a tmp path so no test touches
``~/.config``. The private key is never returned by any endpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lerobot.gui.api import training as training_api
from lerobot.gui.training import nebius_credentials


@pytest.fixture
def nebius_dir(tmp_path: Path, monkeypatch) -> Path:
    d = tmp_path / "nebius"
    monkeypatch.setattr(nebius_credentials, "NEBIUS_DIR", d)
    return d


@pytest.fixture
def client(nebius_dir: Path) -> TestClient:
    app = FastAPI()
    app.include_router(training_api.router)
    return TestClient(app)


def _valid_key() -> str:
    return json.dumps(
        {
            "subject-credentials": {
                "alg": "RS256",
                "private-key": "-----BEGIN PRIVATE KEY-----\nMIIfake\n-----END PRIVATE KEY-----",
                "kid": "publickey-abc",
                "iss": "serviceaccount-xyz",
                "sub": "serviceaccount-xyz",
            }
        }
    )


def test_get_connection_empty(client: TestClient):
    resp = client.get("/api/training/nebius/connection")
    assert resp.status_code == 200
    assert resp.json()["configured"] is False


def test_put_then_get_roundtrip(client: TestClient):
    resp = client.put(
        "/api/training/nebius/connection",
        json={"key_json": _valid_key(), "project_id": "project-e00", "subnet_id": "vpcsubnet-e00"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["configured"] is True
    assert body["service_account_id"] == "serviceaccount-xyz"
    assert body["project_id"] == "project-e00"

    # Re-GET reflects the stored state.
    got = client.get("/api/training/nebius/connection").json()
    assert got["configured"] is True
    assert got["subnet_id"] == "vpcsubnet-e00"


def test_put_console_pieces_assembled_server_side(client: TestClient):
    # Console path: no key_json — send the private key + the two IDs; the
    # server assembles the same subject-credentials JSON.
    resp = client.put(
        "/api/training/nebius/connection",
        json={
            "private_key": "-----BEGIN PRIVATE KEY-----\nMIIfake\n-----END PRIVATE KEY-----",
            "key_id": "publickey-abc",
            "service_account_id": "serviceaccount-xyz",
            "project_id": "project-e00",
            "subnet_id": "vpcsubnet-e00",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["configured"] is True
    assert body["service_account_id"] == "serviceaccount-xyz"
    assert body["key_id"] == "publickey-abc"


def test_put_with_neither_key_nor_pieces_returns_400(client: TestClient):
    resp = client.put(
        "/api/training/nebius/connection",
        json={"project_id": "p", "subnet_id": "s"},
    )
    assert resp.status_code == 400


def test_response_never_contains_private_key(client: TestClient, nebius_dir: Path):
    resp = client.put(
        "/api/training/nebius/connection",
        json={"key_json": _valid_key(), "project_id": "p", "subnet_id": "s"},
    )
    assert "PRIVATE KEY" not in resp.text
    assert "private-key" not in resp.text
    # But the key did land on disk (0600 file).
    assert (nebius_dir / "service_account.json").exists()


def test_put_malformed_key_returns_400(client: TestClient):
    resp = client.put(
        "/api/training/nebius/connection",
        json={"key_json": "garbage", "project_id": "p", "subnet_id": "s"},
    )
    assert resp.status_code == 400
    assert "JSON" in resp.json()["detail"]


def test_put_missing_project_returns_422(client: TestClient):
    # Empty project_id fails pydantic min_length before reaching the store.
    resp = client.put(
        "/api/training/nebius/connection",
        json={"key_json": _valid_key(), "project_id": "", "subnet_id": "s"},
    )
    assert resp.status_code == 422


def test_delete_clears(client: TestClient):
    client.put(
        "/api/training/nebius/connection",
        json={"key_json": _valid_key(), "project_id": "p", "subnet_id": "s"},
    )
    resp = client.delete("/api/training/nebius/connection")
    assert resp.status_code == 200
    assert resp.json()["cleared"] is True
    # Idempotent second clear.
    assert client.delete("/api/training/nebius/connection").json()["cleared"] is False
    assert client.get("/api/training/nebius/connection").json()["configured"] is False
