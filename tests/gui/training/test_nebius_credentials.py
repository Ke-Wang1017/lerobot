# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the server-held Nebius connection store.

All tests write to a tmp dir (never ``~/.config``). Validation is exercised
without importing the optional ``nebius`` SDK — the store mirrors the SDK's
credentials-file shape itself.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from lerobot.gui.training.nebius_credentials import (
    NebiusConnectionStore,
    NebiusCredentialError,
    assemble_authorized_key_json,
)


def _valid_key(sub: str = "serviceaccount-xyz", kid: str = "publickey-abc") -> str:
    return json.dumps(
        {
            "subject-credentials": {
                "alg": "RS256",
                "private-key": "-----BEGIN PRIVATE KEY-----\nMIIfake\n-----END PRIVATE KEY-----",
                "kid": kid,
                "iss": sub,
                "sub": sub,
            }
        }
    )


@pytest.fixture
def store(tmp_path: Path) -> NebiusConnectionStore:
    return NebiusConnectionStore(tmp_path / "nebius")


def test_assemble_authorized_key_json_validates_and_stores(store: NebiusConnectionStore):
    # The console path: pieces (PEM + key id + SA id) → the same subject-
    # credentials JSON, which must pass the store's validation and round-trip.
    key_json = assemble_authorized_key_json(
        private_key="-----BEGIN PRIVATE KEY-----\nMIIfake\n-----END PRIVATE KEY-----",
        key_id="publickey-abc",
        service_account_id="serviceaccount-xyz",
    )
    st = store.set(key_json=key_json, project_id="project-e00", subnet_id="vpcsubnet-e00")
    assert st.configured
    assert st.service_account_id == "serviceaccount-xyz"  # iss == sub == SA id
    assert st.key_id == "publickey-abc"


def test_status_empty_when_nothing_stored(store: NebiusConnectionStore):
    st = store.status()
    assert st.configured is False
    assert st.has_key is False
    assert st.project_id is None and st.subnet_id is None


def test_set_then_status_reports_configured(store: NebiusConnectionStore):
    st = store.set(key_json=_valid_key(), project_id="project-e00", subnet_id="vpcsubnet-e00")
    assert st.configured is True
    assert st.has_key is True
    assert st.service_account_id == "serviceaccount-xyz"
    assert st.key_id == "publickey-abc"
    assert st.project_id == "project-e00"
    assert st.subnet_id == "vpcsubnet-e00"


def test_key_file_is_0600_under_0700_dir(store: NebiusConnectionStore):
    store.set(key_json=_valid_key(), project_id="p", subnet_id="s")
    assert stat.S_IMODE(store.key_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store._dir.stat().st_mode) == 0o700


def test_set_replaces_prior(store: NebiusConnectionStore):
    store.set(key_json=_valid_key(sub="serviceaccount-1"), project_id="p1", subnet_id="s1")
    st = store.set(key_json=_valid_key(sub="serviceaccount-2"), project_id="p2", subnet_id="s2")
    assert st.service_account_id == "serviceaccount-2"
    assert st.project_id == "p2" and st.subnet_id == "s2"


def test_clear_is_idempotent(store: NebiusConnectionStore):
    store.set(key_json=_valid_key(), project_id="p", subnet_id="s")
    assert store.clear() is True
    assert store.clear() is False
    assert store.status().configured is False


def test_status_never_exposes_private_key(store: NebiusConnectionStore):
    store.set(key_json=_valid_key(), project_id="p", subnet_id="s")
    # The status dataclass has no private-key field; assert defensively that
    # none of its values carry the PEM.
    st = store.status()
    assert all("PRIVATE KEY" not in str(v) for v in vars(st).values())


@pytest.mark.parametrize(
    "bad,match",
    [
        ("not json", "not valid JSON"),
        ("{}", "subject-credentials"),
        ('{"subject-credentials": {"alg": "RS256", "kid": "k", "sub": "s"}}', "private-key"),
        ('{"subject-credentials": {"alg": "RS256", "private-key": "nope", "kid": "k", "sub": "s"}}', "PEM"),
        (
            '{"subject-credentials": {"alg": "RS256", "private-key": "-----BEGIN PRIVATE KEY-----x", '
            '"kid": "k", "iss": "a", "sub": "b"}}',
            "issuer must equal subject",
        ),
    ],
)
def test_set_rejects_malformed_key(store: NebiusConnectionStore, bad: str, match: str):
    with pytest.raises(NebiusCredentialError, match=match):
        store.set(key_json=bad, project_id="p", subnet_id="s")
    # Nothing persisted on rejection.
    assert store.status().has_key is False


@pytest.mark.parametrize("project,subnet", [("", "s"), ("p", ""), ("   ", "s")])
def test_set_requires_project_and_subnet(store: NebiusConnectionStore, project: str, subnet: str):
    with pytest.raises(NebiusCredentialError):
        store.set(key_json=_valid_key(), project_id=project, subnet_id=subnet)


def test_key_only_without_project_is_not_configured(store: NebiusConnectionStore):
    # Direct write of just the key file (no connection.json) → has_key but not
    # configured. Exercises the partial-state branch.
    store._dir.mkdir(parents=True, exist_ok=True)
    store.key_path.write_text(_valid_key())
    st = store.status()
    assert st.has_key is True
    assert st.configured is False


def test_corrupt_key_file_reads_as_no_key(store: NebiusConnectionStore):
    store._dir.mkdir(parents=True, exist_ok=True)
    store.key_path.write_text("{ this is not json")
    st = store.status()
    assert st.has_key is False
    assert st.configured is False
