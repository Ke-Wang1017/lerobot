# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the server-held Weights & Biases connection store.

All tests write to a tmp dir (never ``~/.config``) and clear the ambient
``WANDB_API_KEY`` / ``~/.netrc`` sources, so a developer who happens to be
logged into wandb doesn't flip an assertion. Nothing here touches the network:
:func:`verify_api_key` is exercised against a stubbed ``httpx``.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from lerobot.gui.training import wandb_credentials as wc
from lerobot.gui.training.wandb_credentials import (
    SOURCE_ENV,
    SOURCE_NETRC,
    SOURCE_STORED,
    WandbConnectionStore,
    WandbCredentialError,
    mask_key,
    resolve_api_key,
    validate_api_key,
    wandb_enabled,
    wandb_env_for_args,
)

_KEY = "0123456789abcdef0123456789abcdef01234567"  # 40 chars, wandb's own shape
_OTHER_KEY = "fedcba9876543210fedcba9876543210fedcba98"


@pytest.fixture(autouse=True)
def isolated_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No ambient key from any of the three sources unless a test adds one."""
    monkeypatch.setattr(wc, "WANDB_DIR", tmp_path / "wandb")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(wc, "_netrc_key", lambda: None)


# ── Key validation ────────────────────────────────────────────────────────────


def test_validate_strips_surrounding_whitespace() -> None:
    assert validate_api_key(f"  {_KEY}\n") == _KEY


def test_validate_accepts_the_new_wandb_v1_key_format() -> None:
    """wandb.ai now issues ``wandb_v1_<base62>`` (~86 chars) alongside the
    legacy 40-char hex key. Both must pass; underscores are significant."""
    new_style = "wandb_v1_" + "aB3" * 25
    assert validate_api_key(new_style) == new_style


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "   ",
        "short",
        f"wandb login {_KEY}",  # pasted the whole command
        f"{_KEY} {_KEY}",  # two keys / stray text
    ],
)
def test_validate_rejects_malformed_pastes(bad: str) -> None:
    with pytest.raises(WandbCredentialError):
        validate_api_key(bad)


def test_mask_key_reveals_only_the_tail() -> None:
    masked = mask_key(_KEY)
    assert masked == "…4567"
    assert _KEY[:-4] not in masked
    assert mask_key(None) is None


# ── Store ─────────────────────────────────────────────────────────────────────


def test_set_writes_key_0600_under_0700_dir(tmp_path: Path) -> None:
    store = WandbConnectionStore()
    store.set(api_key=_KEY, entity="ada")
    assert store.read_key() == _KEY
    assert stat.S_IMODE(store.key_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.key_path.parent.stat().st_mode) == 0o700


def test_status_reports_stored_source_and_entity() -> None:
    store = WandbConnectionStore()
    store.set(api_key=_KEY, entity="ada")
    st = store.status()
    assert (st.configured, st.source, st.entity) == (True, SOURCE_STORED, "ada")
    assert st.masked_key == "…4567"


def test_replacing_the_key_drops_a_stale_entity() -> None:
    """A new key may belong to a different account; showing the old owner's
    name next to it would be a lie."""
    store = WandbConnectionStore()
    store.set(api_key=_KEY, entity="ada")
    store.set(api_key=_OTHER_KEY)  # verification came back unknown (offline)
    assert store.status().entity is None
    assert store.read_key() == _OTHER_KEY


def test_clear_is_idempotent_and_reports_whether_it_removed_anything() -> None:
    store = WandbConnectionStore()
    store.set(api_key=_KEY, entity="ada")
    assert store.clear() is True
    assert store.clear() is False
    assert store.status().configured is False


# ── Resolution order ──────────────────────────────────────────────────────────


def test_resolve_prefers_stored_over_env_and_netrc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", _OTHER_KEY)
    monkeypatch.setattr(wc, "_netrc_key", lambda: "netrckey")
    WandbConnectionStore().set(api_key=_KEY)
    assert resolve_api_key() == (_KEY, SOURCE_STORED)


def test_resolve_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", f" {_KEY} ")
    monkeypatch.setattr(wc, "_netrc_key", lambda: "netrckey")
    assert resolve_api_key() == (_KEY, SOURCE_ENV)


def test_resolve_falls_back_to_netrc() -> None:
    """`wandb login` on the server is a valid way to be connected — the user
    should never be asked to paste a key they already gave wandb."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(wc, "_netrc_key", lambda: _KEY)
        assert resolve_api_key() == (_KEY, SOURCE_NETRC)


def test_clearing_stored_key_falls_back_to_ambient(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disconnect removes OUR key, not the user's shell env — the UI relies on
    the post-clear status to avoid claiming a run won't be tracked."""
    monkeypatch.setenv("WANDB_API_KEY", _OTHER_KEY)
    store = WandbConnectionStore()
    store.set(api_key=_KEY)
    store.clear()
    st = store.status()
    assert (st.configured, st.source) == (True, SOURCE_ENV)


def test_resolve_returns_none_when_nothing_is_configured() -> None:
    assert resolve_api_key() == (None, None)


# ── Run-args integration ──────────────────────────────────────────────────────


@pytest.mark.parametrize("value", [True, "true", "True", "1", "yes"])
def test_wandb_enabled_accepts_bool_and_string_forms(value) -> None:
    assert wandb_enabled({"wandb.enable": value}) is True


@pytest.mark.parametrize("args", [{}, {"wandb.enable": False}, {"wandb.enable": "false"}])
def test_wandb_not_enabled(args: dict) -> None:
    assert wandb_enabled(args) is False


def test_env_carries_the_key_only_for_a_tracked_run() -> None:
    """An untracked run must not carry the key into its container."""
    WandbConnectionStore().set(api_key=_KEY)
    assert wandb_env_for_args({"wandb.enable": True}) == {"WANDB_API_KEY": _KEY}
    assert wandb_env_for_args({"policy.type": "act"}) == {}


def test_env_is_empty_when_tracking_is_on_but_no_key_resolves() -> None:
    """The API rejects this at submit; if it happens anyway, the trainer should
    fail on wandb's own auth error rather than us inventing a key."""
    assert wandb_env_for_args({"wandb.enable": True}) == {}


# ── Verification (stubbed httpx) ──────────────────────────────────────────────


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _stub_httpx(monkeypatch: pytest.MonkeyPatch, response=None, raises: Exception | None = None) -> None:
    import httpx

    def fake_post(*_args, **_kwargs):
        if raises is not None:
            raise raises
        return response

    monkeypatch.setattr(httpx, "post", fake_post)


def test_verify_returns_the_entity(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_httpx(monkeypatch, _FakeResponse(200, {"data": {"viewer": {"username": "ada", "entity": "acme"}}}))
    assert wc.verify_api_key(_KEY) == "acme"


def test_verify_falls_back_to_username(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_httpx(monkeypatch, _FakeResponse(200, {"data": {"viewer": {"username": "ada", "entity": None}}}))
    assert wc.verify_api_key(_KEY) == "ada"


@pytest.mark.parametrize("status", [401, 403])
def test_verify_raises_on_rejected_key(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    _stub_httpx(monkeypatch, _FakeResponse(status))
    with pytest.raises(WandbCredentialError):
        wc.verify_api_key(_KEY)


def test_verify_raises_on_null_viewer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Some deployments answer a bad key with 200 + nobody home."""
    _stub_httpx(monkeypatch, _FakeResponse(200, {"data": {"viewer": None}}))
    with pytest.raises(WandbCredentialError):
        wc.verify_api_key(_KEY)


def test_verify_is_inconclusive_when_wandb_is_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Offline must not block saving a key — the caller warns instead."""
    import httpx

    _stub_httpx(monkeypatch, raises=httpx.ConnectError("no route to host"))
    assert wc.verify_api_key(_KEY) is None


def test_verify_is_inconclusive_on_unexpected_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_httpx(monkeypatch, _FakeResponse(200, {"errors": [{"message": "???"}]}))
    assert wc.verify_api_key(_KEY) is None
