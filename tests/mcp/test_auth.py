"""Tests for the bearer-token store and scope enforcement."""

from __future__ import annotations

from pathlib import Path

import pytest

from lerobot.mcp.auth import (
    ALL_SCOPES,
    SCOPE_COMMENT,
    SCOPE_OPERATE,
    SCOPE_READ,
    LeRobotTokenVerifier,
    TokenStore,
    requires_scope,
)


@pytest.fixture
def store(tmp_path: Path) -> TokenStore:
    return TokenStore(tmp_path / "tokens.sqlite")


# ── TokenStore ────────────────────────────────────────────────────────────


def test_issue_returns_prefixed_token(store: TokenStore) -> None:
    token = store.issue("alice-laptop", [SCOPE_READ, SCOPE_COMMENT])
    assert token.startswith("sk-lr-")
    assert len(token) > len("sk-lr-") + 20


def test_lookup_returns_name_and_scopes(store: TokenStore) -> None:
    token = store.issue("alice-laptop", [SCOPE_READ, SCOPE_COMMENT])
    info = store.lookup(token)
    assert info == {"name": "alice-laptop", "scopes": [SCOPE_COMMENT, SCOPE_READ]}


def test_lookup_unknown_token_returns_none(store: TokenStore) -> None:
    assert store.lookup("sk-lr-bogus") is None


def test_lookup_revoked_token_returns_none(store: TokenStore) -> None:
    token = store.issue("alice", [SCOPE_READ])
    assert store.revoke("alice") is True
    assert store.lookup(token) is None


def test_revoke_idempotent_after_first(store: TokenStore) -> None:
    store.issue("alice", [SCOPE_READ])
    assert store.revoke("alice") is True
    assert store.revoke("alice") is False  # already revoked → no row to update


def test_revoke_unknown_returns_false(store: TokenStore) -> None:
    assert store.revoke("nope") is False


def test_duplicate_active_name_rejected(store: TokenStore) -> None:
    store.issue("alice", [SCOPE_READ])
    with pytest.raises(ValueError, match="active token"):
        store.issue("alice", [SCOPE_READ])


def test_reissue_after_revoke_succeeds(store: TokenStore) -> None:
    # Revoking frees the name: reissuing the same name is a normal rotation
    # and must produce a fresh, working token (regression — a revoked row used
    # to keep the name reserved via UNIQUE(name)).
    first = store.issue("alice-laptop", [SCOPE_READ])
    assert store.revoke("alice-laptop") is True
    second = store.issue("alice-laptop", [SCOPE_READ])
    assert second != first
    assert store.lookup(first) is None  # old one stays revoked
    assert store.lookup(second)["name"] == "alice-laptop"  # new one works


def test_invalid_scope_rejected(store: TokenStore) -> None:
    with pytest.raises(ValueError, match="Unknown scope"):
        store.issue("alice", ["wizard"])


def test_list_tokens_default_excludes_revoked(store: TokenStore) -> None:
    store.issue("alice", [SCOPE_READ])
    store.issue("bob", [SCOPE_READ, SCOPE_COMMENT])
    store.revoke("alice")
    active = store.list_tokens()
    assert [r["name"] for r in active] == ["bob"]


def test_list_tokens_include_revoked(store: TokenStore) -> None:
    store.issue("alice", [SCOPE_READ])
    store.issue("bob", [SCOPE_READ])
    store.revoke("alice")
    all_tokens = store.list_tokens(include_revoked=True)
    assert {r["name"] for r in all_tokens} == {"alice", "bob"}


def test_lookup_updates_last_used_at(store: TokenStore) -> None:
    token = store.issue("alice", [SCOPE_READ])
    assert store.list_tokens()[0]["last_used_at"] is None
    store.lookup(token)
    assert store.list_tokens()[0]["last_used_at"] is not None


def test_scope_set_normalized_sorted_unique(store: TokenStore) -> None:
    token = store.issue("alice", [SCOPE_OPERATE, SCOPE_READ, SCOPE_READ])
    info = store.lookup(token)
    assert info is not None and info["scopes"] == sorted({SCOPE_OPERATE, SCOPE_READ})


def test_persistence_across_instances(tmp_path: Path) -> None:
    db = tmp_path / "tokens.sqlite"
    s1 = TokenStore(db)
    token = s1.issue("alice", [SCOPE_READ])
    s2 = TokenStore(db)
    assert s2.lookup(token) == {"name": "alice", "scopes": [SCOPE_READ]}


# ── LeRobotTokenVerifier ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_verifier_returns_access_token_with_scopes(store: TokenStore) -> None:
    token = store.issue("alice", [SCOPE_READ, SCOPE_COMMENT])
    verifier = LeRobotTokenVerifier(store)
    access = await verifier.verify_token(token)
    assert access is not None
    assert access.client_id == "alice"
    assert sorted(access.scopes) == [SCOPE_COMMENT, SCOPE_READ]


@pytest.mark.asyncio
async def test_verifier_returns_none_for_unknown(store: TokenStore) -> None:
    verifier = LeRobotTokenVerifier(store)
    assert await verifier.verify_token("sk-lr-bogus") is None


@pytest.mark.asyncio
async def test_verifier_returns_none_for_revoked(store: TokenStore) -> None:
    token = store.issue("alice", [SCOPE_READ])
    store.revoke("alice")
    verifier = LeRobotTokenVerifier(store)
    assert await verifier.verify_token(token) is None


# ── requires_scope decorator ──────────────────────────────────────────────


def test_requires_scope_noop_in_stdio_mode() -> None:
    """No auth context (stdio): decorator must not raise."""

    @requires_scope(SCOPE_OPERATE)
    def operate_tool(x: int) -> int:
        return x * 2

    assert operate_tool(3) == 6


def test_requires_scope_unknown_scope_rejected_at_decoration_time() -> None:
    with pytest.raises(AssertionError):

        @requires_scope("wizard")
        def t() -> None:
            return None


def test_requires_scope_with_token_having_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    """If a token is present and has the required scope, tool runs."""
    from mcp.server.auth.provider import AccessToken

    import lerobot.mcp.auth as auth_mod

    monkeypatch.setattr(
        auth_mod,
        "get_access_token",
        lambda: AccessToken(token="x", client_id="alice", scopes=[SCOPE_READ, SCOPE_COMMENT]),
    )

    @requires_scope(SCOPE_COMMENT)
    def annotate_tool() -> str:
        return "ok"

    assert annotate_tool() == "ok"


def test_requires_scope_with_token_missing_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    """If a token is present but lacks the scope, tool raises PermissionError."""
    from mcp.server.auth.provider import AccessToken

    import lerobot.mcp.auth as auth_mod

    monkeypatch.setattr(
        auth_mod,
        "get_access_token",
        lambda: AccessToken(token="x", client_id="alice", scopes=[SCOPE_READ]),
    )

    @requires_scope(SCOPE_COMMENT)
    def annotate_tool() -> str:
        return "ok"

    with pytest.raises(PermissionError, match="scope 'comment'"):
        annotate_tool()


# ── Regression: default_token_store_path single-source-of-truth ───────────


def test_default_token_store_path_honors_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """All three call sites (ai_setup, _mount_mcp, cli) must agree on the
    path under LEROBOT_MCP_TOKEN_STORE — otherwise a token issued from the
    GUI page is silently invisible to the MCP server.
    """
    from lerobot.mcp.auth import default_token_store_path

    override = tmp_path / "custom_tokens.sqlite"
    monkeypatch.setenv("LEROBOT_MCP_TOKEN_STORE", str(override))
    assert default_token_store_path() == override


def test_default_token_store_path_falls_back_to_hf_home(monkeypatch: pytest.MonkeyPatch) -> None:
    from lerobot.mcp.auth import default_token_store_path
    from lerobot.utils.constants import HF_LEROBOT_HOME

    monkeypatch.delenv("LEROBOT_MCP_TOKEN_STORE", raising=False)
    assert default_token_store_path() == HF_LEROBOT_HOME / "_mcp_tokens.sqlite"


# ── Regression: PRAGMA scope across threads ───────────────────────────────


def test_pragma_synchronous_normal_on_worker_thread(store: TokenStore) -> None:
    """`synchronous=NORMAL` is per-connection — every per-thread connection
    must apply it, not just the constructing thread's.
    """
    import threading

    from_threads: dict[str, int] = {}

    def collect(label: str) -> None:
        # Touch the store from this thread to instantiate its cached connection,
        # then read back the sqlite PRAGMA values for it.
        store.list_tokens()
        conn = store._conn()
        from_threads[label] = conn.execute("PRAGMA synchronous").fetchone()[0]

    collect("main")
    t = threading.Thread(target=collect, args=("worker",))
    t.start()
    t.join()
    # PRAGMA synchronous=NORMAL maps to integer 1. PRAGMA journal_mode=WAL is
    # persistent on the DB file so it survives; synchronous is per-connection.
    assert from_threads["main"] == 1
    assert from_threads["worker"] == 1, (
        "Worker-thread connection silently fell back to synchronous=FULL — "
        "PRAGMA must be applied in _conn() per fresh connection, not only in __init__."
    )


# ── Regression: lookup debounces last_used_at ──────────────────────────────


def test_lookup_debounces_last_used_at(store: TokenStore) -> None:
    """Two lookups in quick succession must NOT write the SQLite row twice —
    that would 2x SQLite write contention on every MCP request.
    """
    token = store.issue("alice", [SCOPE_READ])
    store.lookup(token)
    [row1] = store.list_tokens()
    first_seen = row1["last_used_at"]
    assert first_seen is not None

    # Second lookup, milliseconds later — debounce window is 60s.
    store.lookup(token)
    [row2] = store.list_tokens()
    assert row2["last_used_at"] == first_seen, (
        "last_used_at was bumped within the debounce window — debounce broken"
    )


# ── Regression: legacy scope alias keeps old tokens working ───────────────


def test_legacy_annotate_scope_canonicalizes_to_comment(store: TokenStore) -> None:
    """Tokens issued via the pre-rename codepath (scope 'annotate') still work
    after the 4-tier rename — they look like 'comment' on read.
    """
    # Simulate a token issued before the rename by writing directly through
    # the underlying SQL with the legacy scope name.
    import json

    from lerobot.mcp.auth import _hash, _now_iso

    # Synthetic non-bearer string — never matches the issued-token prefix
    # so it's not mistaken for a real credential by gitleaks-style scans.
    legacy_token = "PYTEST_FIXTURE_legacy_bearer"  # noqa: S105
    store._conn().execute(
        "INSERT INTO tokens(token_hash, name, scopes_json, created_at) VALUES (?, ?, ?, ?)",
        (_hash(legacy_token), "alice-legacy", json.dumps(["read", "annotate"]), _now_iso()),
    )

    info = store.lookup(legacy_token)
    assert info is not None
    # `annotate` was canonicalized to `comment` on read
    assert "annotate" not in info["scopes"]
    assert SCOPE_COMMENT in info["scopes"]
    assert SCOPE_READ in info["scopes"]


def test_legacy_alias_applies_to_list_tokens(store: TokenStore) -> None:
    """list_tokens() also canonicalizes — the GUI's /ai_setup listing must
    show 'comment' for legacy tokens, not 'annotate'.
    """
    import json

    from lerobot.mcp.auth import _hash, _now_iso

    store._conn().execute(
        "INSERT INTO tokens(token_hash, name, scopes_json, created_at) VALUES (?, ?, ?, ?)",
        (_hash("sk-lr-x"), "legacy", json.dumps(["annotate"]), _now_iso()),
    )
    [row] = [r for r in store.list_tokens() if r["name"] == "legacy"]
    assert row["scopes"] == [SCOPE_COMMENT]


def test_validate_scopes_accepts_legacy_input_normalizes_to_canonical(store: TokenStore) -> None:
    """A caller that asks to issue with ['read', 'annotate'] gets a token
    stored with ['comment', 'read'] — silent transition, no error.
    """
    token = store.issue("transitional", ["read", "annotate"])
    info = store.lookup(token)
    assert info is not None
    assert sorted(info["scopes"]) == [SCOPE_COMMENT, SCOPE_READ]


# ── Smoke ──────────────────────────────────────────────────────────────────


def test_all_scopes_constant_consistent() -> None:
    from lerobot.mcp.auth import SCOPE_EDIT

    assert set(ALL_SCOPES) == {SCOPE_READ, SCOPE_COMMENT, SCOPE_EDIT, SCOPE_OPERATE}
