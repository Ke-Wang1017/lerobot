"""Guardrail tests for the /ai_setup GUI route.

Modest coverage: the page loads, posting a form issues a token (one-time
display), the bearer doesn't subsequently appear in the listing, duplicate
names get a 409, and revoking removes a token. Anything beyond this is
verified by manual / smoke testing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lerobot.gui.api.ai_setup import get_token_store, router
from lerobot.mcp.auth import TokenStore


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """An isolated app with the ai_setup router and a fresh token store."""
    app = FastAPI()
    app.include_router(router)
    store = TokenStore(tmp_path / "tokens.sqlite")
    app.dependency_overrides[get_token_store] = lambda: store
    return TestClient(app)


def test_index_loads_with_empty_listing(client: TestClient) -> None:
    r = client.get("/ai_setup")
    assert r.status_code == 200
    assert "No active tokens yet" in r.text
    assert 'name="name"' in r.text  # form is rendered


def test_post_creates_token_and_shows_it_once(client: TestClient) -> None:
    r = client.post("/ai_setup/tokens", data={"name": "alice-laptop", "scope": ["read", "comment"]})
    assert r.status_code == 200
    assert "Token created for" in r.text
    assert "alice-laptop" in r.text
    # The cleartext bearer must appear on this one-time page
    assert "sk-lr-" in r.text
    # Per-tool snippets must be present
    assert "claude mcp add" in r.text
    assert "Codex CLI" in r.text


def test_per_client_snippet_syntax(client: TestClient) -> None:
    """Each client's install snippet uses that tool's real flag syntax.

    Regression guard: an earlier version emitted `claude mcp add --url ... --token`,
    which Claude Code rejects (`unknown option '--url'`). The four clients differ:
    Claude Code / Gemini take a positional URL + `--header`; Codex uses `--url`
    plus a bearer *env var* (no inline token flag); Claude Desktop has no native
    HTTP entry and must be proxied via mcp-remote.
    """
    r = client.post("/ai_setup/tokens", data={"name": "dev-box", "scope": ["read"]})
    assert r.status_code == 200
    text = r.text
    # Claude Code: positional URL, bearer via --header — not the old --url/--token
    assert "claude mcp add --transport http lerobot" in text
    assert "claude mcp add lerobot" not in text
    # Gemini: positional URL + --header — not the old --auth-bearer
    assert "gemini mcp add --transport http lerobot" in text
    assert "--auth-bearer" not in text
    # Codex: --url flag + bearer pulled from an env var
    assert "--bearer-token-env-var LEROBOT_MCP_TOKEN" in text
    assert "export LEROBOT_MCP_TOKEN=" in text
    # Claude Desktop: mcp-remote stdio proxy (no native remote-HTTP entry)
    assert "mcp-remote" in text


def test_index_lists_active_token_but_not_the_bearer(client: TestClient) -> None:
    client.post("/ai_setup/tokens", data={"name": "bob", "scope": ["read"]})
    r = client.get("/ai_setup")
    assert r.status_code == 200
    assert "bob" in r.text
    # Bearer must NOT be reachable on the index — only on the one-time post-create page
    assert "sk-lr-" not in r.text


def test_duplicate_name_returns_409(client: TestClient) -> None:
    client.post("/ai_setup/tokens", data={"name": "alice", "scope": ["read"]})
    r = client.post("/ai_setup/tokens", data={"name": "alice", "scope": ["read"]})
    assert r.status_code == 409


def test_invalid_scope_returns_400(client: TestClient) -> None:
    r = client.post("/ai_setup/tokens", data={"name": "alice", "scope": ["wizard"]})
    assert r.status_code == 400


def test_revoke_removes_token_from_listing(client: TestClient) -> None:
    client.post("/ai_setup/tokens", data={"name": "carol", "scope": ["read"]})
    r = client.post("/ai_setup/tokens/carol/revoke", follow_redirects=False)
    assert r.status_code == 303
    listing = client.get("/ai_setup").text
    assert "carol" not in listing or "Revoke" not in listing  # not in active list


def test_revoke_unknown_returns_404(client: TestClient) -> None:
    r = client.post("/ai_setup/tokens/nope/revoke", follow_redirects=False)
    assert r.status_code == 404


def test_name_pattern_enforced(client: TestClient) -> None:
    r = client.post("/ai_setup/tokens", data={"name": "bad name!", "scope": ["read"]})
    # FastAPI returns 422 for Form validation failure
    assert r.status_code == 422


def test_revoke_path_pattern_enforced(client: TestClient) -> None:
    """The revoke endpoint must reject names that don't match the create
    contract — otherwise URL surface diverges from the create-side rule
    and the 404 becomes a probing oracle for arbitrary path strings.
    """
    r = client.post("/ai_setup/tokens/bad%20name%21/revoke", follow_redirects=False)
    assert r.status_code == 422


def test_operate_scope_is_selectable(client: TestClient) -> None:
    """operate is the highest tier (hardware + training start/stop). It was
    disabled in the issue form until operate-tier tools existed; the training_*
    tools shipped, so the form must now offer it as a real, non-disabled box."""
    import re

    r = client.get("/ai_setup")
    assert r.status_code == 200
    m = re.search(r'<input[^>]*value="operate"[^>]*>', r.text)
    assert m, "operate checkbox not offered"
    assert "disabled" not in m.group(0)  # selectable now, not reserved
