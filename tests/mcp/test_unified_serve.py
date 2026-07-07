"""Pins the unified-process deployment: the GUI app, when started via
``run_server``, also exposes the MCP HTTP transport under ``/mcp``.

Doesn't actually start a uvicorn server — exercises ``_mount_mcp`` in
isolation and asserts the resulting app has the expected routes + that
``/mcp`` enforces bearer-token auth.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def unified_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A unified GUI+MCP app pointed at a tmp token store.

    Reuses the real ``_mount_mcp`` so we test the actual wiring, not a
    rebuilt-just-for-the-test app. Overrides the token-store location via
    ``HF_LEROBOT_HOME`` so the test never touches the user's real one.
    """
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))

    # Re-import constants module so HF_LEROBOT_HOME picks up the env override.
    import importlib

    import lerobot.utils.constants as constants_mod

    importlib.reload(constants_mod)
    # Reload the GUI server module too — it imports HF_LEROBOT_HOME from inside
    # _mount_mcp (lazy import), so no module reload needed for that path; but
    # we DO want a fresh `app` instance per test so routes don't accumulate.
    import lerobot.gui.server as gui_server_mod

    importlib.reload(gui_server_mod)

    # Call _mount_mcp once; subsequent tests get fresh module via reload.
    gui_server_mod._mount_mcp(host="127.0.0.1", port=8000)
    # MUST be used as a context manager so FastAPI's startup hook fires —
    # that's where we enter the MCP session-manager async context. Without
    # it, /mcp requests hit "Task group is not initialized".
    # base_url="http://127.0.0.1:8000" sets the Host header to something
    # FastMCP's transport-security check accepts (it rejects "testserver",
    # which is TestClient's default).
    with TestClient(gui_server_mod.app, base_url="http://127.0.0.1:8000") as c:
        yield c


def _flatten_route_paths(routes) -> set[str]:
    """Collect route paths recursively.

    FastAPI >= 0.139 represents ``include_router`` as a nested ``_IncludedRouter``
    entry (whose ``original_router`` holds the actual routes) instead of
    flattening into ``app.routes``, so a top-level scan alone no longer sees
    the included endpoints.
    """
    paths = set()
    for r in routes:
        p = getattr(r, "path", None)
        if p is not None:
            paths.add(p)
        paths |= _flatten_route_paths(getattr(r, "routes", ()) or ())
        original = getattr(r, "original_router", None)
        if original is not None:
            paths |= _flatten_route_paths(getattr(original, "routes", ()) or ())
    return paths


def test_unified_app_exposes_both_gui_and_mcp(unified_client: TestClient) -> None:
    """All the GUI routes are present AND /mcp is now mounted."""
    import lerobot.gui.server as gui_server_mod

    paths = _flatten_route_paths(gui_server_mod.app.routes)
    # A few GUI routes that we know exist
    assert "/" in paths
    assert "/ai_setup" in paths
    assert "/api/bridge/_dispatch" in paths
    # The MCP mount appears as a Mount with path="/mcp"
    mount_paths = {
        getattr(r, "path", None) for r in gui_server_mod.app.routes if r.__class__.__name__ == "Mount"
    }
    assert "/mcp" in mount_paths


def test_mcp_endpoint_rejects_unauthenticated(unified_client: TestClient) -> None:
    """/mcp on the unified app demands a bearer (same as standalone MCP)."""
    # initialize is the first call any MCP client makes
    r = unified_client.post(
        "/mcp",
        headers={"Accept": "application/json, text/event-stream", "Content-Type": "application/json"},
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"},
            },
        },
    )
    assert r.status_code == 401, f"expected 401 unauthenticated, got {r.status_code}: {r.text[:200]}"


def test_mcp_endpoint_accepts_issued_bearer(unified_client: TestClient, tmp_path: Path) -> None:
    """A token issued into the same store unlocks /mcp."""
    from lerobot.mcp.auth import SCOPE_COMMENT, SCOPE_READ, TokenStore

    store = TokenStore(tmp_path / "_mcp_tokens.sqlite")
    token = store.issue("test-laptop", [SCOPE_READ, SCOPE_COMMENT])

    r = unified_client.post(
        "/mcp",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        },
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"},
            },
        },
    )
    assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text[:200]}"
    # The initialize response is an SSE stream; look for the JSON-RPC result
    body = r.text
    assert '"jsonrpc":"2.0"' in body
    assert '"protocolVersion"' in body
    assert '"serverInfo"' in body


def test_bridge_dispatch_still_loopback(unified_client: TestClient) -> None:
    """Mounting MCP didn't break the bridge dispatch endpoint."""
    import lerobot.gui.api.bridge as bridge_mod

    # TestClient host is "testclient"; relax the loopback gate as the bridge
    # tests do.
    original = bridge_mod._is_loopback
    bridge_mod._is_loopback = lambda host: True  # type: ignore[assignment]
    try:
        r = unified_client.post(
            "/api/bridge/_dispatch",
            json={"client_id": "x", "type": "navigate", "params": {"view": "home"}},
        )
        assert r.status_code == 200
        assert r.json()["delivered"] == 0  # no subscribers
    finally:
        bridge_mod._is_loopback = original
