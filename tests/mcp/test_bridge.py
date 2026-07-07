"""Tests for the GUI bridge (backend half + scope routing) and MCP bridge tools."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lerobot.gui.api.bridge import (
    SUPPORTED_COMMAND_TYPES,
    WILDCARD,
    _reset_registry_for_tests,
    _resolve_target,
    router,
)


@pytest.fixture(autouse=True)
def _fresh_registry(monkeypatch: pytest.MonkeyPatch):
    """Each test starts with an empty subscriber registry.

    Also relaxes the loopback gate by default — under FastAPI's TestClient
    the client host is ``"testclient"``, not ``127.0.0.1``, so the
    production-correct gate would reject every test request. The
    ``test_dispatch_loopback_gate_rejects_remote`` test re-patches it to
    return False to explicitly verify the gate works.
    """
    _reset_registry_for_tests()
    import lerobot.gui.api.bridge as bridge_mod

    monkeypatch.setattr(bridge_mod, "_is_loopback", lambda host: True)
    yield
    _reset_registry_for_tests()


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# Cap WebSocket tests so a misroute can't hang forever again.
pytestmark = pytest.mark.timeout(10)


# ── Pure helpers ──────────────────────────────────────────────────────────


def test_resolve_target_defaults_to_wildcard() -> None:
    assert _resolve_target(None, None) == WILDCARD
    assert _resolve_target("", None) == WILDCARD
    assert _resolve_target("   ", None) == WILDCARD


def test_resolve_target_cookie_wins_over_query() -> None:
    assert _resolve_target("alice", "bob") == "bob"


def test_resolve_target_query_used_when_cookie_absent() -> None:
    assert _resolve_target("alice", None) == "alice"


def test_supported_command_types_includes_expected() -> None:
    assert {"navigate", "notify", "highlight", "filter"} <= SUPPORTED_COMMAND_TYPES


# ── Dispatch endpoint (loopback gate + validation) ────────────────────────


def test_dispatch_rejects_unknown_type(client: TestClient) -> None:
    r = client.post(
        "/api/bridge/_dispatch",
        json={"client_id": "alice", "type": "wizard", "params": {}},
    )
    assert r.status_code == 400


def test_dispatch_loopback_accepted_with_no_subscribers(client: TestClient) -> None:
    # TestClient appears as 127.0.0.1 to the request; loopback gate passes.
    r = client.post(
        "/api/bridge/_dispatch",
        json={"client_id": "alice", "type": "navigate", "params": {"view": "home"}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body == {"delivered": 0, "type": "navigate", "client_id": "alice"}


def test_dispatch_validates_payload_schema(client: TestClient) -> None:
    r = client.post("/api/bridge/_dispatch", json={"type": "navigate"})  # missing client_id
    assert r.status_code == 422


def test_dispatch_loopback_gate_rejects_remote(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Force the loopback gate to look non-loopback and expect 403.

    The autouse fixture relaxes the gate to True; this test re-patches
    it back to False to verify the production-correct behavior.
    """
    import lerobot.gui.api.bridge as bridge_mod

    monkeypatch.setattr(bridge_mod, "_is_loopback", lambda host: False)
    r = client.post(
        "/api/bridge/_dispatch",
        json={"client_id": "alice", "type": "navigate", "params": {}},
    )
    assert r.status_code == 403


def test_dispatch_rejects_proxy_forwarded_for(client: TestClient) -> None:
    """If the request carries X-Forwarded-For, the real client is the proxy,
    not the loopback caller. Reject outright — otherwise an admin who fronts
    the GUI with nginx silently opens /_dispatch to the LAN.
    """
    r = client.post(
        "/api/bridge/_dispatch",
        json={"client_id": "alice", "type": "navigate", "params": {}},
        headers={"X-Forwarded-For": "203.0.113.5"},
    )
    assert r.status_code == 403


def test_dispatch_rejects_proxy_forwarded(client: TestClient) -> None:
    """The RFC-7239 `Forwarded` header gets the same treatment."""
    r = client.post(
        "/api/bridge/_dispatch",
        json={"client_id": "alice", "type": "navigate", "params": {}},
        headers={"Forwarded": "for=203.0.113.5;proto=https"},
    )
    assert r.status_code == 403


# ── WebSocket: hello, scoping, fan-out ────────────────────────────────────


def test_ws_sends_hello_on_connect(client: TestClient) -> None:
    with client.websocket_connect("/api/bridge/ws") as ws:
        msg = json.loads(ws.receive_text())
        assert msg == {"type": "hello", "target": WILDCARD}


def test_ws_target_from_query_param(client: TestClient) -> None:
    with client.websocket_connect("/api/bridge/ws?as=alice-laptop") as ws:
        msg = json.loads(ws.receive_text())
        assert msg == {"type": "hello", "target": "alice-laptop"}


def test_dispatch_reaches_wildcard_subscriber(client: TestClient) -> None:
    with client.websocket_connect("/api/bridge/ws") as ws:
        json.loads(ws.receive_text())  # consume hello
        r = client.post(
            "/api/bridge/_dispatch",
            json={
                "client_id": "alice-laptop",
                "type": "navigate",
                "params": {"view": "episode", "params": {"repo_id": "x/y", "episode_id": 7}},
            },
        )
        assert r.status_code == 200
        assert r.json()["delivered"] == 1
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "navigate"
        assert msg["client_id"] == "alice-laptop"
        assert msg["params"]["params"]["episode_id"] == 7


def test_dispatch_reaches_only_matching_target(client: TestClient) -> None:
    """A tab subscribed as alice receives alice's commands; bob's tab does not."""
    with (
        client.websocket_connect("/api/bridge/ws?as=alice") as ws_alice,
        client.websocket_connect("/api/bridge/ws?as=bob") as ws_bob,
    ):
        json.loads(ws_alice.receive_text())
        json.loads(ws_bob.receive_text())

        r = client.post(
            "/api/bridge/_dispatch",
            json={"client_id": "alice", "type": "navigate", "params": {"view": "home"}},
        )
        assert r.status_code == 200
        assert r.json()["delivered"] == 1

        # Alice receives.
        alice_msg = json.loads(ws_alice.receive_text())
        assert alice_msg["type"] == "navigate"
        assert alice_msg["client_id"] == "alice"

        # Bob has no message pending — verify by sending a second
        # dispatch addressed to bob and confirming THAT arrives next
        # (proving no alice command was queued in front of it).
        r2 = client.post(
            "/api/bridge/_dispatch",
            json={"client_id": "bob", "type": "navigate", "params": {"view": "home"}},
        )
        assert r2.json()["delivered"] == 1
        bob_msg = json.loads(ws_bob.receive_text())
        assert bob_msg["client_id"] == "bob"


def test_wildcard_subscriber_receives_every_client_id(client: TestClient) -> None:
    with client.websocket_connect("/api/bridge/ws") as ws:  # default wildcard
        json.loads(ws.receive_text())
        client.post("/api/bridge/_dispatch", json={"client_id": "a", "type": "navigate", "params": {}})
        client.post(
            "/api/bridge/_dispatch", json={"client_id": "b", "type": "notify", "params": {"title": "x"}}
        )
        m1 = json.loads(ws.receive_text())
        m2 = json.loads(ws.receive_text())
        assert {m1["client_id"], m2["client_id"]} == {"a", "b"}


def test_disconnect_removes_subscriber(client: TestClient) -> None:
    with client.websocket_connect("/api/bridge/ws") as ws:
        json.loads(ws.receive_text())
    # After context exit, the registry should be empty — next dispatch delivers 0.
    r = client.post(
        "/api/bridge/_dispatch",
        json={"client_id": "x", "type": "navigate", "params": {}},
    )
    assert r.status_code == 200
    assert r.json()["delivered"] == 0


# ── MCP bridge tools ──────────────────────────────────────────────────────


def _start_dispatch_capture_server() -> tuple[str, list[dict], threading.Event]:
    """Spin up a tiny HTTP server that records POSTs to /api/bridge/_dispatch.

    Used to verify the MCP bridge tools format their requests correctly
    without depending on the real GUI process being up.
    """
    from http.server import BaseHTTPRequestHandler, HTTPServer

    captured: list[dict] = []
    stop = threading.Event()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *_a, **_kw):  # silence
            pass

        def do_POST(self):  # noqa: N802 — stdlib
            n = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(n).decode("utf-8")
            captured.append(json.loads(body))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"delivered": 1, "type": "test", "client_id": "test"}).encode())

    # Pick a free port.
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    srv = HTTPServer(("127.0.0.1", port), H)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()

    base_url = f"http://127.0.0.1:{port}"
    # Wait briefly for the listener.
    for _ in range(20):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.02)

    def shutdown():
        srv.shutdown()
        srv.server_close()
        stop.set()

    return base_url, captured, threading.Thread(target=shutdown)  # type: ignore[return-value]


def test_bridge_tools_post_expected_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    """Capture what the MCP tools POST when called.

    Verifies the HTTP fallback path's envelope shape — the in-process path
    is disabled here because the test fixture has no real WebSocket
    subscribers, only an HTTP capture server.
    """
    from lerobot.mcp import bridge_tools
    from lerobot.mcp.bridge_tools import configure_bridge

    async def _no_inproc(*a, **k):
        return None

    monkeypatch.setattr(bridge_tools, "_try_inproc_dispatch", _no_inproc)

    base_url, captured, _ = _start_dispatch_capture_server()
    configure_bridge(base_url)

    # Build a FastMCP and grab its tools, then call them via call_tool.
    from lerobot.mcp.server import build_server

    mcp = build_server()

    async def run():
        # navigate_to
        await mcp.call_tool(
            "navigate_to",
            {"view": "episode", "params": {"repo_id": "thewisp/x", "episode_id": 5}},
        )
        # notify_user
        await mcp.call_tool(
            "notify_user",
            {"title": "Done", "body": "10/10 episodes", "deeplink": "#/home", "level": "success"},
        )
        # highlight_in_viewer
        await mcp.call_tool(
            "highlight_in_viewer",
            {"repo_id": "thewisp/x", "episode_ids": [1, 2, 3]},
        )

    asyncio.run(run())
    configure_bridge(None)  # leave the global state clean

    # set_filter is intentionally NOT registered as a tool yet — no viewer
    # has a real filter input. The "filter" command type stays in the wire
    # contract for when the GUI side lands.
    assert len(captured) == 3
    by_type = {c["type"]: c for c in captured}
    assert set(by_type) == {"navigate", "notify", "highlight"}

    nav = by_type["navigate"]
    assert nav["params"]["view"] == "episode"
    assert nav["params"]["params"] == {"repo_id": "thewisp/x", "episode_id": 5}
    # Stdio mode → fixed sentinel client_id.
    assert nav["client_id"] == "stdio-local"

    notify = by_type["notify"]
    assert notify["params"]["title"] == "Done"
    assert notify["params"]["deeplink"] == "#/home"

    highlight = by_type["highlight"]
    assert highlight["params"]["episode_ids"] == [1, 2, 3]


def test_bridge_tools_when_disabled_return_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """No --gui-url configured → tools succeed but report delivered=0.

    Targets the HTTP fallback's 'bridge disabled' branch; in-process is
    bypassed for this contract test.
    """
    from lerobot.mcp import bridge_tools
    from lerobot.mcp.bridge_tools import configure_bridge
    from lerobot.mcp.server import build_server

    async def _no_inproc(*a, **k):
        return None

    monkeypatch.setattr(bridge_tools, "_try_inproc_dispatch", _no_inproc)

    configure_bridge(None)
    mcp = build_server()

    async def run():
        return await mcp.call_tool("navigate_to", {"view": "home", "params": {}})

    _, structured = asyncio.run(run())
    assert structured["delivered"] == 0
    assert "bridge disabled" in structured["reason"]


def test_bridge_tools_when_gui_unreachable_return_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pointed at an unreachable URL → delivered=0 with clear reason."""
    from lerobot.mcp import bridge_tools
    from lerobot.mcp.bridge_tools import configure_bridge
    from lerobot.mcp.server import build_server

    async def _no_inproc(*a, **k):
        return None

    monkeypatch.setattr(bridge_tools, "_try_inproc_dispatch", _no_inproc)

    # 127.0.0.1:1 is reserved; no listener.
    configure_bridge("http://127.0.0.1:1")
    mcp = build_server()

    async def run():
        return await mcp.call_tool("navigate_to", {"view": "home", "params": {}})

    _, structured = asyncio.run(run())
    configure_bridge(None)

    assert structured["delivered"] == 0
    assert "reason" in structured


def test_bridge_tools_inproc_dispatch_when_unified() -> None:
    """In-process path: tool body should ``await registry.dispatch`` directly
    instead of POSTing over HTTP when the GUI module is importable. This is
    the regression for the demo-caught bug where unified-mode HTTP self-loop
    deadlocked the event loop.
    """
    from lerobot.mcp.bridge_tools import _try_inproc_dispatch

    async def run():
        # No subscribers in the test, but the in-process path still resolves
        # the registry and returns delivered=0 cleanly — never falls to HTTP.
        result = await _try_inproc_dispatch("navigate", {"view": "home", "params": {}})
        return result

    result = asyncio.run(run())
    assert result is not None  # in-process resolved
    assert result["delivered"] == 0
    assert result["type"] == "navigate"


# ── Frontend deeplink contract (verified by string match on bridge.js) ───
#
# bridge.js exposes deepLinkFor / parseHash on window.lerobotBridge. We
# don't run JS here, but we lock in the path conventions so the backend
# tests stay aligned with the URLs the frontend produces / parses.


def test_bridge_js_advertises_recognized_views() -> None:
    """Spot-check that the JS recognizes the views the MCP tools emit."""
    from pathlib import Path

    bridge_js = Path(__file__).resolve().parents[2] / "src" / "lerobot" / "gui" / "static" / "bridge.js"
    body = bridge_js.read_text()
    # Each recognized view name must appear in deepLinkFor's branches.
    for view in ("episode", "dataset", "run", "home"):
        assert f'view === "{view}"' in body, f"deepLinkFor missing branch for {view!r}"
    # And the hash-route plumbing functions:
    assert "parseHash" in body
    assert "applyRoute" in body
    # The WebSocket lifecycle wiring is present.
    assert "/api/bridge/ws" in body
    # Notifications are wired (the refocus-on-click pattern).
    assert "Notification" in body
    assert "window.focus" in body


def _unused_compat(*_args: Any) -> None:
    """Placeholder to keep ``Any`` import used."""
    return None
