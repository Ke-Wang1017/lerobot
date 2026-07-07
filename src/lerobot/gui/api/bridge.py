"""GUI bridge — channel for AI-driven UI commands.

The MCP daemon (a separate process) calls bridge tools (``navigate_to``,
``notify_user``, ``highlight_in_viewer``, ``set_filter``) that need to
reach the user's open GUI tab. This module is the GUI-side half of that
bridge:

* ``POST /api/bridge/_dispatch`` — the MCP daemon hits this with a JSON
  command. We localhost-gate it (only loopback callers accepted) and
  fan it out to subscribed WebSocket clients whose declared target
  matches the command's ``client_id`` (or ``*`` wildcards).

* ``WS /api/bridge/ws?as=<client_id|*>`` — the GUI tab connects here on
  load and declares which token's commands it wants to see. ``*`` is
  the single-operator default ("show me everything"); a specific
  ``client_id`` enables per-user scoping for multi-user labs without
  any protocol change.

See the "Isolation model" section of ``src/lerobot/mcp/README.md``
(under "Design rationale") — this is its implementation.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/bridge", tags=["bridge"])

WILDCARD = "*"

# Supported command types — the frontend handler set. Adding a new
# command type requires both a new entry here and a frontend handler.
SUPPORTED_COMMAND_TYPES = frozenset({"navigate", "notify", "highlight", "filter"})


# ── Subscriber registry (in-memory; one process) ─────────────────────────


class _Subscriber:
    """One WebSocket-connected GUI tab.

    Holds the declared target (``client_id`` or ``"*"``) plus the
    WebSocket. ``send`` is a thin awaitable wrapper to keep the
    fan-out loop clean.
    """

    __slots__ = ("ws", "target", "id")

    _next_id = 0

    def __init__(self, ws: WebSocket, target: str):
        _Subscriber._next_id += 1
        self.id = _Subscriber._next_id
        self.ws = ws
        self.target = target

    async def send(self, message: dict) -> None:
        await self.ws.send_text(json.dumps(message))

    def matches(self, command_client_id: str) -> bool:
        return self.target == WILDCARD or self.target == command_client_id


class SubscriberRegistry:
    """Thread-safe in-memory subscriber registry.

    Owns the active WebSocket set. ``dispatch`` is fire-and-forget at
    the per-subscriber level: a single subscriber's send error doesn't
    block others.
    """

    def __init__(self) -> None:
        self._subs: set[_Subscriber] = set()
        self._lock = asyncio.Lock()
        # Stamped on first add() so out-of-loop callers (the in-process
        # bridge-dispatch path in mcp/bridge_tools.py) can find the loop
        # the registry's async ops run on. None until the GUI is up.
        self.event_loop: asyncio.AbstractEventLoop | None = None

    async def add(self, ws: WebSocket, target: str) -> _Subscriber:
        if self.event_loop is None:
            self.event_loop = asyncio.get_running_loop()
        sub = _Subscriber(ws, target)
        async with self._lock:
            self._subs.add(sub)
        logger.info("bridge: subscriber #%d connected (target=%s); count=%d", sub.id, target, len(self._subs))
        return sub

    async def remove(self, sub: _Subscriber) -> None:
        async with self._lock:
            self._subs.discard(sub)
        logger.info("bridge: subscriber #%d disconnected; count=%d", sub.id, len(self._subs))

    async def dispatch(self, command_client_id: str, message: dict) -> int:
        """Fan ``message`` out to subscribers whose target matches.

        Returns the count of recipients (best-effort; a subscriber that
        errored mid-send is still counted before its connection is
        dropped). The MCP daemon uses this count to tell the AI how
        many tabs received the command.
        """
        async with self._lock:
            recipients = [s for s in self._subs if s.matches(command_client_id)]
        delivered = 0
        for sub in recipients:
            try:
                await sub.send(message)
                delivered += 1
            except Exception:  # noqa: BLE001
                logger.warning("bridge: send to subscriber #%d failed; dropping", sub.id, exc_info=True)
                await self.remove(sub)
        return delivered


_REGISTRY = SubscriberRegistry()


def get_registry() -> SubscriberRegistry:
    """FastAPI-style accessor (tests can override via app.dependency_overrides)."""
    return _REGISTRY


# ── Dispatch endpoint (MCP daemon → GUI backend) ─────────────────────────


class DispatchRequest(BaseModel):
    """Wire format the MCP bridge tools POST to ``/api/bridge/_dispatch``."""

    client_id: str = Field(..., description="Originating MCP token's name; routing key.")
    type: str = Field(..., description="Command type; see SUPPORTED_COMMAND_TYPES.")
    params: dict[str, Any] = Field(default_factory=dict)


def _client_host(request: Request) -> str | None:
    if request.client is None:
        return None
    return request.client.host


def _is_loopback(host: str | None) -> bool:
    """Loopback gate: only accept dispatch from the same machine."""
    if host is None:
        return False
    return host in ("127.0.0.1", "::1", "localhost")


@router.post("/_dispatch")
async def dispatch(request: Request, body: DispatchRequest) -> dict[str, Any]:
    """Internal endpoint. Only same-host callers (the MCP daemon) accepted.

    Returns ``{delivered, type, client_id}`` so the calling bridge tool
    can tell the agent whether anyone was listening.
    """
    # Proxy headers indicate the request passed through nginx / Cloudflare /
    # similar — at which point request.client.host is the proxy, not the
    # real client. Reject outright rather than trust either value. The
    # dispatcher is reached only from in-process MCP tools on the same host.
    if request.headers.get("x-forwarded-for") or request.headers.get("forwarded"):
        raise HTTPException(
            status_code=403,
            detail="dispatch is loopback-only; proxy-fronted requests are rejected",
        )
    if not _is_loopback(_client_host(request)):
        raise HTTPException(status_code=403, detail="dispatch is loopback-only")
    if body.type not in SUPPORTED_COMMAND_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"unknown command type {body.type!r}; supported: {sorted(SUPPORTED_COMMAND_TYPES)}",
        )
    message = {"type": body.type, "client_id": body.client_id, "params": body.params}
    delivered = await get_registry().dispatch(body.client_id, message)
    return {"delivered": delivered, "type": body.type, "client_id": body.client_id}


# ── WebSocket endpoint (GUI tab subscribes) ──────────────────────────────


def _resolve_target(query_as: str | None, cookie_as: str | None) -> str:
    """Cookie wins over query (the cookie is the persistent declaration)."""
    raw = cookie_as or query_as or WILDCARD
    # Empty or whitespace falls back to wildcard for robustness.
    raw = raw.strip()
    return raw or WILDCARD


@router.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    """Subscribe a GUI tab. ``as`` query param or ``lerobot_as`` cookie
    declares the target client_id.
    """
    target = _resolve_target(
        ws.query_params.get("as"),
        ws.cookies.get("lerobot_as"),
    )
    await ws.accept()
    sub = await get_registry().add(ws, target)
    # Send a hello so the tab can confirm its declared target was honored.
    try:
        await sub.send({"type": "hello", "target": target})
    except Exception:  # noqa: BLE001
        logger.warning("bridge: initial hello to #%d failed", sub.id, exc_info=True)
        await get_registry().remove(sub)
        return
    try:
        # We don't actually read anything from the client (commands are
        # one-way for now). But we must await receive() to detect close.
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        logger.warning("bridge: ws #%d errored", sub.id, exc_info=True)
    finally:
        await get_registry().remove(sub)


# ── Test helper ──────────────────────────────────────────────────────────


def _reset_registry_for_tests() -> None:
    """Wipe the global registry. Only call from tests."""
    global _REGISTRY
    _REGISTRY = SubscriberRegistry()
