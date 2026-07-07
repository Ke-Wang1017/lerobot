"""MCP-side half of the GUI bridge.

Four tools the AI calls to drive the user's open GUI tab:

* ``navigate_to(view, params)``
* ``notify_user(title, body, deeplink=None, level='info')``
* ``highlight_in_viewer(repo_id, episode_ids)``
* ``set_filter(viewer, filters)``

Each tool POSTs a small JSON envelope to the GUI's
``/api/bridge/_dispatch`` endpoint. The endpoint is loopback-only on the
GUI side, so no auth header is needed — both processes are on the same
host. If the GUI is unreachable the tool returns a clear ``delivered: 0``
result so the agent can tell the user "your GUI isn't open yet."

Identity routing: every dispatch carries the calling MCP token's
``client_id``. GUI tabs that subscribe with ``as=*`` (the v1 default)
receive everything; tabs that subscribe with ``as=<token_name>`` only
receive commands issued under that token. See architectural decision #9.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

from mcp.server.fastmcp import FastMCP

from .auth import SCOPE_READ, requires_scope

logger = logging.getLogger(__name__)

# Set by ``configure_bridge`` at server-build time. None means the bridge
# is disabled (no GUI URL configured) — tools succeed but return
# ``delivered: 0`` with a clear reason.
_GUI_DISPATCH_URL: str | None = None

_STDIO_CLIENT_ID = "stdio-local"


def configure_bridge(gui_url: str | None) -> None:
    """Point bridge tools at the GUI's ``/api/bridge/_dispatch`` endpoint.

    Pass the GUI's *base* URL (e.g. ``http://127.0.0.1:8000``); the
    ``/api/bridge/_dispatch`` suffix is appended internally.
    """
    global _GUI_DISPATCH_URL
    if gui_url is None:
        _GUI_DISPATCH_URL = None
        return
    base = gui_url.rstrip("/")
    _GUI_DISPATCH_URL = f"{base}/api/bridge/_dispatch"
    logger.info("bridge tools configured: dispatch=%s", _GUI_DISPATCH_URL)


def _current_client_id() -> str:
    """Resolve the calling MCP token's ``name``.

    In stdio mode there is no authenticated token; use a fixed sentinel
    so wildcard subscribers still receive the command.
    """
    from mcp.server.auth.middleware.auth_context import get_access_token

    token = get_access_token()
    return token.client_id if token is not None else _STDIO_CLIENT_ID


async def _try_inproc_dispatch(command_type: str, params: dict[str, Any]) -> dict[str, Any] | None:
    """In-process dispatch when MCP is mounted in the same process as the GUI.

    Returns the dispatcher's result dict, or ``None`` if the in-process path
    isn't available (split-process deployments) so the caller falls back to
    HTTP. The tool body is ``async`` so this can ``await`` the registry
    directly without crossing event-loop boundaries.
    """
    try:
        from lerobot.gui.api.bridge import SUPPORTED_COMMAND_TYPES, get_registry
    except ImportError:
        return None
    if command_type not in SUPPORTED_COMMAND_TYPES:
        return {
            "delivered": 0,
            "reason": (
                f"unknown command type {command_type!r}; supported: {sorted(SUPPORTED_COMMAND_TYPES)}"
            ),
        }
    client_id = _current_client_id()
    message = {"type": command_type, "client_id": client_id, "params": params}
    try:
        registry = get_registry()
    except Exception:  # noqa: BLE001
        return None
    delivered = await registry.dispatch(client_id, message)
    return {"delivered": delivered, "type": command_type, "client_id": client_id}


async def _dispatch(command_type: str, params: dict[str, Any]) -> dict[str, Any]:
    """Deliver a bridge command to the GUI.

    Prefers in-process dispatch when running unified (same Python process as
    the GUI); falls back to HTTP for split-process deployments.
    """
    inproc = await _try_inproc_dispatch(command_type, params)
    if inproc is not None:
        return inproc

    if _GUI_DISPATCH_URL is None:
        return {"delivered": 0, "reason": "bridge disabled (no --gui-url configured)"}
    # HTTP fallback path: still synchronous urllib, run in a worker thread so
    # we don't block the event loop while waiting on the loopback POST.
    client_id = _current_client_id()
    payload = json.dumps({"client_id": client_id, "type": command_type, "params": params}).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310  # nosec B310 — admin-supplied http(s) URL, never user input
        _GUI_DISPATCH_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    def _post() -> dict[str, Any]:
        try:
            with urllib.request.urlopen(req, timeout=2.0) as resp:  # noqa: S310  # nosec B310 — see above
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.URLError as e:
            logger.info("bridge dispatch failed: %s", e)
            return {"delivered": 0, "reason": f"GUI unreachable: {e.reason if hasattr(e, 'reason') else e}"}
        except Exception as e:  # noqa: BLE001
            logger.warning("bridge dispatch raised", exc_info=True)
            return {"delivered": 0, "reason": f"dispatch error: {type(e).__name__}: {e}"}

    import asyncio

    return await asyncio.to_thread(_post)


# ── Tool registration ────────────────────────────────────────────────────


def register_bridge_tools(mcp: FastMCP) -> None:
    """Attach the four bridge tools to a FastMCP instance.

    All four are read-scope: they change what the user *sees*, not the
    canonical state on disk.
    """

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    async def navigate_to(view: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Navigate the user's open GUI tab to a specific view.

        Args:
            view: Logical view name. Recognized values: ``'episode'``,
                ``'dataset'``, ``'run'``, ``'home'``, ``'model'``, ``'robot'``.
                Unknown values are forwarded to the GUI as-is.
            params: View-specific params (e.g. ``{'repo_id': '...',
                'episode_id': 47}`` for ``view='episode'``).

        Returns:
            ``{delivered, type, client_id}`` — ``delivered`` is the
            count of GUI tabs that received the command (0 if the GUI
            isn't open).
        """
        return await _dispatch("navigate", {"view": view, "params": params or {}})

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    async def notify_user(
        title: str,
        body: str = "",
        deeplink: str | None = None,
        level: str = "info",
    ) -> dict[str, Any]:
        """Surface a notification in the user's GUI tab.

        If the tab is backgrounded the GUI uses the Web Notifications
        API to display an OS-level banner; clicking it refocuses the
        tab and (if ``deeplink`` is provided) navigates to that view.

        Args:
            title: Short headline (1 line).
            body: Optional detail (1-2 sentences).
            deeplink: Optional fragment/path the click should open
                (e.g. ``'#/episode/.../47'`` or ``'/run/abc123'``).
            level: ``'info'`` | ``'success'`` | ``'warning'`` | ``'error'``.
        """
        return await _dispatch(
            "notify",
            {"title": title, "body": body, "deeplink": deeplink, "level": level},
        )

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    async def highlight_in_viewer(repo_id: str, episode_ids: list[int]) -> dict[str, Any]:
        """Mark a set of episodes as highlighted in the GUI's list/timeline.

        Non-destructive — just a visual emphasis the user can clear.
        """
        return await _dispatch(
            "highlight",
            {"repo_id": repo_id, "episode_ids": list(episode_ids)},
        )

    # set_filter is intentionally NOT registered yet. The bridge dispatches a
    # `filter` command (still in SUPPORTED_COMMAND_TYPES so the wire shape is
    # locked in), but no GUI viewer has a real filter input today. Adding the
    # tool now would advertise a capability that has no visible effect —
    # see plan doc "Tools (design surface)" / Run + Bridge sections, and
    # gui/TODO.md's "filter UX" entry.
