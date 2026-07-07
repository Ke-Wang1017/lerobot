#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""WebXR / WebSocket server that feeds the QuestVRTeleop's cached action.

Runs in a daemon thread launched by the teleoperator's connect(). Serves a
small HTML page over HTTPS (WebXR's "secure context" requirement on Quest
3 prevents plain HTTP from starting an immersive session) and accepts a
WebSocket on /ws. Each XR frame from the Quest carries the controller
poses; we convert them into the teleop's action format and store the
latest under a lock for the teleop's get_action() to read.

Self-signed cert handling: we shell out to ``openssl`` on first run and
cache cert.pem / key.pem in the package directory. The Quest browser will
show a "self-signed cert" warning once per device, then remember the
exception.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import socket
import ssl
import subprocess
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.utils.import_utils import require_package

if TYPE_CHECKING:
    # aiohttp is an optional dependency, imported lazily at runtime inside
    # the server methods. This block makes the `web.*` type annotations
    # resolvable for type-checkers / linters without importing it eagerly.
    from aiohttp import web

logger = logging.getLogger(__name__)

PING_INTERVAL_S = 0.1


def _get_lan_ip() -> str:
    """Best-effort: return the host's LAN IPv4 so we can print a URL the Quest can hit."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


def _ensure_cert(cert_dir: Path) -> tuple[Path, Path]:
    """Generate a self-signed cert in cert_dir if one isn't already there."""
    cert = cert_dir / "cert.pem"
    key = cert_dir / "key.pem"
    if cert.exists() and key.exists():
        return cert, key
    cert_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating self-signed cert for Quest VR teleop ...")
    subprocess.check_call(  # nosec B603 (openssl is a controlled binary)
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "365",
            "-nodes",
            "-subj",
            "/CN=lerobot-quest-vr",
        ]
    )
    return cert, key


class QuestServer:
    """Daemon-thread HTTPS + WSS server. Updates a single shared cached action.

    Lifecycle:
        s = QuestServer(html_path=..., port=8443, on_frame=callback)
        s.start()   # spawns daemon thread, starts asyncio loop + aiohttp app
        # ... read s.cached_action under s.lock from outside ...
        s.stop()    # signals loop to drain, joins the thread

    on_frame(frame_dict) is called for each WebXR frame received. It runs on
    the server's asyncio thread. Use it to update an externally-held cache.
    """

    def __init__(
        self,
        html_path: Path,
        port: int,
        cert_dir: Path,
        on_frame: Any,  # callable taking parsed frame dict
        page_vars: dict[str, str] | None = None,
        get_hold_state: Any = None,  # callable () -> (left_holding, right_holding) | None
    ) -> None:
        self.html_path = Path(html_path)
        self.port = port
        self.cert_dir = Path(cert_dir)
        self.on_frame = on_frame
        # ``{{KEY}}`` markers in the served HTML are substituted with these
        # values at request time. Used to pass per-session config (e.g.
        # ``clutch_button_index`` for the haptic-feedback path) into the
        # page without a back-channel from the WebSocket.
        self.page_vars: dict[str, str] = dict(page_vars or {})
        # Optional hook: every WebXR frame, the WS handler calls this and
        # sends a ``{type: "hold", left, right}`` message to the client
        # whenever the boolean tuple changes. The client uses that to drive
        # the IK-hold continuous-rumble haptic. Polled instead of pushed
        # (no cross-thread queue) — the read is one tuple, atomic under the
        # GIL. None disables the channel.
        self.get_hold_state = get_hold_state
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner: web.AppRunner | None = None
        self._stop_evt = threading.Event()
        self._started = threading.Event()
        self._last_rtt_ms: float | None = None

    @property
    def last_rtt_ms(self) -> float | None:
        return self._last_rtt_ms

    @property
    def url(self) -> str:
        return f"https://{_get_lan_ip()}:{self.port}/"

    @property
    def is_running(self) -> bool:
        """True if the asyncio thread has been started and not yet exited."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        # aiohttp is an optional dependency (lerobot[quest-vr]). Check here,
        # on the calling thread, so a missing install fails with a clear
        # message instead of a generic "server failed to start" timeout
        # raised from deep inside the daemon thread.
        require_package("aiohttp", "quest-vr")
        self._stop_evt.clear()
        self._started.clear()
        self._thread = threading.Thread(target=self._run, name="quest_vr_server", daemon=True)
        self._thread.start()
        # Wait briefly for the loop to bind so callers know if start succeeded.
        if not self._started.wait(timeout=5.0):
            raise RuntimeError("QuestServer failed to start within 5s")

    def stop(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            return
        self._stop_evt.set()
        if self._loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop).result(timeout=3.0)
            except asyncio.CancelledError:
                # Expected — the loop cancelling its own pending tasks at
                # shutdown propagates here as the scheduled coro racing the
                # loop close. Not an error.
                pass
            except Exception as e:
                logger.warning(f"QuestServer shutdown coro raised: {type(e).__name__}: {e}")
        self._thread.join(timeout=3.0)
        if self._thread.is_alive():
            logger.warning("QuestServer thread did not exit within 3s")
        self._thread = None
        self._loop = None
        self._runner = None

    # ── Internal: async loop body ─────────────────────────────────────────

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve_forever())
        except Exception:
            logger.exception("QuestServer asyncio loop crashed")
        finally:
            # Cancel + gather any tasks still around (websocket handlers, ping
            # loops, request handlers mid-request) so closing the loop doesn't
            # log "Task was destroyed but it is pending!" or "Event loop is
            # closed" from aiohttp's late-running cleanup callbacks.
            try:
                pending = [t for t in asyncio.all_tasks(self._loop) if not t.done()]
                for t in pending:
                    t.cancel()
                if pending:
                    self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                # One more pass through the loop to let aiohttp's transport
                # close callbacks settle on the same (still-open) loop rather
                # than firing post-close.
                self._loop.run_until_complete(asyncio.sleep(0))
            except Exception:
                logger.debug("QuestServer task drain raised", exc_info=True)
            self._loop.close()

    async def _serve_forever(self) -> None:
        from aiohttp import web

        cert_path, key_path = _ensure_cert(self.cert_dir)
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))

        app = web.Application()
        app.router.add_get("/", self._handle_index)
        app.router.add_get("/ws", self._handle_ws)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        # Binding 0.0.0.0 here is intentional: the Quest 3 is on a different
        # host (same LAN), so localhost-only would not work. The teleop is a
        # LAN-only dev tool; users open ports manually via their firewall.
        site = web.TCPSite(self._runner, "0.0.0.0", self.port, ssl_context=ssl_ctx)  # nosec B104
        await site.start()
        self._started.set()
        logger.info(f"QuestVR server listening on {self.url}")
        # Park until shutdown.
        while not self._stop_evt.is_set():
            await asyncio.sleep(0.1)

    async def _shutdown(self) -> None:
        # Cleanup tears down the listening socket and waits for in-flight
        # requests / WS sessions to drain. Wrapped in suppress because
        # aiohttp can raise during cleanup if the Quest disconnects rudely.
        if self._runner is not None:
            with contextlib.suppress(Exception):
                await self._runner.cleanup()

    # ── Handlers ──────────────────────────────────────────────────────────

    async def _handle_index(self, request: web.Request) -> web.Response:
        from aiohttp import web

        html = self.html_path.read_text()
        for key, value in self.page_vars.items():
            html = html.replace("{{" + key + "}}", value)
        return web.Response(text=html, content_type="text/html")

    async def _handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        from aiohttp import WSMsgType, web

        ws = web.WebSocketResponse(max_msg_size=64 * 1024)
        await ws.prepare(request)
        logger.info(f"QuestVR client connected from {request.remote}")

        pending_pings: dict[int, float] = {}

        async def ping_loop() -> None:
            seq = 0
            while not ws.closed:
                t = time.perf_counter() * 1000
                pending_pings[seq] = t
                try:
                    await ws.send_json({"type": "ping", "seq": seq, "t_pc_send": t})
                except ConnectionError:
                    return
                seq += 1
                await asyncio.sleep(PING_INTERVAL_S)

        ping_task = asyncio.create_task(ping_loop())
        # Track the last hold-state tuple sent to the client so we only
        # emit a message on transitions. Starts at "neither holding" —
        # matches the client-side initial state, so no spurious first edge.
        last_hold_sent: tuple[bool, bool] = (False, False)
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                data = json.loads(msg.data)
                mtype = data.get("type")
                if mtype == "pong":
                    t_send = pending_pings.pop(data["seq"], None)
                    if t_send is not None:
                        self._last_rtt_ms = time.perf_counter() * 1000 - t_send
                        with contextlib.suppress(ConnectionError):
                            await ws.send_json({"type": "rtt", "ms": self._last_rtt_ms})
                elif mtype == "frame":
                    try:
                        self.on_frame(data)
                    except Exception:
                        logger.exception("on_frame callback raised")
                    # Push a hold-state update if it changed since last send.
                    # Cheap (one bool-tuple read + at most one WS write per
                    # transition); rate-limited by client frame cadence.
                    if self.get_hold_state is not None:
                        try:
                            current = tuple(bool(b) for b in self.get_hold_state())
                        except Exception:
                            logger.exception("get_hold_state raised")
                            current = last_hold_sent
                        if current != last_hold_sent:
                            with contextlib.suppress(ConnectionError):
                                await ws.send_json({"type": "hold", "left": current[0], "right": current[1]})
                            last_hold_sent = current
        finally:
            ping_task.cancel()
            logger.info("QuestVR client disconnected")
        return ws


# Axis-mapping helper kept here (not in teleop module) since it relates to
# the WebXR frame -> robot-frame translation that happens server-side.

# Quest local stage frame: +x = user right, +y = up, +z = toward user.
# Robot base frame:        we map "user forward / right / up" to robot frame
# such that pushing the hand forward extends the arm forward, etc. The
# user is assumed to stand behind the robot facing the same direction the
# arm reaches. ROBOT_FORWARD_IN_URDF + ROBOT_UP_IN_URDF below define what
# "forward" / "up" are in URDF coords; the mapping is derived from them.
ROBOT_FORWARD_IN_URDF = np.array([0.0, -1.0, 0.0])  # SO-107 default; arm reaches in -Y
ROBOT_UP_IN_URDF = np.array([0.0, 0.0, 1.0])
_ROBOT_LEFT_IN_URDF = np.cross(ROBOT_UP_IN_URDF, ROBOT_FORWARD_IN_URDF)

# columns map (quest_x, quest_y, quest_z) unit vectors into URDF frame.
QUEST_TO_ROBOT_M = np.column_stack(
    [
        -_ROBOT_LEFT_IN_URDF,  # quest_x = user_right  -> robot_right
        +ROBOT_UP_IN_URDF,  # quest_y = user_up     -> robot_up
        -ROBOT_FORWARD_IN_URDF,  # quest_z = user_back   -> robot_back
    ]
)


def quest_delta_to_robot(delta_quest: np.ndarray) -> np.ndarray:
    return QUEST_TO_ROBOT_M @ delta_quest


def quest_rot_to_robot(quat_xyzw: list[float]):
    """Quest controller quaternion -> robot-frame scipy Rotation."""
    from scipy.spatial.transform import Rotation

    r_quest = Rotation.from_quat(quat_xyzw).as_matrix()
    return Rotation.from_matrix(QUEST_TO_ROBOT_M @ r_quest @ QUEST_TO_ROBOT_M.T)
