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

"""Screenshot the LeRobot GUI from a Python script.

The boilerplate (start a GUI server, launch Chrome with CDP, wait for
app.js, close CDP cleanly before any iframe attaches, capture via
ffmpeg's x11grab) is the same regardless of what page you're capturing.
This module wraps it so per-task scripts only spell out the *arming*
sequence — the few JS calls that put the page into the state you want
in the shot.

Why CDP for arming + ffmpeg for capture (rather than CDP for both):
once an iframe with its own origin attaches (MeshCat, an embedded
viewer, etc.), Chrome's CDP debugger auto-attaches to the new target
and the parent socket gets stuck — every later `Runtime.evaluate` or
`Page.captureScreenshot` times out. Reading the X server directly side-
steps the deadlock entirely; the only cost is needing a real X display.

Requires:
    - X display (`DISPLAY` set)
    - `ffmpeg` with x11grab support
    - `google-chrome` on PATH
    - Python deps: `requests`, `websocket-client`

Usage (library):

    from screenshot_gui import GuiScreenshotSession

    with GuiScreenshotSession(
        output_path=Path("/tmp/data_tab.png"),
        url_params={"urdfGhost": "on"},
    ) as s:
        s.eval("switchTab('data')")
        s.eval("openDataset('thewisp/intervention_cylinder_ring_assembly')")
        s.wait_until(
            "episodes['thewisp/intervention_cylinder_ring_assembly']?.[0]?.length > 0"
        )
        s.eval("selectEpisode('thewisp/intervention_cylinder_ring_assembly', 0, 385)")
        s.sleep(2)
        s.eval("loadAllFrames(80)")
        # On context exit: close CDP, wait for paint, capture, terminate
        # subprocesses.

Usage (CLI, no arming — just opens the home page):

    python scripts/gui/screenshot_gui.py --output /tmp/home.png

Set `start_gui=False` and pass a `url=` if a GUI is already running on a
known address; otherwise the session starts one on a random port.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests
import websocket

_DEFAULT_WINDOW = (1600, 900)
_APP_JS_READY = "typeof switchTab === 'function'"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] screenshot_gui: {msg}", flush=True)


def _free_port() -> int:
    """Bind a TCP socket to port 0 and return what the OS gave us.

    Closes immediately — we just need to know a port nothing's holding.
    Avoids hard-coding a port that might collide with another agent's
    in-flight GUI server.
    """
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class GuiScreenshotSession:
    """Boot a GUI, drive it through CDP, capture once on close.

    Pre: a usable X display is available; ffmpeg and google-chrome are on
    PATH; ``output_path``'s parent directory exists (or can be created).

    Post (on successful exit): ``output_path`` contains a PNG of the
    Chrome window. The Chrome and GUI subprocesses are terminated. Any
    exception inside the ``with`` block still triggers cleanup.
    """

    def __init__(
        self,
        output_path: Path | str,
        *,
        url: str | None = None,
        url_params: dict[str, str] | None = None,
        gui_port: int | None = None,
        start_gui: bool = True,
        python_bin: str | None = None,
        window_size: tuple[int, int] = _DEFAULT_WINDOW,
        post_close_sleep: float = 8.0,
        app_js_ready_expr: str = _APP_JS_READY,
        boot_timeout_s: float = 30.0,
    ):
        self.output_path = Path(output_path)
        self.window_size = window_size
        self.post_close_sleep = post_close_sleep
        self.app_js_ready_expr = app_js_ready_expr
        self.boot_timeout_s = boot_timeout_s

        self._start_gui = start_gui
        self._gui_port = gui_port if gui_port is not None else _free_port()
        self._python_bin = python_bin or sys.executable
        # url_params let the caller flip iframe defaults (e.g. ?urdfGhost=on)
        # that the page reads on load — useful when the toggle is inside an
        # iframe we'd rather not poke after the fact.
        qs = ""
        if url_params:
            qs = "?" + "&".join(f"{k}={v}" for k, v in url_params.items())
        self._url = url or f"http://127.0.0.1:{self._gui_port}/{qs}"

        self._cdp_port = 9000 + (os.getpid() % 1000)
        self._scratch = Path(tempfile.gettempdir()) / f"lerobot_screenshot_{os.getpid()}"
        self._scratch.mkdir(parents=True, exist_ok=True)

        self._gui_proc: subprocess.Popen | None = None
        self._chrome_proc: subprocess.Popen | None = None
        self._ws: websocket.WebSocket | None = None
        self._mid = 0

    # ---- lifecycle -----------------------------------------------------

    def __enter__(self) -> GuiScreenshotSession:
        try:
            if self._start_gui:
                self._spawn_gui()
                self._wait_for_gui()
            self._spawn_chrome()
            self._attach_cdp()
            self._wait_for_app_js()
        except Exception:
            self._cleanup()
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            # Close CDP *before* any iframe finishes attaching — past that
            # point the socket can wedge and we'd hang in cleanup. We've
            # already armed everything we care about by this point.
            with contextlib.suppress(Exception):
                if self._ws is not None:
                    self._ws.close()
            # Only capture if we exited cleanly (no exception in the body).
            if exc_type is None:
                if self.post_close_sleep > 0:
                    time.sleep(self.post_close_sleep)
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
                self._ffmpeg_capture()
        finally:
            self._cleanup()

    # ---- the small public surface --------------------------------------

    def eval(self, expression: str):
        """Evaluate a JS expression in the page; return its serialised value.

        Pre: session is attached (you're inside the ``with`` block). The
        expression may be sync or return a Promise — we await it either way.
        Post: the JS result is JSON-coerced and returned. Errors raise.
        """
        result = self._cdp(
            "Runtime.evaluate",
            {"expression": expression, "awaitPromise": True, "returnByValue": True},
        )
        result_obj = result.get("result", {}).get("result", {})
        if "exceptionDetails" in result.get("result", {}):
            details = result["result"]["exceptionDetails"]
            raise RuntimeError(f"JS exception evaluating {expression!r}: {details}")
        return result_obj.get("value")

    def wait_until(
        self,
        condition_expr: str,
        timeout: float = 15.0,
        interval: float = 0.25,
    ) -> None:
        """Poll ``condition_expr`` until it returns truthy.

        Pre: ``condition_expr`` is a JS expression evaluable in the page.
        Post: returns when it returns truthy; raises ``TimeoutError`` if
        ``timeout`` seconds elapse without that happening.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.eval(condition_expr):
                return
            time.sleep(interval)
        raise TimeoutError(f"wait_until timed out after {timeout}s: {condition_expr!r}")

    def sleep(self, seconds: float) -> None:
        """Plain ``time.sleep`` — re-exported so arming scripts don't import it."""
        time.sleep(seconds)

    # ---- internals -----------------------------------------------------

    def _spawn_gui(self) -> None:
        _log(f"starting GUI on port {self._gui_port} via {self._python_bin}")
        self._gui_proc = subprocess.Popen(
            [self._python_bin, "-m", "lerobot.gui", "--port", str(self._gui_port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _wait_for_gui(self) -> None:
        deadline = time.monotonic() + self.boot_timeout_s
        url = f"http://127.0.0.1:{self._gui_port}/api/datasets"
        while time.monotonic() < deadline:
            try:
                if requests.get(url, timeout=0.5).status_code == 200:
                    _log("GUI online")
                    return
            except Exception:
                pass
            time.sleep(0.2)
        raise RuntimeError(f"GUI did not come online at {url} within {self.boot_timeout_s}s")

    def _spawn_chrome(self) -> None:
        profile = self._scratch / f"chrome_profile_{self._cdp_port}"
        profile.mkdir(parents=True, exist_ok=True)
        _log(f"launching chrome cdp_port={self._cdp_port} url={self._url}")
        w, h = self.window_size
        self._chrome_proc = subprocess.Popen(
            [
                "google-chrome",
                f"--remote-debugging-port={self._cdp_port}",
                f"--user-data-dir={profile}",
                "--remote-allow-origins=*",
                "--no-first-run",
                "--no-default-browser-check",
                f"--window-size={w},{h}",
                "--window-position=0,0",
                self._url,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _attach_cdp(self) -> None:
        ws_url = None
        deadline = time.monotonic() + self.boot_timeout_s
        while time.monotonic() < deadline:
            try:
                tabs = requests.get(f"http://127.0.0.1:{self._cdp_port}/json", timeout=0.5).json()
                pages = [
                    t
                    for t in tabs
                    if t.get("type") == "page"
                    and (self._gui_port is None or f"127.0.0.1:{self._gui_port}" in t.get("url", ""))
                ]
                if not pages:
                    pages = [t for t in tabs if t.get("type") == "page"]
                if pages:
                    ws_url = pages[0]["webSocketDebuggerUrl"]
                    break
            except Exception:
                pass
            time.sleep(0.2)
        if not ws_url:
            raise RuntimeError("CDP target never appeared")
        self._ws = websocket.create_connection(ws_url, timeout=6.0)
        self._cdp("Page.enable")

    def _wait_for_app_js(self) -> None:
        # app.js loads asynchronously after the page navigates; until
        # `switchTab` is defined, calling it would raise. Wait at most
        # boot_timeout_s.
        deadline = time.monotonic() + self.boot_timeout_s
        while time.monotonic() < deadline:
            try:
                if self.eval(self.app_js_ready_expr):
                    _log("app.js ready")
                    return
            except Exception:
                pass
            time.sleep(0.3)
        _log("app.js not detected, proceeding anyway")

    def _cdp(self, method: str, params: dict | None = None, timeout: float = 6.0) -> dict:
        assert self._ws is not None, "CDP not attached"
        self._mid += 1
        mid = self._mid
        payload: dict = {"id": mid, "method": method}
        if params:
            payload["params"] = params
        self._ws.send(json.dumps(payload))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._ws.settimeout(max(deadline - time.monotonic(), 0.01))
            try:
                msg = json.loads(self._ws.recv())
            except websocket.WebSocketTimeoutException:
                return {}
            if msg.get("id") == mid:
                return msg
        return {}

    def _ffmpeg_capture(self) -> None:
        w, h = self.window_size
        _log(f"ffmpeg x11grab {w}x{h} -> {self.output_path}")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "x11grab",
                "-video_size",
                f"{w}x{h}",
                "-i",
                ":0.0+0,0",
                "-frames:v",
                "1",
                str(self.output_path),
            ],
            check=True,
            timeout=15,
        )

    def _cleanup(self) -> None:
        procs = [self._chrome_proc]
        if self._start_gui:
            procs.append(self._gui_proc)
        for proc in procs:
            if proc is None:
                continue
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()


# ---------------------------------------------------------------------------
# CLI entry point — minimal "open the home page and shoot it" form. For
# anything beyond that, write a few lines using the class above.
# ---------------------------------------------------------------------------


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Screenshot the LeRobot GUI home page.")
    p.add_argument("--output", "-o", required=True, type=Path)
    p.add_argument("--gui-port", type=int, default=None)
    p.add_argument(
        "--no-start-gui",
        action="store_true",
        help="Assume a GUI is already running at --url (or http://127.0.0.1:<gui-port>/).",
    )
    p.add_argument("--url", default=None, help="Page to open. Defaults to the GUI home on --gui-port.")
    p.add_argument("--width", type=int, default=_DEFAULT_WINDOW[0])
    p.add_argument("--height", type=int, default=_DEFAULT_WINDOW[1])
    p.add_argument("--post-close-sleep", type=float, default=8.0)
    args = p.parse_args(argv)

    with GuiScreenshotSession(
        output_path=args.output,
        url=args.url,
        gui_port=args.gui_port,
        start_gui=not args.no_start_gui,
        window_size=(args.width, args.height),
        post_close_sleep=args.post_close_sleep,
    ):
        pass  # No arming — capture as-loaded.
    return 0


if __name__ == "__main__":
    sys.exit(_main())
