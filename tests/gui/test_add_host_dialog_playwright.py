# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Playwright smoke for the Add-SSH-host dialog.

Drives the real GUI (full FastAPI app + static frontend) in a uvicorn
thread, with the training state rewired to tmp dirs and ``probe_ssh``
monkeypatched — no real SSH, no writes outside tmp_path.

State machine under test (see add_host_dialog.js header):

    closed → open(empty, Save disabled) → probing(Test disabled,
    "Testing…") → probed(checklist, Save enabled iff ok) → saved(closed,
    sidebar refreshed)

plus the invalidation edge: editing a field after a green probe must
re-disable Save.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")
import uvicorn  # noqa: E402
from playwright.sync_api import (
    Error as PlaywrightError,  # noqa: E402
    sync_playwright,  # noqa: E402
)

from lerobot.gui.api import training as training_api  # noqa: E402
from lerobot.gui.training.hosts import HostRegistry, TrainingHost  # noqa: E402
from lerobot.gui.training.orchestrator import Orchestrator  # noqa: E402
from lerobot.gui.training.probe import CheckItem, ProbeResult  # noqa: E402
from lerobot.gui.training.runs import RunRegistry  # noqa: E402
from lerobot.gui.training.transport import SubprocessTransport  # noqa: E402

pytestmark = pytest.mark.requires_playwright


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _green_probe_result() -> ProbeResult:
    return ProbeResult(
        ok=True,
        latency_ms=42,
        checks=[
            CheckItem(name="ssh", ok=True, detail="connected in 42 ms"),
            CheckItem(name="docker", ok=True, detail="/usr/bin/docker"),
            CheckItem(name="tmux", ok=True, detail="/usr/bin/tmux"),
            CheckItem(name="nvidia", ok=True, detail="GPU 0: NVIDIA Test GPU"),
        ],
        error_class=None,
        message="All checks passed",
    )


def _auth_fail_probe_result() -> ProbeResult:
    return ProbeResult(
        ok=False,
        latency_ms=99,
        checks=[
            CheckItem(name="ssh", ok=False, detail="SSH authentication failed"),
            CheckItem(name="docker", ok=False, detail="not reached"),
            CheckItem(name="tmux", ok=False, detail="not reached"),
            CheckItem(name="nvidia", ok=False, detail="not reached"),
        ],
        error_class="auth",
        message="SSH authentication failed — check your ssh-agent / ~/.ssh/config",
    )


@pytest.fixture
def gui_server(tmp_path: Path, monkeypatch):
    """Full GUI app in a uvicorn thread, training state on tmp dirs.

    Yields ``(base_url, probe_box)`` where ``probe_box['result']`` /
    ``probe_box['delay_s']`` control what the (mocked) probe returns.
    """
    # Import inside the fixture: pulling in lerobot.gui.server mounts the
    # full app (robot/datasets/etc.) which is slow — only pay it when the
    # test actually runs (not at collection).
    from lerobot.gui import server as gui_server_mod

    monkeypatch.setattr(training_api, "HOSTS_DIR", tmp_path / "training_hosts")

    workstation = TrainingHost(
        id="this-server",
        display_name="This server",
        transport=SubprocessTransport(workdir=tmp_path / "workdir"),
        capabilities={"gpu_name": "Test GPU", "vram_mb": 16384, "gpu_count_detected": 1},
    )
    registry = HostRegistry(hosts=[workstation])
    orch = Orchestrator(host_registry=registry, run_registry=RunRegistry(runs_dir=tmp_path / "runs"))
    training_api.init_state(orch=orch, host_registry=registry)

    probe_box: dict = {"result": _green_probe_result(), "delay_s": 0.0}

    async def fake_probe_ssh(host_spec: str, **kw) -> ProbeResult:
        if probe_box["delay_s"]:
            await asyncio.sleep(probe_box["delay_s"])
        return probe_box["result"]

    monkeypatch.setattr(training_api, "probe_ssh", fake_probe_ssh)

    port = _free_port()
    config = uvicorn.Config(gui_server_mod.app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    import requests

    deadline = time.monotonic() + 15
    base_url = f"http://127.0.0.1:{port}"
    while time.monotonic() < deadline:
        try:
            if requests.get(f"{base_url}/api/training/hosts", timeout=1).status_code == 200:
                break
        except requests.RequestException:
            time.sleep(0.1)
    else:
        server.should_exit = True
        pytest.fail("GUI server did not become ready within 15s")

    yield base_url, probe_box

    server.should_exit = True
    thread.join(timeout=10)
    # init_state is module-global; reset so later tests re-wire explicitly.
    training_api.reset_state_for_testing()


@pytest.fixture
def page(gui_server):
    base_url, probe_box = gui_server
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except PlaywrightError as e:
            pytest.skip(f"chromium not available: {e}")
        pg = browser.new_page(viewport={"width": 1280, "height": 800})
        pg.goto(base_url)
        # The training sidebar lives in the Model tab; the default tab is Data.
        pg.wait_for_function("typeof switchTab === 'function'", timeout=10_000)
        pg.evaluate("switchTab('model')")
        pg.wait_for_selector("#training-add-host-btn", timeout=10_000)
        yield pg, probe_box
        browser.close()


def test_dialog_opens_and_closes(page):
    pg, _ = page
    assert not pg.is_visible("#add-host-overlay")
    pg.click("#training-add-host-btn")
    assert pg.is_visible("#add-host-overlay")
    # Save starts disabled — no probe yet.
    assert pg.is_disabled("#add-host-save-btn")
    pg.click("#add-host-cancel")
    assert not pg.is_visible("#add-host-overlay")
    # Esc also closes.
    pg.click("#training-add-host-btn")
    pg.keyboard.press("Escape")
    assert not pg.is_visible("#add-host-overlay")


def test_test_button_flips_state_during_probe(page):
    pg, probe_box = page
    probe_box["delay_s"] = 1.0  # slow enough to observe the pending state
    pg.click("#training-add-host-btn")
    pg.fill("#add-host-host", "user@example-host")
    pg.click("#add-host-test-btn")
    # While the probe is in flight: label flipped, button disabled.
    pg.wait_for_function(
        "document.getElementById('add-host-test-btn').textContent === 'Testing…'", timeout=2_000
    )
    assert pg.is_disabled("#add-host-test-btn")
    # After it lands: restored label, green checklist, Save enabled.
    pg.wait_for_selector(".probe-check.ok", timeout=5_000)
    assert pg.text_content("#add-host-test-btn") == "Test"
    assert not pg.is_disabled("#add-host-test-btn")
    assert pg.locator(".probe-check.ok").count() == 4
    assert not pg.is_disabled("#add-host-save-btn")


def test_failed_probe_keeps_save_disabled(page):
    pg, probe_box = page
    probe_box["result"] = _auth_fail_probe_result()
    pg.click("#training-add-host-btn")
    pg.fill("#add-host-host", "user@bad-host")
    pg.click("#add-host-test-btn")
    pg.wait_for_selector(".probe-check.fail", timeout=5_000)
    assert pg.is_disabled("#add-host-save-btn")
    status = pg.text_content("#add-host-status")
    assert "authentication failed" in status.lower()


def test_editing_after_green_probe_invalidates(page):
    pg, _ = page
    pg.click("#training-add-host-btn")
    pg.fill("#add-host-host", "user@example-host")
    pg.click("#add-host-test-btn")
    pg.wait_for_selector(".probe-check.ok", timeout=5_000)
    assert not pg.is_disabled("#add-host-save-btn")
    # Change the host after the probe — Save must drop back to disabled.
    pg.fill("#add-host-host", "user@different-host")
    assert pg.is_disabled("#add-host-save-btn")
    assert pg.locator(".probe-check").count() == 0  # checklist cleared


def test_save_adds_host_to_sidebar(page, tmp_path):
    pg, _ = page
    pg.click("#training-add-host-btn")
    pg.fill("#add-host-name", "lab-test")
    pg.fill("#add-host-host", "user@example-host")
    pg.fill("#add-host-display-name", "Lab Test Box")
    pg.click("#add-host-test-btn")
    pg.wait_for_selector(".probe-check.ok", timeout=5_000)
    pg.click("#add-host-save-btn")
    # Dialog closes; sidebar host list refreshes with the new entry.
    pg.wait_for_selector("#add-host-overlay", state="hidden", timeout=5_000)
    pg.wait_for_function(
        "document.getElementById('training-hosts-info').textContent.includes('Lab Test Box')",
        timeout=5_000,
    )
    # Backed by a real file in the (tmp) HOSTS_DIR.
    assert (tmp_path / "training_hosts" / "lab-test.json").exists()


def test_delete_host_from_sidebar(page, tmp_path):
    pg, _ = page
    # Add via API to focus this test on the delete path.
    import requests

    base = pg.url.rstrip("/")
    resp = requests.post(
        f"{base}/api/training/hosts",
        json={"name": "doomed", "host": "u@h", "display_name": "Doomed Host"},
        timeout=5,
    )
    assert resp.status_code == 201
    pg.evaluate("trainingLoadHosts()")
    pg.wait_for_function(
        "document.getElementById('training-hosts-info').textContent.includes('Doomed Host')",
        timeout=5_000,
    )
    pg.on("dialog", lambda d: d.accept())  # confirm() the removal
    pg.click("#training-hosts-info button[title='Remove this host']")
    pg.wait_for_function(
        "!document.getElementById('training-hosts-info').textContent.includes('Doomed Host')",
        timeout=5_000,
    )
    assert not (tmp_path / "training_hosts" / "doomed.json").exists()


def test_ephemeral_host_form_toggles_and_saves(page, tmp_path):
    """Selecting 'Ephemeral cloud' hides the SSH host/Test affordances,
    shows the GPU/disk/TTL fields, and Save (no probe gate) creates a
    transport-less ephemeral host."""
    pg, _ = page
    pg.click("#training-add-host-btn")
    pg.select_option("#add-host-type", value="ephemeral")
    # SSH host field + Test button hidden; ephemeral fields shown; Save live.
    assert not pg.is_visible("#add-host-ssh-fields")
    assert not pg.is_visible("#add-host-test-btn")
    assert pg.is_visible("#add-host-ephemeral-fields")
    assert not pg.is_disabled("#add-host-save-btn")

    pg.fill("#add-host-name", "neb-l40s")
    pg.select_option("#add-host-gpu", value="L40S")
    pg.fill("#add-host-ttl", "12")
    pg.click("#add-host-save-btn")
    pg.wait_for_selector("#add-host-overlay", state="hidden", timeout=5_000)
    pg.wait_for_function(
        "document.getElementById('training-hosts-info').textContent.includes('neb-l40s')",
        timeout=5_000,
    )
    # Persisted as an ephemeral profile (no SSH endpoint).
    import json as _json

    prof = _json.loads((tmp_path / "training_hosts" / "neb-l40s.json").read_text())
    assert prof["provider_id"] == "nebius"
    assert prof["gpu"] == "L40S"
    assert prof["ttl_hours"] == 12
    assert prof["ssh_host"] == ""
