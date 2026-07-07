#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Static regression guards for two reported URDF-viz bugs.

These are static checks (substring / regex on the served JS) rather than
full browser-driven assertions because a Playwright-style test for each
behaviour costs ~80-120 lines of fixture for a 5-10 line fix. Static
guards catch the most common regression mode: a refactor that drops the
key persistence / rendering line. A reviewer fully restoring the bug
would have to also delete the guard comment text — that's the contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_STATIC_DIR = Path(__file__).resolve().parents[2] / "src" / "lerobot" / "gui" / "static"


@pytest.fixture(scope="module")
def app_js() -> str:
    return (_STATIC_DIR / "app.js").read_text()


@pytest.fixture(scope="module")
def urdf_viz_html() -> str:
    return (_STATIC_DIR / "urdf_viz.html").read_text()


# ── Issue 1: ghost toggle persists across episode changes ─────────────────


def test_parent_persists_ghost_toggle_in_session_storage(app_js: str):
    """selectEpisode rebuilds the camera grid via ``grid.innerHTML``,
    destroying the iframe and its module-level ``_ghostOn``. To survive
    that, the parent must store the toggle in sessionStorage (or
    equivalent) when the iframe posts an ``urdfGhostChanged`` message.

    Regression for: "toggle not sticky between episodes."
    """
    assert "urdfGhostChanged" in app_js, (
        "parent must listen for the urdfGhostChanged postMessage from the iframe"
    )
    assert "sessionStorage.setItem('urdfGhost'" in app_js, (
        "parent must persist the toggle state in sessionStorage on change"
    )


def test_parent_reads_persisted_ghost_when_constructing_iframe_src(app_js: str):
    """When a new iframe is created for a new episode, its ``?ghost=``
    URL param must come from the persisted preference, not just the
    parent URL's bookmark default."""
    assert "_urdfGhostPref()" in app_js, (
        "_probeAndAttachUrdfViz must call the persistence-aware preference helper"
    )
    # The helper itself must consult sessionStorage before the URL default.
    assert "sessionStorage.getItem('urdfGhost')" in app_js, (
        "_urdfGhostPref must read the persisted value from sessionStorage"
    )


def test_iframe_posts_ghost_change_to_parent(urdf_viz_html: str):
    """The iframe's toggle handler must inform the parent so the parent
    can persist the new value. Without this, the iframe's local
    ``_ghostOn`` change dies with the iframe on the next episode."""
    assert "urdfGhostChanged" in urdf_viz_html, (
        "iframe ghost-toggle click handler must postMessage urdfGhostChanged"
    )


# ── Issue 2: full-body ghost no longer shown when tube fallback hits in dataset mode ─


def test_tube_fallback_hides_ghost_in_dataset_mode(urdf_viz_html: str):
    """When the trajectory can't render (arm static, end-of-episode
    slice = single point), the previous behaviour fell back to making
    the full body ghost visible. In dataset mode the state robot is
    already animating the current pose, so the redundant full-body
    ghost looked like a bug ("one arm becomes ghost suddenly"). Hide
    it in dataset mode; keep the fallback in live mode where the ghost
    IS the overlay's only signal.

    Regression for: "one arm becomes full body ghost rather than the
    tube" and the static-arm variant.
    """
    # The fallback visibility must depend on MODE, not just _ghostOn.
    assert "fallbackGhostVisible" in urdf_viz_html, (
        "_drawTubeFromPoints must compute a mode-aware fallback visibility"
    )
    assert "MODE !== 'dataset'" in urdf_viz_html, "the fallback must hide ghost in dataset mode"
