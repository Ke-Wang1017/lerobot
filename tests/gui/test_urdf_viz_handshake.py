# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Static-analysis guard for the urdf_viz iframe's failure surfaces.

These tests pin the small set of invariants the data tab's URDF
viewer iframe relies on for not silently hanging in the operator's
face. They're intentionally static-analysis (regex against the
JS/HTML sources) rather than a Playwright end-to-end — fast,
deterministic, no browser needed, and they catch the specific
failure modes that have actually bitten in the wild.

Tests here:

* ``test_iframe_surfaces_webgl_unavailable`` — the try/catch around
  ``new THREE.WebGLRenderer(...)``. Without it, a browser that's
  lost its WebGL context (Chrome's GPU process crashed, hardware
  acceleration off, etc.) sees the iframe stuck on its initial
  ``starting…`` status forever with zero indication of why
  (observed 2026-05-30 after a kernel update left NVIDIA out of
  sync; cost ~1h to diagnose).
* ``test_iframe_starts_with_starting_status`` — the initial visible
  text is ``starting…``. Pinning the literal keeps the regression-
  symptom docs ("if you see 'starting…' forever, …") aligned with
  the code.
* ``test_parent_loads_iframe_with_cache_buster`` — ``iframe.src``
  carries a ``?v=N`` query param. Without it, when we ship a new
  ``urdf_viz.html`` browsers serve the cached old version (URLs
  are identical) and operators see stale broken code. Bump the
  version whenever the iframe script changes materially.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_STATIC = Path(__file__).resolve().parents[2] / "src" / "lerobot" / "gui" / "static"
_URDF_VIZ_HTML = _STATIC / "urdf_viz.html"
_APP_JS = _STATIC / "app.js"


@pytest.fixture(scope="module")
def urdf_viz_html() -> str:
    """Source of ``urdf_viz.html`` (the iframe document)."""
    return _URDF_VIZ_HTML.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def app_js() -> str:
    """Source of ``app.js`` (the parent page driving the data tab)."""
    return _APP_JS.read_text(encoding="utf-8")


def test_iframe_surfaces_webgl_unavailable(urdf_viz_html: str) -> None:
    """``THREE.WebGLRenderer`` MUST be guarded against context-create failure.

    ``new THREE.WebGLRenderer()`` throws synchronously when the browser
    can't grant a WebGL context — hardware acceleration off, sandboxed
    GPU, or driver issue. Without a try/catch around the constructor the
    exception propagates up, the rest of the module script never runs,
    and the operator sees the initial ``starting…`` status forever
    (observed 2026-05-30: Chrome with GL_VENDOR=Disabled, Sandboxed=yes).

    Pinning the guard's shape here keeps a future refactor from
    "simplifying" it away — if the constructor stops being wrapped, the
    failure mode reverts to the silent hang. We don't pin the exact
    user-visible message (it should evolve as a UX concern), only that
    the constructor is wrapped AND setStatus is called from the catch.
    """
    # Pattern: try { ... WebGLRenderer ... } catch { ... setStatus ... }.
    # Allow any whitespace, any other code inside both halves; pin only
    # the constructor + the setStatus call inside the same try-catch
    # using a lazy DOTALL match.
    pattern = re.compile(
        r"try\s*\{[^{}]*?new\s+THREE\.WebGLRenderer[\s\S]*?\}\s*catch[\s\S]*?setStatus\s*\(",
        re.DOTALL,
    )
    assert pattern.search(urdf_viz_html) is not None, (
        "THREE.WebGLRenderer construction isn't inside a try/catch that "
        "calls setStatus on failure. If WebGL is unavailable the script "
        "throws here, init() never runs, and the iframe hangs on the "
        "initial 'starting…' status with no actionable feedback. See the "
        "2026-05-30 bug for the regression-symptom precedent."
    )


def test_iframe_starts_with_starting_status(urdf_viz_html: str) -> None:
    """The initial visible status MUST be ``starting…``.

    This is what the user sees while the module script is loading. If
    the iframe stays on this string after a few seconds, either the
    iframe URL 404'd or the script aborted before reaching its first
    ``setStatus(...)`` call. Pinning the literal here means a future
    refactor that changes the initial string also has to update the
    regression-symptom docs.
    """
    assert "starting" in urdf_viz_html.lower(), (
        "urdf_viz.html no longer ships with a 'starting…' initial status; "
        "update the test (and the user-facing bug-report templates) if "
        "you intentionally changed the placeholder."
    )


def test_parent_loads_iframe_with_cache_buster(app_js: str) -> None:
    """``iframe.src`` MUST carry a ``?v=N`` query param.

    Without the cache-buster, when we ship a new ``urdf_viz.html`` the
    browser serves the cached old version (URLs are identical), and the
    operator sees stale code with whatever bugs that version had —
    exactly the failure mode where "yesterday was fine, today is stuck"
    is reported. Bump this version whenever ``urdf_viz.html``'s script
    changes materially.
    """
    pattern = re.compile(
        r"urdf_viz\.html\?[^'\"`\s]*\bv=\d+",
        re.MULTILINE,
    )
    assert pattern.search(app_js) is not None, (
        "app.js no longer cache-busts urdf_viz.html via ?v=N — browsers "
        "with the old script cached will not pick up new fixes."
    )
