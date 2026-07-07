# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Server-side bug-report storage for the GUI.

Each report is written to its own directory on the GUI server under
``~/.cache/lerobot/bug_reports/<timestamp>_<slug>/`` so reports can be
browsed, zipped, or deleted without parsing a single combined file.
Today the server typically runs on the operator's own machine, but the
storage is server-side either way — that distinction matters once the
GUI moves behind a hosted deployment. A future follow-up (see
``gui/TODO.md``) will add an optional "Upload to GitHub issue" action
that reuses the same directory layout.
"""

from __future__ import annotations

import base64
import json
import logging
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from lerobot.gui.state import AppState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/bug_reports", tags=["bug_reports"])

_app_state: AppState = None  # type: ignore

REPORTS_DIR = Path.home() / ".cache" / "lerobot" / "bug_reports"

# Max accepted screenshot size (base64-decoded). Guards against pathologically
# large uploads from a misbehaving / malicious client. Real screenshots of a
# 4K display compressed as PNG sit comfortably under 8 MB.
_MAX_SCREENSHOT_BYTES = 16 * 1024 * 1024

# Slug derived from the report title; bounded to keep filesystem paths sane.
_SLUG_MAX_LEN = 40


def set_app_state(state: AppState) -> None:
    global _app_state
    _app_state = state


class BugReportRequest(BaseModel):
    """Client-supplied content of a bug report.

    Preconditions:
        - ``title`` is non-empty after stripping whitespace.
        - ``screenshot_data_url``, when present, is a ``data:image/png;base64,...``
          string whose decoded payload is <= ``_MAX_SCREENSHOT_BYTES``.

    All other fields are best-effort context: missing fields are tolerated and
    serialized as ``null`` in ``report.json``.
    """

    title: str
    description: str = ""
    url: str | None = None
    user_agent: str | None = None
    viewport: dict | None = None  # {"width": int, "height": int, "dpr": float}
    active_tab: str | None = None
    screenshot_data_url: str | None = None  # "data:image/png;base64,..."
    client_extra: dict = Field(default_factory=dict)


class BugReportResponse(BaseModel):
    report_id: str
    directory: str
    screenshot_saved: bool


def _slugify(text: str) -> str:
    """Lowercase + ASCII-only + hyphenated slug, bounded length.

    Postconditions: returns a non-empty string of ``[a-z0-9-]`` characters
    no longer than ``_SLUG_MAX_LEN``. Falls back to ``"report"`` when the
    input has no usable characters.
    """
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    slug = slug[:_SLUG_MAX_LEN].rstrip("-")
    return slug or "report"


def _git_sha(repo_root: Path) -> str | None:
    """Return the short HEAD SHA of ``repo_root``, or None if it's not a git checkout."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass
    return None


def _git_dirty(repo_root: Path) -> bool | None:
    """Return True if the working tree has uncommitted changes, None on error."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if out.returncode == 0:
            return bool(out.stdout.strip())
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass
    return None


def _decode_screenshot(data_url: str) -> bytes:
    """Decode a ``data:image/png;base64,...`` URL into raw PNG bytes.

    Raises HTTPException(400) on malformed input or oversize payload.
    """
    if not data_url.startswith("data:image/"):
        raise HTTPException(status_code=400, detail="screenshot_data_url must be a data:image/* URL")
    _, _, payload = data_url.partition(",")
    if not payload:
        raise HTTPException(status_code=400, detail="screenshot_data_url has no payload")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (ValueError, base64.binascii.Error) as e:
        raise HTTPException(status_code=400, detail=f"screenshot decode failed: {e}") from e
    if len(raw) > _MAX_SCREENSHOT_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"screenshot too large: {len(raw)} bytes (max {_MAX_SCREENSHOT_BYTES})",
        )
    return raw


def _server_context() -> dict:
    """Collect server-side context that's useful for triage but tedious for users to type."""
    # __file__ = <repo>/src/lerobot/gui/api/bug_reports.py
    # parents:    0=api  1=gui  2=lerobot  3=src  4=<repo>
    repo_root = Path(__file__).resolve().parents[4]
    return {
        "git_sha": _git_sha(repo_root),
        "git_dirty": _git_dirty(repo_root),
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "repo_root": str(repo_root),
    }


@router.post("", response_model=BugReportResponse)
async def submit_bug_report(req: BugReportRequest) -> BugReportResponse:
    """Persist a bug report to the local filesystem.

    Storage layout: ``REPORTS_DIR/<timestamp>_<slug>/`` containing
    ``report.json`` (everything except the image) and ``screenshot.png``
    (when supplied).

    Returns the report id and absolute directory path so the caller can
    surface them to the user.
    """
    title = req.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="title is required")

    # Decode the screenshot up-front so a bad/oversize payload fails before
    # we touch the filesystem — otherwise a rejected request leaves an empty
    # report directory behind.
    png_bytes: bytes | None = None
    if req.screenshot_data_url:
        png_bytes = _decode_screenshot(req.screenshot_data_url)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = _slugify(title)
    report_id = f"{timestamp}_{slug}"

    out_dir = REPORTS_DIR / report_id
    # Extremely unlikely collision (two reports in the same second with the
    # same slug), but cheap to guard against — suffix with a counter.
    if out_dir.exists():
        for i in range(2, 100):
            candidate = REPORTS_DIR / f"{report_id}-{i}"
            if not candidate.exists():
                out_dir = candidate
                report_id = candidate.name
                break
        else:
            raise HTTPException(status_code=500, detail="could not find a unique report directory")

    out_dir.mkdir(parents=True, exist_ok=False)

    screenshot_saved = False
    if png_bytes is not None:
        (out_dir / "screenshot.png").write_bytes(png_bytes)
        screenshot_saved = True

    report = {
        "schema_version": 1,
        "id": report_id,
        "submitted_at": datetime.now().isoformat(timespec="seconds"),
        "title": title,
        "description": req.description,
        "client": {
            "url": req.url,
            "user_agent": req.user_agent,
            "viewport": req.viewport,
            "active_tab": req.active_tab,
            "extra": req.client_extra,
        },
        "server": _server_context(),
        "screenshot": "screenshot.png" if screenshot_saved else None,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    logger.info("Bug report saved: %s", out_dir)
    return BugReportResponse(
        report_id=report_id,
        directory=str(out_dir),
        screenshot_saved=screenshot_saved,
    )


@router.get("")
async def list_bug_reports() -> dict:
    """List existing bug reports (id, title, submitted_at). Newest first.

    Mainly useful for a future "show my recent reports" panel — kept minimal
    for now: returns just the metadata, not the screenshots.
    """
    if not REPORTS_DIR.exists():
        return {"reports": []}
    items: list[dict] = []
    for d in sorted(REPORTS_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        report_file = d / "report.json"
        if not report_file.exists():
            continue
        try:
            data = json.loads(report_file.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("Skipping unreadable bug report at %s", d, exc_info=True)
            continue
        items.append(
            {
                "id": data.get("id", d.name),
                "title": data.get("title", ""),
                "submitted_at": data.get("submitted_at"),
                "has_screenshot": data.get("screenshot") is not None,
                "directory": str(d),
            }
        )
    return {"reports": items}
