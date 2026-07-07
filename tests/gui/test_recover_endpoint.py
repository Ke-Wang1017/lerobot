# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Smoke test for the GUI ``/api/robot/recover`` endpoint.

We mock the robot builder and ``recover_robot`` so this test exercises the
endpoint wiring (request validation, exception → HTTP mapping, report
serialisation) without touching real serial hardware.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import httpx
import pytest

from lerobot.gui.server import app
from lerobot.motors.recovery import RecoveryReport


@pytest.fixture
def fake_profile() -> dict:
    return {"type": "so107_follower", "port": "/dev/null-test", "cameras": {}}


def _post_recover(profile: dict) -> httpx.Response:
    async def run() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.post("/api/robot/recover", json={"robot": profile})

    return asyncio.run(run())


def test_recover_endpoint_returns_serialized_reports(fake_profile):
    fake_report = RecoveryReport(port="/dev/null-test")
    fake_report.responsive_ids = [1, 2, 3]
    fake_report.torque_disabled_ids = [1, 2, 3]

    with (
        patch("lerobot.gui.api.robot._make_robot_from_profile", return_value=MagicMock()),
        patch("lerobot.motors.recovery.recover_robot", return_value=[fake_report]),
    ):
        resp = _post_recover(fake_profile)

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["reports"] == [fake_report.to_dict()]


def test_recover_endpoint_500s_when_recover_robot_raises(fake_profile):
    """Exceptions inside the recovery flow turn into HTTP 500 with the
    exception message in the body — the GUI surfaces this to the user."""
    with (
        patch("lerobot.gui.api.robot._make_robot_from_profile", return_value=MagicMock()),
        patch(
            "lerobot.motors.recovery.recover_robot",
            side_effect=RuntimeError("boom"),
        ),
    ):
        resp = _post_recover(fake_profile)

    assert resp.status_code == 500
    assert "boom" in resp.json()["detail"]


def test_recover_endpoint_500s_when_robot_build_fails(fake_profile):
    """Profile that can't be turned into a robot (bad type, missing port,
    etc.) shouldn't 200 — the GUI must show the build error."""
    with patch(
        "lerobot.gui.api.robot._make_robot_from_profile",
        side_effect=ValueError("unknown robot type"),
    ):
        resp = _post_recover(fake_profile)

    assert resp.status_code == 500
    assert "unknown robot type" in resp.json()["detail"]
