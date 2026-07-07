#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for BiSO107Follower's camera read strategy."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lerobot.robots.bi_so107_follower.bi_so107_follower import BiSO107Follower
from lerobot.robots.bi_so107_follower.config_bi_so107_follower import BiSO107FollowerConfig


class TestCameraReadStrategy:
    """``camera_read_strategy`` selects between ``read_latest`` (default,
    non-blocking) and ``async_read`` (blocks until next fresh frame). The
    default-on choice is ``latest`` because in a multi-camera setup the
    first camera iterated under ``async_read`` pays the full producer-
    period wait while subsequent cameras' grab threads have already
    cached fresh frames — the result is one super-fresh camera and the
    others randomly stale, plus a halved control rate."""

    def test_config_default_is_latest(self):
        """Robots created without explicitly setting the strategy must
        default to ``latest`` so existing call sites pick up the better
        behaviour automatically."""
        cfg = BiSO107FollowerConfig(left_arm_port="/dev/null", right_arm_port="/dev/null")
        assert cfg.camera_read_strategy == "latest"

    def _read_via_strategy(self, strategy: str) -> tuple[MagicMock, object]:
        """Drive ``BiSO107Follower._read_camera_frame`` with a mock camera
        and report which method was invoked."""
        cam = MagicMock(name="MockCamera")
        cam.read_latest.return_value = "latest_frame"
        cam.async_read.return_value = "async_frame"
        # Bind the unbound method to a SimpleNamespace standing in for the
        # robot — only ``self.config.camera_read_strategy`` is read.
        fake_self = SimpleNamespace(config=SimpleNamespace(camera_read_strategy=strategy))
        result = BiSO107Follower._read_camera_frame(fake_self, cam)
        return cam, result

    def test_latest_calls_read_latest(self):
        cam, result = self._read_via_strategy("latest")
        cam.read_latest.assert_called_once()
        cam.async_read.assert_not_called()
        assert result == "latest_frame"

    def test_wait_for_new_calls_async_read(self):
        cam, result = self._read_via_strategy("wait_for_new")
        cam.async_read.assert_called_once()
        cam.read_latest.assert_not_called()
        assert result == "async_frame"

    def test_latest_falls_back_to_async_on_first_frame(self):
        """``read_latest`` raises until the grab thread has produced its
        first frame. ``_read_camera_frame`` must catch this and block on
        ``async_read`` once so the buffer is populated before subsequent
        ``read_latest`` calls succeed."""
        cam = MagicMock(name="MockCamera")
        cam.read_latest.side_effect = RuntimeError("has not captured any frames yet")
        cam.async_read.return_value = "first_frame"
        fake_self = SimpleNamespace(config=SimpleNamespace(camera_read_strategy="latest"))
        result = BiSO107Follower._read_camera_frame(fake_self, cam)
        cam.read_latest.assert_called_once()
        cam.async_read.assert_called_once()
        assert result == "first_frame"

    def test_unknown_strategy_treated_as_wait_for_new(self):
        """An unrecognised string falls through to ``async_read`` rather
        than failing — preserves the loop in the face of a typo'd config."""
        cam, result = self._read_via_strategy("nonsense_typo")
        cam.async_read.assert_called_once()
        cam.read_latest.assert_not_called()
        assert result == "async_frame"


class TestConfigField:
    """The config field is plain dataclass storage — the test exists so
    a future refactor can't silently rename it without breaking robots
    in the wild."""

    def test_field_can_be_overridden(self):
        cfg = BiSO107FollowerConfig(
            left_arm_port="/dev/null",
            right_arm_port="/dev/null",
            camera_read_strategy="wait_for_new",
        )
        assert cfg.camera_read_strategy == "wait_for_new"

    def test_field_documented_values(self):
        """At least the two documented values must construct cleanly."""
        for value in ("latest", "wait_for_new"):
            cfg = BiSO107FollowerConfig(
                left_arm_port="/dev/null",
                right_arm_port="/dev/null",
                camera_read_strategy=value,
            )
            assert cfg.camera_read_strategy == value


class TestDryRun:
    """``dry_run`` makes ``send_action`` a no-op so automated tooling can
    exercise the full control-loop stack (including get_observation timing
    and any policy under test) without physically driving the motors. Off
    by default — production callers see no change."""

    def test_config_default_is_false(self):
        cfg = BiSO107FollowerConfig(left_arm_port="/dev/null", right_arm_port="/dev/null")
        assert cfg.dry_run is False

    def test_send_action_normal_path_calls_arm_send(self):
        """With dry_run=False, send_action must dispatch to both arms."""
        fake_self = SimpleNamespace(
            config=SimpleNamespace(dry_run=False),
            left_arm=MagicMock(),
            right_arm=MagicMock(),
        )
        fake_self.left_arm.send_action.return_value = {"shoulder.pos": 1.0}
        fake_self.right_arm.send_action.return_value = {"shoulder.pos": 2.0}
        action = {"left_shoulder.pos": 1.0, "right_shoulder.pos": 2.0}
        result = BiSO107Follower.send_action(fake_self, action)
        fake_self.left_arm.send_action.assert_called_once()
        fake_self.right_arm.send_action.assert_called_once()
        assert "left_shoulder.pos" in result and "right_shoulder.pos" in result

    def test_send_action_dry_run_short_circuits(self):
        """With dry_run=True, neither arm's send_action is called and the
        requested action is returned unchanged."""
        fake_self = SimpleNamespace(
            config=SimpleNamespace(dry_run=True),
            left_arm=MagicMock(),
            right_arm=MagicMock(),
        )
        action = {"left_shoulder.pos": 1.0, "right_shoulder.pos": 2.0}
        result = BiSO107Follower.send_action(fake_self, action)
        fake_self.left_arm.send_action.assert_not_called()
        fake_self.right_arm.send_action.assert_not_called()
        assert result == action

    def test_dry_run_warns_only_once(self, caplog):
        """Logging a WARNING on every iteration would spam at 30 Hz. The
        first call logs; subsequent calls stay quiet."""
        fake_self = SimpleNamespace(
            config=SimpleNamespace(dry_run=True),
            left_arm=MagicMock(),
            right_arm=MagicMock(),
        )
        with caplog.at_level("WARNING", logger="lerobot.robots.bi_so107_follower.bi_so107_follower"):
            for _ in range(10):
                BiSO107Follower.send_action(fake_self, {"left_shoulder.pos": 0.0})
        n_warns = sum(1 for r in caplog.records if "dry_run" in r.message)
        assert n_warns == 1, f"Expected one warning, got {n_warns}"

    def test_dry_run_can_be_toggled_via_config(self):
        cfg = BiSO107FollowerConfig(left_arm_port="/dev/null", right_arm_port="/dev/null", dry_run=True)
        assert cfg.dry_run is True


@pytest.mark.skip(
    reason="Constructing BiSO107Follower without hardware needs a substantial mock "
    "setup (motor bus + camera factory). The strategy logic is covered by "
    "TestCameraReadStrategy via _read_camera_frame; full get_observation "
    "wiring was validated by an end-to-end dry run on the white profile."
)
class TestGetObservationIntegration:
    """Placeholder for a full get_observation integration test that would
    construct the robot end-to-end with mocks. Skipped because the value
    over the unit tests above is small."""

    def test_placeholder(self): ...
