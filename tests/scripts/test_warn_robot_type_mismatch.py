"""Tests for ``warn_on_policy_robot_type_mismatch`` (CLI inference embodiment check)."""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from lerobot.common import control_utils
from lerobot.common.control_utils import warn_on_policy_robot_type_mismatch


class _StubRobot:
    def __init__(self, robot_type: str) -> None:
        self.robot_type = robot_type


def _write_train_config(path: Path, repo_id: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "train_config.json", "w") as f:
        json.dump({"dataset": {"repo_id": repo_id}}, f)


def _write_dataset_info(home: Path, repo_id: str, robot_type: str) -> None:
    info_path = home / repo_id / "meta" / "info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(info_path, "w") as f:
        json.dump({"robot_type": robot_type, "fps": 30}, f)


def test_returns_none_when_pretrained_path_is_none(caplog):
    caplog.set_level(logging.INFO)
    result = warn_on_policy_robot_type_mismatch(None, _StubRobot("bi_so107_follower"))
    assert result is None
    assert any("no pretrained_path" in rec.message for rec in caplog.records)


def test_returns_none_when_train_config_missing(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    pretrained = tmp_path / "ckpt"
    pretrained.mkdir()

    result = warn_on_policy_robot_type_mismatch(pretrained, _StubRobot("bi_so107_follower"))
    assert result is None
    assert any("could not determine" in rec.message for rec in caplog.records)


def test_returns_none_when_train_config_malformed(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    pretrained = tmp_path / "ckpt"
    pretrained.mkdir()
    (pretrained / "train_config.json").write_text("{ not json")

    result = warn_on_policy_robot_type_mismatch(pretrained, _StubRobot("bi_so107_follower"))
    assert result is None


def test_returns_none_when_dataset_not_cached(tmp_path, caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    pretrained = tmp_path / "ckpt"
    _write_train_config(pretrained, "user/some_dataset")
    monkeypatch.setattr(control_utils, "Path", Path)  # sanity
    # Point HF_LEROBOT_HOME at an empty dir so info.json lookup misses
    fake_home = tmp_path / "hf_home"
    fake_home.mkdir()
    monkeypatch.setattr("lerobot.utils.constants.HF_LEROBOT_HOME", fake_home)

    result = warn_on_policy_robot_type_mismatch(pretrained, _StubRobot("bi_so107_follower"))
    assert result is None
    assert any("not in the local cache" in rec.message for rec in caplog.records)


def test_match_logs_info_no_warning(tmp_path, caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    pretrained = tmp_path / "ckpt"
    _write_train_config(pretrained, "user/data")
    fake_home = tmp_path / "hf_home"
    _write_dataset_info(fake_home, "user/data", "bi_so107_follower")
    monkeypatch.setattr("lerobot.utils.constants.HF_LEROBOT_HOME", fake_home)

    result = warn_on_policy_robot_type_mismatch(pretrained, _StubRobot("bi_so107_follower"))
    assert result == "bi_so107_follower"
    assert any("embodiment match" in rec.message for rec in caplog.records)
    assert not any(rec.levelno == logging.WARNING for rec in caplog.records)


def test_mismatch_emits_warning_non_interactive(tmp_path, caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    pretrained = tmp_path / "ckpt"
    _write_train_config(pretrained, "user/data")
    fake_home = tmp_path / "hf_home"
    _write_dataset_info(fake_home, "user/data", "bi_so107_follower")
    monkeypatch.setattr("lerobot.utils.constants.HF_LEROBOT_HOME", fake_home)

    result = warn_on_policy_robot_type_mismatch(
        pretrained, _StubRobot("bi_so107_follower_predictive"), interactive=False
    )
    assert result == "bi_so107_follower"
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_records) == 1
    msg = warning_records[0].getMessage()
    assert "EMBODIMENT MISMATCH" in msg
    assert "bi_so107_follower" in msg
    assert "bi_so107_follower_predictive" in msg


def test_mismatch_prompts_when_interactive_yes(tmp_path, monkeypatch):
    pretrained = tmp_path / "ckpt"
    _write_train_config(pretrained, "user/data")
    fake_home = tmp_path / "hf_home"
    _write_dataset_info(fake_home, "user/data", "bi_so107_follower")
    monkeypatch.setattr("lerobot.utils.constants.HF_LEROBOT_HOME", fake_home)

    with patch("builtins.input", return_value="y"):
        result = warn_on_policy_robot_type_mismatch(
            pretrained, _StubRobot("bi_so107_follower_predictive"), interactive=True
        )
    assert result == "bi_so107_follower"


def test_mismatch_aborts_when_interactive_no(tmp_path, monkeypatch):
    pretrained = tmp_path / "ckpt"
    _write_train_config(pretrained, "user/data")
    fake_home = tmp_path / "hf_home"
    _write_dataset_info(fake_home, "user/data", "bi_so107_follower")
    monkeypatch.setattr("lerobot.utils.constants.HF_LEROBOT_HOME", fake_home)

    with patch("builtins.input", return_value=""), pytest.raises(SystemExit):
        warn_on_policy_robot_type_mismatch(
            pretrained, _StubRobot("bi_so107_follower_predictive"), interactive=True
        )


def test_interactive_auto_detects_non_tty(tmp_path, monkeypatch, caplog):
    """When stdin is not a TTY, default behaviour must NOT prompt — never block CI."""
    caplog.set_level(logging.WARNING)
    pretrained = tmp_path / "ckpt"
    _write_train_config(pretrained, "user/data")
    fake_home = tmp_path / "hf_home"
    _write_dataset_info(fake_home, "user/data", "bi_so107_follower")
    monkeypatch.setattr("lerobot.utils.constants.HF_LEROBOT_HOME", fake_home)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))  # not a TTY

    # Should not raise, should not prompt
    sentinel = "PROMPT_CALLED"
    with patch("builtins.input", side_effect=AssertionError(sentinel)):
        result = warn_on_policy_robot_type_mismatch(pretrained, _StubRobot("bi_so107_follower_predictive"))
    assert result == "bi_so107_follower"
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_interactive_respects_lerobot_noninteractive_env(tmp_path, monkeypatch):
    """LEROBOT_NONINTERACTIVE=1 must suppress the prompt even on a real TTY."""
    pretrained = tmp_path / "ckpt"
    _write_train_config(pretrained, "user/data")
    fake_home = tmp_path / "hf_home"
    _write_dataset_info(fake_home, "user/data", "bi_so107_follower")
    monkeypatch.setattr("lerobot.utils.constants.HF_LEROBOT_HOME", fake_home)
    monkeypatch.setenv("LEROBOT_NONINTERACTIVE", "1")

    # Fake a TTY by patching isatty to True
    class _FakeTTY:
        def isatty(self):
            return True

    monkeypatch.setattr("sys.stdin", _FakeTTY())

    with patch("builtins.input", side_effect=AssertionError("should not prompt")):
        result = warn_on_policy_robot_type_mismatch(pretrained, _StubRobot("bi_so107_follower_predictive"))
    assert result == "bi_so107_follower"
