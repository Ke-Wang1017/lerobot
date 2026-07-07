# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the optional ``per_episode: true`` flag on a feature spec.

This is a small format extension: a feature whose info.json entry has
``"per_episode": true`` is treated as episode-wide (same value across
every frame of any given episode). The flag changes interpretation,
not on-disk layout — the column is still per-frame in
``data/chunk-*/file-*.parquet``.

Three layers of behavior pinned here:

1. **Validation**: rejects ``per_episode: true`` on dtypes/shapes where
   it doesn't make sense (image/video/non-scalar).
2. **Write-time enforcement**: the writer's ``add_frame`` raises if a
   frame's value for a declared per-episode feature disagrees with the
   value pinned by the first frame of that episode.
3. **Helper predicates**: ``is_per_episode_declared(ft)`` is the canonical
   reader-side test.
"""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.datasets.feature_utils import (
    is_per_episode_declared,
    validate_per_episode_flag,
)

# ── Helper predicate ───────────────────────────────────────────────────────


class TestIsPerEpisodeDeclared:
    def test_returns_true_when_flag_set(self) -> None:
        ft = {"dtype": "bool", "shape": [1], "names": None, "per_episode": True}
        assert is_per_episode_declared(ft) is True

    def test_returns_false_when_flag_absent(self) -> None:
        ft = {"dtype": "bool", "shape": [1], "names": None}
        assert is_per_episode_declared(ft) is False

    def test_returns_false_when_flag_explicitly_false(self) -> None:
        ft = {"dtype": "bool", "shape": [1], "names": None, "per_episode": False}
        assert is_per_episode_declared(ft) is False

    def test_only_true_value_counts(self) -> None:
        # Defensive: a truthy-but-not-True value (e.g. 1, "yes") should not
        # silently activate the flag. Strict identity comparison.
        ft = {"dtype": "bool", "shape": [1], "names": None, "per_episode": 1}
        assert is_per_episode_declared(ft) is False


# ── validate_per_episode_flag (info.json schema check) ─────────────────────


class TestValidatePerEpisodeFlag:
    def test_passes_when_flag_absent(self) -> None:
        ft = {"dtype": "float32", "shape": [1], "names": None}
        assert validate_per_episode_flag("reward", ft) == ""

    def test_passes_for_scalar_bool(self) -> None:
        ft = {"dtype": "bool", "shape": [1], "names": None, "per_episode": True}
        assert validate_per_episode_flag("success", ft) == ""

    def test_passes_for_scalar_int(self) -> None:
        ft = {"dtype": "int64", "shape": [1], "names": ["ee", "joint"], "per_episode": True}
        assert validate_per_episode_flag("control_mode", ft) == ""

    def test_passes_for_empty_shape(self) -> None:
        # Shape [] is also legal for a scalar.
        ft = {"dtype": "bool", "shape": [], "names": None, "per_episode": True}
        assert validate_per_episode_flag("success", ft) == ""

    def test_rejects_image(self) -> None:
        ft = {"dtype": "image", "shape": [3, 96, 96], "names": None, "per_episode": True}
        err = validate_per_episode_flag("camera", ft)
        assert err
        assert "per_episode" in err
        assert "image" in err

    def test_rejects_video(self) -> None:
        ft = {"dtype": "video", "shape": [3, 480, 640], "names": None, "per_episode": True}
        err = validate_per_episode_flag("camera", ft)
        assert err
        assert "video" in err

    def test_rejects_vector(self) -> None:
        ft = {"dtype": "float32", "shape": [14], "names": None, "per_episode": True}
        err = validate_per_episode_flag("state", ft)
        assert err
        assert "shape" in err
        assert "[14]" in err

    def test_rejects_matrix(self) -> None:
        ft = {"dtype": "float32", "shape": [3, 3], "names": None, "per_episode": True}
        err = validate_per_episode_flag("rotation", ft)
        assert err
        assert "shape" in err


# ── Feature-name validator integration ─────────────────────────────────────


class TestValidateFeatureNamesIntegration:
    """``_validate_feature_names`` (called by ``LeRobotDatasetMetadata.create``)
    should reject malformed per_episode declarations alongside the existing
    name-character check, so a recording session can't accidentally ship an
    invalid declaration."""

    def test_rejects_per_episode_on_vector_feature(self) -> None:
        from lerobot.utils.feature_utils import _validate_feature_names

        features = {
            "state": {"dtype": "float32", "shape": [14], "names": None, "per_episode": True},
        }
        with pytest.raises(ValueError, match="per_episode"):
            _validate_feature_names(features)

    def test_accepts_valid_declarations(self) -> None:
        from lerobot.utils.feature_utils import _validate_feature_names

        features = {
            "success": {"dtype": "bool", "shape": [1], "names": None, "per_episode": True},
            "reward": {"dtype": "float32", "shape": [1], "names": None},
        }
        _validate_feature_names(features)  # should not raise


# ── Write-time enforcement in DatasetWriter.add_frame ──────────────────────


class TestAddFrameEnforcesPerEpisodeUniformity:
    """``add_frame`` rejects calls that violate a declared per-episode
    invariant. The error fires at the violating call site so the recording
    pipeline author sees exactly which call set the second value, instead
    of a parquet-write failure later."""

    def test_first_frame_pins_value_then_consistent_frames_pass(
        self, tmp_path, empty_lerobot_dataset_factory
    ):
        ds = empty_lerobot_dataset_factory(
            root=tmp_path / "ds",
            features={
                "state": {"dtype": "float32", "shape": (2,), "names": None},
                "success": {
                    "dtype": "bool",
                    "shape": (1,),
                    "names": None,
                    "per_episode": True,
                },
            },
        )
        # 3 frames, all matching value — should not raise.
        for _ in range(3):
            ds.add_frame(
                {"state": np.zeros(2, dtype=np.float32), "success": np.array([True], dtype=bool), "task": "t"}
            )

    def test_inconsistent_value_raises(self, tmp_path, empty_lerobot_dataset_factory):
        ds = empty_lerobot_dataset_factory(
            root=tmp_path / "ds",
            features={
                "state": {"dtype": "float32", "shape": (2,), "names": None},
                "success": {
                    "dtype": "bool",
                    "shape": (1,),
                    "names": None,
                    "per_episode": True,
                },
            },
        )
        ds.add_frame(
            {"state": np.zeros(2, dtype=np.float32), "success": np.array([True], dtype=bool), "task": "t"}
        )
        with pytest.raises(ValueError, match="per_episode"):
            ds.add_frame(
                {
                    "state": np.zeros(2, dtype=np.float32),
                    "success": np.array([False], dtype=bool),
                    "task": "t",
                }
            )

    def test_error_message_identifies_feature_and_values(self, tmp_path, empty_lerobot_dataset_factory):
        ds = empty_lerobot_dataset_factory(
            root=tmp_path / "ds",
            features={
                "state": {"dtype": "float32", "shape": (2,), "names": None},
                "control_mode": {
                    "dtype": "int64",
                    "shape": (1,),
                    "names": ["ee", "joint"],
                    "per_episode": True,
                },
            },
        )
        ds.add_frame(
            {
                "state": np.zeros(2, dtype=np.float32),
                "control_mode": np.array([0], dtype=np.int64),
                "task": "t",
            }
        )
        with pytest.raises(ValueError) as exc_info:
            ds.add_frame(
                {
                    "state": np.zeros(2, dtype=np.float32),
                    "control_mode": np.array([1], dtype=np.int64),
                    "task": "t",
                }
            )
        msg = str(exc_info.value)
        assert "control_mode" in msg
        # The error names both the pinned and the offending value.
        assert "0" in msg and "1" in msg

    def test_undeclared_feature_can_vary_freely(self, tmp_path, empty_lerobot_dataset_factory):
        """Sanity / back-compat: features without the flag are NOT bound
        by the consistency check, so an existing per-frame ``reward`` keeps
        accepting different values across frames."""
        ds = empty_lerobot_dataset_factory(
            root=tmp_path / "ds",
            features={
                "state": {"dtype": "float32", "shape": (2,), "names": None},
                "reward": {"dtype": "float32", "shape": (1,), "names": None},
            },
        )
        # Reward changes per frame — should not raise.
        for r in (0.1, 0.2, 0.3):
            ds.add_frame(
                {
                    "state": np.zeros(2, dtype=np.float32),
                    "reward": np.array([r], dtype=np.float32),
                    "task": "t",
                }
            )

    def test_value_match_is_dtype_tolerant(self, tmp_path, empty_lerobot_dataset_factory):
        """Two array constructions of the same value (different in-memory
        layouts) should NOT be flagged as inconsistent. The consistency
        check uses ``np.array_equal`` so differences in stride / dtype-of-
        the-array but same scalar content pass."""
        ds = empty_lerobot_dataset_factory(
            root=tmp_path / "ds",
            features={
                "state": {"dtype": "float32", "shape": (2,), "names": None},
                "success": {
                    "dtype": "bool",
                    "shape": (1,),
                    "names": None,
                    "per_episode": True,
                },
            },
        )
        # Two equivalent arrays — same value, different construction path.
        ds.add_frame(
            {
                "state": np.zeros(2, dtype=np.float32),
                "success": np.array([True], dtype=bool),
                "task": "t",
            }
        )
        ds.add_frame(
            {
                "state": np.zeros(2, dtype=np.float32),
                "success": np.array([True], dtype=bool).copy(),
                "task": "t",
            }
        )
