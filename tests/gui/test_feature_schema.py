# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for ``DatasetInfo.features_schema`` and ``_build_features_schema``.

The schema dict is what powers the GUI's renderer registry — the front-end
dispatches widgets on (dtype, ndim) and uses ``names`` for vector-component
labels. These tests pin the JSON shape so a frontend that reads
``features_schema[name].dtype`` etc. can rely on it.
"""

from __future__ import annotations

import math

import numpy as np

from lerobot.gui.api.datasets import (
    FeatureSchema,
    _build_features_schema,
    _scalar_observed_extrema,
)


def test_scalar_features() -> None:
    features = {
        "reward": {"dtype": "float32", "shape": (1,), "names": None},
        "success": {"dtype": "bool", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    }
    schema = _build_features_schema(features)
    assert set(schema.keys()) == {"reward", "success", "frame_index"}
    assert schema["reward"].dtype == "float32"
    assert schema["reward"].shape == [1]  # tuple coerced to list for JSON
    assert schema["reward"].names is None
    assert schema["success"].dtype == "bool"


def test_vector_with_component_names() -> None:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": [
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
                "gripper",
                "j8",
                "j9",
                "j10",
                "j11",
                "j12",
                "j13",
                "j14",
            ],
        }
    }
    schema = _build_features_schema(features)
    assert schema["observation.state"].shape == [14]
    assert schema["observation.state"].names is not None
    assert len(schema["observation.state"].names) == 14
    assert schema["observation.state"].names[0] == "joint1"


def test_image_feature() -> None:
    features = {
        "observation.images.front": {
            "dtype": "image",
            "shape": [3, 480, 640],
            "names": None,
        }
    }
    schema = _build_features_schema(features)
    assert schema["observation.images.front"].dtype == "image"
    assert schema["observation.images.front"].shape == [3, 480, 640]


def test_string_feature() -> None:
    features = {"subtask": {"dtype": "string", "shape": (1,), "names": None}}
    schema = _build_features_schema(features)
    assert schema["subtask"].dtype == "string"
    # No coercion of names for scalars/strings — stays None.
    assert schema["subtask"].names is None


def test_dict_names_collapsed_to_list() -> None:
    """Some upstream features encode names as ``{"motors": [...]}`` —
    flatten to a single list so the GUI doesn't have to handle two shapes."""
    features = {
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": {"motors": ["m1", "m2", "m3", "m4", "m5", "m6"]},
        }
    }
    schema = _build_features_schema(features)
    assert schema["action"].names == ["m1", "m2", "m3", "m4", "m5", "m6"]


def test_dict_names_unflattenable_drops_to_none() -> None:
    features = {
        "weird": {
            "dtype": "float32",
            "shape": (2,),
            "names": {"a": "scalar", "b": "scalar"},  # non-list values inside dict
        }
    }
    schema = _build_features_schema(features)
    assert schema["weird"].names is None


def test_missing_dtype_or_shape_doesnt_crash() -> None:
    """``info.json`` for older datasets may have partial entries —
    we should produce a valid (if degraded) FeatureSchema."""
    features = {"weird": {}}
    schema = _build_features_schema(features)
    assert schema["weird"].dtype == ""
    assert schema["weird"].shape == []
    assert schema["weird"].names is None


def test_feature_schema_round_trips_through_json() -> None:
    features = {
        "reward": {"dtype": "float32", "shape": [1], "names": None},
        "observation.state": {"dtype": "float32", "shape": [14], "names": ["a"] * 14},
    }
    schema = _build_features_schema(features)
    # Pydantic round-trip — confirms the response is JSON-serializable.
    payload = {name: fs.model_dump() for name, fs in schema.items()}
    re_parsed = {name: FeatureSchema.model_validate(d) for name, d in payload.items()}
    assert re_parsed["reward"].dtype == "float32"
    assert re_parsed["observation.state"].shape == [14]


# ── Subtask synthesis: subtask_index + meta/subtasks.parquet → "subtask" string ──


def test_subtask_synthesis_replaces_storage_with_string_display() -> None:
    """When ``subtask_synthesis=True``, ``subtask_index`` is removed from the
    schema and replaced by a synthetic ``subtask`` (string) entry.

    The user always thinks in terms of strings. Backend rule: synthesize iff
    BOTH ``subtask_index`` is in ``features`` AND the dataset has a lookup
    table (``meta/subtasks.parquet``) — caller passes ``subtask_synthesis=True``
    when that condition is met.
    """
    features = {
        "subtask_index": {"dtype": "int64", "shape": [1], "names": None},
        "reward": {"dtype": "float32", "shape": [1], "names": None},
    }
    schema = _build_features_schema(features, subtask_synthesis=True)
    assert "subtask_index" not in schema, "storage feature must be hidden"
    assert "subtask" in schema, "display feature must be synthesized"
    sub = schema["subtask"]
    assert sub.dtype == "string"
    assert sub.shape == [1]
    assert sub.names is None
    # Reward is unrelated; should pass through.
    assert "reward" in schema
    assert schema["reward"].dtype == "float32"


def test_subtask_synthesis_disabled_keeps_storage_name() -> None:
    """``subtask_synthesis=False`` (the default for datasets without the
    lookup table) leaves ``subtask_index`` in the schema as-is."""
    features = {
        "subtask_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    schema = _build_features_schema(features, subtask_synthesis=False)
    assert "subtask" not in schema
    assert "subtask_index" in schema
    assert schema["subtask_index"].dtype == "int64"


def test_subtask_synthesis_when_no_subtask_index_is_noop() -> None:
    """Passing ``subtask_synthesis=True`` for a dataset that doesn't have
    ``subtask_index`` should NOT invent a fake ``subtask`` entry."""
    features = {"reward": {"dtype": "float32", "shape": [1], "names": None}}
    schema = _build_features_schema(features, subtask_synthesis=True)
    assert "subtask" not in schema
    assert "reward" in schema


def test_subtask_synthesis_inherits_per_episode_flag() -> None:
    """If ``subtask_index`` is detected as per-episode-broadcast, the
    synthetic ``subtask`` inherits the flag — coercion happens at the
    storage layer, but the user-facing card needs to know to render the
    'whole episode' note."""
    features = {
        "subtask_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    schema = _build_features_schema(features, per_episode={"subtask_index"}, subtask_synthesis=True)
    assert "subtask" in schema
    assert schema["subtask"].is_per_episode is True


def test_subtask_synthesis_does_not_set_per_episode_when_storage_isnt() -> None:
    features = {
        "subtask_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    schema = _build_features_schema(features, per_episode=set(), subtask_synthesis=True)
    assert schema["subtask"].is_per_episode is False


# ── Observed (dataset-wide) min/max extrema ───────────────────────────────


class TestScalarObservedExtrema:
    """``_scalar_observed_extrema`` pulls the dataset-wide ``(min, max)`` from
    ``meta/stats.json`` for scalar numeric features. The schema endpoint
    surfaces this so the GUI can show a stable ``reward [min … max]`` chip
    next to the feature name and use it as the slider scale (instead of the
    per-episode observed range, which jumps when switching episodes).
    """

    def test_pulls_min_max_from_scalar_stats(self) -> None:
        stats = {
            "reward": {
                "min": np.array([-1.5]),
                "max": np.array([0.0]),
                "mean": np.array([-0.5]),
                "std": np.array([0.3]),
                "count": np.array([100]),
            },
        }
        mn, mx = _scalar_observed_extrema(stats, "reward", [1])
        assert mn == -1.5
        assert mx == 0.0

    def test_handles_empty_shape(self) -> None:
        # Some features land with shape=[] rather than shape=[1] for scalars.
        stats = {"reward": {"min": np.array(0.0), "max": np.array(1.0)}}
        mn, mx = _scalar_observed_extrema(stats, "reward", [])
        assert mn == 0.0
        assert mx == 1.0

    def test_returns_none_for_vector_feature(self) -> None:
        # Component-wise stats exist but we deliberately don't surface them —
        # one chip per dimension would clutter the card header.
        stats = {
            "action": {
                "min": np.array([-1.0, -1.0, 0.0]),
                "max": np.array([1.0, 1.0, 0.5]),
            },
        }
        mn, mx = _scalar_observed_extrema(stats, "action", [3])
        assert mn is None
        assert mx is None

    def test_returns_none_when_feature_missing_from_stats(self) -> None:
        stats = {"reward": {"min": np.array([0.0]), "max": np.array([1.0])}}
        mn, mx = _scalar_observed_extrema(stats, "no_such_feature", [1])
        assert (mn, mx) == (None, None)

    def test_returns_none_when_stats_dict_is_none(self) -> None:
        # Older datasets / freshly-created ones may not have stats yet.
        mn, mx = _scalar_observed_extrema(None, "reward", [1])
        assert (mn, mx) == (None, None)

    def test_filters_nan_and_inf(self) -> None:
        # NaN / inf would break JSON serialization and make no sense as a
        # slider bound; replace with None so the GUI falls back to the series.
        stats = {
            "reward": {
                "min": np.array([float("nan")]),
                "max": np.array([float("inf")]),
            },
        }
        mn, mx = _scalar_observed_extrema(stats, "reward", [1])
        assert mn is None
        assert mx is None


class TestSchemaWithStats:
    """``_build_features_schema(stats=...)`` populates ``observed_min/max``
    on scalar numeric features and leaves vectors / non-numeric alone."""

    def test_scalar_feature_gets_observed_extrema(self) -> None:
        features = {"reward": {"dtype": "float32", "shape": [1], "names": None}}
        stats = {"reward": {"min": np.array([-1.0]), "max": np.array([0.5])}}
        schema = _build_features_schema(features, stats=stats)
        assert schema["reward"].observed_min == -1.0
        assert schema["reward"].observed_max == 0.5

    def test_vector_feature_has_none_observed_extrema(self) -> None:
        features = {"action": {"dtype": "float32", "shape": [6], "names": None}}
        stats = {
            "action": {
                "min": np.array([-1.0] * 6),
                "max": np.array([1.0] * 6),
            },
        }
        schema = _build_features_schema(features, stats=stats)
        assert schema["action"].observed_min is None
        assert schema["action"].observed_max is None

    def test_no_stats_dict_leaves_observed_extrema_none(self) -> None:
        features = {"reward": {"dtype": "float32", "shape": [1], "names": None}}
        schema = _build_features_schema(features, stats=None)
        assert schema["reward"].observed_min is None
        assert schema["reward"].observed_max is None

    def test_string_feature_has_none_observed_extrema(self) -> None:
        # Strings shouldn't have numeric stats; even if some appear, we don't
        # surface them.
        features = {"label": {"dtype": "string", "shape": [1], "names": None}}
        stats = {"label": {"min": np.array([0]), "max": np.array([0])}}
        schema = _build_features_schema(features, stats=stats)
        assert schema["label"].observed_min == 0  # well-formed numeric stats pass through
        # But a missing entry stays None — verifying via a separate feature:
        features2 = {"label": {"dtype": "string", "shape": [1], "names": None}}
        schema2 = _build_features_schema(features2, stats={})
        assert schema2["label"].observed_min is None

    def test_serializes_to_json_friendly_floats(self) -> None:
        # numpy float64 must coerce to Python float so pydantic + JSON happy.
        features = {"reward": {"dtype": "float32", "shape": [1], "names": None}}
        stats = {"reward": {"min": np.array([-1.0], dtype=np.float64), "max": np.array([1.0])}}
        schema = _build_features_schema(features, stats=stats)
        assert isinstance(schema["reward"].observed_min, float)
        assert isinstance(schema["reward"].observed_max, float)
        # Round-trip through pydantic + dict to make sure no numpy leaks.
        dumped = schema["reward"].model_dump()
        assert isinstance(dumped["observed_min"], float)
        assert math.isfinite(dumped["observed_min"])


# ── Declared bounds (info.json min/max → FeatureSchema.declared_min/max) ───


class TestDeclaredBoundsInSchema:
    def test_min_max_populate_declared_fields(self) -> None:
        features = {
            "quality": {"dtype": "int64", "shape": [1], "names": None, "min": 1, "max": 5},
        }
        schema = _build_features_schema(features)
        assert schema["quality"].declared_min == 1.0
        assert schema["quality"].declared_max == 5.0

    def test_only_min_declared(self) -> None:
        features = {"loss": {"dtype": "float32", "shape": [1], "names": None, "min": 0.0}}
        schema = _build_features_schema(features)
        assert schema["loss"].declared_min == 0.0
        assert schema["loss"].declared_max is None

    def test_no_bounds_declared_yields_none(self) -> None:
        features = {"reward": {"dtype": "float32", "shape": [1], "names": None}}
        schema = _build_features_schema(features)
        assert schema["reward"].declared_min is None
        assert schema["reward"].declared_max is None

    def test_non_numeric_bound_silently_dropped(self) -> None:
        # Defensive: a malformed info.json that has min: "low" shouldn't crash.
        features = {"x": {"dtype": "float32", "shape": [1], "names": None, "min": "low"}}
        schema = _build_features_schema(features)
        assert schema["x"].declared_min is None

    def test_categorical_names_pass_through(self) -> None:
        # ``names`` for an int feature is the categorical labels list — should
        # land in FeatureSchema.names so the frontend can render the dropdown.
        features = {
            "control_mode": {"dtype": "int64", "shape": [1], "names": ["ee", "joint"]},
        }
        schema = _build_features_schema(features)
        assert schema["control_mode"].names == ["ee", "joint"]

    def test_serializes_to_json_friendly_floats(self) -> None:
        features = {"q": {"dtype": "int64", "shape": [1], "names": None, "min": 1, "max": 5}}
        schema = _build_features_schema(features)
        dumped = schema["q"].model_dump()
        assert isinstance(dumped["declared_min"], float)
        assert isinstance(dumped["declared_max"], float)
