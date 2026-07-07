# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the optional ``min`` / ``max`` / ``names`` keys on a feature spec.

These are the format extensions added so the GUI can:

* enforce a discrete bounded range (e.g. quality ∈ [1, 5]) at ``add_frame``
  and at GUI staging time, instead of relying on observed stats and hoping;
* render and edit categorical integer features (``names=["ee","joint"]``)
  via dropdown and per-segment colored band, with the on-disk value being
  the integer index.

Both fields are *optional* — features without them validate identically to
the pre-extension behavior.
"""

from __future__ import annotations

import numpy as np

from lerobot.datasets.feature_utils import (
    validate_feature_dtype_and_shape,
    validate_feature_numeric_bounds,
)

# ── Declared bounds (numeric min/max) ──────────────────────────────────────


class TestDeclaredBoundsValidation:
    def test_value_inside_bounds_passes(self) -> None:
        feature = {"dtype": "int64", "shape": (1,), "min": 1, "max": 5}
        assert validate_feature_dtype_and_shape("quality", feature, np.array([3], dtype=np.int64)) == ""

    def test_value_at_bounds_passes(self) -> None:
        # Bounds are inclusive on both ends — declared "1 to 5" lets you
        # pick 1 and 5. (Half-open here would surprise annotators.)
        feature = {"dtype": "int64", "shape": (1,), "min": 1, "max": 5}
        assert validate_feature_dtype_and_shape("quality", feature, np.array([1], dtype=np.int64)) == ""
        assert validate_feature_dtype_and_shape("quality", feature, np.array([5], dtype=np.int64)) == ""

    def test_value_below_min_rejected(self) -> None:
        feature = {"dtype": "int64", "shape": (1,), "min": 1, "max": 5}
        err = validate_feature_dtype_and_shape("quality", feature, np.array([0], dtype=np.int64))
        assert err
        assert "below declared min" in err
        assert "quality" in err

    def test_value_above_max_rejected(self) -> None:
        feature = {"dtype": "int64", "shape": (1,), "min": 1, "max": 5}
        err = validate_feature_dtype_and_shape("quality", feature, np.array([7], dtype=np.int64))
        assert err
        assert "above declared max" in err

    def test_only_min_declared(self) -> None:
        feature = {"dtype": "float32", "shape": (1,), "min": 0.0}
        # Below min: rejected.
        err = validate_feature_dtype_and_shape("loss", feature, np.array([-1.0], dtype=np.float32))
        assert "below declared min" in err
        # Anything ≥ 0 passes (no max declared).
        assert validate_feature_dtype_and_shape("loss", feature, np.array([1e6], dtype=np.float32)) == ""

    def test_only_max_declared(self) -> None:
        feature = {"dtype": "float32", "shape": (1,), "max": 1.0}
        err = validate_feature_dtype_and_shape("p", feature, np.array([1.5], dtype=np.float32))
        assert "above declared max" in err
        assert validate_feature_dtype_and_shape("p", feature, np.array([-1e6], dtype=np.float32)) == ""

    def test_no_bounds_declared_is_unbounded(self) -> None:
        feature = {"dtype": "float32", "shape": (1,), "names": None}
        # Should match the pre-extension "no bounds" behavior — anything
        # finite passes.
        assert validate_feature_dtype_and_shape("reward", feature, np.array([1e9], dtype=np.float32)) == ""
        assert validate_feature_dtype_and_shape("reward", feature, np.array([-1e9], dtype=np.float32)) == ""

    def test_vector_feature_checks_each_component(self) -> None:
        feature = {"dtype": "float32", "shape": (3,), "min": 0.0, "max": 1.0}
        # All inside.
        assert (
            validate_feature_dtype_and_shape("rgb_norm", feature, np.array([0.1, 0.5, 0.9], dtype=np.float32))
            == ""
        )
        # One out — error mentions one of the violations.
        err = validate_feature_dtype_and_shape(
            "rgb_norm", feature, np.array([0.1, 0.5, 1.5], dtype=np.float32)
        )
        assert "above declared max" in err

    def test_dtype_or_shape_violation_takes_precedence_over_bounds(self) -> None:
        # If the value isn't even the right shape, we surface that error
        # first — bounds checking on a wrong-shape array would just be
        # noise.
        feature = {"dtype": "float32", "shape": (1,), "min": 0.0, "max": 1.0}
        err = validate_feature_dtype_and_shape("p", feature, np.array([0.5, 0.5], dtype=np.float32))
        assert "shape" in err
        assert "below" not in err and "above" not in err


# ── Categorical (int + names) ─────────────────────────────────────────────


class TestCategoricalValidation:
    def test_legal_index_passes(self) -> None:
        feature = {"dtype": "int64", "shape": (1,), "names": ["ee", "joint"]}
        assert validate_feature_dtype_and_shape("control_mode", feature, np.array([0], dtype=np.int64)) == ""
        assert validate_feature_dtype_and_shape("control_mode", feature, np.array([1], dtype=np.int64)) == ""

    def test_negative_index_rejected(self) -> None:
        feature = {"dtype": "int64", "shape": (1,), "names": ["ee", "joint"]}
        err = validate_feature_dtype_and_shape("control_mode", feature, np.array([-1], dtype=np.int64))
        assert "outside categorical range" in err

    def test_index_at_or_above_n_classes_rejected(self) -> None:
        feature = {"dtype": "int64", "shape": (1,), "names": ["ee", "joint"]}
        err = validate_feature_dtype_and_shape("control_mode", feature, np.array([2], dtype=np.int64))
        assert "outside categorical range" in err
        assert "[0, 2)" in err

    def test_names_on_float_feature_does_not_trigger_categorical_check(self) -> None:
        # ``names`` only carries categorical semantics for INT dtype features —
        # for float vectors it's just "component names". A float feature with
        # names but no min/max should not be bound-checked.
        feature = {
            "dtype": "float32",
            "shape": (3,),
            "names": ["x", "y", "z"],
        }
        assert (
            validate_feature_numeric_bounds("pos", feature, np.array([0.5, 99.0, -50.0], dtype=np.float32))
            == ""
        )

    def test_vector_int_with_component_names_is_not_categorical(self) -> None:
        """Backward-compat regression: ``names`` on a non-scalar int feature is
        component labels (legacy LeRobot convention), NOT categorical labels.
        Without the scalar gate, an ``int64[3]`` with names=["x","y","z"] would
        be wrongly bounds-checked against [0, 3) and reject any value ≥ 3.
        """
        feature = {"dtype": "int64", "shape": (3,), "names": ["x", "y", "z"]}
        # Values way outside [0, len(names)=3) — should still pass because
        # categorical mode is gated to scalars only.
        assert (
            validate_feature_dtype_and_shape(
                "joint_indices", feature, np.array([100, 200, 300], dtype=np.int64)
            )
            == ""
        )
        # And via the inner helper directly:
        assert (
            validate_feature_numeric_bounds(
                "joint_indices", feature, np.array([100, 200, 300], dtype=np.int64)
            )
            == ""
        )

    def test_categorical_combined_with_explicit_max_uses_both(self) -> None:
        # If someone declares both names and an explicit max (unusual but
        # legal), both checks fire. This is just a defensive test — the
        # categorical check uses len(names), the explicit max uses the
        # declared value, and either failure reports.
        feature = {
            "dtype": "int64",
            "shape": (1,),
            "names": ["a", "b", "c", "d"],
            "max": 1,  # explicit max < n_classes
        }
        err = validate_feature_dtype_and_shape("x", feature, np.array([2], dtype=np.int64))
        # Either failure mode is acceptable; we just assert one fires.
        assert "above declared max" in err or "outside categorical range" in err
