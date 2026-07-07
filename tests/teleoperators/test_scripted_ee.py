#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for ScriptedBimanualEETeleop.

Pure-Python, no hardware. Covers the surface a follower's
``attach_teleop`` matches against (Cartesian-VR-shaped action features),
the wall-clock trajectory advancement, and the ``is_exhausted`` flip
that signals the end of a single-shot trajectory to loop drivers.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from lerobot.teleoperators.scripted_ee import (
    ScriptedBimanualEETeleop,
    ScriptedBimanualEETeleopConfig,
)


def _build(shape: str = "heart", **kwargs) -> ScriptedBimanualEETeleop:
    cfg = ScriptedBimanualEETeleopConfig(
        shape=shape,
        size_m=kwargs.pop("size_m", 0.050),
        n_waypoints=kwargs.pop("n_waypoints", 8),
        ramp_ticks=kwargs.pop("ramp_ticks", 2),
        loop_hz=kwargs.pop("loop_hz", 100.0),  # fast so tests finish quickly
        **kwargs,
    )
    return ScriptedBimanualEETeleop(cfg)


def test_action_features_include_cartesian_vr_keys():
    """Follower attach_teleop probes ``left_target_x`` / ``right_target_x``."""
    t = _build()
    names = t.action_features["names"]
    for k in ("left_target_x", "left_target_y", "left_target_z", "left_gripper_pos"):
        assert k in names, f"missing {k}"
    for k in ("right_target_x", "right_target_y", "right_target_z", "right_gripper_pos"):
        assert k in names, f"missing {k}"


def test_idle_before_connect():
    """Before connect(), the teleop returns zero deltas and is not exhausted."""
    t = _build()
    assert not t.is_connected
    assert not t.is_exhausted
    raw = t.get_action_raw()
    assert raw["left_target_x"] == 0.0
    assert raw["right_target_x"] == 0.0


def test_connect_starts_trajectory_and_advances_with_wall_clock():
    """Wall-clock advancement: after connect, deltas progress through ramp."""
    t = _build(ramp_ticks=2, n_waypoints=8, loop_hz=100.0)
    t.connect()
    try:
        # Tick 0 is the very start (delta ≈ 0). With wall-clock interpolation
        # the read just after connect captures a sub-millisecond of elapsed
        # time, so allow ~1 mm absolute slack rather than exact zero.
        raw0 = t.get_action_raw()
        assert raw0["left_target_x"] == pytest.approx(0.0, abs=1e-3)
        assert raw0["left_target_y"] == pytest.approx(0.0, abs=1e-3)
        # Sleep one tick (10 ms at 100 Hz). Should be inside the ramp_in.
        # Tolerance is generous because ``time.sleep`` overshoots its
        # target and the interpolation now samples continuously rather
        # than at discrete tick boundaries — a few ms of jitter maps to a
        # few mm of slack in the ramp values.
        time.sleep(0.011)
        raw1 = t.get_action_raw()
        # Ramp is linear from 0 to offset (in robot base frame: +up + +forward).
        # With forward = (0,-1,0), up = (0,0,1), offset = (0, -0.05, +0.05),
        # at ~50-60% ramp: target_y in [-0.030, -0.010], target_z in [+0.010, +0.030].
        assert raw1["left_target_x"] == pytest.approx(0.0, abs=1e-9)
        assert -0.045 < raw1["left_target_y"] < -0.005
        assert 0.005 < raw1["left_target_z"] < 0.045
    finally:
        t.disconnect()


def test_is_exhausted_flips_after_trajectory():
    """After ramp_in + shape + ramp_out the teleop flags exhausted."""
    cfg = ScriptedBimanualEETeleopConfig(
        shape="circle", size_m=0.05, n_waypoints=4, ramp_ticks=2, loop_hz=1000.0
    )
    t = ScriptedBimanualEETeleop(cfg)
    t.connect()
    try:
        assert not t.is_exhausted
        # Total ticks = ramp_in + shape + ramp_out = 2 + 4 + 2 = 8.
        # At 1000 Hz that's 8 ms. Sleep well past it.
        time.sleep(0.030)
        assert t.is_exhausted, "trajectory should have completed"
        # After exhaustion, deltas are zeros (held at seed).
        raw = t.get_action_raw()
        assert raw["left_target_x"] == 0.0
        assert raw["left_target_y"] == 0.0
        assert raw["left_target_z"] == 0.0
    finally:
        t.disconnect()


def test_set_action_transform_applies_in_get_action():
    """Installed transform converts raw EE-deltas to whatever it returns."""
    t = _build()
    t.connect()
    try:
        # Identity transform proves the wiring; the bench installs the IK
        # transform / adapter-cache wrapper here in real usage.
        t.set_action_transform(lambda d: {**d, "left_target_x": 99.0})
        post = t.get_action()
        assert post["left_target_x"] == 99.0
        # get_action_raw bypasses the transform.
        raw = t.get_action_raw()
        assert raw["left_target_x"] != 99.0
    finally:
        t.disconnect()


def test_unknown_shape_rejected():
    """connect() raises if the shape isn't one of the three supported."""
    cfg = ScriptedBimanualEETeleopConfig(shape="triangle", size_m=0.05)
    t = ScriptedBimanualEETeleop(cfg)
    with pytest.raises(ValueError, match="unknown shape"):
        t.connect()


def test_circle_traces_a_circle_in_base_frame():
    """The circle shape, projected onto its (forward, lateral) plane, has
    constant radius around the anchor."""
    cfg = ScriptedBimanualEETeleopConfig(
        shape="circle",
        size_m=0.05,
        n_waypoints=32,
        ramp_ticks=0,
        loop_hz=1000.0,
        offset_forward_m=0.0,
        offset_up_m=0.0,
    )
    t = ScriptedBimanualEETeleop(cfg)
    t.connect()
    try:
        # Pull all 32 shape waypoints. The trajectory has no ramps so
        # tick 0..31 are the shape.
        forward = np.asarray(cfg.forward_axis, dtype=float)
        lateral = np.asarray(cfg.lateral_axis, dtype=float)
        for i in range(32):
            time.sleep(0.0012)  # > 1 ms each so the tick index advances
            raw = t.get_action_raw()
            d = np.array([raw["left_target_x"], raw["left_target_y"], raw["left_target_z"]])
            f = float(d @ forward)
            ell = float(d @ lateral)
            # Circle in local frame: center at (+r, 0), radius r. So
            # (forward - r)^2 + lateral^2 == r^2. Tolerance accommodates
            # wall-clock interpolation between adjacent waypoints — the
            # interpolated point is on a chord of the circle, with sagitta
            # ~ r * (1 - cos(2π / n)) — for n=32, ~5e-3 r ≈ 2.5e-4 m,
            # squared into the residual ≈ 2.5e-5.
            r = cfg.size_m
            residual = abs((f - r) ** 2 + ell**2 - r**2)
            assert residual < 1e-3, f"tick {i}: residual {residual:.6f}"
    finally:
        t.disconnect()
