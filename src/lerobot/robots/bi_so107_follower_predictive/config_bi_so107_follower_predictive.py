#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass

from ..bi_so107_follower.config_bi_so107_follower import BiSO107FollowerConfig
from ..config import RobotConfig
from ..predictive.config import PredictiveControllerConfig


@RobotConfig.register_subclass("bi_so107_follower_predictive")
@dataclass
class BiSO107FollowerPredictiveConfig(PredictiveControllerConfig, BiSO107FollowerConfig):
    """Config for the bimanual SO-107 follower with predictive-lookahead.

    Inherits every field of ``BiSO107FollowerConfig`` (per-arm ports,
    cameras, ``camera_read_strategy``, ``dry_run``, etc.) — recorded
    ``bi_so107_follower`` profiles and ``.trajectory.json`` files
    load against this config unchanged.

    Controller knobs come from :class:`PredictiveControllerConfig` and
    are shared by both arms. Splitting them per-arm is a small follow-up
    if asymmetric tuning is ever needed; until then keeping them shared
    keeps the YAML / CLI surface small and matches the prototype's
    symmetric tuning on bi_so107 + white profile.

    Why this is registered as a distinct ``robot_type``: dataset action /
    state alignment differs from ``bi_so107_follower`` (controller
    compensates motor τ → ``state(t) ≈ intent(t)`` instead of
    ``state(t) ≈ leader(t − τ)``). Treating it as a separate embodiment
    prevents accidental mixing of the two recording regimes in training.
    See ``SO107FollowerPredictiveRobotConfig`` for the long-form rationale.
    """
