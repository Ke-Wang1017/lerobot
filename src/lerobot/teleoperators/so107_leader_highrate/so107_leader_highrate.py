#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""SO-107 leader with a background bus-read thread.

Drop-in replacement for :class:`SO107Leader`. Connect, calibrate,
intervention, feedback — all inherited unchanged. The only behavioral
difference is that ``get_action()`` no longer reads the bus; a
background thread (started in ``connect()``) reads at
``config.read_rate_hz`` and caches the latest pose. ``get_action()``
returns the cache.

Implementation: this class is now a thin composition over
:class:`HighRateLeaderMixin` and :class:`SO107Leader`. All thread /
cache logic lives in the mixin under ``lerobot.teleoperators.highrate``.
Adding a new ``XXLeaderHighRate`` variant for different motor counts
(e.g. SO-100 6-motor — see :class:`SOLeaderHighRate`) follows the same
pattern with no code duplication.
"""

from __future__ import annotations

from ..highrate.mixin import HighRateLeaderMixin
from ..so_leader.so_leader import SO107Leader
from .config_so107_leader_highrate import SO107LeaderHighRateConfig


class SO107LeaderHighRate(HighRateLeaderMixin, SO107Leader):
    """SO-107 leader that polls the leader bus in a background thread."""

    config_class = SO107LeaderHighRateConfig
    name = "so107_leader_highrate"
