#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, TypedDict

import numpy as np
import torch


class TransitionKey(str, Enum):
    """Keys for accessing EnvTransition dictionary components."""

    # TODO(Steven): Use consts
    OBSERVATION = "observation"
    ACTION = "action"
    REWARD = "reward"
    DONE = "done"
    TRUNCATED = "truncated"
    INFO = "info"
    COMPLEMENTARY_DATA = "complementary_data"


PolicyAction = torch.Tensor
RobotAction = dict[str, Any]
EnvAction = np.ndarray
RobotObservation = dict[str, Any]
BatchType = dict[str, Any]


@dataclass(frozen=True)
class ActionChunk:
    """A fixed-cadence horizon of intent samples for ``Robot.send_action``.

    Conceptual model: ``frames[0]`` is the action that should be applied
    at the receiver's "now" (the wall-clock moment ``send_action`` runs).
    ``frames[k]`` is the action for ``now + k / fps``. Frame indices are
    intentionally NOT timestamped — the receiver stamps its own "now"
    when the chunk arrives, so any send→receive delivery delay becomes
    invisible to the lookup. This is robust in a way that absolute
    timestamps (which assume a shared clock and zero delivery delay) and
    relative offsets (which need a separate "anchor moment") are not.

    Intended senders:
      * ``TrajectoryReplayTeleop`` — slices a window from the recorded
        trajectory starting at the current playback position.
      * Chunked policy inference — emits the action chunk that the
        policy produced, indexed from the current control tick.

    Intended receivers:
      * ``SO107FollowerPredictive`` — interpolates the lookahead target
        at index ``L * fps``; falls back to chunk-tail velocity
        extrapolation past the last frame.
      * Plain robots — default ``Robot.send_action`` unpacks ``frames[0]``
        as the single intent; chunk semantics are opt-in.

    Postconditions on construction:
      * ``fps > 0`` — required for the index→time mapping.
      * ``frames`` non-empty — frame 0 is the canonical "current intent",
        recorded in the dataset and returned from ``send_action``.
      * Every frame must carry the same key set (validated at the
        receiver where the motor map is known, not here).
    """

    fps: float
    frames: tuple[dict[str, float], ...]

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError(f"ActionChunk.fps must be positive, got {self.fps}")
        if not self.frames:
            raise ValueError("ActionChunk.frames must be non-empty")


def action_first_frame(action: RobotAction | ActionChunk) -> RobotAction:
    """Return the implicit-now intent dict.

    Robots that don't natively consume ``ActionChunk`` call this at the
    top of ``send_action`` for back-compat — the chunk's other frames
    are discarded and the robot behaves as if a single intent was sent.
    Predictive / chunk-aware robots use the chunk directly instead of
    going through this helper.
    """
    if isinstance(action, ActionChunk):
        return dict(action.frames[0])
    return action


EnvTransition = TypedDict(
    "EnvTransition",
    {
        TransitionKey.OBSERVATION.value: RobotObservation | None,
        TransitionKey.ACTION.value: PolicyAction | RobotAction | EnvAction | None,
        TransitionKey.REWARD.value: float | torch.Tensor | None,
        TransitionKey.DONE.value: bool | torch.Tensor | None,
        TransitionKey.TRUNCATED.value: bool | torch.Tensor | None,
        TransitionKey.INFO.value: dict[str, Any] | None,
        TransitionKey.COMPLEMENTARY_DATA.value: dict[str, Any] | None,
    },
)
