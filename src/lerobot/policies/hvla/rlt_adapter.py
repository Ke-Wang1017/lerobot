# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""RLT policy adapter for HVLA S1 (flow-matching).

Lives in the ``hvla`` package rather than ``rlt`` because it depends on the
concrete ``FlowMatchingS1Policy``; ``rlt`` stays policy-agnostic. This is the
reference implementation of :class:`RLTPolicyAdapter` and preserves the exact
operations the pre-adapter RLT integration performed against the S1 policy.
"""

from __future__ import annotations

from torch import Tensor

from lerobot.policies.rlt.policy_adapter import RLTPolicyAdapter


class HVLAS1Adapter(RLTPolicyAdapter):
    """Adapt a ``FlowMatchingS1Policy`` to the RLT interface.

    S1 stores normalization stats on the policy instance (``_state_mean`` etc.)
    and exposes ``encode_observations`` on the inner ``.model``; ``context_dim``
    is the S1 ``hidden_dim``. The reference chunk comes from
    ``predict_action_chunk``, which accepts a precomputed ``context`` so the
    DINOv2 forward runs once per inference step.
    """

    def __init__(self, policy):
        self._policy = policy

    @property
    def policy(self):
        """The wrapped S1 policy (for call sites that still need it directly)."""
        return self._policy

    # --- shape metadata ---
    @property
    def context_dim(self) -> int:
        return self._policy.config.hidden_dim

    @property
    def action_dim(self) -> int:
        return self._policy.config.action_dim

    @property
    def state_dim(self) -> int:
        return self._policy.config.state_dim

    @property
    def chunk_size(self) -> int:
        return self._policy.config.chunk_size

    # --- RTC capability surface ---
    @property
    def supports_rtc(self) -> bool:
        return getattr(self._policy, "supports_rtc", False)

    @property
    def rtc_prefix_length(self) -> int:
        return getattr(self._policy, "rtc_prefix_length", 5)

    def prefix_drift(self) -> float | None:
        inner_model = self._policy.model if hasattr(self._policy, "model") else self._policy
        return getattr(inner_model, "_last_prefix_drift", None)

    # --- core ops ---
    def encode_context(self, batch: dict) -> Tensor:
        # CRITICAL: use the policy's shared prep helper so state is z-scored
        # the same way it was during RL-token-encoder training. Encoding an
        # un-normalized state produces OOD z_rl.
        prepared = self._policy.prepare_batch_for_encode_observations(batch)
        return self._policy.model.encode_observations(prepared)

    def predict_reference_chunk(self, batch: dict, context: Tensor | None = None, **kwargs) -> Tensor:
        # The ``context`` kwarg is only added when present so policy variants
        # without that parameter (mocks, older variants) aren't forced to
        # accept it — mirrors the pre-adapter call site.
        if context is not None:
            kwargs["context"] = context
        return self._policy.predict_action_chunk(batch, **kwargs)

    def normalize_state(self, state: Tensor) -> Tensor:
        mean = self._policy._state_mean
        if mean is None:
            return state
        return (state - mean.to(state.device)) / self._policy._state_std.to(state.device)

    def normalize_action(self, action: Tensor) -> Tensor:
        mean = self._policy._action_mean
        if mean is None:
            return action
        return (action - mean.to(action.device)) / self._policy._action_std.to(action.device)

    def denormalize_action(self, action: Tensor) -> Tensor:
        mean = self._policy._action_mean
        if mean is None:
            return action
        return action * self._policy._action_std.to(action.device) + mean.to(action.device)

    def reset(self) -> None:
        self._policy.reset()
