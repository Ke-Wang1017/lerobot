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

"""Policy-facing seam for RLT online RL fine-tuning.

RLT ("RL Token") fine-tunes a *frozen* base policy online with a lightweight
TD3 actor-critic. To do that it needs four things from the policy:

  1. an encoded-observation *context* it can compress into an RL token
     (``encode_context`` → ``[B, N, D]``; N may be 1 for policies whose
     conditioning is a single pooled vector),
  2. the policy's own predicted *reference* action chunk, used as the actor's
     behavior-cloning anchor (``predict_reference_chunk`` → ``[B, T, A]`` in
     raw action space),
  3. state/action *normalization* into the actor's shared space, and
  4. shape *metadata* (context/action/state dims, chunk length).

``RLTPolicyAdapter`` is the ONLY surface through which RLT touches the base
policy, so the RL machinery (token encoder, actor-critic, replay buffer,
intervention recorder) stays policy-agnostic and each policy supplies a thin
concrete adapter. See :class:`~lerobot.policies.hvla.rlt_adapter.HVLAS1Adapter`
(in the hvla package, because it depends on ``FlowMatchingS1Policy``) for the
reference implementation.
"""

from __future__ import annotations

import abc

from torch import Tensor


class RLTPolicyAdapter(abc.ABC):
    """Adapt an arbitrary base policy to the interface RLT depends on."""

    # ------------------------------------------------------------------
    # Shape metadata
    # ------------------------------------------------------------------
    @property
    @abc.abstractmethod
    def context_dim(self) -> int:
        """Channel dim D of the ``encode_context`` output ``[B, N, D]``.

        Also the default RL-token bottleneck width unless overridden.
        """

    @property
    @abc.abstractmethod
    def action_dim(self) -> int:
        """Real (unpadded) action dimensionality A."""

    @property
    @abc.abstractmethod
    def state_dim(self) -> int:
        """Proprioceptive state dimensionality."""

    @property
    @abc.abstractmethod
    def chunk_size(self) -> int:
        """Length T of the policy's predicted action chunk.

        Must be >= the RLT ``rl_chunk_length`` C (the actor refines the
        first C frames after the RTC prefix).
        """

    # ------------------------------------------------------------------
    # Optional real-time-chunking (RTC) capability surface.
    # Policies without RTC keep the defaults.
    # ------------------------------------------------------------------
    @property
    def supports_rtc(self) -> bool:
        return False

    @property
    def rtc_prefix_length(self) -> int:
        return 0

    def prefix_drift(self) -> float | None:
        """Latest RTC prefix-drift diagnostic, or ``None`` if unavailable."""
        return None

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def encode_context(self, batch: dict) -> Tensor:
        """Return the encoded-observation context ``[B, N, context_dim]``.

        Implementations own any batch preparation/normalization the policy's
        encoder expects (e.g. z-scoring state, arranging image keys) so the
        context matches what the RL-token encoder was trained on.
        """

    @abc.abstractmethod
    def predict_reference_chunk(self, batch: dict, context: Tensor | None = None, **kwargs) -> Tensor:
        """Return the policy's reference action chunk ``[B, T, action_dim]`` (raw space).

        ``context`` is the (optional) precomputed :meth:`encode_context` output —
        adapters that can reuse it to skip a redundant vision forward should;
        those that cannot may ignore it. Extra ``kwargs`` (e.g. ``num_steps``)
        pass through to the policy's sampler.
        """

    @abc.abstractmethod
    def normalize_state(self, state: Tensor) -> Tensor:
        """Z-score raw proprioceptive state into actor input space (identity if no stats)."""

    @abc.abstractmethod
    def normalize_action(self, action: Tensor) -> Tensor:
        """Normalize a raw action into actor space (identity if no stats)."""

    @abc.abstractmethod
    def denormalize_action(self, action: Tensor) -> Tensor:
        """Inverse of :meth:`normalize_action` — actor output → raw action space."""

    def reset(self) -> None:  # noqa: B027
        """Clear per-episode policy state (e.g. action queue). Default: no-op."""
