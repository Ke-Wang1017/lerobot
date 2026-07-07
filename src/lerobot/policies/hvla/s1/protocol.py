"""S1 Policy Protocol — interface that any S1 action policy must satisfy.

Two implementations planned:
  1. ACTWithVLM (CVAE) — needs temporal ensembling, no RTC support
  2. FlowMatchingS1 — native RTC via prefix conditioning, no ensembling needed
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

# Batch keys for S2 conditioning
S2_LATENT_KEY = "observation.s2_latent"
S2_AGE_KEY = "observation.s2_latent_age"
# Batch key for RTC prefix (last K executed actions)
ACTION_PREFIX_KEY = "action.prev_chunk"


@runtime_checkable
class S1Policy(Protocol):
    """Contract for S1 action policies in HVLA.

    Any policy plugged into the HVLA system must implement this interface.
    The policy receives observation batches that include:
      - Camera images (policy-specific keys)
      - "observation.state" [B, action_dim] — joint positions
      - "observation.s2_latent" [B, 2048] — S2 scene latent
      - "observation.s2_latent_age" [B, 1] — staleness of S2 latent in seconds

    For RTC-capable policies, the batch may also contain:
      - "action.prev_chunk" [B, K, action_dim] — last K executed actions
    """

    @property
    def supports_rtc(self) -> bool:
        """Whether this policy uses previous action prefix for chunk continuity.

        If True, the inference loop will populate ACTION_PREFIX_KEY in the batch
        with the last K executed actions from the robot.
        """
        ...

    @property
    def needs_temporal_ensemble(self) -> bool:
        """Whether chunks are noisy and need external temporal ensembling.

        ACT (CVAE) produces inconsistent chunks → needs ensembling (adds lag).
        Flow matching with RTC → consistent chunks, no ensembling needed.
        """
        ...

    @property
    def rtc_prefix_length(self) -> int:
        """Number of past executed actions to feed as RTC prefix (K).
        Only meaningful if supports_rtc is True. Default: 0.
        """
        ...

    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a full action chunk from observation.

        Args:
            batch: Observation dict. May include ACTION_PREFIX_KEY if supports_rtc.

        Returns:
            [B, chunk_size, action_dim] — predicted future actions.
        """
        ...

    def reset(self) -> None:
        """Reset episode state (ensembler, action queue, denoising state, etc)."""
        ...

    def to(self, device: torch.device | str) -> S1Policy: ...

    def eval(self) -> S1Policy: ...
