# S1 action policy for Hierarchical VLA.
#
# Two implementations:
#   1. ACTWithVLM (CVAE) — default, needs temporal ensembling
#   2. FlowMatchingS1 — RTC prefix conditioning, no ensembling needed
#
# The contract is defined in protocol.py. Both implementations accept
# S2 latent + age in the input batch.

from lerobot.policies.hvla.s1.protocol import (
    ACTION_PREFIX_KEY,
    S2_AGE_KEY,
    S2_LATENT_KEY,
    S1Policy,
)


# Lazy imports to avoid loading unnecessary dependencies
def load_act_policy(checkpoint_path: str):
    """Load ACTWithVLM policy (default S1)."""
    from lerobot.policies.act_vlm.modeling_act_vlm import ACTWithVLMPolicy

    return ACTWithVLMPolicy.from_pretrained(pretrained_name_or_path=checkpoint_path)


def load_flow_matching_policy(checkpoint_path: str, config_overrides: dict | None = None):
    """Load Flow Matching S1 policy."""
    from lerobot.policies.hvla.s1.flow_matching import FlowMatchingS1Config, FlowMatchingS1Policy

    config = FlowMatchingS1Config(**(config_overrides or {}))
    return FlowMatchingS1Policy.from_pretrained(checkpoint_path, config=config)


__all__ = [
    "S1Policy",
    "S2_LATENT_KEY",
    "S2_AGE_KEY",
    "ACTION_PREFIX_KEY",
    "load_act_policy",
    "load_flow_matching_policy",
]
