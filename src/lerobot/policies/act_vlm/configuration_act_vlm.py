#!/usr/bin/env python
"""Configuration for ACT with VLM conditioning (dual-system S1 policy)."""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("act_vlm")
@dataclass
class ACTWithVLMConfig(ACTConfig):
    """Configuration for ACT + VLM latent conditioning.

    Extends ACTConfig with:
    - S2 latent injection (from Pi0.5 prefix encoding)
    - Optional DINOv2 vision backbone (replacing ResNet18)
    """

    # S2 (VLM) latent conditioning
    s2_latent_dim: int = 2048
    s2_projector_hidden_dim: int = 1024
    s2_age_embedding_dim: int = 64  # intermediate dim for age MLP (1 → 64 → dim_model)

    # DINOv2 backbone option
    use_dino_backbone: bool = False
    dino_model: str = "dinov2_vits14"
    dino_output_dim: int = 384
    freeze_vision_backbone: bool = False

    # Override default vision backbone to allow non-resnet when using dino
    vision_backbone: str = "resnet18"

    # Training defaults matching vanilla ACT
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    def __post_init__(self):
        # Skip ACTConfig's resnet validation if using DINOv2
        if self.use_dino_backbone:
            # Temporarily set vision_backbone to pass parent validation
            saved_backbone = self.vision_backbone
            self.vision_backbone = "resnet18"
            super().__post_init__()
            self.vision_backbone = saved_backbone
        else:
            super().__post_init__()

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")
