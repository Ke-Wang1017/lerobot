"""Configuration for Flow Matching S1 action policy.

Action expert conditioned on:
  - DINOv2 image features (same backbone as ACTWithVLM)
  - S2 latent [2048] + age embedding
  - Robot state [action_dim]
  - Training-time RTC: simulated delay + inpainting prefix

No VLM in S1 — S2 handles scene understanding via shared latent.

References:
  - Flow Matching: Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
  - Pi0 action expert: Black et al., "π₀: A Vision-Language-Action Flow Model", 2024
  - SmolVLA: Luo et al., "SmolVLA: A Small Vision-Language-Action Model", 2025
  - Training-time RTC: Mees et al., "Training-Time Action Conditioning for Efficient
    Real-Time Chunking", arXiv:2512.05964, 2025
  - Inference-time RTC: Moeglich et al., "Real-Time Execution of Action Chunking
    Flow Policies", arXiv:2506.07339, 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FlowMatchingS1Config:
    """Config for FlowMatchingS1Policy.

    Architecture follows Pi0/SmolVLA action expert design:
    - Observation encoder (DINOv2 + state + S2 latent) → context tokens
    - Flow matching decoder with cross-attention to context
    - Action+timestep fusion via concat → MLP(SiLU) (matching Pi0/SmolVLA)
    """

    # --- Action space ---
    action_dim: int = 14  # 7 joints × 2 arms
    chunk_size: int = 50  # predict 50 future actions (~1.67s at 30Hz)
    n_action_steps: int = 50  # execute full chunk (RTC handles continuity)

    # --- Model architecture ---
    # Targeting ~30M params (excl. frozen DINOv2 86M).
    # Helix uses 80M for humanoid; we need less for bimanual 14-DOF.
    hidden_dim: int = 768  # transformer hidden dimension
    num_heads: int = 8  # attention heads
    num_encoder_layers: int = 4  # observation encoder depth
    num_decoder_layers: int = 6  # flow matching decoder depth
    dim_feedforward: int = 2048  # FFN intermediate size
    dropout: float = 0.1

    # --- Image backbone ---
    use_dino_backbone: bool = True  # DINOv2 ViT-B/14 (same as ACTWithVLM)
    image_features: dict = field(
        default_factory=lambda: {
            "observation.images.front": 224,
            "observation.images.left_wrist": 224,
            "observation.images.right_wrist": 224,
            "observation.images.top": 224,
        }
    )
    dino_model: str = (
        "dinov2_vits14"  # ViT-S (22M) — same as ACTWithVLM. Use "dinov2_vitb14" for ViT-B (86M).
    )
    freeze_backbone: bool = False  # finetune DINOv2 (required for good performance)
    backbone_gradient_checkpointing: bool = True  # saves ~40% activation memory for DINOv2
    backbone_dim: int = 384  # DINOv2 ViT-S output dim (768 for ViT-B)

    # --- S2 conditioning ---
    s2_latent_dim: int = 2048  # S2 prefix latent dimension
    s2_proj_hidden: int = 1024  # S2 projection MLP intermediate
    use_s2_age_embedding: bool = False  # disabled — old ACT worked without it

    # --- Flow matching ---
    num_inference_steps: int = 10  # denoising steps at inference (10 = best quality/speed tradeoff)
    time_sampling_beta_alpha: float = 1.5  # Beta distribution for training time sampling
    time_sampling_beta_beta: float = 1.0
    time_min: float = 0.001  # minimum timestep
    time_max: float = 1.0  # maximum timestep

    # --- Training-time RTC (arXiv:2512.05964, Ψ₀ arXiv:2603.12263) ---
    # Simulates inference delay during training by replacing the first D actions
    # in x_t with ground-truth (unnoised) actions, and setting their per-position
    # timestep to t=0 (clean). Prefix positions excluded from loss.
    # At inference, overlap actions from the previous chunk serve as prefix.
    # d sampled as Uniform(1, rtc_max_delay) with rtc_drop_prob chance of d=0.
    rtc_max_delay: int = 6  # max simulated delay in frames (15 denoise steps ≈ 5 frames)
    rtc_drop_prob: float = 0.2  # probability of no prefix (simulates first chunk)

    # --- Robot state ---
    robot_state_feature: bool = True
    state_dim: int = 14  # same as action_dim for bimanual

    # --- Training ---
    # LR references: Pi0=2.5e-5, ACT=1e-5, SmolVLA=1e-4, Pi0.5+LoRA=1.2e-4
    lr: float = 2.5e-5  # peak LR (cosine schedule)
    lr_decay: float = 2.5e-6  # final LR after cosine decay
    weight_decay: float = 1e-4
    warmup_steps: int = 1000

    @property
    def num_images(self) -> int:
        return len(self.image_features)
