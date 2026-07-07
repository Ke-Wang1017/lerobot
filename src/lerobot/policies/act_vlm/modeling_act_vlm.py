#!/usr/bin/env python
"""ACT with VLM conditioning — S1 policy for dual-system VLA.

Extends ACT with:
1. S2 latent token: Pi0.5 prefix encoding [2048] → projected → injected as encoder token
2. Optional DINOv2-S backbone: Replaces ResNet18 for faster, higher-quality visual features
"""

from collections import deque
from itertools import chain

import einops
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.policies.act.modeling_act import (
    ACTDecoder,
    ACTEncoder,
    ACTSinusoidalPositionEmbedding2d,
    ACTTemporalEnsembler,
    create_sinusoidal_pos_embedding,
)
from lerobot.policies.act_vlm.configuration_act_vlm import ACTWithVLMConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

S2_LATENT_KEY = "observation.s2_latent"
S2_AGE_KEY = "observation.s2_latent_age"  # age in seconds (scalar per batch item)


class ACTWithVLMPolicy(PreTrainedPolicy):
    """ACT policy conditioned on S2 VLM latents for dual-system VLA."""

    config_class = ACTWithVLMConfig
    name = "act_vlm"

    def __init__(self, config: ACTWithVLMConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = ACTWithVLM(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> dict:
        backbone_params = []
        other_params = []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith("model.backbone") or n.startswith("model.dino_backbone"):
                backbone_params.append(p)
            else:
                other_params.append(p)
        return [
            {"params": other_params},
            {"params": backbone_params, "lr": self.config.optimizer_lr_backbone},
        ]

    def reset(self):
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], action_offset: int = 0) -> Tensor:
        """Return one action to execute.

        Args:
            batch: Observation batch.
            action_offset: Number of actions to skip into the chunk to compensate
                for inference delay. E.g., if inference took 40ms at 30fps,
                action_offset=1 skips the first (stale) action.
                With temporal ensembling, shifts the ensembler output index.
        """
        self.eval()

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)  # [B, chunk_size, action_dim]
            # Time-shift: drop first `action_offset` actions so the ensembler
            # blends over actions that correspond to "now" instead of "40ms ago".
            # Pad the end with the last action to keep chunk_size consistent.
            if action_offset > 0 and actions.shape[1] > action_offset:
                shifted = actions[:, action_offset:]
                pad = actions[:, -1:].expand(-1, action_offset, -1)
                actions = torch.cat([shifted, pad], dim=1)
            action = self.temporal_ensembler.update(actions)
            return action

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)
            # Skip first `action_offset` actions, then take n_action_steps
            start = min(action_offset, actions.shape[1] - self.config.n_action_steps)
            start = max(start, 0)
            end = start + self.config.n_action_steps
            self._action_queue.extend(actions[:, start:end].transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        actions = self.model(batch)[0]
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae:
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        return loss, loss_dict


class ACTWithVLM(nn.Module):
    """ACT model extended with S2 VLM latent conditioning and optional DINOv2 backbone.

    Architecture changes from ACT:
    1. S2 projector: MLP that maps [s2_latent_dim] → [dim_model], injected as first encoder token
    2. Optional DINOv2 backbone: Replaces ResNet18 with frozen DINOv2-S (22M params, ~5ms/image)
       - Outputs patch tokens [B, N_patches, 384] instead of conv feature maps
       - Projected to [B, N_patches, dim_model] via linear layer
    """

    def __init__(self, config: ACTWithVLMConfig):
        super().__init__()
        self.config = config

        # --- S2 latent projector ---
        self.s2_projector = nn.Sequential(
            nn.Linear(config.s2_latent_dim, config.s2_projector_hidden_dim),
            nn.GELU(),
            nn.Linear(config.s2_projector_hidden_dim, config.dim_model),
        )
        # Age embedding: encodes latent staleness (seconds) → added to projected s2 token.
        # Output layer is zero-initialized so age_embedding(0.0) = zeros,
        # preserving backward compatibility with checkpoints that lack these weights.
        age_out_layer = nn.Linear(config.s2_age_embedding_dim, config.dim_model)
        nn.init.zeros_(age_out_layer.weight)
        nn.init.zeros_(age_out_layer.bias)
        self.s2_age_embedding = nn.Sequential(
            nn.Linear(1, config.s2_age_embedding_dim),
            nn.GELU(),
            age_out_layer,
        )

        # --- VAE encoder (same as ACT) ---
        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0],
                config.dim_model,
            )
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        # --- Vision backbone ---
        if self.config.image_features:
            if config.use_dino_backbone:
                self.dino_backbone = torch.hub.load(
                    "facebookresearch/dinov2", config.dino_model, pretrained=True
                )
                if config.freeze_vision_backbone:
                    for p in self.dino_backbone.parameters():
                        p.requires_grad = False
                    self.dino_backbone.eval()
                else:
                    # Wrap each ViT block with gradient checkpointing to reduce
                    # activation memory during backward (~40% savings).
                    from torch.utils.checkpoint import checkpoint

                    for block in self.dino_backbone.blocks:
                        original_forward = block.forward
                        block.forward = lambda x, _fwd=original_forward: checkpoint(
                            _fwd, x, use_reentrant=False
                        )
                self.dino_proj = nn.Linear(config.dino_output_dim, config.dim_model)
                # Learnable position embedding for DINOv2 patch tokens
                # For 224×224 with patch_size=14: 16×16 = 256 patches
                self.dino_patch_pos_embed = None  # Created dynamically on first forward
            else:
                backbone_model = getattr(torchvision.models, config.vision_backbone)(
                    replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                    weights=config.pretrained_backbone_weights,
                    norm_layer=FrozenBatchNorm2d,
                )
                self.backbone = IntermediateLayerGetter(
                    backbone_model, return_layers={"layer4": "feature_map"}
                )

        # --- Transformer encoder/decoder ---
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        # Encoder input projections
        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)

        if self.config.image_features and not config.use_dino_backbone:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )

        # Positional embeddings for 1D tokens: [s2_latent, vae_latent, (state), (env_state)]
        n_1d_tokens = 2  # s2_latent + vae_latent (ACT has 1 for just vae_latent)
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)

        if self.config.image_features and not config.use_dino_backbone:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # Decoder
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

        self._reset_parameters()

    def _reset_parameters(self):
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _get_dino_features(self, img: Tensor) -> tuple[Tensor, Tensor]:
        """Extract DINOv2 patch tokens and create positional embeddings.

        Args:
            img: (B, C, H, W) image tensor

        Returns:
            features: (N_patches, B, dim_model) projected patch tokens
            pos_embed: (N_patches, 1, dim_model) positional embeddings
        """
        if self.config.freeze_vision_backbone:
            with torch.no_grad():
                out = self.dino_backbone.forward_features(img)
                patch_tokens = out["x_norm_patchtokens"]  # [B, N_patches, dino_dim]
        else:
            out = self.dino_backbone.forward_features(img)
            patch_tokens = out["x_norm_patchtokens"]

        n_patches = patch_tokens.shape[1]
        features = self.dino_proj(patch_tokens)  # [B, N_patches, dim_model]

        # Create or reuse positional embeddings
        if self.dino_patch_pos_embed is None or self.dino_patch_pos_embed.shape[0] != n_patches:
            pos = create_sinusoidal_pos_embedding(n_patches, self.config.dim_model).to(
                device=features.device, dtype=features.dtype
            )
            self.dino_patch_pos_embed = pos.unsqueeze(1)  # [N_patches, 1, dim_model]

        # Rearrange to (N_patches, B, dim_model)
        features = features.permute(1, 0, 2)
        pos_embed = self.dino_patch_pos_embed.to(device=features.device, dtype=features.dtype)

        return features, pos_embed

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        if self.config.use_vae and self.training:
            assert ACTION in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]

        # --- S2 latent ---
        if S2_LATENT_KEY in batch:
            s2_latent = batch[S2_LATENT_KEY]  # [B, s2_latent_dim]
        else:
            # Graceful degradation: zero latent if not provided
            s2_latent = torch.zeros(
                batch_size,
                self.config.s2_latent_dim,
                dtype=torch.float32,
                device=next(self.parameters()).device,
            )

        s2_token = self.s2_projector(s2_latent)  # [B, dim_model]

        # --- S2 latent age embedding (additive) ---
        if S2_AGE_KEY in batch:
            age_seconds = batch[S2_AGE_KEY]  # [B, 1] or [B]
            if age_seconds.ndim == 1:
                age_seconds = age_seconds.unsqueeze(-1)  # [B, 1]
            s2_token = s2_token + self.s2_age_embedding(age_seconds)  # [B, dim_model]

        # --- VAE encoder (same as ACT) ---
        if self.config.use_vae and ACTION in batch and self.training:
            cls_embed = einops.repeat(self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size)
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE])
                robot_state_embed = robot_state_embed.unsqueeze(1)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])

            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            pos_embed = self.vae_encoder_pos_enc.clone().detach()
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch[OBS_STATE].device if self.config.robot_state_feature else s2_latent.device,
            )
            key_padding_mask = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], axis=1)

            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                s2_latent.device
            )

        # --- Assemble encoder input tokens ---
        # Token order: [s2_token, vae_latent, (state), (env_state), image_features...]
        encoder_in_tokens = [
            s2_token,  # NEW: S2 conditioning token
            self.encoder_latent_input_proj(latent_sample),
        ]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        # --- Image features ---
        if self.config.image_features:
            if self.config.use_dino_backbone:
                for img in batch[OBS_IMAGES]:
                    cam_features, cam_pos_embed = self._get_dino_features(img)
                    encoder_in_tokens.extend(list(cam_features))
                    encoder_in_pos_embed.extend(list(cam_pos_embed))
            else:
                for img in batch[OBS_IMAGES]:
                    cam_features = self.backbone(img)["feature_map"]
                    cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                    cam_features = self.encoder_img_feat_input_proj(cam_features)
                    cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                    cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
                    encoder_in_tokens.extend(list(cam_features))
                    encoder_in_pos_embed.extend(list(cam_pos_embed))

        # Stack all tokens
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        # --- Transformer forward ---
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        decoder_out = decoder_out.transpose(0, 1)
        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)
