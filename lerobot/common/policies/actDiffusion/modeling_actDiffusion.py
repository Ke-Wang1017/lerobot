#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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
"""Action Chunking Transformer Policy

As per Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (https://arxiv.org/abs/2304.13705).
The majority of changes here involve removing unused code, unifying naming, and adding helpful comments.
"""

import math
from collections import deque
from itertools import chain
from typing import Callable

import einops
import numpy as np
from typing import List
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.common.policies.actDiffusion.configuration_actDiffusion import ACTDiffusionConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
)

from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


class ACTDiffusionPolicy(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="lerobot",
    repo_url="https://github.com/huggingface/lerobot",
    tags=["robotics", "actDiffusion"],
):
    """
    ALOHA Unleashed 🌋: A Simple Recipe for Robot Dexterity
   (paper: https://aloha-unleashed.github.io/assets/aloha_unleashed.pdf)
    """

    name = "actDiffusion"

    def __init__(
        self,
        config: ACTDiffusionConfig | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__()
        if config is None:
            config = ACTDiffusionConfig()
        self.config: ACTDiffusionConfig = config

        self.normalize_inputs = Normalize(
            config.input_shapes, config.input_normalization_modes, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        # this will be modified later to be a diffusion model
        self.model = ACT(config)

        self.expected_image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]

        # if config.temporal_ensemble_coeff is not None:
        #     self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        # if self.config.temporal_ensemble_coeff is not None:
        #     self.temporal_ensembler.reset()
        # else:
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()

        batch = self.normalize_inputs(batch)
        if len(self.expected_image_keys) > 0:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            # actions = self.model(batch)[0][:, : self.config.n_action_steps]
            actions = self.model.generate_actions(batch)

            # TODO(rcadene): make _forward return output dictionary?
            actions = self.unnormalize_outputs({"action": actions})["action"]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()
 
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        if len(self.expected_image_keys) > 0:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)
        batch = self.normalize_targets(batch)
        # actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        # l1_loss = (
        #     F.l1_loss(batch["action"], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        # ).mean()

        loss = (self.model.compute_loss(batch)* ~batch["action_is_pad"].unsqueeze(-1)).mean()

        return  {"loss": loss}


def _make_noise_scheduler(name: str, **kwargs: dict) -> DDPMScheduler | DDIMScheduler:
    """
    Factory for noise scheduler instances of the requested type. All kwargs are passed
    to the scheduler.
    """
    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    elif name == "DDIM":
        return DDIMScheduler(**kwargs)
    else:
        raise ValueError(f"Unsupported noise scheduler type {name}")


class ACT(nn.Module):
    """Action Chunking Transformer: The underlying neural network for ACTPolicy.

    Note: In this code we use the terms `vae_encoder`, 'encoder', `decoder`. The meanings are as follows.
        - The `vae_encoder` is, as per the literature around variational auto-encoders (VAE), the part of the
          model that encodes the target data (a sequence of actions), and the condition (the robot
          joint-space).
        - A transformer with an `encoder` (not the VAE encoder) and `decoder` (not the VAE decoder) with
          cross-attention is used as the VAE decoder. For these terms, we drop the `vae_` prefix because we
          have an option to train this model without the variational objective (in which case we drop the
          `vae_encoder` altogether, and nothing about this model has anything to do with a VAE).

                                 Transformer
                                 Used alone for inference
                                 (acts as VAE decoder
                                  during training)
                                ┌───────────────────────┐
                                │             Outputs   │
                                │                ▲      │
                                │     ┌─────►┌───────┐  │
                   ┌──────┐     │     │      │Transf.│  │
                   │      │     │     ├─────►│decoder│  │
              ┌────┴────┐ │     │     │      │       │  │
              │         │ │     │ ┌───┴───┬─►│       │  │
              │ VAE     │ │     │ │       │  └───────┘  │
              │ encoder │ │     │ │Transf.│             │
              │         │ │     │ │encoder│             │
              └───▲─────┘ │     │ │       │             │
                  │       │     │ └▲──▲─▲─┘             │
                  │       │     │  │  │ │               │
                inputs    └─────┼──┘  │ image emb.      │
                                │    state emb.         │
                                └───────────────────────┘
    """

    def __init__(self, config: ACTDiffusionConfig):
        super().__init__()
        self.config = config
        # BERT style VAE encoder with input tokens [cls, robot_state, *action_sequence].
        # The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).
        self.use_robot_state = "observation.state" in config.input_shapes
        self.use_images = any(k.startswith("observation.image") for k in config.input_shapes)
        self.use_env_state = "observation.environment_state" in config.input_shapes

        # Backbone for image feature extraction.
        if self.use_images:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            # Note: The assumption here is that we are using a ResNet model (and hence layer4 is the final
            # feature map).
            # Note: The forward method of this returns a dict: {"feature_map": output}.
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # Transformer (acts as VAE decoder when training with the variational objective).
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)
        self.timestep_embedder = TimestepEmbeddingGenerator(config.num_train_timesteps, config.dim_model)

        # Transformer encoder input projections. The tokens will be structured like
        # [latent, (robot_state), (env_state), (image_feature_map_pixels)].
        # TODO: state and env_state might go through a small MLP
        if self.use_robot_state:
            self.encoder_robot_state_input_proj = nn.Linear(
                config.input_shapes["observation.state"][0], config.dim_model
            )
        if self.use_env_state:
            self.encoder_env_state_input_proj = nn.Linear(
                config.input_shapes["observation.environment_state"][0], config.dim_model
            )
        if self.use_images:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
        # Transformer encoder positional embeddings.
        n_1d_tokens = 0  # for the latent
        if self.use_robot_state:
            n_1d_tokens += 1
        if self.use_env_state:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.use_images:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # Transformer decoder.
        # Learnable positional embedding for the transformer's decoder (in the style of DETR object queries).
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model) 
        # linear projection for actions in decoder
        self.action_input_project = nn.Linear(config.output_shapes["action"][0], config.dim_model)
        # Final action regression head on the output of the transformer's decoder.
        self.action_head = nn.Linear(config.dim_model, config.output_shapes["action"][0])
        # pos embedding on one-hot time encoding
        self.timestep_embed_generator = TimestepEmbeddingGenerator(config.num_train_timesteps, config.dim_model)
        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )

        if config.num_inference_steps is None:
            self.num_inference_steps = config.num_train_timesteps
        else:
            self.num_inference_steps = config.num_inference_steps

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encoding(self, batch: dict[str, Tensor]) ->  tuple[Tensor, Tensor]:
        """A forward pass through the Action Chunking Transformer encoder.

        `batch` should have the following structure:
        {
            "observation.state" (optional): (B, state_dim) batch of robot states.

            "observation.images": (B, n_cameras, C, H, W) batch of images.
                AND/OR
            "observation.environment_state": (B, env_dim) batch of environment states.

            "action" (optional, only if training with VAE): (B, chunk_size, action dim) batch of actions.
        }

        Returns:
            (chunk_size, Batch, dim_model) batch of encoder embeddings and the positional embedding
        """
                # When not using the VAE encoder, we set the latent to be all zeros.
        mu = log_sigma_x2 = None

        # Prepare transformer encoder inputs.
        encoder_in_tokens = []
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        # Robot state token.
        if self.use_robot_state:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch["observation.state"].squeeze()))
        # Environment state token.
        if self.use_env_state:
            encoder_in_tokens.append(
                self.encoder_env_state_input_proj(batch["observation.environment_state"])
            )

        # Camera observation features and positional embeddings.
        if self.use_images:
            all_cam_features = []
            all_cam_pos_embeds = []

            for cam_index in range(batch["observation.images"].shape[-4]):
                cam_features = self.backbone(batch["observation.images"][:, cam_index].squeeze())["feature_map"]
                # TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use
                # buffer
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)  # (B, C, h, w)
                all_cam_features.append(cam_features)
                all_cam_pos_embeds.append(cam_pos_embed)
            # Concatenate camera observation feature maps and positional embeddings along the width dimension,
            # and move to (sequence, batch, dim).
            all_cam_features = torch.cat(all_cam_features, axis=-1)
            encoder_in_tokens.extend(einops.rearrange(all_cam_features, "b c h w -> (h w) b c"))
            all_cam_pos_embeds = torch.cat(all_cam_pos_embeds, axis=-1)
            encoder_in_pos_embed.extend(einops.rearrange(all_cam_pos_embeds, "b c h w -> (h w) b c"))

        # Stack all tokens along the sequence dimension.
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        # Forward pass through the transformer modules.
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        return encoder_out, encoder_in_pos_embed
 

    def decoding(self, action: Tensor, encoder_out: Tensor, timestep_embed: Tensor, encoder_in_pos_embed: Tensor,
                  ) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        # change to (Chunk size, Batch_size, Channel_size)
        action = action.transpose(0,1)
        decoder_in = self.action_input_project(action)
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            timestep_embed,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        # Move back to (B, S, C).
        decoder_out = decoder_out.transpose(0, 1)

        actions = self.action_head(decoder_out)

        return actions

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)

            "action": (B, horizon, action_dim)
        }
        """
        # Input validation.
        assert set(batch).issuperset({"observation.state", "action", "action_is_pad"})
        assert "observation.images" in batch or "observation.environment_state" in batch
        horizon = batch["action"].shape[1]
        assert horizon == self.config.chunk_size

        # Forward diffusion.
        trajectory = batch["action"]
        # Sample noise to add to the trajectory.
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        # Sample a random noising timestep for each item in the batch.
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        )
        # Add noise to the clean trajectories according to the noise magnitude at each timestep.
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)
        timestep_embed = self.timestep_embed_generator.create_timestep_embedding(timesteps)
        # prediction throught encoding and decoding
        encoder_out, encoder_in_pos_embed = self.encoding(batch)
        pred = self.decoding(noisy_trajectory, encoder_out, timestep_embed, encoder_in_pos_embed)

        # Compute the loss.
        # The target is either the original trajectory, or the noise.
        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = batch["action"]
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")

        return loss
    
    # ========= inference  ============
    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have:
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)
        }
        """
        # run sampling
        actions = self.sample_action(batch)

        # Extract `n_action_steps` steps worth of actions (from the current observation).
        actions = actions[:, :self.config.n_action_steps]

        return actions
    
    def sample_action(
        self, batch: dict[str, Tensor], generator: torch.Generator | None = None
    ) -> Tensor:
        batch_size= batch["observation.state"].shape[0]
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        # Sample prior.
        sample = torch.randn(
            size=(batch_size, self.config.chunk_size, self.config.output_shapes["action"][0]),
            dtype=dtype,
            device=device,
            generator=generator,
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        encoder_out, encoder_in_pos_embed = self.encoding(batch)

        for t in self.noise_scheduler.timesteps:
            # Predict model output.
            timesteps = torch.ones(size=(sample.shape[0],),
                                   dtype=torch.int64, 
                                   device=device)
            timesteps = timesteps * t
            timestep_embed = self.timestep_embed_generator.create_timestep_embedding(timesteps)
            model_output = self.decoding(sample, encoder_out, timestep_embed, encoder_in_pos_embed)
            # Compute previous image: x_t -> x_t-1
            sample = self.noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample

        return sample


class ACTEncoder(nn.Module):
    """Convenience module for running multiple encoder layers, maybe followed by normalization."""

    def __init__(self, config: ACTDiffusionConfig, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class ACTEncoderLayer(nn.Module):
    def __init__(self, config: ACTDiffusionConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]  # note: [0] to select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class ACTDecoder(nn.Module):
    def __init__(self, config: ACTDiffusionConfig):
        """Convenience module for running multiple decoder layers followed by normalization."""
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        timestep_embed: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x, encoder_out, timestep_embed=timestep_embed, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed
            )
        if self.norm is not None:
            x = self.norm(x)
        return x


class ACTDecoderLayer(nn.Module):
    def __init__(self, config: ACTDiffusionConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        timestep_embed: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (Decoder Sequence, Batch, Channel) tensor of input tokens.
            encoder_out: (Encoder Sequence, B, C) output features from the last layer of the encoder we are
                cross-attending with.
            decoder_pos_embed: (ES, 1, C) positional embedding for keys (from the encoder).
            encoder_pos_embed: (DS, 1, C) Positional_embedding for the queries (from the decoder).
        Returns:
            (DS, B, C) tensor of decoder output features.
        """
        skip = x
        # if self.pre_norm:
        #     x = self.norm1(x)
        # q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        # x = self.self_attn(q, k, value=x)[0]  # select just the output, not the attention weights
        # x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        input_1 = torch.concat((self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),timestep_embed), dim=0)
        input_2 = torch.concat((encoder_out,timestep_embed), dim=0)
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=input_1,
            value=input_2,
        )[0]  # select just the output, not the attention weights
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x

class TimestepEmbeddingGenerator:
    def __init__(self, num_timesteps: int, dim_model: int):
        self.num_timesteps = num_timesteps
        self.dim_model = dim_model
        self.embedding = nn.Embedding(num_timesteps, dim_model)
        self.timestep_encode = nn.Sequential(
            SinusoidalPosEmb(self.dim_model),
            nn.Linear(self.dim_model, self.dim_model * 4),
            nn.Mish(),
            nn.Linear(self.dim_model * 4, self.dim_model),
        )

    def generate_one_hot_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Generate one-hot encoded tensor for timesteps.
        
        Args:
            timesteps (Tensor): 1D tensor of shape (batch_size,) containing timestep indices.
        
        Returns:
            Tensor: One-hot encoded tensor of shape (batch_size, num_timesteps).
        """
        batch_size = timesteps.shape[0]
        one_hot = torch.zeros(batch_size, self.num_timesteps, device=timesteps.device)
        one_hot.scatter_(1, timesteps.unsqueeze(1), 1)
        return one_hot

    def create_timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal positional embeddings for the given timesteps.
        
        Args:
            timesteps (Tensor): 1D tensor of shape (batch_size,) containing timestep indices.
        
        Returns:
            Tensor: Sinusoidal positional embeddings of shape (1, batch_size, dim_model).
        """
        # learned embeddings
        # self.embedding.to(timesteps.device)
        # timestep_embed = self.embedding(timesteps)
        # timestep_embed = timestep_embed.repeat(8, 1, 1)  
        # return timestep_embed
        #sin pos embedding
        # onehot_timestep = self.generate_one_hot_timesteps(timesteps).long().to(timesteps.device)
        # onehot_timestep = onehot_timestep.detach().cpu().numpy()
        # onehot_timestep = torch.from_numpy(onehot_timestep).long()
        # timestep_indices = torch.argmax(onehot_timestep, dim=1)
        # timestep_indices = timestep_indices.detach().cpu().numpy()
        # timestep_embed = self.create_sinusoidal_pos_embedding_for_timestep(timestep_indices)
        # timestep_embed = timestep_embed.unsqueeze(0).to(timesteps.device)
        # return timestep_embed
        # sin pos + MLP
        self.timestep_encode.to(timesteps.device)
        return self.timestep_encode(timesteps).unsqueeze(0)
    
    def create_sinusoidal_pos_embedding_for_timestep(self, positions: np.ndarray) -> torch.Tensor:
        """
        Generate sinusoidal positional embeddings for given positions.
        
        Args:
            positions: Array of positions for which to generate embeddings.
        
        Returns:
            Tensor: Sinusoidal positional embeddings.
        """
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / self.dim_model) for hid_j in range(self.dim_model)]

        sinusoid_table = np.array([get_position_angle_vec(pos) for pos in positions])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.from_numpy(sinusoid_table).float()


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    

class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings similar to what's presented in Attention Is All You Need.

    The variation is that the position indices are normalized in [0, 2π] (not quite: the lower bound is 1/H
    for the vertical direction, and 1/W for the horizontal direction.
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: The desired dimension of the embeddings.
        """
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        # Inverse "common ratio" for the geometric progression in sinusoid frequencies.
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: A (B, C, H, W) batch of 2D feature map to generate the embeddings for.
        Returns:
            A (1, C, H, W) batch of corresponding sinusoidal positional embeddings.
        """
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        # Note: These are like range(1, H+1) and range(1, W+1) respectively, but in most implementations
        # they would be range(0, H) and range(0, W). Keeping it at as is to match the original code.
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        # "Normalize" the position index such that it ranges in [0, 2π].
        # Note: Adding epsilon on the denominator should not be needed as all values of y_embed and x_range
        # are non-zero by construction. This is an artifact of the original code.
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)

        # Note: this stack then flatten operation results in interleaved sine and cosine terms.
        # pos_embed_x and pos_embed_y are (1, H, W, C // 2).
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)

        return pos_embed


def get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
