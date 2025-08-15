#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team.
# All rights reserved.
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

import math
from collections import deque
from collections.abc import Callable
from dataclasses import asdict
from typing import Literal

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch.distributions import (
    Categorical,
    MultivariateNormal,
    TanhTransform,
    Transform,
    TransformedDistribution,
)
from transformers import (
    AutoModel,
    AutoModelForImageTextToText,
    SmolVLMForConditionalGeneration,
)

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.fqlvla.configuration_fqlvla import FQLVLAConfig, is_image_feature
from lerobot.policies.normalize import NormalizeBuffer, UnnormalizeBuffer
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
from lerobot.policies.utils import get_device_from_parameters

DISCRETE_DIMENSION_INDEX = -1  # Gripper is always the last dimension


class FQLVLAPolicy(
    PreTrainedPolicy,
):
    config_class = FQLVLAConfig
    name = "fqlvla"

    def __init__(
        self,
        config: FQLVLAConfig | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Determine action dimension and initialize all components
        continuous_action_dim = config.output_features["action"].shape[0]
        self._init_normalization(dataset_stats)
        self._init_encoders()
        self._init_critics(continuous_action_dim)
        self._init_actor_bc_flow(continuous_action_dim)
        self._init_actor_onestep_flow(continuous_action_dim)
        self._init_temperature()

        self.reset()

    def get_optim_params(self) -> dict:
        optim_params = {
            "actor_bc_flow": [
                p
                for n, p in self.actor_bc_flow.named_parameters()
                if not n.startswith("encoder") or not self.shared_encoder
            ],
            "actor_onestep_flow": [
                p
                for n, p in self.actor_onestep_flow.named_parameters()
                if not n.startswith("encoder") or not self.shared_encoder
            ],
            "critic": self.critic_ensemble.parameters(),
            # "temperature": self.log_alpha,
        }
        if self.config.num_discrete_actions is not None:
            optim_params["discrete_critic"] = self.discrete_critic.parameters()
        raise ValueError("Not used")
        return optim_params

    def reset(self):
        """Reset the policy"""
        self.actor_bc_flow.encoder.vla.reset()
        self.actor_onestep_flow.encoder.vla.reset()
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        raise NotImplementedError("FQLVLAPolicy does not support action chunking. It returns single actions!")

    # @torch.no_grad()
    # def compute_flow_actions(self, batch: dict[str, Tensor], noises: Tensor) -> Tensor:
    #     observations_features = None
    #     if self.shared_encoder and self.actor_bc_flow.encoder.has_images:
    #         # Cache and normalize image features
    #         observations_features = self.actor_bc_flow.encoder.get_cached_image_features(batch, normalize=True)

    #     actions = noises
    #     flow_steps = self.config.flow_steps
    #     dt = 1.0 / flow_steps

    #     # Euler method.
    #     for i in range(flow_steps):
    #         t_val = float(i) / flow_steps
    #         t = torch.full((actions.shape[0], 1), t_val, device=noises.device)
    #         vels, _, _ = self.actor_bc_flow(batch, observations_features, actions, t)
    #         actions = actions + vels / flow_steps

    #     # actions, discrete_logits = torch.split(actions, [30, 30], dim=-1)
    #     # actions = actions.reshape(actions.shape[0], -1, 4)  # Reshape to match action dimensions
    #     # continuous_actions = actions[:, :, :3]
    #     # discrete_actions = actions[:, :, 3:]
    #     # continuous_actions = torch.clamp(continuous_actions, -1.0, 1.0)
    #     # discrete_actions = torch.clamp(discrete_actions, 0.0, 2.0)
    #     # actions = torch.cat([continuous_actions, discrete_actions], dim=-1)
    #     # actions = actions.reshape(actions.shape[0], -1)  # Flatten the action dimension
    #     # actions = torch.clamp(actions, -1.0, 1.0)

    #     return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select action for inference/evaluation"""

        observations_features = None
        if self.shared_encoder and self.actor_onestep_flow.encoder.has_images:
            # Cache and normalize image features
            observations_features = self.actor_onestep_flow.encoder.get_cached_image_features(
                batch, normalize=True
            )

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            # batch_shape = list(observations[list(observations.keys())[0]].shape[:-1])
            batch_shape = batch["observation.state"].shape[0]
            action_dim = self.actor_onestep_flow.action_dim  # self.config['action_dim']
            device = batch["observation.state"].device

            noises = torch.randn(batch_shape, action_dim, device=device)

            actions, _, _ = self.actor_onestep_flow(batch, observations_features, noises)

            actions = actions.reshape(batch_shape, -1, 4)
            actions = torch.clamp(actions, -1.0, 1.0)
            # actions = self.actor_bc_flow.encoder.vla.unnormalize_outputs({"action":actions})["action"]
            actions = self.unnormalize_targets({"action": actions})["action"]

            self._action_queue.extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        actions = self._action_queue.popleft()

        # if self.config.num_discrete_actions is not None:
        #     discrete_action, _, _ = self.discrete_actor(batch, observations_features)
        #     # discrete_action_value = self.discrete_critic(batch, observations_features)
        #     # discrete_action = torch.argmax(discrete_action_value, dim=-1, keepdim=True)
        #     actions = torch.cat([actions, discrete_action.unsqueeze(-1)], dim=-1)

        return actions

    def critic_forward(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        use_target: bool = False,
        observation_features: Tensor | None = None,
        do_output_normalization: bool = True,
    ) -> Tensor:
        """Forward pass through a critic network ensemble

        Args:
            observations: Dictionary of observations
            actions: Action tensor
            use_target: If True, use target critics, otherwise use ensemble critics

        Returns:
            Tensor of Q-values from all critics
        """

        critics = self.critic_target if use_target else self.critic_ensemble
        q_values = critics(observations, actions, observation_features, do_output_normalization)
        return q_values

    # def discrete_critic_forward(
    #     self, observations, use_target=False, observation_features=None
    # ) -> torch.Tensor:
    #     """Forward pass through a discrete critic network

    #     Args:
    #         observations: Dictionary of observations
    #         use_target: If True, use target critics, otherwise use ensemble critics
    #         observation_features: Optional pre-computed observation features to avoid recomputing encoder output

    #     Returns:
    #         Tensor of Q-values from the discrete critic network
    #     """
    #     discrete_critic = self.discrete_critic_target if use_target else self.discrete_critic
    #     q_values = discrete_critic(observations, observation_features)
    #     return q_values

    def forward(
        self,
        batch: dict[str, Tensor | dict[str, Tensor]],
        model: Literal["actor", "critic", "discrete_critic", "discrete_actor"] = "critic",
    ) -> dict[str, Tensor]:
        """Compute the loss for the given model

        Args:
            batch: Dictionary containing:
                - action: Action tensor
                - reward: Reward tensor
                - state: Observations tensor dict
                - next_state: Next observations tensor dict
                - done: Done mask tensor
                - observation_feature: Optional pre-computed observation features
                - next_observation_feature: Optional pre-computed next observation features
            model: Which model to compute the loss for ("actor", "critic", "discrete_critic", or "temperature")

        Returns:
            The computed loss tensor
        """
        # Extract common components from batch
        actions: Tensor = batch["action"]
        actions_is_pad = batch["actions_is_pad"]
        observations: dict[str, Tensor] = batch["state"]
        observation_features: Tensor = batch.get("observation_feature")

        if model == "critic":
            # Extract critic-specific components
            rewards: Tensor = batch["reward_nsteps"]
            discounts: Tensor = batch["discount_nsteps"]
            next_observations: dict[str, Tensor] = batch["next_state_nsteps"]
            done: Tensor = batch["done_nsteps"]
            next_observation_features: Tensor = batch.get("next_observation_feature")

            loss_critic, info = self.compute_loss_critic(
                observations=observations,
                actions=actions,
                actions_is_pad=actions_is_pad,
                rewards=rewards,
                discounts=discounts,
                next_observations=next_observations,
                done=done,
                observation_features=observation_features,
                next_observation_features=next_observation_features,
            )

            return {
                "loss_critic": loss_critic,
                "info": info,
            }

        # if model == "discrete_critic" and self.config.num_discrete_actions is not None:
        #     # Extract critic-specific components
        #     rewards: Tensor = batch["reward"]
        #     next_observations: dict[str, Tensor] = batch["next_state"]
        #     done: Tensor = batch["done"]
        #     next_observation_features: Tensor = batch.get("next_observation_feature")
        #     complementary_info = batch.get("complementary_info")
        #     loss_discrete_critic, info = self.compute_loss_discrete_critic(
        #         observations=observations,
        #         actions=actions[:, 0],
        #         rewards=rewards,
        #         next_observations=next_observations,
        #         done=done,
        #         observation_features=observation_features,
        #         next_observation_features=next_observation_features,
        #         complementary_info=complementary_info,
        #     )
        #     return {"loss_discrete_critic": loss_discrete_critic, "info": info}
        # if model == "actor":
        #     return {
        #         "loss_actor": self.compute_loss_actor(
        #             observations=observations,
        #             observation_features=observation_features,
        #             actions=actions,
        #         )
        #     }
        if model == "actor_bc_flow":
            loss_actor_bc_flow, info = self.compute_loss_actor_bc_flow(
                observations=observations,
                observation_features=observation_features,
                actions=actions,
                actions_is_pad=actions_is_pad,
            )
            return {
                "loss_actor_bc_flow": loss_actor_bc_flow,
                "info": info,
            }
        if model == "actor_onestep_flow":
            actor_onestep_flow_loss, info = self.compute_loss_actor_onestep_flow(
                observations=observations,
                observation_features=observation_features,
                actions=actions,
                actions_is_pad=actions_is_pad,
            )
            return {
                "loss_actor_onestep_flow": actor_onestep_flow_loss,
                "info": info,
            }

        # if model == "discrete_actor":
        #     if self.config.num_discrete_actions is None:
        #         raise ValueError("Discrete actor is not configured for this policy.")

        #     loss_discrete_actor, info = self.compute_loss_discrete_actor(
        #         observations=observations,
        #         observation_features=observation_features,
        #     )
        #     return {"loss_discrete_actor": loss_discrete_actor, "info": info}

        # if model == "temperature":
        #     return {
        #         "loss_temperature": self.compute_loss_temperature(
        #             observations=observations,
        #             observation_features=observation_features,
        #         )
        #     }

        raise ValueError(f"Unknown model type: {model}")

    def update_target_networks(self):
        """Update target networks with exponential moving average"""
        for target_param, param in zip(
            self.critic_target.parameters(),
            self.critic_ensemble.parameters(),
            strict=True,
        ):
            target_param.data.copy_(
                param.data * self.config.critic_target_update_weight
                + target_param.data * (1.0 - self.config.critic_target_update_weight)
            )
        # if self.config.num_discrete_actions is not None:
        #     for target_param, param in zip(
        #         self.discrete_critic_target.parameters(),
        #         self.discrete_critic.parameters(),
        #         strict=True,
        #     ):
        #         target_param.data.copy_(
        #             param.data * self.config.critic_target_update_weight
        #             + target_param.data * (1.0 - self.config.critic_target_update_weight)
        #         )

    def update_temperature(self):
        self.temperature = self.log_alpha.exp().item()

    def compute_loss_critic(
        self,
        observations,
        actions,
        actions_is_pad: Tensor,
        rewards,
        discounts,
        next_observations,
        done,
        observation_features: Tensor | None = None,
        next_observation_features: Tensor | None = None,
    ) -> Tensor:
        actions = self.normalize_targets({"action": actions})["action"]

        with torch.no_grad():
            batch_shape = next_observations["observation.state"].shape[0]
            action_dim = self.actor_onestep_flow.action_dim  # self.config['action_dim']

            noises = torch.randn(
                batch_shape, action_dim, device=next_observations["observation.state"].device
            )
            next_actions, _, _ = self.actor_onestep_flow(next_observations, next_observation_features, noises)

            next_actions = torch.clamp(next_actions, -1.0, 1.0)

            # 2- compute q targets
            next_qs = self.critic_forward(
                observations=next_observations,
                actions=next_actions,
                use_target=True,
                observation_features=next_observation_features,
                do_output_normalization=False,
            )  # (critic_ensemble_size, batch_size)

            # subsample critics to prevent overfitting if use high UTD (update to date)
            # TODO: Get indices before forward pass to avoid unnecessary computation
            if self.config.num_subsample_critics is not None:
                raise NotImplementedError(
                    "Subsampling critics is not implemented yet. "
                    "Please set num_subsample_critics to None or implement the subsampling logic."
                )
                # indices = torch.randperm(self.config.num_critics)
                # indices = indices[: self.config.num_subsample_critics]
                # next_qs = next_qs[indices]

            # critics subsample size
            if self.config.q_agg == "min":
                next_q, _ = next_qs.min(dim=0)  # Get values from min operation
            else:
                next_q = next_qs.mean(dim=0)

            # if self.config.use_backup_entropy:
            #     min_q = min_q - (self.temperature * next_log_probs)

            td_target = rewards.squeeze(-1) + (1 - done) * discounts.squeeze(-1) * next_q
            # td_target = rewards.squeeze(-1) + (1 - done) * self.config.discount * next_q

        # 3- compute predicted qs
        # if self.config.num_discrete_actions is not None:
        #     # NOTE: We only want to keep the continuous action part
        #     # In the buffer we have the full action space (continuous + discrete)
        #     # We need to split them before concatenating them in the critic forward
        #     actions: Tensor = actions[:, :, :DISCRETE_DIMENSION_INDEX]

        # actions = pad_vector(actions, 32)
        # actions = actions[:, 0, :3] # TODO: use all chunks
        # actions = actions * (~actions_is_pad).unsqueeze(-1)
        actions = actions[:, :, :].reshape(actions.shape[0], -1)  # [32, 150]

        q_preds = self.critic_forward(
            observations=observations,
            actions=actions,
            use_target=False,
            observation_features=observation_features,
            do_output_normalization=False,
        )

        # 4- Calculate loss
        # Compute state-action value loss (TD loss) for all of the Q functions in the ensemble.
        td_target_duplicate = einops.repeat(td_target, "b -> e b", e=q_preds.shape[0])
        # You compute the mean loss of the batch for each critic and then to compute the final loss you sum them up

        q_preds = q_preds[:, ~actions_is_pad[:, -1]]
        td_target_duplicate = td_target_duplicate[:, ~actions_is_pad[:, -1]]

        critics_loss = (
            F.mse_loss(
                input=q_preds,
                target=td_target_duplicate,
                reduction="none",
            ).mean(dim=1)
        ).sum()

        info = {
            # "critic_loss": critic_loss,
            "predicted_qs": torch.mean(q_preds),
            "target_qs": torch.mean(td_target_duplicate),
            "rewards": rewards.mean(),
            "actions_is_pad": torch.mean(actions_is_pad.float()),
        }

        return critics_loss, info

    # def compute_loss_discrete_critic(
    #     self,
    #     observations,
    #     actions,
    #     rewards,
    #     next_observations,
    #     done,
    #     observation_features=None,
    #     next_observation_features=None,
    #     complementary_info=None,
    # ):
    #     # NOTE: We only want to keep the discrete action part
    #     # In the buffer we have the full action space (continuous + discrete)
    #     # We need to split them before concatenating them in the critic forward
    #     actions_discrete: Tensor = actions[:, DISCRETE_DIMENSION_INDEX:].clone()
    #     actions_discrete = torch.round(actions_discrete)
    #     actions_discrete = actions_discrete.long()

    #     with torch.no_grad():
    #         # last_next_observations = {key: val[:, -1] for key, val in next_observations.items()}
    #         # action, log_prob, action_probs
    #         _, next_log_probs, next_action_probs = self.discrete_actor(
    #             next_observations, next_observation_features
    #         )
    #         # next_action_preds [228]
    #         # next_state_action_probs [228, 3]
    #         # best_next_discrete_action [228, 3]

    #         # Get target Q-values from target network
    #         target_next_discrete_qs = self.discrete_critic_forward(
    #             observations=next_observations,
    #             # actions=next_action_preds,
    #             use_target=True,
    #             observation_features=next_observation_features,
    #         )  # [2, 228, 3]

    #         # critics subsample size
    #         target_next_discrete_q, _ = target_next_discrete_qs.min(dim=0)  # Get values from min operation
    #         if self.config.use_backup_entropy:
    #             target_next_discrete_q = target_next_discrete_q - (self.temperature * next_log_probs)

    #         target_next_discrete_q = target_next_discrete_q * next_action_probs

    #         # adapt Q-target for discrete Q-function
    #         target_next_discrete_q = target_next_discrete_q.sum(dim=1)

    #         # # Use gather to select Q-values for best actions
    #         # target_next_discrete_q = torch.gather(
    #         #     target_next_discrete_qs, dim=1, index=next_action_preds.unsqueeze(-1)
    #         # ).squeeze(-1)

    #         # Compute target Q-value with Bellman equation
    #         rewards_discrete = rewards
    #         # if discrete_penalties is not None:
    #         #     rewards_discrete = rewards + discrete_penalties
    #         target_discrete_q = rewards_discrete + (1 - done) * self.config.discount * target_next_discrete_q

    #     # 3- compute predicted qs
    #     # if self.config.num_discrete_actions is not None:
    #     #     # NOTE: We only want to keep the continuous action part
    #     #     # In the buffer we have the full action space (continuous + discrete)
    #     #     # We need to split them before concatenating them in the critic forward
    #     #     actions: Tensor = actions[:, :DISCRETE_DIMENSION_INDEX]

    #     predicted_discrete_qs = self.discrete_critic_forward(
    #         observations=observations,
    #         # actions=actions_discrete,
    #         use_target=False,
    #         observation_features=observation_features,
    #     )

    #     # Use gather to select Q-values for taken actions
    #     # Expand actions_discrete to match the critic ensemble dimension
    #     actions_discrete_expanded = actions_discrete.unsqueeze(0).expand(
    #         predicted_discrete_qs.size(0), -1, -1
    #     )
    #     predicted_discrete_q = torch.gather(
    #         predicted_discrete_qs, dim=2, index=actions_discrete_expanded
    #     ).squeeze(-1)

    #     target_discrete_q_expanded = einops.repeat(
    #         target_discrete_q, "b -> e b", e=predicted_discrete_qs.shape[0]
    #     )

    #     # Compute MSE loss between predicted and target Q-values
    #     # discrete_critic_loss = F.mse_loss(
    #     #     input=predicted_discrete_q, target=target_discrete_q_expanded
    #     # )
    #     discrete_critic_loss = (
    #         F.mse_loss(
    #             input=predicted_discrete_q,
    #             target=target_discrete_q_expanded,
    #             reduction="none",
    #         ).mean(dim=1)
    #     ).sum()

    #     info = {
    #         "predicted_qs": torch.mean(predicted_discrete_q),
    #         "target_qs": torch.mean(target_discrete_q_expanded),
    #         "rewards": rewards_discrete.mean(),
    #     }

    #     return discrete_critic_loss, info

    # def compute_loss_temperature(self, observations, observation_features: Tensor | None = None) -> Tensor:
    #     """Compute the temperature loss"""
    #     # calculate temperature loss
    #     with torch.no_grad():
    #         _, log_probs, _ = self.discrete_actor(observations, observation_features)
    #     temperature_loss = (-self.log_alpha.exp() * (log_probs + self.target_entropy)).mean()
    #     return temperature_loss

    def compute_loss_actor_bc_flow(
        self,
        observations,
        observation_features: Tensor | None,
        actions: Tensor | None,
        actions_is_pad: Tensor | None,
    ) -> Tensor:
        actions = self.normalize_targets({"action": actions})["action"]
        # actions = actions[:, :, :DISCRETE_DIMENSION_INDEX]
        bc_flow_loss, _, _ = self.actor_bc_flow(observations, observation_features, actions, actions_is_pad)

        info = {}

        return bc_flow_loss, info

    def compute_loss_actor_onestep_flow(
        self,
        observations,
        observation_features: Tensor | None,
        actions: Tensor | None,
        actions_is_pad: Tensor | None,
    ) -> Tensor:
        batch_size = actions.shape[0]
        # action_dim = self.actor_onestep_flow.action_dim # self.config['action_dim']
        # action_dim = self.actor_bc_flow.action_dim

        # Distillation loss.
        # noises = torch.randn(batch_size, action_dim, device=observations['observation.state'].device)
        with torch.no_grad():
            noises = self.actor_bc_flow.encoder.vla.model.sample_noise(
                (
                    batch_size,
                    self.actor_bc_flow.encoder.vla.model.config.chunk_size,
                    self.actor_bc_flow.encoder.vla.model.config.max_action_dim,
                ),
                device=observations["observation.state"].device,
            )

        target_flow_actions, _, _ = self.actor_bc_flow.sample_actions(
            observations, observation_features, noises=noises
        )

        target_flow_actions = target_flow_actions[:, :, :4]
        # target_flow_actions = self.actor_bc_flow.encoder.vla.unnormalize_outputs({"action": target_flow_actions})["action"]

        # target_flow_actions = torch.clamp(target_flow_actions, -1.0, 1.0)
        target_flow_actions = target_flow_actions.reshape(target_flow_actions.shape[0], -1)
        # _, _, actor_actions  = self.actor_onestep_flow(observations, observation_features, noises)
        actor_actions, _, _ = self.actor_onestep_flow(
            observations, observation_features, noises[:, :, :4].reshape(batch_size, -1)
        )

        distill_loss = F.mse_loss(input=actor_actions, target=target_flow_actions)

        # Q loss.
        actor_actions = torch.clamp(actor_actions, -1.0, 1.0)

        q_preds = self.critic_forward(
            observations=observations,
            actions=actor_actions,
            use_target=False,
            observation_features=observation_features,
            do_output_normalization=False,
        )
        # min_q_preds = q_preds.min(dim=0)[0]
        min_q_preds = q_preds.mean(dim=0, keepdim=True)
        q_loss = -min_q_preds.mean()

        if self.config.normalize_q_loss:
            lam = 1.0 / q_preds.abs().mean().detach()
            q_loss = lam * q_loss

        # actor_loss = ((self.temperature * log_probs) - min_q_preds).mean()
        actor_onestep_flow_loss = self.config.alpha * distill_loss + q_loss

        info = {
            "q_loss": q_loss,
            "predicted_qs": torch.mean(q_preds),
            "distill_loss": distill_loss,
            "q": torch.mean(min_q_preds),
        }

        return actor_onestep_flow_loss, info

    # def compute_loss_discrete_actor(
    #     self,
    #     observations,
    #     observation_features: Tensor | None = None,
    # ) -> Tensor:
    #     _, log_probs, action_probs = self.discrete_actor(observations, observation_features)
    #     # actions_pi [228]
    #     # log_probs [228, 3]
    #     # action_probs [228, 3]

    #     q_preds = self.discrete_critic_forward(
    #         observations=observations,
    #         use_target=False,
    #         observation_features=observation_features,
    #     )  # [2, 228, 3]

    #     # min_q_preds = q_preds
    #     min_q_preds = q_preds.min(dim=0)[0]  # [3]
    #     # min_q_preds = torch.gather(q_preds, dim=1, index=actions_pi.unsqueeze(-1))

    #     actor_loss = (action_probs * ((self.temperature * log_probs) - min_q_preds)).mean()
    #     return actor_loss, {
    #         "q_loss": torch.mean(-min_q_preds),
    #         "predicted_qs": torch.mean(q_preds),
    #     }

    def _init_normalization(self, dataset_stats):
        """Initialize input/output normalization modules."""
        self.normalize_inputs = nn.Identity()
        self.normalize_targets = nn.Identity()
        if self.config.dataset_stats is not None:
            params = _convert_normalization_params_to_tensor(self.config.dataset_stats)
            self.normalize_inputs = NormalizeBuffer(
                self.config.input_features, self.config.normalization_mapping, params
            )
            stats = dataset_stats or params
            self.normalize_targets = NormalizeBuffer(
                self.config.output_features, self.config.normalization_mapping, stats
            )
            self.unnormalize_targets = UnnormalizeBuffer(
                self.config.output_features, self.config.normalization_mapping, stats
            )

    def _init_encoders(self):
        """Initialize shared or separate encoders for actor and critic."""
        self.shared_encoder = self.config.shared_encoder
        self.encoder_critic = SACObservationEncoder(self.config, self.normalize_inputs)
        # self.encoder_discrete_critic = (
        #     self.encoder_critic
        #     if self.shared_encoder
        #     else SACObservationEncoder(self.config, self.normalize_inputs)
        # )

        cfg_policy: SmolVLAConfig = PreTrainedConfig.from_pretrained("lerobot/smolvla_base")
        # cfg_policy.n_obs_steps: int = 1
        cfg_policy.chunk_size = self.config.chunk_size
        cfg_policy.n_action_steps = self.config.chunk_size
        # cfg_policy.train_state_proj = False
        # cfg_policy.max_action_dim = 3
        # cfg_policy.max_state_dim = 18

        cfg_policy.normalization_mapping = {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            # "ENV": NormalizationMode.MIN_MAX,
            # "ACTION": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.IDENTITY,
        }
        kwargs = {}
        # kwargs["pretrained_name_or_path"] = "lerobot/smolvla_base"

        # from lerobot.configs.train import TrainRLServerPipelineConfig
        # from lerobot.datasets.factory import make_dataset

        # cfg = TrainRLServerPipelineConfig.from_pretrained("/home/jpizarrom/Projects/lerobot-hil-serl-configs/fql/train_gym_hil_env_fqlvla_lilkm_pushfreq50.json")
        # cfg.policy.chunk_size = self.config.chunk_size
        # offline_dataset = make_dataset(cfg)
        # for k in ["min", "max", "mean", "std"]:
        #     offline_dataset.meta.stats["action"][k] = offline_dataset.meta.stats["action"][k][:-1]
        # kwargs["dataset_stats"] = offline_dataset.meta.stats
        kwargs["dataset_stats"] = _convert_normalization_params_to_tensor(self.config.dataset_stats)
        # features = env_to_policy_features(cfg.env)
        cfg_policy.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(4,))}
        cfg_policy.input_features = {
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
            "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(18,)),
        }
        # cfg_policy.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
        # cfg_policy.input_features = {key: ft for key, ft in features.items() if key not in cfg_policy.output_features}
        kwargs["config"] = cfg_policy

        self.vlm: SmolVLMForConditionalGeneration = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name_or_path="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            device_map="auto",
            torch_dtype="bfloat16",
            low_cpu_mem_usage=True,
        )
        # self.vlm.model.text_model.layers = self.vlm.model.text_model.layers[:cfg_policy.num_vlm_layers]

        # self.encoder_actor_onestep_flow = SACObservationEncoderVLA(self.config, SmolVLAPolicy(vlm=self.vlm, **kwargs))
        # self.encoder_actor_onestep_flow = SACObservationEncoderVLA(self.config, SmolVLAPolicy.from_pretrained(pretrained_name_or_path="lerobot/smolvla_base", vlm=self.vlm, **kwargs))
        self.encoder_actor_bc_flow = SACObservationEncoderVLA(
            self.config,
            SmolVLAPolicy.from_pretrained(
                pretrained_name_or_path="lerobot/smolvla_base", vlm=self.vlm, **kwargs
            ),
        )

        # self.encoder_actor_bc_flow = (
        #     self.encoder_critic
        #     if self.shared_encoder
        #     else SACObservationEncoder(self.config, self.normalize_inputs)
        # )

        self.encoder_actor_onestep_flow = (
            self.encoder_critic
            if self.shared_encoder
            else SACObservationEncoder(self.config, self.normalize_inputs)
        )

        # self.encoder_discrete_actor = (
        #     self.encoder_critic
        #     if self.shared_encoder
        #     else SACObservationEncoder(self.config, self.normalize_inputs)
        # )

        # self.encoder_actor_bc_flow = (
        #     self.encoder_critic
        #     if self.shared_encoder
        #     else SACObservationEncoder(self.config, self.normalize_inputs)
        # )
        # self.encoder_actor_onestep_flow = (
        #     self.encoder_critic
        #     if self.shared_encoder
        #     else SACObservationEncoder(self.config, self.normalize_inputs)
        # )

    def _init_critics(self, continuous_action_dim):
        """Build critic ensemble, targets, and optional discrete critic."""
        heads = [
            CriticHead(
                input_dim=self.encoder_critic.output_dim
                + (continuous_action_dim + 1) * self.config.chunk_size,
                **asdict(self.config.critic_network_kwargs),
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_ensemble = CriticEnsemble(
            encoder=self.encoder_critic, ensemble=heads, output_normalization=self.normalize_targets
        )
        target_heads = [
            CriticHead(
                input_dim=self.encoder_critic.output_dim
                + (continuous_action_dim + 1) * self.config.chunk_size,
                **asdict(self.config.critic_network_kwargs),
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_target = CriticEnsemble(
            encoder=self.encoder_critic, ensemble=target_heads, output_normalization=self.normalize_targets
        )
        self.critic_target.load_state_dict(self.critic_ensemble.state_dict())

        if self.config.use_torch_compile:
            self.critic_ensemble = torch.compile(self.critic_ensemble)
            self.critic_target = torch.compile(self.critic_target)

        # if self.config.num_discrete_actions is not None:
        #     self._init_discrete_critics()

    # def _init_discrete_critics(self):
    #     """Build discrete discrete critic ensemble and target networks."""
    #     heads = [
    #         DiscreteCriticHead(
    #             input_dim=self.encoder_discrete_critic.output_dim,
    #             output_dim=self.config.num_discrete_actions,
    #             **asdict(self.config.discrete_critic_network_kwargs),
    #         )
    #         for _ in range(self.config.num_critics)
    #     ]
    #     self.discrete_critic = DiscreteCriticEnsemble(encoder=self.encoder_discrete_critic, ensemble=heads)
    #     target_heads = [
    #         DiscreteCriticHead(
    #             input_dim=self.encoder_discrete_critic.output_dim,
    #             output_dim=self.config.num_discrete_actions,
    #             **asdict(self.config.discrete_critic_network_kwargs),
    #         )
    #         for _ in range(self.config.num_critics)
    #     ]
    #     self.discrete_critic_target = DiscreteCriticEnsemble(
    #         encoder=self.encoder_discrete_critic, ensemble=target_heads
    #     )

    #     # TODO: (maractingi, azouitine) Compile the discrete critic
    #     self.discrete_critic_target.load_state_dict(self.discrete_critic.state_dict())

    def _init_actor_bc_flow(self, continuous_action_dim):
        """Initialize policy actor network and default target entropy."""
        # NOTE: The actor select only the continuous action part
        # params = asdict(self.config.actor_network_kwargs)
        # action_out_proj = copy.deepcopy(self.encoder_actor_bc_flow.vla.model.action_out_proj)
        # Linear(in_features=720, out_features=32, bias=True)
        self.actor_bc_flow = ActorVectorFieldPolicyVLA(
            encoder=self.encoder_actor_bc_flow,
            # network=MLP(input_dim=self.encoder_actor_bc_flow.output_dim + continuous_action_dim + 1, **params),
            # network=MLP(input_dim=720, **params),
            # network=action_out_proj,
            action_dim=continuous_action_dim,
            encoder_is_shared=self.shared_encoder,
            **asdict(self.config.policy_kwargs),
        )

        # params = asdict(self.config.actor_network_kwargs)
        # self.actor_bc_flow = ActorVectorFieldPolicy(
        #     encoder=self.encoder_actor_bc_flow,
        #     network=MLP(input_dim=self.encoder_actor_bc_flow.output_dim + continuous_action_dim *self.config.chunk_size + 1, **params),
        #     action_dim=continuous_action_dim * self.config.chunk_size,
        #     encoder_is_shared=self.shared_encoder,
        #     **asdict(self.config.policy_kwargs),
        # )

        self.target_entropy = self.config.target_entropy
        if self.target_entropy is None:
            dim = self.config.num_discrete_actions if self.config.num_discrete_actions is not None else 0
            self.target_entropy = -np.prod(dim) / 2

        # if self.config.num_discrete_actions is not None:
        #     self._init_discrete_actor()

    def _init_actor_onestep_flow(self, continuous_action_dim):
        """Initialize policy actor network and default target entropy."""
        # NOTE: The actor select only the continuous action part
        params = asdict(self.config.actor_network_kwargs)
        self.actor_onestep_flow = ActorVectorFieldPolicy(
            encoder=self.encoder_actor_onestep_flow,
            network=MLP(
                input_dim=self.encoder_actor_onestep_flow.output_dim
                + (continuous_action_dim + 1) * self.config.chunk_size,
                **params,
            ),
            action_dim=(continuous_action_dim + 1) * self.config.chunk_size,
            # num_discrete_actions=self.config.num_discrete_actions* self.config.chunk_size,
            encoder_is_shared=self.shared_encoder,
            **asdict(self.config.policy_kwargs),
        )
        # action_out_proj = copy.deepcopy(self.encoder_actor_onestep_flow.vla.model.action_out_proj)
        # self.actor_onestep_flow = ActorVectorFieldPolicyVLAOneStep(
        #     encoder=self.encoder_actor_onestep_flow,
        #     # network=MLP(input_dim=self.encoder_actor_onestep_flow.output_dim + continuous_action_dim, **params),
        #     # network=MLP(input_dim=720, **params),
        #     # network=action_out_proj,
        #     action_dim=continuous_action_dim,
        #     encoder_is_shared=self.shared_encoder,
        #     **asdict(self.config.policy_kwargs),
        # )

    # def _init_discrete_actor(self):
    #     self.discrete_actor = DiscretePolicy(
    #         encoder=self.encoder_discrete_actor,
    #         network=MLP(
    #             input_dim=self.encoder_discrete_actor.output_dim,
    #             **asdict(self.config.discrete_actor_network_kwargs),
    #         ),
    #         action_dim=self.config.num_discrete_actions,
    #         encoder_is_shared=self.shared_encoder,
    #         **asdict(self.config.discrete_policy_kwargs),
    #     )

    def _init_temperature(self):
        """Set up temperature parameter and initial log_alpha."""
        temp_init = self.config.temperature_init
        self.log_alpha = nn.Parameter(torch.tensor([math.log(temp_init)]))
        self.temperature = self.log_alpha.exp().item()


class SACObservationEncoderVLA(nn.Module):
    """Encode image and/or state vector observations."""

    def __init__(self, config: FQLVLAConfig, vla: SmolVLAPolicy) -> None:
        super().__init__()
        self.config = config
        self.vla = vla

    def forward(
        self, obs: dict[str, Tensor], cache: dict[str, Tensor] | None = None, detach: bool = False
    ) -> Tensor:
        batch = self.vla._prepare_batch(obs)
        images, img_masks = self.vla.prepare_images(batch)
        state = self.vla.prepare_state(batch)
        batch["task"] = "pick up the pink cube"
        lang_tokens, lang_masks = self.vla.prepare_language(batch)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.vla.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )

        return prefix_embs, prefix_pad_masks, prefix_att_masks, state


class SACObservationEncoder(nn.Module):
    """Encode image and/or state vector observations."""

    def __init__(self, config: FQLVLAConfig, input_normalizer: nn.Module) -> None:
        super().__init__()
        self.config = config
        self.input_normalization = input_normalizer
        self._init_image_layers()
        self._init_state_layers()
        self._compute_output_dim()

    def _init_image_layers(self) -> None:
        self.image_keys = [k for k in self.config.input_features if is_image_feature(k)]
        self.has_images = bool(self.image_keys)
        if not self.has_images:
            return

        if self.config.vision_encoder_name is not None:
            self.image_encoder = PretrainedImageEncoder(self.config)
        else:
            self.image_encoder = DefaultImageEncoder(self.config)

        if self.config.freeze_vision_encoder:
            freeze_image_encoder(self.image_encoder)

        dummy = torch.zeros(1, *self.config.input_features[self.image_keys[0]].shape)
        with torch.no_grad():
            _, channels, height, width = self.image_encoder(dummy).shape

        self.spatial_embeddings = nn.ModuleDict()
        self.post_encoders = nn.ModuleDict()

        for key in self.image_keys:
            name = key.replace(".", "_")
            self.spatial_embeddings[name] = SpatialLearnedEmbeddings(
                height=height,
                width=width,
                channel=channels,
                num_features=self.config.image_embedding_pooling_dim,
            )
            self.post_encoders[name] = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(
                    in_features=channels * self.config.image_embedding_pooling_dim,
                    out_features=self.config.latent_dim,
                ),
                nn.LayerNorm(normalized_shape=self.config.latent_dim),
                nn.Tanh(),
            )

    def _init_state_layers(self) -> None:
        self.has_env = "observation.environment_state" in self.config.input_features
        self.has_state = "observation.state" in self.config.input_features
        if self.has_env:
            dim = self.config.input_features["observation.environment_state"].shape[0]
            self.env_encoder = nn.Sequential(
                nn.Linear(dim, self.config.latent_dim),
                nn.LayerNorm(self.config.latent_dim),
                nn.Tanh(),
            )
        if self.has_state:
            dim = self.config.input_features["observation.state"].shape[0]
            self.state_encoder = nn.Sequential(
                nn.Linear(dim, self.config.latent_dim),
                nn.LayerNorm(self.config.latent_dim),
                nn.Tanh(),
            )

    def _compute_output_dim(self) -> None:
        out = 0
        if self.has_images:
            out += len(self.image_keys) * self.config.latent_dim
        if self.has_env:
            out += self.config.latent_dim
        if self.has_state:
            out += self.config.latent_dim
        self._out_dim = out

    def forward(
        self, obs: dict[str, Tensor], cache: dict[str, Tensor] | None = None, detach: bool = False
    ) -> Tensor:
        obs = self.input_normalization(obs)
        parts = []
        if self.has_images:
            if cache is None:
                cache = self.get_cached_image_features(obs, normalize=False)
            parts.append(self._encode_images(cache, detach))
        if self.has_env:
            parts.append(self.env_encoder(obs["observation.environment_state"]))
        if self.has_state:
            # if obs["observation.state"].ndim == 3:
            #     # TODO: remove this when we have a proper dimension
            #     parts.append(self.state_encoder(obs["observation.state"].squeeze(1)))
            # else:
            #     parts.append(self.state_encoder(obs["observation.state"]))
            parts.append(self.state_encoder(obs["observation.state"]))
        if parts:
            return torch.cat(parts, dim=-1)

        raise ValueError(
            "No parts to concatenate, you should have at least one image or environment state or state"
        )

    def get_cached_image_features(self, obs: dict[str, Tensor], normalize: bool = False) -> dict[str, Tensor]:
        """Extract and optionally cache image features from observations.

        This function processes image observations through the vision encoder once and returns
        the resulting features.
        When the image encoder is shared between actor and critics AND frozen, these features can be safely cached and
        reused across policy components (actor, critic, discrete_critic), avoiding redundant forward passes.

        Performance impact:
        - The vision encoder forward pass is typically the main computational bottleneck during training and inference
        - Caching these features can provide 2-4x speedup in training and inference

        Normalization behavior:
        - When called from inside forward(): set normalize=False since inputs are already normalized
        - When called from outside forward(): set normalize=True to ensure proper input normalization

        Usage patterns:
        - Called in select_action() with normalize=True
        - Called in learner.py's get_observation_features() to pre-compute features for all policy components
        - Called internally by forward() with normalize=False

        Args:
            obs: Dictionary of observation tensors containing image keys
            normalize: Whether to normalize observations before encoding
                      Set to True when calling directly from outside the encoder's forward method
                      Set to False when calling from within forward() where inputs are already normalized

        Returns:
            Dictionary mapping image keys to their corresponding encoded features
        """
        if normalize:
            obs = self.input_normalization(obs)
        batched = torch.cat([obs[k] for k in self.image_keys], dim=0)
        out = self.image_encoder(batched)
        chunks = torch.chunk(out, len(self.image_keys), dim=0)
        return dict(zip(self.image_keys, chunks, strict=False))

    def _encode_images(self, cache: dict[str, Tensor], detach: bool) -> Tensor:
        """Encode image features from cached observations.

        This function takes pre-encoded image features from the cache and applies spatial embeddings and post-encoders.
        It also supports detaching the encoded features if specified.

        Args:
            cache (dict[str, Tensor]): The cached image features.
            detach (bool): Usually when the encoder is shared between actor and critics,
            we want to detach the encoded features on the policy side to avoid backprop through the encoder.
            More detail here `https://cdn.aaai.org/ojs/17276/17276-13-20770-1-2-20210518.pdf`

        Returns:
            Tensor: The encoded image features.
        """
        feats = []
        for k, feat in cache.items():
            safe_key = k.replace(".", "_")
            x = self.spatial_embeddings[safe_key](feat)
            x = self.post_encoders[safe_key](x)
            if detach:
                x = x.detach()
            feats.append(x)
        return torch.cat(feats, dim=-1)

    @property
    def output_dim(self) -> int:
        return self._out_dim


class MLP(nn.Module):
    """Multi-layer perceptron builder.

    Dynamically constructs a sequence of layers based on `hidden_dims`:
      1) Linear (in_dim -> out_dim)
      2) Optional Dropout if `dropout_rate` > 0 and (not final layer or `activate_final`)
      3) LayerNorm on the output features
      4) Activation (standard for intermediate layers, `final_activation` for last layer if `activate_final`)

    Arguments:
        input_dim (int): Size of input feature dimension.
        hidden_dims (list[int]): Sizes for each hidden layer.
        activations (Callable or str): Activation to apply between layers.
        activate_final (bool): Whether to apply activation at the final layer.
        dropout_rate (Optional[float]): Dropout probability applied before normalization and activation.
        final_activation (Optional[Callable or str]): Activation for the final layer when `activate_final` is True.

    For each layer, `in_dim` is updated to the previous `out_dim`. All constructed modules are
    stored in `self.net` as an `nn.Sequential` container.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.GELU(),
        activate_final: bool = False,
        # dropout_rate: float | None = None,
        final_activation: Callable[[torch.Tensor], torch.Tensor] | str | None = None,
        layer_norm: bool = False,
        default_init: float | None = None,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        total = len(hidden_dims)

        for idx, out_dim in enumerate(hidden_dims):
            # 1) linear transform
            layers.append(nn.Linear(in_dim, out_dim))

            if default_init is not None:
                nn.init.uniform_(layers[-1].weight, -default_init, default_init)
                nn.init.uniform_(layers[-1].bias, -default_init, default_init)
            else:
                orthogonal_init()(layers[-1].weight)
                # nn.init.zeros_(layers[-1].bias)

            is_last = idx == total - 1
            # 2-4) optionally add dropout, normalization, and activation
            if not is_last or activate_final:
                # if dropout_rate and dropout_rate > 0:
                #     layers.append(nn.Dropout(p=dropout_rate))
                act_cls = final_activation if is_last and final_activation else activations
                act = act_cls if isinstance(act_cls, nn.Module) else getattr(nn, act_cls)()
                layers.append(act)

                if layer_norm:
                    layers.append(nn.LayerNorm(out_dim))

            in_dim = out_dim

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CriticHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.GELU(),
        activate_final: bool = False,
        # dropout_rate: float | None = None,
        default_init: float | None = None,
        init_final: float | None = None,
        final_activation: Callable[[torch.Tensor], torch.Tensor] | str | None = None,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.net = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activations=activations,
            activate_final=activate_final,
            # dropout_rate=dropout_rate,
            final_activation=final_activation,
            layer_norm=layer_norm,
            default_init=default_init,
        )
        self.output_layer = nn.Linear(in_features=hidden_dims[-1], out_features=1)
        if init_final is not None:
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.output_layer.weight)
            # nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.net(x))


class CriticEnsemble(nn.Module):
    """
    CriticEnsemble wraps multiple CriticHead modules into an ensemble.

    Args:
        encoder (SACObservationEncoder): encoder for observations.
        ensemble (List[CriticHead]): list of critic heads.
        output_normalization (nn.Module): normalization layer for actions.
        init_final (float | None): optional initializer scale for final layers.

    Forward returns a tensor of shape (num_critics, batch_size) containing Q-values.
    """

    def __init__(
        self,
        encoder: SACObservationEncoder,
        ensemble: list[CriticHead],
        output_normalization: nn.Module,
        init_final: float | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.init_final = init_final
        self.output_normalization = output_normalization
        self.critics = nn.ModuleList(ensemble)

    def forward(
        self,
        observations: dict[str, torch.Tensor],
        actions: torch.Tensor,
        observation_features: torch.Tensor | None = None,
        do_output_normalization: bool = True,
    ) -> torch.Tensor:
        device = get_device_from_parameters(self)
        # Move each tensor in observations to device
        observations = {k: v.to(device) for k, v in observations.items()}
        if do_output_normalization:
            # NOTE: We normalize actions it helps for sample efficiency
            actions = actions.reshape(actions.shape[0], -1, 4)  # Reshape to match action dimensions
            actions: dict[str, torch.tensor] = {"action": actions}
            # NOTE: Normalization layer took dict in input and outputs a dict that why
            actions = self.output_normalization(actions)["action"]
            actions = actions.reshape(actions.shape[0], -1)  # Flatten the action dimension
        actions = actions.to(device)

        obs_enc = self.encoder(observations, cache=observation_features)

        inputs = torch.cat([obs_enc, actions], dim=-1)

        # Loop through critics and collect outputs
        q_values = []
        for critic in self.critics:
            q_values.append(critic(inputs))

        # Stack outputs to match expected shape [num_critics, batch_size]
        q_values = torch.stack([q.squeeze(-1) for q in q_values], dim=0)
        return q_values


# class DiscreteCriticHead(nn.Module):
#     def __init__(
#         self,
#         # encoder: nn.Module,
#         input_dim: int,
#         hidden_dims: list[int],
#         output_dim: int = 3,
#         activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.GELU(),
#         activate_final: bool = False,
#         # dropout_rate: float | None = None,
#         default_init: float | None = None,
#         init_final: float | None = None,
#         final_activation: Callable[[torch.Tensor], torch.Tensor] | str | None = None,
#         layer_norm: bool = False,
#     ):
#         super().__init__()
#         # self.encoder = encoder
#         self.output_dim = output_dim

#         self.net = MLP(
#             input_dim=input_dim,
#             hidden_dims=hidden_dims,
#             activations=activations,
#             activate_final=activate_final,
#             # dropout_rate=dropout_rate,
#             final_activation=final_activation,
#             layer_norm=layer_norm,
#             default_init=default_init,
#         )

#         self.output_layer = nn.Linear(in_features=hidden_dims[-1], out_features=self.output_dim)
#         if init_final is not None:
#             nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
#             nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
#         else:
#             orthogonal_init()(self.output_layer.weight)
#             # nn.init.zeros_(self.output_layer.bias)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # device = get_device_from_parameters(self)
#         # observations = {k: v.to(device) for k, v in observations.items()}
#         # obs_enc = self.encoder(observations, cache=observation_features)
#         return self.output_layer(self.net(x))


# class DiscreteCriticEnsemble(nn.Module):
#     """
#     CriticEnsemble wraps multiple CriticHead modules into an ensemble.

#     Args:
#         encoder (SACObservationEncoder): encoder for observations.
#         ensemble (List[CriticHead]): list of critic heads.
#         output_normalization (nn.Module): normalization layer for actions.
#         init_final (float | None): optional initializer scale for final layers.

#     Forward returns a tensor of shape (num_critics, batch_size) containing Q-values.
#     """

#     def __init__(
#         self,
#         encoder: SACObservationEncoder,
#         ensemble: list[CriticHead],
#         # output_normalization: nn.Module,
#         init_final: float | None = None,
#     ):
#         super().__init__()
#         self.encoder = encoder
#         self.init_final = init_final
#         # self.output_normalization = output_normalization
#         self.critics = nn.ModuleList(ensemble)

#     def forward(
#         self,
#         observations: dict[str, torch.Tensor],
#         observation_features: torch.Tensor | None = None,
#     ) -> torch.Tensor:
#         device = get_device_from_parameters(self)
#         # Move each tensor in observations to device
#         observations = {k: v.to(device) for k, v in observations.items()}
#         # NOTE: We normalize actions it helps for sample efficiency
#         # actions: dict[str, torch.tensor] = {"action": actions}
#         # NOTE: Normalization layer took dict in input and outputs a dict that why
#         # actions = self.output_normalization(actions)["action"]
#         # actions = actions.to(device)

#         obs_enc = self.encoder(observations, cache=observation_features)

#         # inputs = torch.cat([obs_enc, actions], dim=-1)

#         # Loop through critics and collect outputs
#         q_values = []
#         for critic in self.critics:
#             q_values.append(critic(obs_enc))

#         # Stack outputs to match expected shape [num_critics, batch_size]
#         q_values = torch.stack([q.squeeze(-1) for q in q_values], dim=0)
#         return q_values


# class Policy(nn.Module):
#     def __init__(
#         self,
#         encoder: SACObservationEncoder,
#         network: nn.Module,
#         action_dim: int,
#         std_min: float = -5,
#         std_max: float = 2,
#         fixed_std: torch.Tensor | None = None,
#         init_final: float | None = None,
#         use_tanh_squash: bool = False,
#         encoder_is_shared: bool = False,
#     ):
#         super().__init__()
#         self.encoder: SACObservationEncoder = encoder
#         self.network = network
#         self.action_dim = action_dim
#         self.std_min = std_min
#         self.std_max = std_max
#         self.fixed_std = fixed_std
#         self.use_tanh_squash = use_tanh_squash
#         self.encoder_is_shared = encoder_is_shared

#         # Find the last Linear layer's output dimension
#         for layer in reversed(network.net):
#             if isinstance(layer, nn.Linear):
#                 out_features = layer.out_features
#                 break
#         # Mean layer
#         self.mean_layer = nn.Linear(out_features, action_dim)
#         if init_final is not None:
#             nn.init.uniform_(self.mean_layer.weight, -init_final, init_final)
#             nn.init.uniform_(self.mean_layer.bias, -init_final, init_final)
#         else:
#             orthogonal_init()(self.mean_layer.weight)

#         # Standard deviation layer or parameter
#         if fixed_std is None:
#             self.std_layer = nn.Linear(out_features, action_dim)
#             if init_final is not None:
#                 nn.init.uniform_(self.std_layer.weight, -init_final, init_final)
#                 nn.init.uniform_(self.std_layer.bias, -init_final, init_final)
#             else:
#                 orthogonal_init()(self.std_layer.weight)

#     def forward(
#         self,
#         observations: torch.Tensor,
#         observation_features: torch.Tensor | None = None,
#     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         # We detach the encoder if it is shared to avoid backprop through it
#         # This is important to avoid the encoder to be updated through the policy
#         obs_enc = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)

#         # Get network outputs
#         outputs = self.network(obs_enc)
#         means = self.mean_layer(outputs)

#         # Compute standard deviations
#         if self.fixed_std is None:
#             log_std = self.std_layer(outputs)
#             std = torch.exp(log_std)  # Match JAX "exp"
#             std = torch.clamp(std, self.std_min, self.std_max)  # Match JAX default clip
#         else:
#             std = self.fixed_std.expand_as(means)

#         # Build transformed distribution
#         dist = TanhMultivariateNormalDiag(loc=means, scale_diag=std)

#         # Sample actions (reparameterized)
#         actions = dist.rsample()

#         # Compute log_probs
#         log_probs = dist.log_prob(actions)

#         return actions, log_probs, means

#     def get_features(self, observations: torch.Tensor) -> torch.Tensor:
#         """Get encoded features from observations"""
#         device = get_device_from_parameters(self)
#         observations = observations.to(device)
#         if self.encoder is not None:
#             with torch.inference_mode():
#                 return self.encoder(observations)
#         return observations


class ActorVectorFieldPolicy(nn.Module):
    """
    Actor vector field network for flow matching.

    Args:
        hidden_dims (list[int]): Hidden layer dimensions.
        action_dim (int): Action dimension.
        layer_norm (bool): Whether to apply layer normalization.
        encoder (nn.Module, optional): Optional encoder module to encode the inputs.
    """

    def __init__(
        self,
        encoder: SACObservationEncoder,
        network: nn.Module,
        action_dim: int,
        init_final: float | None = None,
        encoder_is_shared: bool = False,
    ):
        super().__init__()
        self.encoder: SACObservationEncoder = encoder
        self.network = network
        self.action_dim = action_dim
        self.encoder_is_shared = encoder_is_shared

        # Find the last Linear layer's output dimension
        for layer in reversed(network.net):
            if isinstance(layer, nn.Linear):
                out_features = layer.out_features
                break

        self.output_layer = nn.Linear(out_features, action_dim)
        if init_final is not None:
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.output_layer.weight)
            # nn.init.zeros_(self.output_layer.bias)

    def forward(
        self,
        observations: torch.Tensor,
        observation_features: torch.Tensor | None,
        actions: torch.Tensor,
        times: torch.Tensor = None,
        # is_encoded: bool = False,
    ) -> torch.Tensor:
        """
        Return the vectors at the given states, actions, and times (optional).

        Args:
            observations (Tensor): Observations.
            actions (Tensor): Actions.
            times (Tensor, optional): Times.
            is_encoded (bool): Whether the observations are already encoded.
        """
        # if not is_encoded and self.encoder is not None:
        #     observations = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)
        obs_enc = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)
        inputs = [obs_enc, actions]
        if times is not None:
            inputs.append(times)
        x = torch.cat(inputs, dim=-1)

        # Get network outputs
        outputs = self.output_layer(self.network(x))
        return outputs, None, None  # Return None for log_probs and means as they are not used in this context


class ActorVectorFieldPolicyVLA(nn.Module):
    """
    Actor vector field network for flow matching.

    Args:
        hidden_dims (list[int]): Hidden layer dimensions.
        action_dim (int): Action dimension.
        layer_norm (bool): Whether to apply layer normalization.
        encoder (nn.Module, optional): Optional encoder module to encode the inputs.
    """

    def __init__(
        self,
        encoder: SACObservationEncoderVLA,
        # network: nn.Module,
        action_dim: int,
        init_final: float | None = None,
        encoder_is_shared: bool = False,
    ):
        super().__init__()
        self.encoder: SACObservationEncoderVLA = encoder
        # self.network = network
        self.action_dim = action_dim
        self.encoder_is_shared = encoder_is_shared

        # # Find the last Linear layer's output dimension
        # for layer in reversed(network.net):
        #     if isinstance(layer, nn.Linear):
        #         out_features = layer.out_features
        #         break

        # # self.output_layer = nn.Linear(out_features, action_dim)
        # self.output_layer = nn.Linear(out_features, 32)
        # if init_final is not None:
        #     nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
        #     nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        # else:
        #     orthogonal_init()(self.output_layer.weight)

    def forward(
        self,
        observations: torch.Tensor,
        observation_features: torch.Tensor | None,
        actions: torch.Tensor,
        actions_is_pad: torch.Tensor,
        # times: torch.Tensor = None,
        # is_encoded: bool = False,
        # noises: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Return the vectors at the given states, actions, and times (optional).

        Args:
            observations (Tensor): Observations.
            actions (Tensor): Actions.
            times (Tensor, optional): Times.
            is_encoded (bool): Whether the observations are already encoded.
        """
        # if not is_encoded and self.encoder is not None:
        #     observations = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)
        # obs_enc = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)

        # prefix_embs, prefix_pad_masks, prefix_att_masks, state = obs_enc
        # bsize = state.shape[0]

        observations_with_task = {
            "task": "pick up the pink cube",
            "action": actions,  # Assuming actions are part of the observations
            "actions_is_pad": actions_is_pad,
            **observations,
        }
        loss, loss_dict, v_t = self.encoder.vla.forward(observations_with_task, noise=None)

        return (
            loss,
            loss_dict,
            v_t,
        )  # Return None for log_probs and means as they are not used in this context

    @torch.no_grad()
    def sample_actions(
        self,
        observations: torch.Tensor,
        observation_features: torch.Tensor | None,
        actions: torch.Tensor = None,
        times: torch.Tensor = None,
        noises: torch.Tensor | None = None,
        # is_encoded: bool = False,
    ) -> torch.Tensor:
        """
        Return the vectors at the given states, actions, and times (optional).

        Args:
            observations (Tensor): Observations.
            actions (Tensor): Actions.
            times (Tensor, optional): Times.
            is_encoded (bool): Whether the observations are already encoded.
        """
        # if not is_encoded and self.encoder is not None:
        #     observations = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)
        # obs_enc = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)

        # prefix_embs, prefix_pad_masks, prefix_att_masks, state = obs_enc
        # bsize = state.shape[0]

        observations_with_task = {
            "task": "pick up the pink cube",
            **observations,
        }

        batch = self.encoder.vla._prepare_batch(observations_with_task)
        images, img_masks = self.encoder.vla.prepare_images(batch)
        state = self.encoder.vla.prepare_state(batch)
        lang_tokens, lang_masks = self.encoder.vla.prepare_language(batch)

        actions = self.encoder.vla.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noises
        )

        # Unpad actions
        # original_action_dim = self.encoder.vla.config.action_feature.shape[0]
        # actions = actions[:, :, :original_action_dim]

        # actions = self.encoder.vla.unnormalize_outputs({"action": actions})["action"]

        actions = torch.clamp(actions, -1.0, 1.0)

        return actions, None, None  # Return None for log_probs and means as they are not used in this context


class ActorVectorFieldPolicyVLAOneStep(nn.Module):
    """
    Actor vector field network for flow matching.

    Args:
        hidden_dims (list[int]): Hidden layer dimensions.
        action_dim (int): Action dimension.
        layer_norm (bool): Whether to apply layer normalization.
        encoder (nn.Module, optional): Optional encoder module to encode the inputs.
    """

    def __init__(
        self,
        encoder: SACObservationEncoderVLA,
        # network: nn.Module,
        action_dim: int,
        init_final: float | None = None,
        encoder_is_shared: bool = False,
    ):
        super().__init__()
        self.encoder: SACObservationEncoderVLA = encoder
        # self.network = network
        self.action_dim = action_dim
        self.encoder_is_shared = encoder_is_shared

        # # Find the last Linear layer's output dimension
        # for layer in reversed(network.net):
        #     if isinstance(layer, nn.Linear):
        #         out_features = layer.out_features
        #         break

        # # self.output_layer = nn.Linear(out_features, action_dim)
        # self.output_layer = nn.Linear(out_features, 32)
        # if init_final is not None:
        #     nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
        #     nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        # else:
        #     orthogonal_init()(self.output_layer.weight)

    @torch.no_grad()
    def forward(
        self,
        observations: torch.Tensor,
        observation_features: torch.Tensor | None,
        # actions: torch.Tensor = None,
        # times: torch.Tensor = None,
        noises: torch.Tensor | None = None,
        # is_encoded: bool = False,
    ) -> torch.Tensor:
        """
        Return the vectors at the given states, actions, and times (optional).

        Args:
            observations (Tensor): Observations.
            actions (Tensor): Actions.
            times (Tensor, optional): Times.
            is_encoded (bool): Whether the observations are already encoded.
        """
        # if not is_encoded and self.encoder is not None:
        #     observations = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)
        # obs_enc = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)

        # prefix_embs, prefix_pad_masks, prefix_att_masks, state = obs_enc
        # bsize = state.shape[0]

        observations_with_task = {
            "task": "pick up the pink cube",
            **observations,
        }

        batch = self.encoder.vla._prepare_batch(observations_with_task)
        images, img_masks = self.encoder.vla.prepare_images(batch)
        state = self.encoder.vla.prepare_state(batch)
        lang_tokens, lang_masks = self.encoder.vla.prepare_language(batch)

        # actions = self.encoder.vla.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noises)
        bsize = state.shape[0]
        device = state.device

        if noises is None:
            actions_shape = (
                bsize,
                self.encoder.vla.model.config.chunk_size,
                self.encoder.vla.model.config.max_action_dim,
            )
            noise = self.encoder.vla.model.sample_noise(actions_shape, device)
        else:
            noise = noises
        # time = self.encoder.vla.model.sample_time(noise.shape[0], noise.device)

        x_t = noise
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.encoder.vla.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.encoder.vla.model.embed_suffix_notime(x_t)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.encoder.vla.model.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.encoder.vla.model.config.chunk_size :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.encoder.vla.model.action_out_proj(suffix_out)

        return None, None, v_t  # Return None for log_probs and means as they are not used in this context


class DefaultImageEncoder(nn.Module):
    def __init__(self, config: FQLVLAConfig):
        super().__init__()
        image_key = next(key for key in config.input_features if is_image_feature(key))
        self.image_enc_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=config.input_features[image_key].shape[0],
                out_channels=config.image_encoder_hidden_dim,
                kernel_size=7,
                stride=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.image_encoder_hidden_dim,
                out_channels=config.image_encoder_hidden_dim,
                kernel_size=5,
                stride=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.image_encoder_hidden_dim,
                out_channels=config.image_encoder_hidden_dim,
                kernel_size=3,
                stride=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.image_encoder_hidden_dim,
                out_channels=config.image_encoder_hidden_dim,
                kernel_size=3,
                stride=2,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.image_enc_layers(x)
        return x


class DiscretePolicy(nn.Module):
    def __init__(
        self,
        encoder: SACObservationEncoder,
        network: nn.Module,
        action_dim: int,
        # std_min: float = -5,
        # std_max: float = 2,
        # fixed_std: torch.Tensor | None = None,
        init_final: float | None = None,
        # use_tanh_squash: bool = False,
        encoder_is_shared: bool = False,
    ):
        super().__init__()
        self.encoder: SACObservationEncoder = encoder
        self.network = network
        self.action_dim = action_dim
        # self.std_min = std_min
        # self.std_max = std_max
        # self.fixed_std = fixed_std
        # self.use_tanh_squash = use_tanh_squash
        self.encoder_is_shared = encoder_is_shared

        # Find the last Linear layer's output dimension
        for layer in reversed(network.net):
            if isinstance(layer, nn.Linear):
                out_features = layer.out_features
                break
        # logits layer
        self.output_layer = nn.Linear(out_features, action_dim)
        if init_final is not None:
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.output_layer.weight)

    def forward(
        self,
        observations: torch.Tensor,
        observation_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # We detach the encoder if it is shared to avoid backprop through it
        # This is important to avoid the encoder to be updated through the policy
        obs_enc = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)

        # Get network outputs
        outputs = self.network(obs_enc)
        logits = self.output_layer(outputs)

        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)

        return action, log_prob, action_probs

    def get_features(self, observations: torch.Tensor) -> torch.Tensor:
        """Get encoded features from observations"""
        device = get_device_from_parameters(self)
        observations = observations.to(device)
        if self.encoder is not None:
            with torch.inference_mode():
                return self.encoder(observations)
        return observations


def freeze_image_encoder(image_encoder: nn.Module):
    """Freeze all parameters in the encoder"""
    for param in image_encoder.parameters():
        param.requires_grad = False


class PretrainedImageEncoder(nn.Module):
    def __init__(self, config: FQLVLAConfig):
        super().__init__()

        self.image_enc_layers, self.image_enc_out_shape = self._load_pretrained_vision_encoder(config)

    def _load_pretrained_vision_encoder(self, config: FQLVLAConfig):
        """Set up CNN encoder"""

        self.image_enc_layers = AutoModel.from_pretrained(config.vision_encoder_name, trust_remote_code=True)

        if hasattr(self.image_enc_layers.config, "hidden_sizes"):
            self.image_enc_out_shape = self.image_enc_layers.config.hidden_sizes[-1]  # Last channel dimension
        elif hasattr(self.image_enc_layers, "fc"):
            self.image_enc_out_shape = self.image_enc_layers.fc.in_features
        else:
            raise ValueError("Unsupported vision encoder architecture, make sure you are using a CNN")
        return self.image_enc_layers, self.image_enc_out_shape

    def forward(self, x):
        enc_feat = self.image_enc_layers(x).last_hidden_state
        return enc_feat


def orthogonal_init():
    return lambda x: torch.nn.init.orthogonal_(x, gain=1.0)


class SpatialLearnedEmbeddings(nn.Module):
    def __init__(self, height, width, channel, num_features=8):
        """
        PyTorch implementation of learned spatial embeddings

        Args:
            height: Spatial height of input features
            width: Spatial width of input features
            channel: Number of input channels
            num_features: Number of output embedding dimensions
        """
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.num_features = num_features

        self.kernel = nn.Parameter(torch.empty(channel, height, width, num_features))

        nn.init.kaiming_normal_(self.kernel, mode="fan_in", nonlinearity="linear")

    def forward(self, features):
        """
        Forward pass for spatial embedding

        Args:
            features: Input tensor of shape [B, C, H, W] where B is batch size,
                     C is number of channels, H is height, and W is width
        Returns:
            Output tensor of shape [B, C*F] where F is the number of features
        """

        features_expanded = features.unsqueeze(-1)  # [B, C, H, W, 1]
        kernel_expanded = self.kernel.unsqueeze(0)  # [1, C, H, W, F]

        # Element-wise multiplication and spatial reduction
        output = (features_expanded * kernel_expanded).sum(dim=(2, 3))  # Sum over H,W dimensions

        # Reshape to combine channel and feature dimensions
        output = output.view(output.size(0), -1)  # [B, C*F]

        return output


class RescaleFromTanh(Transform):
    def __init__(self, low: float = -1, high: float = 1):
        super().__init__()

        self.low = low

        self.high = high

    def _call(self, x):
        # Rescale from (-1, 1) to (low, high)

        return 0.5 * (x + 1.0) * (self.high - self.low) + self.low

    def _inverse(self, y):
        # Rescale from (low, high) back to (-1, 1)

        return 2.0 * (y - self.low) / (self.high - self.low) - 1.0

    def log_abs_det_jacobian(self, x, y):
        # log|d(rescale)/dx| = sum(log(0.5 * (high - low)))

        scale = 0.5 * (self.high - self.low)

        return torch.sum(torch.log(scale), dim=-1)


class TanhMultivariateNormalDiag(TransformedDistribution):
    def __init__(self, loc, scale_diag, low=None, high=None):
        base_dist = MultivariateNormal(loc, torch.diag_embed(scale_diag))

        transforms = [TanhTransform(cache_size=1)]

        if low is not None and high is not None:
            low = torch.as_tensor(low)

            high = torch.as_tensor(high)

            transforms.insert(0, RescaleFromTanh(low, high))

        super().__init__(base_dist, transforms)

    def mode(self):
        # Mode is mean of base distribution, passed through transforms

        x = self.base_dist.mean

        for transform in self.transforms:
            x = transform(x)

        return x

    def stddev(self):
        std = self.base_dist.stddev

        x = std

        for transform in self.transforms:
            x = transform(x)

        return x


def _convert_normalization_params_to_tensor(normalization_params: dict) -> dict:
    converted_params = {}
    for outer_key, inner_dict in normalization_params.items():
        converted_params[outer_key] = {}
        for key, value in inner_dict.items():
            converted_params[outer_key][key] = torch.tensor(value)
            if "image" in outer_key:
                converted_params[outer_key][key] = converted_params[outer_key][key].view(3, 1, 1)

    return converted_params
