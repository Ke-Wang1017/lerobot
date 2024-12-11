from functools import partial
from typing import Dict, Optional, Tuple, FrozenSet, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from serl_launcher.networks.actor_critic_nets_torch import Policy, Critic
from serl_launcher.networks.lagrange_torch import GeqLagrangeMultiplier
from serl_launcher.networks.mlp_torch import MLP

class SACAgent:
    """
    PyTorch implementation of Soft Actor-Critic (SAC) agent.
    Supports:
     - SAC (default)
     - TD3 (policy_kwargs={"std_parameterization": "fixed", "fixed_std": 0.1})
     - REDQ (critic_ensemble_size=10, critic_subsample_size=2)
     - SAC-ensemble (critic_ensemble_size>>1)
    """
    
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        critic_target: nn.Module,
        temp: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        temp_optimizer: torch.optim.Optimizer,
        config: dict
    ):
        self.actor = actor
        self.critic = critic
        self.critic_target = critic_target
        self.temp = temp
        
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.temp_optimizer = temp_optimizer
        
        self.config = config
        self.device = next(actor.parameters()).device

    def _compute_next_actions(
        self, 
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute next actions and their log probs for the SAC update"""
        next_action_distribution = self.actor(batch["next_observations"])
        next_actions, next_actions_log_probs = next_action_distribution.sample_and_log_prob()
        
        assert next_actions.shape == batch["actions"].shape
        assert next_actions_log_probs.shape == (batch["actions"].shape[0],)
        
        return next_actions, next_actions_log_probs

    def critic_loss_fn(
        self, 
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute critic loss and info dict"""
        batch_size = batch["rewards"].shape[0]
        
        with torch.no_grad():
            next_actions, next_actions_log_probs = self._compute_next_actions(batch)
            
            # Get target Q-values
            target_qs = self.critic_target(batch["next_observations"], next_actions)
            
            # Subsample critics if requested
            if self.config["critic_subsample_size"] is not None:
                indices = torch.randperm(self.config["critic_ensemble_size"])
                indices = indices[:self.config["critic_subsample_size"]]
                target_qs = target_qs[indices]
            
            # Compute target using min Q
            target_q = target_qs.min(dim=0)[0]
            assert target_q.shape == (batch_size,)
            
            # Compute backup
            target = (
                batch["rewards"] + 
                self.config["discount"] * batch["masks"] * target_q
            )
            
            if self.config["backup_entropy"]:
                temperature = self.temp()
                target = target - temperature * next_actions_log_probs

        # Compute critic loss
        current_qs = self.critic(batch["observations"], batch["actions"])
        assert current_qs.shape == (self.config["critic_ensemble_size"], batch_size)
        
        critic_loss = F.mse_loss(
            current_qs, 
            target.unsqueeze(0).expand(self.config["critic_ensemble_size"], -1)
        )

        info = {
            "critic_loss": critic_loss.item(),
            "q_values": current_qs.mean().item(),
            "target_q": target.mean().item(),
        }
        
        return critic_loss, info

    def actor_loss_fn(
        self, 
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute actor loss and info dict"""
        temperature = self.temp()
        
        dist = self.actor(batch["observations"])
        actions, log_probs = dist.sample_and_log_prob()
        
        # Get Q-values for the sampled actions
        q_values = self.critic(batch["observations"], actions)
        q_values = q_values.mean(dim=0)  # Average across ensemble
        
        actor_loss = (temperature * log_probs - q_values).mean()
        
        info = {
            "actor_loss": actor_loss.item(),
            "entropy": -log_probs.mean().item(),
            "temperature": temperature.item(),
        }
        
        return actor_loss, info

    def temperature_loss_fn(
        self, 
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute temperature loss and info dict"""
        with torch.no_grad():
            _, next_actions_log_probs = self._compute_next_actions(batch)
            entropy = -next_actions_log_probs.mean()
            
        temperature_loss = self.temp(
            lhs=entropy,
            rhs=self.config["target_entropy"]
        )
        
        info = {"temperature_loss": temperature_loss.item()}
        return temperature_loss, info

    def update(
        self,
        batch: Dict[str, torch.Tensor],
        networks_to_update: FrozenSet[str] = frozenset({"actor", "critic", "temperature"})
    ) -> Tuple["SACAgent", Dict]:
        """Perform one update step for the agent"""
        
        # Move batch to device
        batch = {k: torch.as_tensor(v, device=self.device) for k, v in batch.items()}
        
        info = {}
        
        # Update critic
        if "critic" in networks_to_update:
            self.critic_optimizer.zero_grad()
            critic_loss, critic_info = self.critic_loss_fn(batch)
            critic_loss.backward()
            self.critic_optimizer.step()
            info.update(critic_info)
            
            # Update target network
            with torch.no_grad():
                tau = self.config["soft_target_update_rate"]
                for target, source in zip(
                    self.critic_target.parameters(), 
                    self.critic.parameters()
                ):
                    target.data.mul_(1 - tau)
                    target.data.add_(tau * source.data)
        
        # Update actor
        if "actor" in networks_to_update:
            self.actor_optimizer.zero_grad()
            actor_loss, actor_info = self.actor_loss_fn(batch)
            actor_loss.backward()
            self.actor_optimizer.step()
            info.update(actor_info)
        
        # Update temperature
        if "temperature" in networks_to_update:
            self.temp_optimizer.zero_grad()
            temp_loss, temp_info = self.temperature_loss_fn(batch)
            temp_loss.backward()
            self.temp_optimizer.step()
            info.update(temp_info)
            
        return self, info

    @torch.no_grad()
    def sample_actions(
        self,
        observations: Dict[str, torch.Tensor],
        argmax: bool = False
    ) -> torch.Tensor:
        """Sample actions from policy"""
        observations = {
            k: torch.as_tensor(v, device=self.device) 
            for k, v in observations.items()
        }
        
        dist = self.actor(observations)
        if argmax:
            return dist.mode()
        return dist.sample()

    @classmethod
    def create(
        cls,
        observations: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        actor_def: nn.Module,
        critic_def: nn.Module,
        temp_def: nn.Module,
        actor_optimizer_kwargs: dict = {"lr": 3e-4},
        critic_optimizer_kwargs: dict = {"lr": 3e-4},
        temp_optimizer_kwargs: dict = {"lr": 3e-4},
        discount: float = 0.99,
        soft_target_update_rate: float = 0.005,
        target_entropy: Optional[float] = None,
        backup_entropy: bool = True,
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        device: str = "cuda",
        **kwargs
    ) -> "SACAgent":
        """Create a new SAC agent"""
        device = torch.device(device)
        
        # Create networks
        actor = actor_def.to(device)
        critic = critic_def.to(device)
        critic_target = deepcopy(critic_def).to(device)
        temp = temp_def.to(device)
        
        # Create optimizers
        actor_optimizer = torch.optim.Adam(
            actor.parameters(), **actor_optimizer_kwargs
        )
        critic_optimizer = torch.optim.Adam(
            critic.parameters(), **critic_optimizer_kwargs
        )
        temp_optimizer = torch.optim.Adam(
            temp.parameters(), **temp_optimizer_kwargs
        )
        
        # Set target entropy if not specified
        if target_entropy is None:
            target_entropy = -np.prod(actions.shape[-1])
        
        config = {
            "discount": discount,
            "soft_target_update_rate": soft_target_update_rate,
            "target_entropy": target_entropy,
            "backup_entropy": backup_entropy,
            "critic_ensemble_size": critic_ensemble_size,
            "critic_subsample_size": critic_subsample_size,
            **kwargs
        }
        
        return cls(
            actor=actor,
            critic=critic,
            critic_target=critic_target,
            temp=temp,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            temp_optimizer=temp_optimizer,
            config=config
        )

 