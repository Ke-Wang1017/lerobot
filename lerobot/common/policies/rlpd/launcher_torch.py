# !/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np

from serl_launcher.common.wandb import WandBLogger
from serl_launcher.agents.continuous.bc import BCAgent
from serl_launcher.agents.continuous.sac_torch import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.vision.data_augmentations_torch import batched_random_crop
from serl_launcher.networks.mlp_torch import MLP

def make_bc_agent(
    seed: int, 
    sample_obs: dict, 
    sample_action: torch.Tensor, 
    image_keys: tuple = ("image",), 
    encoder_type: str = "resnet-pretrained",
    device: str = "cuda"
) -> BCAgent:
    torch.manual_seed(seed)
    
    return BCAgent.create(
        sample_obs,
        sample_action,
        network_kwargs={
            "activation": nn.Tanh(),
            "use_layer_norm": True,
            "hidden_dims": [512, 512, 512],
            "dropout_rate": 0.25,
        },
        policy_kwargs={
            "tanh_squash_distribution": False,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        use_proprio=True,
        encoder_type=encoder_type,
        image_keys=image_keys,
        augmentation_function=make_batch_augmentation_func(image_keys),
        device=device
    )

def make_sac_pixel_agent(
    seed: int,
    sample_obs: dict,
    sample_action: torch.Tensor,
    image_keys: tuple = ("image",),
    encoder_type: str = "resnet-pretrained",
    reward_bias: float = 0.0,
    target_entropy: float = None,
    discount: float = 0.97,
    device: str = "cuda"
) -> SACAgent:
    torch.manual_seed(seed)
    
    agent = SACAgent.create_pixels(
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activation": nn.Tanh(),
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activation": nn.Tanh(),
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=make_batch_augmentation_func(image_keys),
        device=device
    )
    return agent

def make_sac_pixel_agent_hybrid_single_arm(
    seed: int,
    sample_obs: dict,
    sample_action: torch.Tensor,
    image_keys: tuple = ("image",),
    encoder_type: str = "resnet-pretrained",
    reward_bias: float = 0.0,
    target_entropy: float = None,
    discount: float = 0.97,
    device: str = "cuda"
) -> SACAgentHybridSingleArm:
    torch.manual_seed(seed)
    
    agent = SACAgentHybridSingleArm.create_pixels(
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activation": nn.Tanh(),
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        grasp_critic_network_kwargs={
            "activation": nn.Tanh(),
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activation": nn.Tanh(),
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=make_batch_augmentation_func(image_keys),
        device=device
    )
    return agent

def make_sac_pixel_agent_hybrid_dual_arm(
    seed: int,
    sample_obs: dict,
    sample_action: torch.Tensor,
    image_keys: tuple = ("image",),
    encoder_type: str = "resnet-pretrained",
    reward_bias: float = 0.0,
    target_entropy: float = None,
    discount: float = 0.97,
    device: str = "cuda"
) -> SACAgentHybridDualArm:
    torch.manual_seed(seed)
    
    agent = SACAgentHybridDualArm.create_pixels(
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activation": nn.Tanh(),
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        grasp_critic_network_kwargs={
            "activation": nn.Tanh(),
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activation": nn.Tanh(),
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=make_batch_augmentation_func(image_keys),
        device=device
    )
    return agent

def linear_schedule(step: int) -> float:
    init_value = 10.0
    end_value = 50.0
    decay_steps = 15_000

    linear_step = min(step, decay_steps)
    decayed_value = init_value + (end_value - init_value) * (linear_step / decay_steps)
    return decayed_value
    
def make_batch_augmentation_func(image_keys: tuple) -> callable:
    def data_augmentation_fn(observations: dict, seed: int) -> dict:
        torch.manual_seed(seed)
        for pixel_key in image_keys:
            observations = {
                **observations,
                pixel_key: batched_random_crop(
                    observations[pixel_key], 
                    seed=seed, 
                    padding=4, 
                    num_batch_dims=2
                )
            }
        return observations
    
    def augment_batch(batch: dict, seed: int) -> dict:
        obs_seed = seed
        next_obs_seed = seed + 1
        
        obs = data_augmentation_fn(batch["observations"], obs_seed)
        next_obs = data_augmentation_fn(batch["next_observations"], next_obs_seed)
        
        return {
            **batch,
            "observations": obs,
            "next_observations": next_obs,
        }
    
    return augment_batch

def make_trainer_config(port_number: int = 5588, broadcast_port: int = 5589) -> dict:
    return {
        "port_number": port_number,
        "broadcast_port": broadcast_port,
        "request_types": ["send-stats"],
    }

def make_wandb_logger(
    project: str = "hil-serl",
    description: str = "serl_launcher",
    debug: bool = False,
) -> WandBLogger:
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update({
        "project": project,
        "exp_descriptor": description,
        "tag": description,
    })
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant={},
        debug=debug,
    )
    return wandb_logger
