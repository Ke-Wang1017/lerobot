"""S2 VLM-only configuration for Hierarchical VLA.

S2 uses PaliGemma (SigLIP vision + Gemma 2B language model) for scene understanding.
No action expert — S1 handles action generation.
"""

from dataclasses import dataclass


@dataclass
class S2VLMConfig:
    # PaliGemma VLM architecture
    paligemma_variant: str = "gemma_2b"
    dtype: str = "bfloat16"

    # Vocabulary
    vocab_size: int = 257152
    image_token_index: int = 257152

    # Gemma 2B language model
    hidden_size: int = 2048
    intermediate_size: int = 16384
    num_attention_heads: int = 8
    head_dim: int = 256
    num_hidden_layers: int = 18
    num_key_value_heads: int = 1
    hidden_activation: str = "gelu_pytorch_tanh"

    # SigLIP vision encoder
    vision_intermediate_size: int = 4304
    vision_projection_dim: int = 2048
    vision_projector_hidden_act: str = "gelu_fast"

    # Image preprocessing
    image_resolution: tuple[int, int] = (224, 224)

    # Tokenizer
    max_token_len: int = 256

    # Latent output
    latent_dim: int = 2048  # mean-pooled prefix output dimension

    # AR subtask decoding
    subtask_max_decoding_steps: int = 25
    subtask_temperature: float = 0.0  # 0 = greedy argmax

    # Training
    subtask_loss_weight: float = 10.0
    fast_token_loss_weight: float = 1.0
    fast_tokenizer_name: str = "physical-intelligence/fast"

    # LoRA
    lora_rank: int = 32
    lora_targets_vlm: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    lora_targets_vision: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "out_proj")
    lora_finetune_vision: bool = True

    # Inference
    compile_model: bool = False

    # Training only (do NOT enable for inference — adds recomputation overhead)
    gradient_checkpointing: bool = False

    # Constants
    EOS_TOKEN: int = 1
    NEG_INF: float = -2.3819763e38
