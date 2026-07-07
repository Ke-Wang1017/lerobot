"""S2 VLM-only model for Hierarchical VLA.

Wraps PaliGemma (SigLIP + Gemma 2B) for scene understanding.
No action expert — only prefix encoding, latent extraction, and AR subtask decoding.

Reuses LeRobot's pi_gemma.py for the PaliGemma implementation.
"""

import logging
import math

import torch
from torch import Tensor, nn
from transformers.models.auto import CONFIG_MAPPING

from lerobot.policies.hvla.s2.config import S2VLMConfig
from lerobot.policies.pi_gemma import PaliGemmaForConditionalGenerationWithPiGemma

logger = logging.getLogger(__name__)


def make_att_2d_masks(pad_masks: Tensor, att_masks: Tensor) -> Tensor:
    """Construct 2D attention masks from padding and AR masks.

    Tokens attend to valid input tokens with cumulative mask_ar <= their own.
    All-zero att_masks = full (prefix-LM) attention between all valid tokens.
    """
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d & pad_2d


class S2VLMModel(nn.Module):
    """PaliGemma VLM-only model for S2 latent extraction and subtask decoding.

    Loads from a Pi0.5 checkpoint, stripping action expert weights.
    Only the VLM branch (SigLIP + Gemma 2B) is used.
    """

    def __init__(self, config: S2VLMConfig):
        super().__init__()
        self.config = config

        # Build HF PaliGemma config
        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = config.vocab_size  # noqa: SLF001
        vlm_config_hf.image_token_index = config.image_token_index
        vlm_config_hf.text_config.hidden_size = config.hidden_size
        vlm_config_hf.text_config.intermediate_size = config.intermediate_size
        vlm_config_hf.text_config.num_attention_heads = config.num_attention_heads
        vlm_config_hf.text_config.head_dim = config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = config.num_hidden_layers
        vlm_config_hf.text_config.num_key_value_heads = config.num_key_value_heads
        vlm_config_hf.text_config.hidden_activation = config.hidden_activation
        vlm_config_hf.text_config.dtype = "float32"
        vlm_config_hf.text_config.vocab_size = config.vocab_size
        vlm_config_hf.text_config.use_adarms = False
        vlm_config_hf.text_config.adarms_cond_dim = None
        vlm_config_hf.vision_config.image_size = config.image_resolution[0]
        vlm_config_hf.vision_config.intermediate_size = config.vision_intermediate_size
        vlm_config_hf.vision_config.projection_dim = config.vision_projection_dim
        vlm_config_hf.vision_config.projector_hidden_act = config.vision_projector_hidden_act
        vlm_config_hf.vision_config.dtype = "float32"

        self.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(config=vlm_config_hf)
        self._apply_precision(config.dtype)

    def _apply_precision(self, precision: str):
        """Apply mixed precision: bf16 for most params, fp32 for vision path and norms."""
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        keep_fp32 = [
            "vision_tower",
            "multi_modal_projector",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]
        for name, param in self.named_parameters():
            if any(s in name for s in keep_fp32):
                param.data = param.data.to(dtype=torch.float32)

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for training (reduces memory, slower)."""
        self.paligemma.model.language_model.gradient_checkpointing_enable()

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def embed_image(self, image: Tensor) -> Tensor:
        """Encode images through SigLIP → patch embeddings [B, num_patches, hidden_size]."""
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        image_outputs = self.paligemma.model.get_image_features(image)
        features = image_outputs.pooler_output * self.config.hidden_size**0.5
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features

    def embed_language_tokens(self, tokens: Tensor) -> Tensor:
        """Embed token IDs → [B, seq_len, hidden_size]."""
        return self.paligemma.model.language_model.embed_tokens(tokens)

    def deembed(self, embeddings: Tensor) -> Tensor:
        """Hidden states → vocab logits [B, seq_len, vocab_size]."""
        return torch.matmul(
            embeddings,
            self.paligemma.model.language_model.embed_tokens.weight.T,
        )

    # ------------------------------------------------------------------
    # Prefix encoding
    # ------------------------------------------------------------------

    def embed_prefix(
        self,
        images: list[Tensor],
        image_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Embed images + language tokens for PaliGemma prefix.

        Returns: (embeddings, pad_masks, att_masks) all [B, seq_len, ...]
        """
        embs = []
        pad_masks = []
        att_mask_list: list[int] = []

        for img, img_mask in zip(images, image_masks, strict=True):
            img_emb = self.embed_image(img)
            bsize, num_patches = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_patches))
            att_mask_list += [0] * num_patches

        # Language tokens with sqrt(dim) scaling
        lang_emb = self.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_mask_list += [0] * lang_emb.shape[1]

        embs_cat = torch.cat(embs, dim=1)
        pad_masks_cat = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_mask_list, dtype=torch.bool, device=pad_masks_cat.device)
        att_masks = att_masks[None, :].expand(pad_masks_cat.shape[0], -1)

        return embs_cat, pad_masks_cat, att_masks

    def _prepare_attention_masks_4d(self, att_2d_masks: Tensor) -> Tensor:
        """Convert 2D boolean masks to 4D float masks for transformer."""
        att_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_4d, 0.0, self.config.NEG_INF)

    def _vlm_forward(
        self,
        inputs_embeds: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
        past_key_values=None,
        use_cache: bool = False,
    ):
        """Run PaliGemma language model forward (VLM-only, no action expert)."""
        self.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        output = self.paligemma.model.language_model.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        return output.last_hidden_state, output.past_key_values

    # ------------------------------------------------------------------
    # Latent extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_prefix_latent(
        self,
        images: list[Tensor],
        image_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
    ) -> Tensor:
        """Extract scene-understanding latent by mean-pooling prefix output.

        Returns: [B, latent_dim] tensor (default 2048).
        """
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, image_masks, lang_tokens, lang_masks
        )
        att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att_4d = self._prepare_attention_masks_4d(att_2d)

        prefix_out, _ = self._vlm_forward(
            inputs_embeds=prefix_embs,
            attention_mask=att_4d,
            position_ids=position_ids,
            use_cache=False,
        )

        mask = prefix_pad_masks[:, :, None].to(dtype=prefix_out.dtype)
        pooled = (prefix_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled

    # ------------------------------------------------------------------
    # Fused latent + subtask AR decoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_prefix_latent_and_subtask(
        self,
        images: list[Tensor],
        image_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        max_decoding_steps: int | None = None,
        temperature: float | None = None,
    ) -> tuple[Tensor, list[int], list[list[tuple[int, float]]], float]:
        """Single prefix forward → latent (mean-pool) + AR subtask decoding from KV cache.

        Returns: (latent [B, 2048], output_token_ids, topk_per_step, confidence)
            confidence: product of softmax probabilities of chosen tokens (0–1).
        """
        if max_decoding_steps is None:
            max_decoding_steps = self.config.subtask_max_decoding_steps
        if temperature is None:
            temperature = self.config.subtask_temperature

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, image_masks, lang_tokens, lang_masks
        )
        att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att_4d = self._prepare_attention_masks_4d(att_2d)

        prefix_out, past_kv = self._vlm_forward(
            inputs_embeds=prefix_embs,
            attention_mask=att_4d,
            position_ids=position_ids,
            use_cache=True,
        )

        # Latent via mean-pooling
        mask = prefix_pad_masks[:, :, None].to(dtype=prefix_out.dtype)
        pooled = (prefix_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        # AR decode subtask from KV cache
        device = prefix_embs.device
        last_valid_idx = int(torch.where(prefix_pad_masks[0])[0][-1].item())
        logits = self.deembed(prefix_out[:, last_valid_idx : last_valid_idx + 1])
        next_token = self._sample_token(logits[:, 0], temperature)

        prefix_valid_len = int(prefix_pad_masks[0].sum().item())
        emb_scale = math.sqrt(prefix_embs.shape[-1])

        output_tokens: list[int] = []
        topk_per_step: list[list[tuple[int, float]]] = []
        # Per-token log-probs for confidence (geometric mean, so length-
        # independent). Stored per step so s2_process can compute confidence
        # over just the subtask portion (before ";") after detokenization.
        token_log_probs: list[float] = []

        # Top-k for first token + log-prob of chosen token
        first_topk = torch.topk(logits[0, 0], k=min(5, logits.shape[-1]))
        topk_per_step.append(list(zip(first_topk.indices.tolist(), first_topk.values.tolist(), strict=False)))
        first_log_probs = torch.log_softmax(logits[0, 0].float(), dim=-1)
        token_log_probs.append(first_log_probs[next_token[0, 0]].item())

        for step in range(max_decoding_steps):
            tok = next_token[0, 0].item()
            if tok == self.config.EOS_TOKEN:
                break
            output_tokens.append(tok)

            token_emb = self.embed_language_tokens(next_token) * emb_scale
            token_emb = token_emb.to(dtype=prefix_embs.dtype)
            pos_ids = torch.tensor([[prefix_valid_len + step]], device=device, dtype=torch.long)

            # Attention: attend to valid prefix + all decoded AR tokens
            cross = torch.where(prefix_pad_masks[0], 0.0, self.config.NEG_INF)
            ar_part = torch.zeros(step + 1, device=device, dtype=torch.float32)
            ar_att_4d = torch.cat([cross, ar_part], dim=0).reshape(1, 1, 1, -1)

            ar_out, past_kv = self._vlm_forward(
                inputs_embeds=token_emb,
                attention_mask=ar_att_4d,
                position_ids=pos_ids,
                past_key_values=past_kv,
                use_cache=True,
            )

            logits = self.deembed(ar_out[:, -1:])
            step_topk = torch.topk(logits[0, 0], k=min(5, logits.shape[-1]))
            topk_per_step.append(
                list(zip(step_topk.indices.tolist(), step_topk.values.tolist(), strict=False))
            )
            next_token = self._sample_token(logits[:, 0], temperature)
            step_log_probs = torch.log_softmax(logits[0, 0].float(), dim=-1)
            token_log_probs.append(step_log_probs[next_token[0, 0]].item())
        return pooled, output_tokens, topk_per_step, token_log_probs

    @staticmethod
    def _sample_token(logits_2d: Tensor, temperature: float) -> Tensor:
        """Sample or argmax from [B, vocab] logits."""
        if temperature <= 0.0:
            return logits_2d.argmax(dim=-1, keepdim=True)
        probs = torch.softmax(logits_2d / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    # ------------------------------------------------------------------
    # Weight loading (VLM-only from Pi0.5 checkpoint)
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, config: S2VLMConfig | None = None) -> "S2VLMModel":
        """Load VLM-only weights from a Pi0.5 checkpoint, skipping action expert."""
        from safetensors.torch import load_file

        if config is None:
            config = S2VLMConfig()

        model = cls(config)

        state_dict = load_file(checkpoint_path)

        # Filter out action expert keys
        skip_prefixes = (
            "gemma_expert",
            "action_in_proj",
            "action_out_proj",
            "time_mlp",
            "state_proj",
        )

        # Remap keys: checkpoint may use "paligemma_with_expert.paligemma.*"
        # but our model expects "paligemma.*"
        vlm_keys = {}
        for k, v in state_dict.items():
            if any(k.startswith(p) or f".{p}" in k for p in skip_prefixes):
                continue
            # Strip wrapper prefix if present
            new_k = k
            if new_k.startswith("paligemma_with_expert.paligemma."):
                new_k = "paligemma." + new_k[len("paligemma_with_expert.paligemma.") :]
            elif new_k.startswith("paligemma_with_expert."):
                new_k = new_k[len("paligemma_with_expert.") :]
            vlm_keys[new_k] = v

        missing, unexpected = model.load_state_dict(vlm_keys, strict=False)
        if missing:
            logger.warning("Missing keys when loading VLM weights: %s", missing[:10])
        if unexpected:
            logger.warning("Unexpected keys (skipped): %s", unexpected[:10])

        skipped_count = len(state_dict) - len(vlm_keys)
        logger.info(
            "Loaded %d VLM params, skipped %d action expert params",
            len(vlm_keys),
            skipped_count,
        )
        return model
