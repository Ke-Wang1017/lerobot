"""LoRA modules for HVLA flow-matching decoder.

Adds rank-r adapters to the decoder's attention (Q,K,V,O) and FFN linears
while freezing the rest of the model. The base nn.MultiheadAttention is
replaced with a Q/K/V-separated version so each projection can carry its
own LoRA — slicing the original packed `in_proj_weight` keeps the math
identical at init (LoRA B is zero, so `forward(x) == base(x)` at attach
time).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """nn.Linear with frozen base + trainable rank-r delta `B @ A`.

    forward(x) = base(x) + scaling * (dropout(x) @ A^T @ B^T)
    A: [r, in], Kaiming-uniform init (small).
    B: [out, r], zero init — so the wrapped module reproduces the base at attach.
    scaling = alpha / r.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        for p in self.base.parameters():
            p.requires_grad = False

        self.rank = rank
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return out + self.scaling * delta


class LoRAMultiheadAttention(nn.Module):
    """Drop-in replacement for nn.MultiheadAttention(batch_first=True).

    Splits the packed `in_proj_weight` into separate Q/K/V linears, each
    LoRA-wrapped, plus a LoRA-wrapped out_proj. Uses
    F.scaled_dot_product_attention for the attention computation.

    Returns (output, None) to match nn.MultiheadAttention's
    (output, attn_weights) signature when need_weights=False.
    """

    def __init__(self, base: nn.MultiheadAttention, rank: int, alpha: float):
        super().__init__()
        assert base.batch_first, "Only batch_first=True is supported"
        d = base.embed_dim
        self.embed_dim = d
        self.num_heads = base.num_heads
        self.head_dim = d // self.num_heads
        self.dropout_p = base.dropout

        # Slice packed in_proj_weight [3d, d] → separate Q, K, V linears
        has_bias = base.in_proj_bias is not None
        q_proj = nn.Linear(d, d, bias=has_bias)
        k_proj = nn.Linear(d, d, bias=has_bias)
        v_proj = nn.Linear(d, d, bias=has_bias)
        with torch.no_grad():
            w = base.in_proj_weight.data
            q_proj.weight.copy_(w[:d])
            k_proj.weight.copy_(w[d : 2 * d])
            v_proj.weight.copy_(w[2 * d :])
            if has_bias:
                b = base.in_proj_bias.data
                q_proj.bias.copy_(b[:d])
                k_proj.bias.copy_(b[d : 2 * d])
                v_proj.bias.copy_(b[2 * d :])

        self.q_proj = LoRALinear(q_proj, rank, alpha)
        self.k_proj = LoRALinear(k_proj, rank, alpha)
        self.v_proj = LoRALinear(v_proj, rank, alpha)
        self.out_proj = LoRALinear(base.out_proj, rank, alpha)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, None]:
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.reshape(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        # Combine attn_mask + key_padding_mask if either is set. For the
        # HVLA decoder both are None in normal training (chunk-bidirectional
        # self-attn, full cross-attn over context).
        attn_bias = None
        if key_padding_mask is not None:
            kpm = key_padding_mask
            if kpm.dtype == torch.bool:
                attn_bias = torch.zeros(B, 1, Lq, Lk, device=q.device, dtype=q.dtype)
                attn_bias.masked_fill_(kpm[:, None, None, :], float("-inf"))
            else:
                attn_bias = kpm[:, None, None, :].to(q.dtype)
        if attn_mask is not None:
            attn_bias = attn_mask if attn_bias is None else attn_bias + attn_mask

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, Lq, self.embed_dim)
        return self.out_proj(attn_out), None


def apply_lora_to_decoder(
    policy,
    rank: int = 16,
    alpha: float = 32.0,
    ffn: bool = True,
) -> tuple[int, int]:
    """Replace decoder attention/FFN with LoRA-wrapped versions, freeze the rest.

    Operates in place on `policy.model.decoder_layers`. Must be called AFTER
    loading the pretrained state dict (so original weights are present to
    copy into the LoRA modules' frozen bases).

    Returns (n_trainable_params, n_total_params).
    """
    inner = policy.model

    for layer in inner.decoder_layers:
        layer.self_attn = LoRAMultiheadAttention(layer.self_attn, rank, alpha)
        layer.multihead_attn = LoRAMultiheadAttention(layer.multihead_attn, rank, alpha)
        if ffn:
            layer.linear1 = LoRALinear(layer.linear1, rank, alpha)
            layer.linear2 = LoRALinear(layer.linear2, rank, alpha)

    # Freeze everything, then unfreeze LoRA params only
    for p in policy.parameters():
        p.requires_grad = False
    for n, p in policy.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            p.requires_grad = True

    n_total = sum(p.numel() for p in policy.parameters())
    n_lora = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    return n_lora, n_total
