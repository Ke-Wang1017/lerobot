"""RL Token encoder and decoder.

The encoder compresses S1's observation context tokens into a single bottleneck
vector (z_rl). The decoder reconstructs the original tokens for training the
encoder. At runtime only the encoder is used (frozen).

Architecture follows Xu et al. 2026, Sec IV-A:
  encoder: append learned readout embedding to S1 context, process with small
           transformer, take output at readout position → z_rl
  decoder: autoregressive reconstruction of context tokens from z_rl
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from lerobot.policies.hvla.rlt.config import RLTConfig

# Fields that determine encoder/decoder tensor shapes. Saved to
# ``config.json`` alongside each trained checkpoint so a loader can
# rebuild the exact architecture regardless of what's in the live
# ``RLTConfig`` defaults. Non-shape fields (dropout, LR, etc.) are
# omitted — they don't affect loading, only training.
_SHAPE_FIELDS = (
    "rl_token_dim",
    "context_dim",
    "token_encoder_layers",
    "token_decoder_layers",
    "token_num_heads",
    "token_ffn_dim",
)


def save_rlt_token_config(ckpt_dir: Path | str, rlt_config: RLTConfig) -> None:
    """Write the shape-defining subset of ``rlt_config`` to
    ``{ckpt_dir}/config.json`` alongside encoder.pt / decoder.pt. Must be
    called at save time for every checkpoint that wants to be loadable
    with a non-default architecture."""
    ckpt_dir = Path(ckpt_dir)
    arch = {k: getattr(rlt_config, k) for k in _SHAPE_FIELDS}
    (ckpt_dir / "config.json").write_text(json.dumps(arch, indent=2))


def load_rlt_token_config(
    ckpt_dir: Path | str,
    base: RLTConfig | None = None,
) -> RLTConfig:
    """Return an ``RLTConfig`` whose shape fields match the saved
    checkpoint. Reads the architecture manifest from
    ``{ckpt_dir}/config.json``. Any non-shape fields on ``base`` are
    preserved — callers supply a config already populated with runtime
    fields (lr, beta, sigma) and the manifest only overrides
    shape-affecting fields.

    Raises ``FileNotFoundError`` when no manifest is present. Legacy
    checkpoints from the deprecated 2L d=768 architecture predate the
    manifest format and are no longer supported — retrain with the
    canonical 4L d=2048 recipe (see ``train_token.py`` docstring) or
    add a ``config.json`` by hand if you really need the old arch.
    """
    cfg = base if base is not None else RLTConfig()
    config_path = Path(ckpt_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No architecture manifest at {config_path}. Pre-manifest "
            f"checkpoints (the deprecated 2L d=768 family) are no longer "
            f"loadable. Retrain the token encoder with the canonical 4L "
            f"d=2048 recipe (`python -m lerobot.policies.hvla.rlt."
            f"train_token --encoder-layers 4 --decoder-layers 4 "
            f"--rl-token-dim 2048 ...`) or write a config.json with the "
            f"saved architecture's shape fields."
        )
    arch = json.loads(config_path.read_text())
    for k, v in arch.items():
        if k in _SHAPE_FIELDS and hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


class RLTokenEncoder(nn.Module):
    """Compress S1 context tokens [B, N, C] → z_rl [B, D] via learned readout.

    C is ``config.context_dim`` (defaults to ``rl_token_dim`` for symmetric
    setups — context channels already match transformer d_model, no
    projection). When C ≠ rl_token_dim an input projection widens/narrows
    the context to the transformer's residual stream. This is what lets
    the bottleneck dim (rl_token_dim) differ from the S1 hidden_dim that
    produced the context tokens.
    """

    def __init__(self, config: RLTConfig):
        super().__init__()
        d = config.rl_token_dim
        # When config.context_dim is None, assume context already has d
        # channels (original behavior — backward compat with checkpoints
        # trained before this field existed). nn.Identity adds no
        # parameters, so old state_dicts load unchanged.
        ctx_d = config.context_dim if config.context_dim is not None else d
        self.input_proj = nn.Linear(ctx_d, d) if ctx_d != d else nn.Identity()

        # Learned readout embedding appended to context sequence
        self.readout_embed = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.token_num_heads,
            dim_feedforward=config.token_ffn_dim,
            dropout=config.token_dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.token_encoder_layers,
        )

    def forward(self, context: Tensor) -> Tensor:
        """
        Args:
            context: [B, N_ctx, C] from S1.encode_observations (detached).
                C = config.context_dim (or rl_token_dim when None).
        Returns:
            z_rl: [B, D] compressed representation, D = rl_token_dim.
        """
        context = self.input_proj(context)  # [B, N, D]
        B = context.shape[0]
        readout = self.readout_embed.expand(B, -1, -1)  # [B, 1, D]
        augmented = torch.cat([context, readout], dim=1)  # [B, N+1, D]
        out = self.transformer(augmented)  # [B, N+1, D]
        z_rl = out[:, -1, :]  # [B, D]
        return z_rl


class RLTokenDecoder(nn.Module):
    """Autoregressive reconstruction of S1 context tokens from z_rl.

    Used only during Phase 1 training. Teacher-forced: at training time,
    ground-truth context tokens are shifted and fed as input, with causal
    masking so position i only sees positions < i.
    """

    def __init__(self, config: RLTConfig):
        super().__init__()
        d = config.rl_token_dim
        ctx_d = config.context_dim if config.context_dim is not None else d

        # Teacher-forced targets arrive in context space (dim=ctx_d).
        # Project into the transformer's residual stream (dim=d) when they
        # differ; Identity when symmetric (backward compat).
        self.target_proj = nn.Linear(ctx_d, d) if ctx_d != d else nn.Identity()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=config.token_num_heads,
            dim_feedforward=config.token_ffn_dim,
            dropout=config.token_dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.token_decoder_layers,
        )
        # Output head projects transformer hidden (d) back to context
        # space (ctx_d) so reconstruction is compared in the same units
        # as the target. For symmetric setups this is the existing d→d
        # linear; when widened, it also narrows from d back to ctx_d.
        self.output_proj = nn.Linear(d, ctx_d)

    def forward(self, z_rl: Tensor, target_context: Tensor) -> Tensor:
        """Teacher-forced autoregressive reconstruction.

        Args:
            z_rl: [B, D] RL token (used as memory for cross-attention,
                D = rl_token_dim)
            target_context: [B, N, C] ground-truth context tokens
                (C = context_dim, stop-grad)
        Returns:
            reconstructed: [B, N, C] predicted context tokens in the same
                dimensionality as ``target_context`` (so the reconstruction
                loss compares apples to apples).
        """
        B, N, _ = target_context.shape

        # Project targets into the transformer's residual stream (dim=D).
        # For symmetric (C == D) setups this is Identity — unchanged
        # behavior.
        target_proj = self.target_proj(target_context)  # [B, N, D]

        # Memory: z_rl as single token for cross-attention
        memory = z_rl.unsqueeze(1)  # [B, 1, D]

        # Decoder input: shift right — prepend z_rl, drop last target pos.
        # Position 0 input = z_rl, position i input = target_proj[:, i-1].
        decoder_input = torch.cat(
            [
                z_rl.unsqueeze(1),  # [B, 1, D]
                target_proj[:, :-1, :],  # [B, N-1, D]
            ],
            dim=1,
        )  # [B, N, D]

        # Causal mask: position i can only attend to positions <= i
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            N,
            device=z_rl.device,
            dtype=z_rl.dtype,
        )

        out = self.transformer(decoder_input, memory, tgt_mask=causal_mask)
        reconstructed = self.output_proj(out)  # [B, N, C]
        return reconstructed


def rl_token_reconstruction_loss(
    encoder: RLTokenEncoder,
    decoder: RLTokenDecoder,
    context: Tensor,
) -> Tensor:
    """Compute L_rto (paper Eq. 2): reconstruction loss for RL token.

    Per-token L2 norm squared, summed over tokens, averaged over batch.
    Not F.mse_loss (which averages over all dims — too weak).

    Args:
        encoder: RL token encoder
        decoder: RL token decoder
        context: [B, N, D] from S1.encode_observations (already detached / stop-grad)
    Returns:
        scalar loss
    """
    target = context.detach()
    z_rl = encoder(target)
    reconstructed = decoder(z_rl, target)
    # Paper Eq. 2: ||h(d([z_rl, z̃_{1:i-1}]))_i - z̃_i||^2 summed over i
    loss = ((reconstructed - target) ** 2).sum(dim=-1).mean()
    return loss
