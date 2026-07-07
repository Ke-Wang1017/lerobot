"""Visualize the RLT replay buffer for diagnostic analysis.

Produces a 2x3 panel PNG with three views of the same z_rl t-SNE
projection (top row) and three diagnostic plots (bottom row):

Top row — three perspectives on the same z_rl embedding:
  1. Colored by REWARD CATEGORY of the stored transition: shows
     where +1, -1, and r=0 transitions sit in the embedding.
  2. Colored by V(s) ≈ Q(s, π_θ(s)): the value of each state
     under the CURRENT actor's policy. This is the "is this
     state region good or bad to be in?" view. Note: states are
     rarely intrinsically bad — usually only specific actions in
     a state are bad, so V(s) tends to be more uniformly positive
     than Q(s, a_stored) below.
  3. Colored by Q(s, a_stored): the critic's evaluation of the
     SPECIFIC action recorded in the buffer for each transition.
     Red here means "the recorded action led to low Q" (e.g.
     abort transitions where the actor was producing OOD wild
     actions). Different from V(s) because state value depends
     on the actor's CURRENT policy, not the recorded action.

Diagnostic plots (bottom row):
  4. Q distribution histogram by reward category — visual sanity
     check that the critic separates +1, -1, and r=0 cleanly.
  5. Actor deviation |a-ref| vs Q(s, a_stored) — does the actor
     deviate from S1 ref where Q says to, or randomly?
  6. Terminal position timeline — where in the buffer (oldest →
     newest) do +1 and -1 anchors live?

Usage:
    python src/lerobot/policies/hvla/scripts/rlt_visualize.py outputs/rlt_online_v2_widened/latest

Optional --tsne-perplexity (default 30), --max-mid-episode N (cap the
mid-episode points so terminal markers don't get visually drowned).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE


def _label_categories(reward: np.ndarray, done: np.ndarray) -> np.ndarray:
    """Bucket each transition into:
    0 = mid-episode (r=0, done=False, OR timeout's last frame)
    1 = success terminal (+1, done=True)
    2 = abort terminal (-1, done=True)
    """
    cat = np.zeros(len(reward), dtype=int)
    cat[(reward > 0.5) & (done > 0.5)] = 1
    cat[(reward < -0.5) & (done > 0.5)] = 2
    return cat


def _load_actor_critic(ckpt_dir: Path, z_rl_dim: int, state_dim: int, action_dim: int):
    """Load both saved networks. Caller resolves missing-file cases."""
    from lerobot.policies.hvla.rlt.actor_critic import RLTActor, RLTCritic
    from lerobot.policies.hvla.rlt.config import RLTConfig

    cfg = RLTConfig(rl_token_dim=z_rl_dim)
    actor = critic = None
    if (ckpt_dir / "actor.pt").exists():
        actor = RLTActor(cfg, state_dim, action_dim)
        actor.load_state_dict(torch.load(str(ckpt_dir / "actor.pt"), weights_only=True, map_location="cpu"))
        actor.eval()
    if (ckpt_dir / "critic.pt").exists():
        critic = RLTCritic(cfg, state_dim, action_dim)
        critic.load_state_dict(torch.load(str(ckpt_dir / "critic.pt"), weights_only=True, map_location="cpu"))
        critic.eval()
    return cfg, actor, critic


def _critic_q_on_stored(critic, cfg, z_rl, state, action, action_dim) -> np.ndarray:
    """Q(s, a_stored) — what the critic predicts for the SPECIFIC action
    recorded in the buffer. Reflects "what the critic learned from this
    training data point" rather than "how good the state is now."
    """
    n = len(z_rl)
    q_min = np.zeros(n, dtype=np.float32)
    bs = 256
    with torch.no_grad():
        for i in range(0, n, bs):
            j = min(i + bs, n)
            z = torch.from_numpy(z_rl[i:j])
            s = torch.from_numpy(state[i:j])
            a = torch.from_numpy(action[i:j]).reshape(-1, cfg.rl_chunk_length, action_dim)
            qs = critic(z, s, a)
            q_stacked = torch.stack([q.squeeze(-1) for q in qs])
            q_min[i:j] = q_stacked.min(dim=0).values.numpy()
    return q_min


def _value_under_actor(actor, critic, cfg, z_rl, state, ref, action_dim) -> np.ndarray:
    """V(s) ≈ Q(s, π_θ(s)) — the value of each state under the CURRENT
    actor's policy. Run actor deterministically (no exploration noise) to
    get its chosen action, then critic on that action. This is the
    "where would the actor land Q if asked to act in this state?" view —
    different from Q(s, a_stored) because state value depends on the
    actor's current policy, not the action that happened to be recorded.
    """
    n = len(z_rl)
    v = np.zeros(n, dtype=np.float32)
    bs = 256
    with torch.no_grad():
        for i in range(0, n, bs):
            j = min(i + bs, n)
            z = torch.from_numpy(z_rl[i:j])
            s = torch.from_numpy(state[i:j])
            r = torch.from_numpy(ref[i:j]).reshape(-1, cfg.rl_chunk_length, action_dim)
            actor_action = actor(z, s, r, deterministic=True)  # μ, no noise
            qs = critic(z, s, actor_action)
            q_stacked = torch.stack([q.squeeze(-1) for q in qs])
            v[i:j] = q_stacked.min(dim=0).values.numpy()
    return v


def _scatter_by_category(ax, xy: np.ndarray, cat: np.ndarray, title: str, max_mid: int = 1000):
    """Plot a 2D scatter coloured by reward category. Mid-episode dots
    are subsampled (default: cap at 1000) so terminal markers stand
    out — otherwise the +1 / -1 dots get visually drowned by the
    thousands of r=0 dots."""
    counts = {0: int((cat == 0).sum()), 1: int((cat == 1).sum()), 2: int((cat == 2).sum())}
    rng = np.random.default_rng(0)

    # Mid-episode (subsampled)
    mid_mask = cat == 0
    mid_idx = np.where(mid_mask)[0]
    if len(mid_idx) > max_mid:
        mid_idx = rng.choice(mid_idx, max_mid, replace=False)
    ax.scatter(
        xy[mid_idx, 0],
        xy[mid_idx, 1],
        c="#bbbbbb",
        s=4,
        alpha=0.4,
        label=f"mid-episode (n={counts[0]}{'⁎' if counts[0] > max_mid else ''})",
    )

    # Success +1
    if counts[1] > 0:
        m = cat == 1
        ax.scatter(
            xy[m, 0],
            xy[m, 1],
            c="#2ecc71",
            s=80,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
            label=f"success (+1) terminal (n={counts[1]})",
        )
    # Abort -1
    if counts[2] > 0:
        m = cat == 2
        ax.scatter(
            xy[m, 0],
            xy[m, 1],
            c="#e74c3c",
            s=80,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
            marker="X",
            label=f"abort (-1) terminal (n={counts[2]})",
        )
    ax.set_xlabel("t-SNE1")
    ax.set_ylabel("t-SNE2")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "checkpoint_dir", type=Path, help="path to a checkpoint dir containing replay_buffer.pt + critic.pt"
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output PNG path (default: <checkpoint_dir>/rlt_visualizations.png)",
    )
    p.add_argument("--tsne-perplexity", type=int, default=30)
    p.add_argument(
        "--max-mid-episode",
        type=int,
        default=1000,
        help="subsample r=0 transitions when plotting (defaults 1000)",
    )
    args = p.parse_args()

    ckpt = args.checkpoint_dir
    if not (ckpt / "replay_buffer.pt").exists():
        print(f"ERROR: {ckpt}/replay_buffer.pt not found", file=sys.stderr)
        return 1
    out = args.out or (ckpt / "rlt_visualizations.png")

    print(f"loading {ckpt}/replay_buffer.pt ...")
    blob = torch.load(str(ckpt / "replay_buffer.pt"), weights_only=False)
    n = int(blob["size"])
    if n == 0:
        print("ERROR: empty replay buffer", file=sys.stderr)
        return 1
    z_rl = blob["z_rl"][:n].numpy().astype(np.float32)
    state = blob["state"][:n].numpy().astype(np.float32)
    action = blob["action"][:n].numpy().astype(np.float32)
    ref = blob["ref"][:n].numpy().astype(np.float32)
    reward = blob["reward"][:n, 0].numpy()
    done = blob["done"][:n, 0].numpy()
    print(f"loaded {n} transitions  z_rl_dim={z_rl.shape[1]}  state_dim={state.shape[1]}")

    cat = _label_categories(reward, done)
    counts = {0: int((cat == 0).sum()), 1: int((cat == 1).sum()), 2: int((cat == 2).sum())}
    print(f"categories: mid-episode={counts[0]}  success+1={counts[1]}  abort-1={counts[2]}")

    # Load networks. Caller can run with critic-only (V undefined) or
    # neither (only the categorical view). We compute both Q(s, a_stored)
    # and V(s) ≈ Q(s, π_θ(s)) — they tell different stories. See module
    # docstring.
    action_dim_inferred = action.shape[1] // 10  # rl_chunk_length default
    cfg, actor_net, critic_net = _load_actor_critic(
        ckpt,
        z_rl_dim=z_rl.shape[1],
        state_dim=state.shape[1],
        action_dim=action_dim_inferred,
    )
    action_dim = action.shape[1] // cfg.rl_chunk_length

    q_stored = None
    if critic_net is not None:
        try:
            print("computing Q(s, a_stored) on every transition ...")
            q_stored = _critic_q_on_stored(critic_net, cfg, z_rl, state, action, action_dim)
            print(
                f"  Q(s,a_stored) range: [{q_stored.min():.2f}, {q_stored.max():.2f}]  "
                f"mean={q_stored.mean():.2f}"
            )
        except Exception as e:
            print(f"  Q(s,a_stored) eval failed: {e}", file=sys.stderr)

    v_actor = None
    if actor_net is not None and critic_net is not None:
        try:
            print("computing V(s) ≈ Q(s, π_θ(s)) on every transition ...")
            v_actor = _value_under_actor(
                actor_net,
                critic_net,
                cfg,
                z_rl,
                state,
                ref,
                action_dim,
            )
            print(f"  V(s) range: [{v_actor.min():.2f}, {v_actor.max():.2f}]  mean={v_actor.mean():.2f}")
        except Exception as e:
            print(f"  V(s) eval failed: {e}", file=sys.stderr)

    # t-SNE on z_rl
    print(f"running t-SNE on z_rl (perplexity={args.tsne_perplexity}) ...")
    z_rl_2d = TSNE(
        n_components=2,
        perplexity=args.tsne_perplexity,
        random_state=42,
        init="pca",
        learning_rate="auto",
    ).fit_transform(z_rl)

    # Figure: 2x3, all of row 1 share the same z_rl t-SNE coordinates
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(
        f"RLT diagnostic — {ckpt.parent.name}/{ckpt.name}  (n={n})",
        fontsize=14,
        y=0.995,
    )

    # (1) z_rl t-SNE colored by reward category
    _scatter_by_category(
        axes[0, 0],
        z_rl_2d,
        cat,
        title="z_rl by reward category",
        max_mid=args.max_mid_episode,
    )

    # (2) z_rl t-SNE colored by V(s) — actor's policy value at each state.
    # This is the "is this STATE region good or bad to be in?" view —
    # different from panel 3 (Q on stored action). Panel 3 reflects
    # training labels; V(s) reflects how the current actor evaluates each
    # state via Q(s, π_θ(s)). For OOD-shake aborts, panel 3 shows red
    # because the recorded wild action got Q=-1, but if the actor's
    # current policy chooses a different action at the same state, V(s)
    # may well be positive there.
    ax = axes[0, 1]
    if v_actor is not None:
        sc = ax.scatter(
            z_rl_2d[:, 0], z_rl_2d[:, 1], c=v_actor, s=6, cmap="RdYlGn", vmin=-1.2, vmax=1.2, alpha=0.7
        )
        plt.colorbar(sc, ax=ax, label="V(s) ≈ Q(s, π(s))")
        ax.set_title("z_rl colored by V(s) [actor's policy value]")
    else:
        ax.text(
            0.5,
            0.5,
            "actor or critic not available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            color="gray",
        )
        ax.set_title("V(s) unavailable")
    ax.set_xlabel("t-SNE1")
    ax.set_ylabel("t-SNE2")

    # (3) z_rl t-SNE colored by Q(s, a_stored) — what the critic predicts
    # for the action that was actually recorded in this transition. For
    # abort transitions, that's the wild action right before the operator
    # pressed LEFT. Red here = "the recorded action led to low Q" — does
    # not imply the state is intrinsically bad.
    ax = axes[0, 2]
    if q_stored is not None:
        sc = ax.scatter(
            z_rl_2d[:, 0], z_rl_2d[:, 1], c=q_stored, s=6, cmap="RdYlGn", vmin=-1.2, vmax=1.2, alpha=0.7
        )
        plt.colorbar(sc, ax=ax, label="Q(s, a_stored)")
        ax.set_title("z_rl colored by Q on stored action [training labels]")
    else:
        ax.text(
            0.5,
            0.5,
            "critic not available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            color="gray",
        )
        ax.set_title("Q(s, a_stored) unavailable")
    ax.set_xlabel("t-SNE1")
    ax.set_ylabel("t-SNE2")

    # (4) Q histograms by reward category — Q on the stored action.
    # Cleanly-separated peaks at +1 / 0.x / -1 mean the critic learned
    # discriminative training labels.
    ax = axes[1, 0]
    if q_stored is not None:
        bins = np.linspace(-1.5, 1.5, 50)
        if counts[0] > 0:
            ax.hist(
                q_stored[cat == 0],
                bins=bins,
                alpha=0.45,
                color="#888888",
                label=f"mid-episode (n={counts[0]})",
            )
        if counts[1] > 0:
            ax.hist(
                q_stored[cat == 1],
                bins=bins,
                alpha=0.6,
                color="#2ecc71",
                label=f"+1 terminal (n={counts[1]})",
            )
        if counts[2] > 0:
            ax.hist(
                q_stored[cat == 2],
                bins=bins,
                alpha=0.6,
                color="#e74c3c",
                label=f"-1 terminal (n={counts[2]})",
            )
        ax.axvline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.axvline(1, color="#2ecc71", linestyle=":", linewidth=0.7, alpha=0.5)
        ax.axvline(-1, color="#e74c3c", linestyle=":", linewidth=0.7, alpha=0.5)
        ax.set_xlabel("Q(s, a_stored)")
        ax.set_ylabel("count")
        ax.set_title("Q(s, a_stored) by reward category")
        ax.legend(loc="best", fontsize=8)
    else:
        ax.set_axis_off()

    # (5) Actor deviation vs Q on stored action. Shows whether large
    # actor deviations correlate with bad outcomes (red Xs at high
    # delta) or good ones (green at high delta).
    ax = axes[1, 1]
    if q_stored is not None:
        delta = np.abs(action - ref).mean(axis=1)
        ax.scatter(delta[cat == 0], q_stored[cat == 0], c="#bbbbbb", s=4, alpha=0.3, label="mid-ep")
        if counts[1] > 0:
            ax.scatter(
                delta[cat == 1],
                q_stored[cat == 1],
                c="#2ecc71",
                s=40,
                edgecolor="black",
                linewidth=0.4,
                alpha=0.85,
                label="+1",
            )
        if counts[2] > 0:
            ax.scatter(
                delta[cat == 2],
                q_stored[cat == 2],
                c="#e74c3c",
                s=40,
                edgecolor="black",
                linewidth=0.4,
                alpha=0.85,
                marker="X",
                label="-1",
            )
        ax.set_xlabel("|actor - ref| (mean per transition)")
        ax.set_ylabel("Q(s, a_stored)")
        ax.set_title("Actor deviation vs Q on stored action")
        ax.legend(loc="best", fontsize=8)
    else:
        ax.set_axis_off()

    # (6) Episode-position timeline (which buffer slot did each terminal land at?)
    ax = axes[1, 2]
    idx = np.arange(n)
    ax.scatter(idx[cat == 0], np.zeros((cat == 0).sum()), c="#cccccc", s=2, alpha=0.3, label="mid-ep")
    if counts[1] > 0:
        ax.scatter(
            idx[cat == 1],
            np.ones((cat == 1).sum()),
            c="#2ecc71",
            s=40,
            edgecolor="black",
            linewidth=0.4,
            label="+1",
        )
    if counts[2] > 0:
        ax.scatter(
            idx[cat == 2],
            -np.ones((cat == 2).sum()),
            c="#e74c3c",
            s=40,
            edgecolor="black",
            linewidth=0.4,
            marker="X",
            label="-1",
        )
    ax.set_xlabel("buffer index (oldest → newest)")
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["abort", "mid-ep", "success"])
    ax.set_ylim(-1.5, 1.5)
    ax.set_title("Terminal positions in buffer (x = recency)")
    ax.legend(loc="lower center", fontsize=8, ncol=3)
    ax.grid(axis="x", alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
