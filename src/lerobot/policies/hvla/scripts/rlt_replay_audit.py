"""Quick audit of replay_buffer.pt: composition, terminal counts, Q stats.

Answers questions like:
  - How many transitions are terminal (done=True)?
  - Of terminals, how many are +1 (success), -1 (abort), 0 (other)?
  - How many non-terminal r=0 transitions (failure rollouts + intervention windows)?
  - What does the critic predict on a sample of transitions?

Usage:
    python src/lerobot/policies/hvla/scripts/rlt_replay_audit.py outputs/rlt_online_v2_widened/latest
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: rlt_replay_audit.py <checkpoint_dir>", file=sys.stderr)
        return 1
    ckpt = Path(sys.argv[1])
    rb_path = ckpt / "replay_buffer.pt"
    actor_path = ckpt / "actor.pt"
    critic_path = ckpt / "critic.pt"

    print(f"loading {rb_path} ...")
    blob = torch.load(rb_path, map_location="cpu", weights_only=False)
    size = int(blob["size"])
    reward = blob["reward"][:size, 0]
    done = blob["done"][:size, 0]

    print(f"buffer size: {size} (capacity {blob['capacity']})")
    print()
    print("=== Reward / done composition ===")
    n_done = int((done > 0.5).sum())
    n_pos1 = int(((reward > 0.5) & (done > 0.5)).sum())
    n_neg1 = int(((reward < -0.5) & (done > 0.5)).sum())
    n_zero_terminal = int(((reward.abs() < 0.5) & (done > 0.5)).sum())
    n_nonterm = size - n_done
    print(f"  terminal (done=True)        : {n_done}")
    print(f"    +1.0 (autonomous success) : {n_pos1}")
    print(f"    -1.0 (abort/disaster)     : {n_neg1}")
    print(f"    other terminal            : {n_zero_terminal}")
    print(f"  non-terminal (done=False)   : {n_nonterm}")
    print(f"    r=0 (most common)         : {int(((reward.abs() < 1e-6) & (done < 0.5)).sum())}")
    print(f"    r!=0 unexpected           : {int(((reward.abs() > 1e-6) & (done < 0.5)).sum())}")

    # Distinguish RL transitions (z_rl from inference thread) vs intervention.
    # Intervention chunks have action == ref (BC penalty zero). RL transitions
    # have action = actor output, ref = S1 reference — these only match in
    # warmup mode (actor REPLACES with ref) or when actor delta = 0.
    action = blob["action"][:size]
    ref = blob["ref"][:size]
    delta = (action - ref).abs().mean(dim=1)
    n_action_eq_ref = int((delta < 1e-6).sum())
    n_action_neq_ref = size - n_action_eq_ref
    print()
    print("=== Action vs Ref alignment ===")
    print(f"  action == ref (intervention or warmup): {n_action_eq_ref}")
    print(f"  action != ref (post-warmup RL actor)  : {n_action_neq_ref}")
    print(f"  delta stats: mean={delta.mean():.4f} max={delta.max():.4f}")

    print()
    print("=== Reward by episode-end ===")
    # When does done occur? Position in buffer.
    done_idx = torch.nonzero(done > 0.5, as_tuple=False).squeeze(-1).tolist()
    print(f"  terminal positions (first 30): {done_idx[:30]}")
    if len(done_idx) > 30:
        print(f"  ... and {len(done_idx) - 30} more")
    for i in done_idx[:20]:
        print(f"    idx={i:5d}  reward={reward[i].item():+.2f}  delta={delta[i].item():.3f}")

    # Spot-check Q values on terminal vs non-terminal samples by loading the
    # critic and running forward. This gives us a sanity check on Q range.
    if critic_path.exists() and actor_path.exists():
        try:
            from lerobot.policies.hvla.rlt.actor_critic import RLTCritic
            from lerobot.policies.hvla.rlt.config import RLTConfig
        except Exception as e:
            print(f"\n(skipping critic eval — import failed: {e})")
            return 0

        # Recover rl_token_dim from buffer; cfg only exposes the default.
        rl_token_dim = blob["z_rl"].shape[1]
        state_dim = blob["state"].shape[1]
        action_total = blob["action"].shape[1]
        cfg = RLTConfig(rl_token_dim=rl_token_dim)
        action_dim = action_total // cfg.rl_chunk_length
        if action_dim * cfg.rl_chunk_length != action_total:
            print(
                f"  action_dim mismatch: total={action_total} not divisible by C={cfg.rl_chunk_length}; skipping Q eval"
            )
            return 0

        critic = RLTCritic(cfg, state_dim, action_dim)
        critic.load_state_dict(torch.load(critic_path, map_location="cpu", weights_only=True))
        critic.eval()

        # Sample a few transitions from each category
        def q_on(indices):
            if not indices:
                return None
            idx = torch.tensor(indices)
            with torch.no_grad():
                z = blob["z_rl"][idx]
                s = blob["state"][idx]
                a = blob["action"][idx]
                # critic forward expects [B, ...]
                q_list = critic(z, s, a.reshape(-1, cfg.rl_chunk_length, action_dim))
            # critic returns list of [B, 1] tensors (one per ensemble)
            return torch.stack([q.squeeze(-1) for q in q_list]).min(dim=0).values

        # Group indices
        idx_pos1 = [i for i, (r, d) in enumerate(zip(reward, done, strict=False)) if r > 0.5 and d > 0.5]
        idx_neg1 = [i for i, (r, d) in enumerate(zip(reward, done, strict=False)) if r < -0.5 and d > 0.5]
        idx_nonterm = [i for i, d in enumerate(done) if d < 0.5][:200]
        idx_human = torch.nonzero(delta < 1e-6, as_tuple=False).squeeze(-1).tolist()[:200]

        print()
        print("=== Critic Q values on stored transitions (min over ensemble) ===")
        for label, idx_list in [
            (f"terminal +1 ({len(idx_pos1)})", idx_pos1),
            (f"terminal -1 ({len(idx_neg1)})", idx_neg1),
            ("non-terminal r=0 (sample 200)", idx_nonterm),
            ("action==ref ie intervention or warmup (sample 200)", idx_human),
        ]:
            q = q_on(idx_list)
            if q is None or q.numel() == 0:
                print(f"  {label}: no samples")
                continue
            print(f"  {label}: mean={q.mean():+.3f} min={q.min():+.3f} max={q.max():+.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
