"""Phase E (r=16) vs Phase F (r=4) — capacity comparison."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
df_e = pd.read_csv(HERE / "phase_e" / "capacity_curves_old10_fl25.csv")
df_f = pd.read_csv(HERE / "phase_f" / "capacity_curves_old10_fl25.csv")


def per_iter_stats(df: pd.DataFrame, n_iters: int):
    new_fit_delta = np.zeros(n_iters)
    mean_old = np.zeros(n_iters)
    max_old = np.zeros(n_iters)
    forget_loss = np.zeros(n_iters)
    for it in range(1, n_iters + 1):
        rows = df[df["iter"] == it]
        trained_ep = int(rows["trained_ep"].iloc[0])
        new_loss = float(rows[rows["eval_ep"] == str(trained_ep)]["loss"].iloc[0])
        baseline = float(rows[rows["eval_ep"] == str(trained_ep)]["baseline"].iloc[0])
        new_fit_delta[it - 1] = 100 * (1 - new_loss / max(baseline, 1e-6))
        forget_loss[it - 1] = float(rows[rows["eval_ep"] == "forget"]["loss"].iloc[0])
        old = []
        for _, r in rows.iterrows():
            if r["eval_ep"] == "forget":
                continue
            try:
                ep = int(r["eval_ep"])
            except ValueError:
                continue
            if ep != trained_ep:
                old.append(float(r["loss"]))
        mean_old[it - 1] = float(np.mean(old)) if old else 0.0
        max_old[it - 1] = float(np.max(old)) if old else 0.0
    return new_fit_delta, mean_old, max_old, forget_loss


n_iters = int(df_e["iter"].max())
fit_e, old_e, max_e, fgt_e = per_iter_stats(df_e, n_iters)
fit_f, old_f, max_f, fgt_f = per_iter_stats(df_f, n_iters)

iters_x = np.arange(1, n_iters + 1)

fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))

axes[0].plot(iters_x, fit_e, "-o", color="#1f77b4", label="r=16", lw=1.6, ms=4)
axes[0].plot(iters_x, fit_f, "-s", color="#d62728", label="r=4", lw=1.6, ms=4)
axes[0].set_title("New-fit Δ% (just-added episode)\nhigher is better")
axes[0].set_xlabel("iteration")
axes[0].set_ylabel("Δ% from baseline")
axes[0].set_ylim(50, 100)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(iters_x, old_e, "-o", color="#1f77b4", label="r=16", lw=1.6, ms=4)
axes[1].plot(iters_x, old_f, "-s", color="#d62728", label="r=4", lw=1.6, ms=4)
axes[1].set_title("Mean old-episode loss\nlower is better")
axes[1].set_xlabel("iteration")
axes[1].set_ylabel("loss")
axes[1].axhline(0.04, color="gray", ls=":", lw=0.8, alpha=0.6)
axes[1].set_ylim(0, 0.13)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(iters_x, max_e, "-o", color="#1f77b4", label="r=16", lw=1.6, ms=4)
axes[2].plot(iters_x, max_f, "-s", color="#d62728", label="r=4", lw=1.6, ms=4)
axes[2].set_title("Max old-episode loss\n(worst-case retention)")
axes[2].set_xlabel("iteration")
axes[2].set_ylabel("loss")
axes[2].set_ylim(0, 0.18)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

axes[3].plot(iters_x, fgt_e, "-o", color="#1f77b4", label="r=16", lw=1.6, ms=4)
axes[3].plot(iters_x, fgt_f, "-s", color="#d62728", label="r=4", lw=1.6, ms=4)
axes[3].set_title("Forget loss (training-set holdout)")
axes[3].set_xlabel("iteration")
axes[3].set_ylabel("loss")
axes[3].set_ylim(0.01, 0.06)
axes[3].legend()
axes[3].grid(True, alpha=0.3)

fig.suptitle(
    "Phase E (r=16) vs Phase F (r=4) — same N=30 episodes, same 10/25/65 mix, 60 steps/iter",
    fontsize=11,
    y=1.00,
)
plt.tight_layout()
out = HERE / "phase_ef_compare.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"Saved {out}")

print()
print("=== r=4 vs r=16 summary ===")
print(
    f"new-fit (last-5):   r=16 {fit_e[-5:].mean():.1f}%  vs  r=4 {fit_f[-5:].mean():.1f}%   "
    f"(gap {fit_e[-5:].mean() - fit_f[-5:].mean():+.1f}pp)"
)
print(
    f"old_avg (last-5):   r=16 {old_e[-5:].mean():.4f}  vs  r=4 {old_f[-5:].mean():.4f}   "
    f"(gap {(old_f[-5:].mean() / old_e[-5:].mean() - 1) * 100:+.0f}%)"
)
print(
    f"old_worst (peak):   r=16 {max_e.max():.4f}  vs  r=4 {max_f.max():.4f}   "
    f"(gap {(max_f.max() / max_e.max() - 1) * 100:+.0f}%)"
)
print(f"forget (last-5):    r=16 {fgt_e[-5:].mean():.4f}  vs  r=4 {fgt_f[-5:].mean():.4f}")
