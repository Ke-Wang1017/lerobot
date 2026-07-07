"""Phase E saturation analysis: heatmap + summary curves over N=30 iterations.

Produces:
  phase_e_heatmap.png  — per-episode loss heatmap (rows = episodes in order
    of addition, cols = iterations, color = held-out loss). The diagonal
    shows fit quality at addition time; off-diagonal shows retention as more
    episodes are added.
  phase_e_curves.png   — three summary lines: (a) new-fit drop %% per iter,
    (b) mean loss across all flashed episodes per iter (cumulative quality),
    (c) max loss across flashed episodes (worst-case retention).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
df = pd.read_csv(HERE / "phase_e" / "capacity_curves_old10_fl25.csv")

# Episodes in order of first appearance
ep_order: list[int] = []
for ep in df["eval_ep"]:
    if ep == "forget":
        continue
    try:
        ep_int = int(ep)
    except ValueError:
        continue
    if ep_int not in ep_order:
        ep_order.append(ep_int)

n_eps = len(ep_order)
n_iters = int(df["iter"].max())

# Build N×N matrix: rows = episodes (added order), cols = iterations
loss_matrix = np.full((n_eps, n_iters), np.nan)
for _, r in df.iterrows():
    if r["eval_ep"] == "forget":
        continue
    try:
        ep = int(r["eval_ep"])
    except ValueError:
        continue
    row = ep_order.index(ep)
    col = int(r["iter"]) - 1
    loss_matrix[row, col] = float(r["loss"])

# --- Heatmap ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 7))
im = ax.imshow(loss_matrix, aspect="auto", cmap="viridis_r", vmin=0.02, vmax=0.20, origin="upper")
cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
cbar.set_label("held-out loss", fontsize=10)
ax.set_xticks(range(n_iters))
ax.set_xticklabels(range(1, n_iters + 1), fontsize=8)
ax.set_yticks(range(n_eps))
ax.set_yticklabels(
    [f"ep {ep} (added @ {i + 1})" for i, ep in enumerate(ep_order)],
    fontsize=8,
)
ax.set_xlabel("iteration")
ax.set_title(
    f"Phase E heatmap: per-episode held-out loss across {n_iters} flash iterations\n"
    "Diagonal = fit quality when added • off-diagonal (right of diagonal) = retention as more added",
    fontsize=11,
)

# Mark the diagonal (when each episode was added)
for i in range(n_eps):
    ax.add_patch(plt.Rectangle((i - 0.45, i - 0.45), 0.9, 0.9, fill=False, edgecolor="red", lw=1.2))

plt.tight_layout()
plt.savefig(HERE / "phase_e_heatmap.png", dpi=140, bbox_inches="tight")
print(f"Saved {HERE / 'phase_e_heatmap.png'}")
plt.close(fig)

# --- Summary curves -----------------------------------------------------------
# (a) new_fit Δ% per iter
# (b) mean loss across flashed eps (eval_ep ∈ all eps seen so far) per iter
# (c) max loss across flashed eps per iter
# (d) forget loss per iter

iters_x = np.arange(1, n_iters + 1)
new_fit_delta = np.zeros(n_iters)
mean_old = np.zeros(n_iters)
max_old = np.zeros(n_iters)
forget_loss = np.zeros(n_iters)
mean_all = np.zeros(n_iters)  # average over ALL flashed eps including newest

for it in range(1, n_iters + 1):
    rows = df[df["iter"] == it]
    trained_ep = int(rows["trained_ep"].iloc[0])
    new_loss = float(rows[rows["eval_ep"] == str(trained_ep)]["loss"].iloc[0])
    baseline = float(rows[rows["eval_ep"] == str(trained_ep)]["baseline"].iloc[0])
    new_fit_delta[it - 1] = 100 * (1 - new_loss / max(baseline, 1e-6))
    forget_loss[it - 1] = float(rows[rows["eval_ep"] == "forget"]["loss"].iloc[0])

    flashed_losses = []
    flashed_old_losses = []
    for _, r in rows.iterrows():
        if r["eval_ep"] == "forget":
            continue
        try:
            ep = int(r["eval_ep"])
        except ValueError:
            continue
        flashed_losses.append(float(r["loss"]))
        if ep != trained_ep:
            flashed_old_losses.append(float(r["loss"]))
    mean_all[it - 1] = float(np.mean(flashed_losses)) if flashed_losses else 0.0
    mean_old[it - 1] = float(np.mean(flashed_old_losses)) if flashed_old_losses else 0.0
    max_old[it - 1] = float(np.max(flashed_old_losses)) if flashed_old_losses else 0.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot(iters_x, new_fit_delta, "-o", color="#1f77b4", lw=2, label="new-fit drop %% (just-added ep)")
ax1.set_xlabel("iteration")
ax1.set_ylabel("new-fit drop %% from baseline")
ax1.set_title("Fit capacity: does new-episode fit degrade as N grows?")
ax1.set_ylim(50, 100)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(1, n_iters + 1, 2))

# Mean line for first 5 iters and last 5 iters
early_mean = float(np.mean(new_fit_delta[:5]))
late_mean = float(np.mean(new_fit_delta[-5:]))
ax1.axhline(early_mean, ls="--", color="green", alpha=0.5, lw=1)
ax1.axhline(late_mean, ls="--", color="red", alpha=0.5, lw=1)
ax1.text(n_iters * 0.6, early_mean + 1, f"avg first-5: {early_mean:.1f}%%", fontsize=8, color="green")
ax1.text(n_iters * 0.6, late_mean - 2, f"avg last-5: {late_mean:.1f}%%", fontsize=8, color="red")

ax2.plot(iters_x, mean_old, "-o", color="#d62728", lw=2, label="mean old-episode loss")
ax2.plot(iters_x, max_old, "-^", color="#ff7f0e", lw=1.5, label="max old-episode loss")
ax2.plot(iters_x, mean_all, "-s", color="#9467bd", lw=1.5, label="mean of all flashed (incl newest)")
ax2.plot(iters_x, forget_loss, "--", color="black", lw=1.5, label="forget (training-set)")
ax2.axhline(0.04, color="gray", ls=":", lw=0.8, alpha=0.6)
ax2.text(n_iters - 4, 0.043, "easy-ep floor", fontsize=8, color="gray")
ax2.set_xlabel("iteration")
ax2.set_ylabel("held-out loss")
ax2.set_title("Retention: how does the LoRA hold accumulated corrections?")
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(1, n_iters + 1, 2))
ax2.legend(loc="upper left", fontsize=9)

fig.suptitle(
    f"Phase E saturation analysis (N={n_iters}, 10% old + 25% flashed + 65% new, rank 16, 60 steps/iter)",
    fontsize=11,
    y=1.00,
)
plt.tight_layout()
plt.savefig(HERE / "phase_e_curves.png", dpi=140, bbox_inches="tight")
print(f"Saved {HERE / 'phase_e_curves.png'}")
plt.close(fig)

# --- Numeric summary ---------------------------------------------------------
print("\n=== Phase E saturation summary ===")
print(f"Iterations: {n_iters}")
print(
    f"new-fit Δ%%: first-5 avg = {early_mean:.1f}%%, last-5 avg = {late_mean:.1f}%%, "
    f"slope = {late_mean - early_mean:+.1f}pp"
)
print(
    f"mean old-ep loss: iter1=N/A, iter5={mean_old[4]:.4f}, iter15={mean_old[14] if n_iters >= 15 else float('nan'):.4f}, "
    f"iter25={mean_old[24] if n_iters >= 25 else float('nan'):.4f}, iter{n_iters}={mean_old[-1]:.4f}"
)
print(
    f"max old-ep loss: iter5={max_old[4]:.4f}, iter15={max_old[14] if n_iters >= 15 else float('nan'):.4f}, "
    f"iter25={max_old[24] if n_iters >= 25 else float('nan'):.4f}, iter{n_iters}={max_old[-1]:.4f}"
)
print(f"forget: iter1={forget_loss[0]:.4f}, iter{n_iters}={forget_loss[-1]:.4f}")
