"""Plot training trajectories in (new-loss, old-loss) space — one panel per replay %."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
CURVE_DIR = HERE / "phase_b"

EPISODES = [247, 174, 299]
REPLAY_PCTS = [0, 10, 25]

EP_COLORS = {247: "#d62728", 174: "#1f77b4", 299: "#2ca02c"}
EP_LABELS = {
    247: "ep 247 (hard)",
    174: "ep 174 (hard)",
    299: "ep 299 (control)",
}

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True, sharey=True)

for ax, rp in zip(axes, REPLAY_PCTS, strict=False):
    for ep in EPISODES:
        path = CURVE_DIR / f"ep{ep}_rp{rp:03d}_curve.csv"
        df = pd.read_csv(path)
        x = df["fit_val_loss"].values
        y = df["forget_val_loss"].values
        color = EP_COLORS[ep]
        ax.plot(x, y, "-", color=color, lw=1.6, alpha=0.85, label=EP_LABELS[ep])
        # Start: hollow circle (baseline, step 0)
        ax.plot(x[0], y[0], "o", mfc="white", mec=color, mew=1.8, ms=9)
        # End: filled circle (final, step 100)
        ax.plot(x[-1], y[-1], "o", color=color, ms=9)

    ax.set_title(f"{rp}% replay", fontsize=13)
    ax.set_xlabel("loss on new episode (lower → better fit)")
    ax.grid(True, alpha=0.3)
    ax.axvline(0.04, color="gray", ls=":", lw=0.8, alpha=0.7)
    ax.axhline(0.04, color="gray", ls=":", lw=0.8, alpha=0.7)
    ax.set_xlim(0, 0.34)
    ax.set_ylim(0, 0.12)

axes[0].set_ylabel("loss on prior data (lower → less forgetting)")
axes[0].legend(loc="upper right", fontsize=9, framealpha=0.95)

# Annotation: "good corner" in the bottom-left of leftmost panel
axes[0].annotate(
    "ideal\n(both losses low)",
    xy=(0.012, 0.012),
    xytext=(0.07, 0.01),
    fontsize=9,
    color="#2a7a2a",
    ha="left",
    va="bottom",
    arrowprops={"arrowstyle": "->", "color": "#2a7a2a", "lw": 1.2},
)

# Direction-of-training annotation on rightmost panel
axes[2].annotate(
    "training\n(○ start → ● end)",
    xy=(0.27, 0.10),
    xytext=(0.18, 0.10),
    fontsize=9,
    color="#444",
    ha="left",
    va="center",
)

fig.suptitle(
    "Flash-DAgger LoRA: training trajectory in (new, old) loss space — 100 steps, rank 16",
    fontsize=12,
    y=1.00,
)
plt.tight_layout()

out_path = HERE / "trajectories.png"
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"Saved {out_path}")
