"""Phase C vs Phase D capacity-test trajectories — side by side."""

from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
df_c = pd.read_csv(HERE / "phase_c" / "capacity_curves.csv")
df_d = pd.read_csv(HERE / "phase_d" / "capacity_curves_old10_fl25.csv")


def collect_ep_order(df: pd.DataFrame) -> list[int]:
    seen: list[int] = []
    for ep in df["eval_ep"]:
        if ep == "forget":
            continue
        try:
            ep_int = int(ep)
        except ValueError:
            continue
        if ep_int not in seen:
            seen.append(ep_int)
    return seen


ep_order = collect_ep_order(df_c)
n_eps = len(ep_order)
colors = cm.viridis(np.linspace(0.05, 0.9, n_eps))
ep_color = {ep: colors[i] for i, ep in enumerate(ep_order)}


def plot_phase(ax, df: pd.DataFrame, title: str):
    for ep, color in ep_color.items():
        mask = df["eval_ep"].astype(str) == str(ep)
        sub = df[mask].sort_values("iter")
        if sub.empty:
            continue
        iters = sub["iter"].astype(int).values
        losses = sub["loss"].astype(float).values
        added_at = sub["added_at_iter"].iloc[0]
        ax.plot(
            iters, losses, "-o", color=color, lw=1.6, ms=5, alpha=0.9, label=f"ep {ep} (added @ {added_at})"
        )
        if iters[0] == added_at:
            ax.plot(iters[0], losses[0], "s", color=color, ms=9, mec="black", mew=0.8)
    forget = df[df["eval_ep"] == "forget"].sort_values("iter")
    ax.plot(forget["iter"], forget["loss"], "--", color="black", lw=2.0, label="forget (training-set)")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("iteration")
    ax.set_xticks(range(1, 11))
    ax.grid(True, alpha=0.3)
    ax.axhline(0.04, color="gray", ls=":", lw=0.8, alpha=0.6)
    ax.set_ylim(0, 0.36)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
plot_phase(ax1, df_c, "Phase C — 10% old replay only (no rehearsal)")
plot_phase(ax2, df_d, "Phase D — 10% old + 25% flashed + 65% new (3-way mix)")
ax1.set_ylabel("held-out loss")
ax1.legend(loc="upper right", fontsize=7, framealpha=0.95, ncol=2)
ax1.text(0.5, 0.043, "≈ easy-episode floor", fontsize=8, color="gray")

fig.suptitle(
    "Capacity test: same 10 episodes flashed sequentially. ■ = added at this iter; "
    "subsequent points = retention as more episodes are added.",
    fontsize=11,
    y=1.00,
)
plt.tight_layout()

out_path = HERE / "phase_cd_compare.png"
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"Saved {out_path}")
