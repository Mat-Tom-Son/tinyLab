#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "reports" / "pythia_head_trajectory.parquet"
OUT_PDF = ROOT / "paper" / "figures" / "pythia_emergence_curves.pdf"
OUT_PNG = OUT_PDF.with_suffix(".png")


def main() -> None:
    df = pd.read_parquet(DATA)
    df = df[(df.layer == 0) & (df.metric == "logit_diff")]
    # Pythia-160M has 12 heads per layer: 0..11
    heads = [2, 4, 7, 11]
    conditions = ["facts", "neg", "cf", "logic"]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=False)
    axes = axes.flatten()
    colors = {2: "C0", 4: "C1", 7: "C2", 11: "C3"}

    for ax, cond in zip(axes, conditions):
        sub = df[df["condition"] == cond]
        steps = sorted(sub.checkpoint_step.unique())
        for h in heads:
            hs = sub[sub["head"] == h].sort_values("checkpoint_step")
            if hs.empty:
                continue
            ax.plot(
                hs.checkpoint_step,
                hs.value,
                marker="o",
                label=f"0:{h}",
                color=colors.get(h, None),
            )
        ax.set_title(cond.capitalize())
        ax.set_xlabel("Training Step")
        ax.set_ylabel("ΔLD (head ablated)")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Pythia-160M: Emergence of Layer-0 Suppressor Effects")
    fig.tight_layout(rect=[0, 0.0, 1, 0.96])

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT_PDF} and {OUT_PNG}")


if __name__ == "__main__":
    main()
mpl.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'savefig.dpi': 200,
    'lines.linewidth': 1.8,
})
mpl.rcParams['axes.prop_cycle'] = cycler(color=[
    '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9', '#E69F00', '#000000'
])
