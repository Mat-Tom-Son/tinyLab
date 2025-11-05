"""Overlay baseline entropy profile with Δ entropy from sharpener scan.

Reads a JSON produced by lab.analysis.layer_entropy_and_sharpener_scan and
generates a figure with:
  - Baseline layer-wise output entropy (line)
  - Scatter (or small bars) of per-head Δ entropy on the last-K layers

Runtime tip: negligible; plotting only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from cycler import cycler


def load_scan(path: Path) -> Dict:
    return json.loads(path.read_text())


def plot_entropy_overlay(data: Dict, out: Path) -> None:
    prof = data["layer_entropy_profile"]
    heads = data["sharpener_scan"]

    layers = [int(x["layer"]) for x in prof]
    ent = [float(x["entropy"]) for x in prof]

    # Styling
    mpl.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "savefig.dpi": 200,
        "lines.linewidth": 1.8,
    })
    mpl.rcParams["axes.prop_cycle"] = cycler(color=["#0072B2", "#D55E00", "#009E73", "#CC79A7"])

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.plot(layers, ent, marker="o", label="Baseline entropy (last position)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Entropy")
    ax.set_title("Entropy Profile with Δ Entropy from Last-K Head Ablations")

    # Overlay Δ entropy (ablated − baseline) for scanned heads
    # Use a secondary y-axis for clarity
    ax2 = ax.twinx()
    ax2.set_ylabel("Δ Entropy (ablated − baseline)")

    # Group by layer
    by_layer: Dict[int, List[float]] = {}
    for r in heads:
        L = int(r["layer"])
        by_layer.setdefault(L, []).append(float(r["d_entropy_final"]))

    xs, ys = [], []
    for L, vals in by_layer.items():
        xs.extend([L] * len(vals))
        ys.extend(vals)
    ax2.scatter(xs, ys, color="#D55E00", alpha=0.5, s=12, label="Δ entropy per head")

    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    print(f"Wrote {out}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("figs/entropy_overlay"))
    args = ap.parse_args()

    data = load_scan(args.input)
    stem = args.input.stem
    out = args.out.parent / f"{stem}_overlay"
    plot_entropy_overlay(data, out)


if __name__ == "__main__":
    main()

