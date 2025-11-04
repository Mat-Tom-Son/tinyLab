import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from cycler import cycler

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASK_FILES = {
    "Facts": ROOT / "reports" / "activation_entropy_gpt2medium_facts_robust.json",
    "Counterfactual": ROOT / "reports" / "activation_entropy_gpt2medium_cf_robust.json",
    "Negation": ROOT / "reports" / "activation_entropy_gpt2medium_neg_robust.json",
    "Logic": ROOT / "reports" / "activation_entropy_gpt2medium_logic_robust.json",
}


def load_outputs() -> Dict[str, Dict]:
    data = {}
    for task, path in TASK_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing report for {task}: {path}")
        data[task] = json.loads(path.read_text())
    return data


def gather_random_deltas(data: Dict[str, Dict]) -> List[float]:
    vals: List[float] = []
    for task, d in data.items():
        rc = d.get("random_control", {})
        delta_out = rc.get("delta_output_entropy", {})
        vals.extend(delta_out.get("values", []))
    return [float(x) for x in vals]


def fig_geometric_signature(data: Dict[str, Dict]) -> Path:
    # Styling
    mpl.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "savefig.dpi": 200,
        "lines.linewidth": 1.8,
    })
    mpl.rcParams["axes.prop_cycle"] = cycler(
        color=["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # Panel A: Output entropy (random null + actual deltas)
    rand = np.array(gather_random_deltas(data), dtype=float)
    if rand.size == 0:
        raise RuntimeError("No random control deltas found in reports.")
    # Lower-tail percentiles (predicted negative direction)
    p1, p5 = np.percentile(rand, [1, 5])

    ax1.hist(rand, bins=30, color="lightgray", edgecolor="white", label="Random L0 heads")
    ax1.axvline(p5, color="gray", linestyle=":", label="5th pct (lower tail)")
    ax1.axvline(p1, color="gray", linestyle="--", label="1st pct (lower tail)")

    for task, d in data.items():
        delta = float(d["deltas"]["output_entropy"])  # ablated - baseline
        ax1.axvline(delta, label=task)

    ax1.set_xlabel("Δ Output Entropy (ablated − baseline; lower = sharper)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Output Distribution Flattening under Suppressor Ablation")
    ax1.legend(loc="upper right", fontsize=8)

    # Panel B: Curvature reduction (early)
    tasks = list(data.keys())
    curv_early = [float(data[t]["deltas"]["curv_early"]) for t in tasks]
    x = np.arange(len(tasks))
    ax2.bar(x, curv_early, color=[mpl.rcParams["axes.prop_cycle"].by_key()["color"][i % 8] for i in range(len(tasks))])
    ax2.set_xticks(x)
    ax2.set_xticklabels(tasks, rotation=15)
    ax2.set_ylabel("Δ Curvature (early)\n(ablated − baseline; lower = straighter)")
    ax2.set_title("Trajectory Curvature Reduction (Layer‑0 Residuals)")
    ax2.axhline(0.0, color="black", linewidth=0.8)

    fig.tight_layout()
    out = OUT_DIR / "geometric_signature.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"Wrote {out}")
    return out


def main() -> None:
    data = load_outputs()
    fig_geometric_signature(data)


if __name__ == "__main__":
    main()

