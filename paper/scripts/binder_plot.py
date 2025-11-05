import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler


def load_rows(path: Path) -> List[Dict]:
    data = json.loads(path.read_text())
    return data.get("rows", [])


def topk_by_metric(rows: List[Dict], metric: str, k: int) -> List[Dict]:
    # Most negative first for deltas
    sorted_rows = sorted(rows, key=lambda r: float(r.get(metric, 0.0)))
    return sorted_rows[:k]


def fig_binder_bar(rows: List[Dict], title: str, out: Path, ylabel: str) -> None:
    # Styling similar to geometric_signature.py
    mpl.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "savefig.dpi": 200,
        "lines.linewidth": 1.8,
    })
    mpl.rcParams["axes.prop_cycle"] = cycler(color=["#0072B2"])  # blue

    labels = [f"L{int(r['layer'])}H{int(r['head'])}" for r in rows]
    vals = [float(r["d_acc"]) for r in rows]

    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = range(len(vals))
    ax.bar(x, vals)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0.0, color="black", linewidth=0.8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))


def write_markdown(rows: List[Dict], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_md.open("w") as f:
        f.write("| Layer | Head | d_acc | d_ld | acc_base | acc_abl | p_drop | kl_div |\n")
        f.write("| --- | --- | ---:| ---:| ---:| ---:| ---:| ---:|\n")
        for r in rows:
            f.write(
                f"| {int(r['layer'])} | {int(r['head'])} | {float(r['d_acc']):+.3f} | {float(r.get('d_ld', 0.0)):+.3f} | "
                f"{float(r.get('acc_base', 0.0)):.3f} | {float(r.get('acc_abl', 0.0)):.3f} | "
                f"{float(r.get('p_drop', float('nan'))):.3f} | {float(r.get('kl_div', float('nan'))):.3f} |\n"
            )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="binder_sweep JSON path")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--metric", type=str, default="d_ld", choices=["d_ld", "d_acc"], help="Ranking metric")
    ap.add_argument("--outdir", type=Path, default=Path("figs"))
    args = ap.parse_args()

    rows = load_rows(args.input)
    if not rows:
        raise SystemExit(f"No rows found in {args.input}")
    top = topk_by_metric(rows, args.metric, args.top_k)

    stem = args.input.stem.replace(".json", "")
    out_base = args.outdir / f"binder_{stem}_top{args.top_k}"
    ylab = "Δ Logit Diff (ablated − baseline)" if args.metric == "d_ld" else "Δ Accuracy (ablated − baseline)"
    fig_binder_bar(top, title=f"Binder Sweep: Top {args.top_k} by {args.metric} (lower is worse)", out=out_base, ylabel=ylab)
    write_markdown(top, out_base.with_suffix(".md"))
    print(f"Wrote {out_base}.pdf/.png and {out_base}.md")


if __name__ == "__main__":
    main()
