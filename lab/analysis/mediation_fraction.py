"""Compute mediation fractions from H6 activation-patch runs.

For each child run under a cross-condition H6 parent, normalizes the per-layer
patched logit-diff against the clean↔corrupt gap to estimate the fraction of the
effect mediated by that layer/window.

Usage:
  python -m lab.analysis.mediation_fraction \
    --parent lab/runs/h6_layer_targets_window_balanced_gpt2small_<hash>/manifest.json \
    --tag facts \
    --out reports/mediation_fraction_gpt2small_facts.json

Notes:
  - Assumes H6 batteries used patch_direction=corrupt->clean.
  - Reads per-layer LD from metrics/layer_impact.parquet (seed-tagged) and
    recomputes LD_clean and LD_corrupt once for the dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from ..src.components import datasets as D, load_model, metrics as M
from ..src.utils import io


def mean_ci(x: np.ndarray, alpha: float = 0.05):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan"), (float("nan"), float("nan"))
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    # Normal approx
    z = 1.959963984540054
    half = z * (s / max(np.sqrt(x.size), 1.0))
    return m, (m - half, m + half)


def compute_ld(model, dset, field: str, cfg: Dict) -> float:
    texts = [ex[cfg["dataset"][field]] for ex in dset]
    toks = model.to_tokens(texts)
    with torch.no_grad():
        logits = model(toks)
    t_ids, f_ids = M._first_token_ids(model, dset, cfg)
    return float(M.logit_diff_first_token(logits, t_ids, f_ids))


def load_child(child_manifest: Dict) -> Dict:
    run_dir = Path(child_manifest["run_dir"])
    cfg = io.load_json(run_dir / "config.json")
    dset, _, _ = D.load_split(cfg["dataset"])  # uses embedded dataset cfg
    device = cfg.get("device", "cuda")
    model = load_model.load_transformerlens(cfg["model"], device=device)
    model.eval()
    return {
        "run_dir": run_dir,
        "cfg": cfg,
        "model": model,
        "dset": dset,
    }


def mediation_for_child(child_manifest: Dict) -> Dict:
    loaded = load_child(child_manifest)
    run_dir: Path = loaded["run_dir"]
    cfg = loaded["cfg"]
    model = loaded["model"]
    dset = loaded["dset"]

    # Baselines
    ld_clean = compute_ld(model, dset, "clean_field", cfg)
    ld_corrupt = compute_ld(model, dset, "corrupt_field", cfg)
    denom = ld_clean - ld_corrupt

    # Per-layer patched LD (seed-tagged table)
    table = pd.read_parquet(run_dir / "metrics" / "layer_impact.parquet")
    table = table[table["metric"] == "logit_diff"].copy()

    out_rows = []
    for layer, g in table.groupby("layer"):
        # Each row's value is LD_patched for that seed/layer
        fracs = []
        for _, row in g.iterrows():
            ld_patched = float(row["value"])  # already mean over batch
            frac = (ld_patched - ld_corrupt) / denom if denom != 0 else float("nan")
            fracs.append(frac)
        m, (lo, hi) = mean_ci(np.array(fracs))
        out_rows.append({"layer": int(layer), "n": len(fracs), "mean": m, "ci95": [lo, hi]})

    out = {
        "run_dir": str(run_dir),
        "ld_clean": ld_clean,
        "ld_corrupt": ld_corrupt,
        "denom": denom,
        "layers": sorted(out_rows, key=lambda r: r["layer"]),
    }
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", type=Path, required=True, help="Path to H6 parent manifest.json")
    ap.add_argument("--tag", type=str, required=True, help="Condition tag to summarise (facts|neg|cf|logic)")
    ap.add_argument("--window", type=str, default="", help="Comma-separated window layers to aggregate (e.g., '8,9,10,11')")
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    parent = json.loads(Path(args.parent).read_text())
    children = {c["tag"]: c for c in parent["child_runs"]}
    if args.tag not in children:
        raise SystemExit(f"Tag {args.tag} not in parent manifest")
    res = mediation_for_child(children[args.tag])

    # Aggregate window if provided
    if args.window:
        want = {int(x.strip()) for x in args.window.split(",") if x.strip()}
        sel = [r for r in res["layers"] if r["layer"] in want]
        means = np.array([r["mean"] for r in sel], dtype=float)
        m, (lo, hi) = mean_ci(means)
        res["window"] = {"layers": sorted(want), "mean": m, "ci95": [lo, hi]}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

