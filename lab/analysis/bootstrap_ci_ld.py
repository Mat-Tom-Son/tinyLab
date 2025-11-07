"""Bootstrap CI for ΔLD under a specified head subset ablation.

Computes per-example logit-diff on clean prompts and under a heads_subset_zero
ablation, then bootstraps the mean ΔLD with replacement.

Usage:
  python -m lab.analysis.bootstrap_ci_ld \
    --config lab/configs/run_h1_cross_condition_balanced_gpt2small.json \
    --tag neg \
    --model-name gpt2-small \
    --device cuda \
    --layer 0 --heads 2 4 7 \
    --boot 5000 \
    --out reports/bootstrap_ci_gpt2small_neg_triplet.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch

from ..src.components import datasets as D, load_model
from ..src.components import metrics as M
from ..src.utils import io


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--model-name", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--heads", type=int, nargs="+", required=True)
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args()


def per_example_ld(logits: torch.Tensor, t_ids: torch.Tensor, f_ids: torch.Tensor) -> np.ndarray:
    last = logits[:, -1, :]
    t = last[torch.arange(last.size(0)), t_ids]
    f = last[torch.arange(last.size(0)), f_ids]
    return (t - f).detach().cpu().numpy()


def main():
    args = parse_args()
    cfg = io.load_json(args.config)
    shared = cfg["shared"]

    # Select tag dataset
    dcfg = shared["dataset"].copy()
    for cond in cfg["conditions"]:
        if cond.get("tag") == args.tag:
            dcfg.update(cond.get("dataset", {}))
            break
    else:
        raise SystemExit(f"Tag {args.tag} not found in config")

    dset, _, _ = D.load_split(dcfg)

    model = load_model.load_transformerlens({"name": args.model_name, "dtype": shared["model"]["dtype"]}, device=args.device)
    model.eval()

    clean_texts = [ex[dcfg["clean_field"]] for ex in dset]
    toks = model.to_tokens(clean_texts)
    with torch.no_grad():
        logits_clean = model(toks)

    # Ablated (subset-zero) forward
    node = f"blocks.{args.layer}.attn.hook_z"
    def zero_subset(z, hook):
        z = z.clone()
        for h in args.heads:
            z[:, :, h, :] = 0.0
        return z
    with torch.no_grad():
        logits_abl = model.run_with_hooks(toks, fwd_hooks=[(node, zero_subset)])

    # Per-example LDs and diffs
    t_ids, f_ids = M._first_token_ids(model, dset, {"dataset": dcfg})
    ld_c = per_example_ld(logits_clean, t_ids, f_ids)
    ld_a = per_example_ld(logits_abl, t_ids, f_ids)
    diffs = ld_a - ld_c

    # Bootstrap
    rng = np.random.default_rng(123)
    n = diffs.shape[0]
    means = []
    for _ in range(args.boot):
        idx = rng.integers(0, n, size=n)
        means.append(float(np.mean(diffs[idx])))
    means = np.array(means, dtype=float)
    lo, hi = np.percentile(means, [2.5, 97.5])
    out = {
        "tag": args.tag,
        "layer": args.layer,
        "heads": args.heads,
        "n_examples": n,
        "boot": args.boot,
        "mean_delta_ld": float(np.mean(diffs)),
        "ci95": [float(lo), float(hi)]
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

