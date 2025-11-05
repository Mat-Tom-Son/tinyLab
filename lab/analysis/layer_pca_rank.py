"""Layer-wise PCA rank curve (variance coverage) and optional entropy.

Estimates intrinsic dimension (k PCs to explain var_frac) of layer residuals
at the last token across prompts. Plots layer index vs. k and saves CSV.

Usage:
  python -m lab.analysis.layer_pca_rank \
    --config lab/configs/run_h1_cross_condition_balanced.json \
    --tag facts \
    --model-name gpt2-medium \
    --device mps \
    --samples 128 \
    --var-frac 0.9 \
    --output reports/layer_pca_rank_gpt2medium_facts.json

Note: Uses only the clean prompts from the specified condition.

Runtime tip: GPT-2 Medium, 1k examples on MPS runs in ~1–3 minutes
with <1 GB memory; scale samples accordingly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from ..src.components import load_model, datasets
from ..src.utils import io


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True, help="Cross-condition or single-run config JSON")
    p.add_argument("--tag", type=str, required=True, help="Condition tag (for cross-condition configs)")
    p.add_argument("--model-name", type=str, default=None, help="Override model name (default from config)")
    p.add_argument("--device", type=str, default=None, help="Override device")
    p.add_argument("--samples", type=int, default=256, help="Max examples to sample")
    p.add_argument("--var-frac", type=float, default=0.90, help="Variance fraction for PCA rank")
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def build_child_cfg(cfg_path: Path, tag: str) -> Dict:
    cfg = io.load_json(cfg_path)
    if "shared" in cfg and "conditions" in cfg:
        shared = cfg["shared"].copy()
        for cond in cfg["conditions"]:
            if cond.get("tag") == tag:
                ds = shared["dataset"].copy()
                ds.update(cond.get("dataset", {}))
                shared["dataset"] = ds
                return shared
        raise ValueError(f"Tag '{tag}' not found in {cfg_path}")
    else:
        # Single-run style
        return {
            "device": cfg.get("device", "auto"),
            "model": cfg["model"],
            "dataset": cfg["dataset"],
            "seed": cfg.get("seed", 0),
        }


def svd_rank_for_fraction(x: np.ndarray, var_frac: float = 0.9) -> int:
    """Return PCs needed to cover var_frac variance (economical SVD on samples).

    x is [N, D]. We compute SVD on centered data and convert singular values to
    covariance eigenvalues: lambda = s^2 / (N-1).
    """
    if x.ndim != 2 or x.shape[0] < 2:
        return 0
    xc = x - x.mean(axis=0, keepdims=True)
    try:
        U, s, Vt = np.linalg.svd(xc, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0
    n = x.shape[0]
    eigvals = (s ** 2) / max(n - 1, 1)
    total = float(np.sum(eigvals))
    if total <= 0:
        return 0
    csum = np.cumsum(eigvals)
    k = int(np.searchsorted(csum, var_frac * total) + 1)
    return max(1, k)


def main() -> None:
    args = parse_args()
    child = build_child_cfg(args.config, args.tag)
    if args.model_name:
        child["model"]["name"] = args.model_name
    if args.device:
        child["device"] = args.device

    rows, split_info, data_hash = datasets.load_split(child["dataset"])
    if args.samples and len(rows) > args.samples:
        rows = rows[: args.samples]

    clean_texts = [ex[child["dataset"]["clean_field"]] for ex in rows]

    model = load_model.load_transformerlens(child["model"], device=child.get("device", "auto"))
    model.eval()
    toks = model.to_tokens(clean_texts)

    with torch.no_grad():
        logits, cache = model.run_with_cache(toks)

    n_layers = model.cfg.n_layers
    layer_ranks: List[Tuple[int, int]] = []

    for layer in range(n_layers):
        resid = cache[f"blocks.{layer}.hook_resid_post"]  # [B, S, D]
        mat = resid[:, -1, :].to(torch.float32).cpu().numpy()
        k = svd_rank_for_fraction(mat, var_frac=args.var_frac)
        layer_ranks.append((layer, k))

    out = {
        "config": str(args.config),
        "tag": args.tag,
        "model": child["model"],
        "device": child.get("device", "auto"),
        "data_hash": data_hash,
        "n_examples": len(rows),
        "var_frac": args.var_frac,
        "layer_ranks": [{"layer": l, "k": k} for (l, k) in layer_ranks],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))

    # Save CSV and a quick plot
    try:
        import pandas as pd
        import matplotlib.pyplot as plt

        df = pd.DataFrame(out["layer_ranks"])  # type: ignore[index]
        csv_path = args.output.with_suffix(".csv")
        df.to_csv(csv_path, index=False)

        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        ax.plot(df["layer"], df["k"], marker="o")
        ax.set_xlabel("Layer")
        ax.set_ylabel(f"k PCs @ {args.var_frac:.2f} var frac")
        ax.set_title("Layer-wise PCA Rank (last-position residuals)")
        fig.tight_layout()
        fig_path = args.output.with_suffix(".pdf")
        fig.savefig(fig_path)
        fig.savefig(fig_path.with_suffix(".png"))
    except Exception:
        pass

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
