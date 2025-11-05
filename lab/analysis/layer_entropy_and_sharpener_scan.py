"""Layer-wise output-entropy profile + late-layer sharpener head scan.

Two parts:
  1) Baseline entropy after each layer (implied logits from ln_final+W_U).
  2) Scan heads in the last K layers; for each head, ablate and report
     Δ entropy at the final position (ablated − baseline). Heads that increase
     entropy on ablation are candidates for "sharpeners".

Usage:
  python -m lab.analysis.layer_entropy_and_sharpener_scan \
    --config lab/configs/run_h1_cross_condition_balanced.json \
    --tag facts \
    --model-name gpt2-medium \
    --device mps \
    --samples 128 \
    --last-k 3 \
    --output reports/layer_entropy_scan_gpt2medium_facts.json
Runtime tip: GPT‑2 Medium with 512 examples and last‑k=3 runs in ~1–3 minutes
on MPS with <1 GB memory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from ..src.components import load_model, datasets
from ..src.utils import io


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--model-name", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--last-k", type=int, default=3, help="Scan last K layers for sharpeners")
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
        return {
            "device": cfg.get("device", "auto"),
            "model": cfg["model"],
            "dataset": cfg["dataset"],
            "seed": cfg.get("seed", 0),
        }


def softmax_entropy_last(logits: torch.Tensor) -> float:
    """Shannon entropy at last position (if 3D) or over vectors (if 2D).

    Accepts shapes:
      - [B, S, V]: uses last position S-1
      - [B, V]: uses the given logits directly
    """
    if logits.dim() == 3:
        last = logits[:, -1, :].to(torch.float32)
    elif logits.dim() == 2:
        last = logits.to(torch.float32)
    else:
        raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")
    p = torch.softmax(last, dim=-1)
    p = p.clamp_min(1e-30)
    h = -(p * p.log()).sum(dim=-1).mean().item()
    return float(h)


def get_unembed_params(model):
    # Try common TransformerLens param locations
    W_U = getattr(model, "W_U", None)
    b_U = getattr(model, "b_U", None)
    if W_U is None or b_U is None:
        unembed = getattr(model, "unembed", None)
        if unembed is not None:
            W_U = getattr(unembed, "W_U", None)
            b_U = getattr(unembed, "b_U", None)
    if W_U is None or b_U is None:
        raise RuntimeError("Could not locate unembed parameters (W_U, b_U)")
    return W_U, b_U


def logits_from_resid(model, resid_last: torch.Tensor) -> torch.Tensor:
    # Apply ln_final then unembed
    W_U, b_U = get_unembed_params(model)
    x = model.ln_final(resid_last)
    # x: [B, D], W_U: [D, V]
    logits = x @ W_U + b_U
    return logits


def zero_head_hook(layer: int, head: int):
    node = f"blocks.{layer}.attn.hook_z"

    def fn(z, hook):
        z = z.clone()
        z[:, :, head, :] = 0.0
        return z

    return (node, fn)


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

    # Baseline cache once
    with torch.no_grad():
        logits_final, cache = model.run_with_cache(toks)

    n_layers = model.cfg.n_layers
    # Panel 1: per-layer entropy profile (implied logits)
    layer_entropy = []
    for layer in range(n_layers):
        resid = cache[f"blocks.{layer}.hook_resid_post"]  # [B, S, D]
        resid_last = resid[:, -1, :]
        with torch.no_grad():
            logits_L = logits_from_resid(model, resid_last)
        H = softmax_entropy_last(logits_L)
        layer_entropy.append({"layer": layer, "entropy": H})

    # Baseline final entropy
    H_final_base = softmax_entropy_last(logits_final)

    # Panel 2: last-K sharpener scan
    last_k = max(1, min(args.last_k, n_layers))
    layers_to_scan = list(range(n_layers - last_k, n_layers))
    head_rows: List[Dict] = []
    for L in layers_to_scan:
        for h in range(model.cfg.n_heads):
            hook = zero_head_hook(L, h)
            with torch.no_grad():
                logits_a = model.run_with_hooks(toks, fwd_hooks=[hook])
            H_a = softmax_entropy_last(logits_a)
            head_rows.append({
                "layer": int(L),
                "head": int(h),
                "entropy_final_abl": float(H_a),
                "entropy_final_base": float(H_final_base),
                "d_entropy_final": float(H_a - H_final_base),
            })

    out = {
        "config": str(args.config),
        "tag": args.tag,
        "model": child["model"],
        "device": child.get("device", "auto"),
        "n_examples": len(rows),
        "data_hash": data_hash,
        "layer_entropy_profile": layer_entropy,
        "sharpener_scan": head_rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))

    # Save convenience CSVs and a profile plot
    try:
        import pandas as pd
        import matplotlib.pyplot as plt

        dfL = pd.DataFrame(layer_entropy)
        dfL.to_csv(args.output.with_suffix(".profile.csv"), index=False)

        dfH = pd.DataFrame(head_rows)
        dfH.to_csv(args.output.with_suffix(".heads.csv"), index=False)

        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        ax.plot(dfL["layer"], dfL["entropy"], marker="o")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Output entropy (implied)")
        ax.set_title("Layer-wise Output Entropy Profile (last position)")
        fig.tight_layout()
        fig.savefig(args.output.with_suffix(".pdf"))
        fig.savefig(args.output.with_suffix(".png"))
    except Exception:
        pass

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
