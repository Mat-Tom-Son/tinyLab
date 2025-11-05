"""Binder-head sweep across middle layers on a synthetic binding dataset.

Goal
-----
Surface heads (esp. mid-layers) whose ablation disrupts adjective–noun binding.

What it does
------------
- Generates a synthetic dataset by combining adjective buckets (colors/sizes/
  materials) with noun buckets (vehicles/animals/furniture/objects), including
  plural variants and random ordering.
- For each example ("A red car and a blue truck … The red"), target is the bound
  noun (" car") and foil is the distractor noun (" truck").
- Sweeps heads in a layer window (default 8:12), computes Δ metrics per head.

Runtime and memory (rough order of magnitude)
---------------------------------------------
- GPT‑2 Medium (MPS, float16), 5k examples, layers 8–12 (≈64 heads): ~3–10 min
  depending on batch size (64–128). Memory peaks at ~1–2 GB on Apple Silicon.

Usage
-----
  python -m lab.analysis.binder_sweep \
    --model-name gpt2-medium \
    --device mps \
    --layer-start 8 --layer-end 12 \
    --max-examples 5000 --batch-size 64 \
    --output reports/binder_sweep_gpt2medium.json

Notes
-----
- We filter out examples where target/foil are not single tokens for the model.
- Reports baseline accuracy and Δ metrics per head (ΔLD, Δacc, Δp_drop, ΔKL).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from random import Random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from ..src.components import load_model, metrics as M


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", "--model", type=str, default="gpt2-medium")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])
    p.add_argument("--layer-range", type=str, default="8:12", help="Inclusive:exclusive layer range, e.g. 8:12")
    p.add_argument("--layer-start", type=int, default=None, help="Optional start layer (overrides --layer-range)")
    p.add_argument("--layer-end", type=int, default=None, help="Optional end layer (exclusive; overrides --layer-range)")
    p.add_argument("--scale", type=float, default=0.0, help="Zeroing scale (0.0 = full ablation)")
    p.add_argument("--max-examples", type=int, default=5000, help="Max synthetic examples after filtering")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size for forwards")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--output", "--out", type=Path, required=True)
    return p.parse_args()


def synthetic_binding_dataset(max_examples: int, seed: int) -> List[Dict[str, str]]:
    """Generate randomized adjective–noun binding prompts with distractors.

    Returns a superset (may exceed max_examples); callers should filter on
    single-token target/foil and truncate to the desired size.
    """
    rng = Random(seed)
    colors = [
        "red", "blue", "green", "black", "white", "yellow", "purple", "pink", "orange", "brown",
        "silver", "gold"
    ]
    sizes = ["big", "small", "tiny", "huge", "little", "massive"]
    materials = ["wooden", "metal", "plastic", "leather", "paper"]
    adjs = colors + sizes + materials

    nouns = [
        "car", "truck", "boat", "bike", "dog", "cat", "chair", "table", "sofa", "bed",
        "lamp", "hat", "coat", "bag", "shoe", "book", "cup"
    ]

    def pluralize(n: str) -> str:
        if n.endswith(("s", "x")) or n.endswith("ch") or n.endswith("sh"):
            return n + "es"
        if n.endswith("y") and (len(n) > 1 and n[-2] not in "aeiou"):
            return n[:-1] + "ies"
        return n + "s"

    exs: List[Dict[str, str]] = []
    attempts = 0
    max_attempts = max_examples * 10
    while len(exs) < max_examples and attempts < max_attempts:
        attempts += 1
        a1, a2 = rng.sample(adjs, 2)
        n1, n2 = rng.sample(nouns, 2)
        use_plural = rng.random() < 0.5
        n1p = pluralize(n1) if use_plural else n1
        n2p = pluralize(n2) if use_plural else n2

        order_first = rng.random() < 0.5
        if order_first:
            clean = f"A {a1} {n1p} and a {a2} {n2p} were nearby. The {a1}"
        else:
            clean = f"A {a2} {n2p} and a {a1} {n1p} were nearby. The {a1}"
        if rng.random() < 0.5:
            clean = clean.replace("were nearby", "were parked")
        target = f" {n1p}"
        foil = f" {n2p}"
        exs.append({"clean": clean, "target": target, "foil": foil})
    return exs


def head_zero_hook(layer: int, head: int, scale: float = 0.0):
    node = f"blocks.{layer}.attn.hook_z"

    def fn(z, hook):
        z = z.clone()
        z[:, :, head, :] = z[:, :, head, :] * scale
        return z

    return (node, fn)


@dataclass
class ModelSpec:
    name: str
    dtype: str


def to_model_cfg(spec: ModelSpec) -> Dict:
    return {"name": spec.name, "dtype": spec.dtype}


def compute_baseline(model, dset: List[Dict[str, str]], batch_size: int):
    total = 0
    sum_ld = 0.0
    sum_acc = 0.0
    baseline_logits_batches: List[torch.Tensor] = []
    t_ids_batches: List[torch.Tensor] = []

    for i in range(0, len(dset), batch_size):
        chunk = dset[i : i + batch_size]
        toks = model.to_tokens([ex["clean"] for ex in chunk])
        with torch.no_grad():
            clean_logits = model(toks)
        t_ids, f_ids = M._first_token_ids(model, chunk, {
            "dataset": {"target_field": "target", "foil_field": "foil"}
        })
        ld = M.logit_diff_first_token(clean_logits, t_ids, f_ids)
        preds = torch.softmax(clean_logits[:, -1, :], dim=-1).argmax(dim=-1)
        acc = (preds == t_ids).float().mean().item()
        n = clean_logits.size(0)
        total += n
        sum_ld += float(ld) * n
        sum_acc += float(acc) * n
        baseline_logits_batches.append(clean_logits)
        t_ids_batches.append(t_ids)

    ld_base = sum_ld / max(1, total)
    acc_base = sum_acc / max(1, total)
    return baseline_logits_batches, t_ids_batches, float(ld_base), float(acc_base)


def run() -> None:
    args = parse_args()
    if args.layer_start is not None and args.layer_end is not None:
        layer_lo, layer_hi = args.layer_start, args.layer_end
    else:
        layer_lo, layer_hi = [int(x) for x in args.layer_range.split(":", 1)]

    # Load model
    spec = ModelSpec(name=args.model_name, dtype=args.dtype)
    model = load_model.load_transformerlens(to_model_cfg(spec), device=args.device)
    model.eval()

    # Build dataset and drop non-single-token examples
    raw = synthetic_binding_dataset(args.max_examples * 2, args.seed)
    # Filter by single-token compatibility
    ok: List[Dict[str, str]] = []
    for ex in raw:
        try:
            tid = model.to_single_token(ex["target"])  # type: ignore[arg-type]
            fid = model.to_single_token(ex["foil"])    # type: ignore[arg-type]
        except Exception:
            tid, fid = None, None
        if tid is not None and fid is not None:
            ok.append(ex)

    if len(ok) < 128:
        raise RuntimeError("Too few single-token binding examples after filtering.")
    ok = ok[: args.max_examples]

    baseline_logits_batches, t_ids_batches, ld_base, acc_base = compute_baseline(model, ok, args.batch_size)

    results = {
        "model": args.model_name,
        "dtype": args.dtype,
        "device": args.device,
        "layer_range": [layer_lo, layer_hi],
        "scale": args.scale,
        "n_examples": len(ok),
        "baseline": {"logit_diff": ld_base, "acc": acc_base},
        "rows": [],
    }

    # Sweep heads
    n_heads = model.cfg.n_heads

    for layer in range(layer_lo, min(layer_hi, model.cfg.n_layers)):
        for head in range(n_heads):
            hook = head_zero_hook(layer, head, scale=args.scale)

            total = 0
            sum_ld = 0.0
            sum_pdrop = 0.0
            sum_kl = 0.0
            sum_afr = 0.0
            sum_acc = 0.0

            for bi, i0 in enumerate(range(0, len(ok), args.batch_size)):
                chunk = ok[i0 : i0 + args.batch_size]
                toks = model.to_tokens([ex["clean"] for ex in chunk])
                with torch.no_grad():
                    ablated_logits = model.run_with_hooks(toks, fwd_hooks=[hook])

                clean_logits = baseline_logits_batches[bi]
                t_ids = t_ids_batches[bi]
                summary, _ = M.evaluate_outputs(model, clean_logits, ablated_logits, chunk, {
                    "dataset": {"target_field": "target", "foil_field": "foil"}
                })
                n = ablated_logits.size(0)
                total += n
                sum_ld += float(summary.get("logit_diff", 0.0)) * n
                sum_pdrop += float(summary.get("p_drop", 0.0)) * n
                sum_kl += float(summary.get("kl_div", 0.0)) * n
                sum_afr += float(summary.get("acc_flip_rate", 0.0)) * n

                preds_a = torch.softmax(ablated_logits[:, -1, :], dim=-1).argmax(dim=-1)
                acc_a = (preds_a == t_ids).float().mean().item()
                sum_acc += float(acc_a) * n

            ld_abl = sum_ld / max(1, total)
            acc_abl = sum_acc / max(1, total)
            row = {
                "layer": int(layer),
                "head": int(head),
                "ld_abl": float(ld_abl),
                "ld_base": ld_base,
                "d_ld": float(ld_abl - ld_base),
                "acc_abl": float(acc_abl),
                "acc_base": float(acc_base),
                "d_acc": float(acc_abl - acc_base),
                "p_drop": float(sum_pdrop / max(1, total)),
                "kl_div": float(sum_kl / max(1, total)),
                "acc_flip_rate": float(sum_afr / max(1, total)),
            }
            results["rows"].append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    # Write a CSV for convenience
    try:
        import pandas as pd

        df = pd.DataFrame(results["rows"])  # type: ignore[arg-type]
        df.to_csv(args.output.with_suffix(".csv"), index=False)
    except Exception:
        pass

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    run()
