"""Natural-corpus binder probe using corpus-derived adjective/noun pools.

Approach: scan a lightweight corpus (wikitext-2-raw) for occurrences of
adjective–noun pairs from seed buckets (colors/sizes/materials × nouns).
Harvest observed adjectives and nouns, then synthesize binding prompts using
those pools (two compositions per example) as in binder_sweep, to keep the
causal structure while drawing from natural tokens.

Usage:
  python -m lab.analysis.binder_corpus_probe \
    --model-name gpt2-medium --device mps \
    --layer-start 8 --layer-end 12 \
    --max-examples 5000 --batch-size 64 \
    --output reports/binder_sweep_corpus_gpt2medium.json

Runtime tip: ~5–10 minutes for 5k examples on MPS with float16.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import datasets as hfds
import torch

from ..src.components import load_model, metrics as M
from .binder_sweep import head_zero_hook


SEED_ADJS = {
    "colors": [
        "red", "blue", "green", "black", "white", "yellow", "purple", "pink", "orange", "brown",
        "silver", "gold"
    ],
    "sizes": ["big", "small", "tiny", "huge", "little", "massive"],
    "materials": ["wooden", "metal", "plastic", "leather", "paper"],
}

SEED_NOUNS = [
    "car", "truck", "boat", "bike", "dog", "cat", "chair", "table", "sofa", "bed",
    "lamp", "hat", "coat", "bag", "shoe", "book", "cup", "house", "tree", "train"
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", type=str, default="gpt2-medium")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="float16")
    p.add_argument("--layer-start", type=int, default=8)
    p.add_argument("--layer-end", type=int, default=12)
    p.add_argument("--max-examples", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--noun-classes",
        type=str,
        default="",
        help="Comma-separated subset of noun classes (vehicles,furniture)",
    )
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def pluralize(n: str) -> str:
    if n.endswith(("s", "x")) or n.endswith("ch") or n.endswith("sh"):
        return n + "es"
    if n.endswith("y") and (len(n) > 1 and n[-2] not in "aeiou"):
        return n[:-1] + "ies"
    return n + "s"


def harvest_pools() -> tuple[list[str], list[str]]:
    ds = hfds.load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = "\n".join(ds["text"]).lower()
    base_adjs = set(sum(SEED_ADJS.values(), []))
    base_nouns = set(SEED_NOUNS)
    found_adjs, found_nouns = set(), set()
    words = texts.split()
    for i in range(len(words) - 1):
        a, n = words[i], words[i + 1]
        if a in base_adjs:
            found_adjs.add(a)
            if n.isalpha():
                found_nouns.add(n)
        if n in base_nouns and a.isalpha():
            found_nouns.add(n)
            if a.isalpha():
                found_adjs.add(a)
    # Keep a modest number to avoid OOV drift
    # Restrict nouns to our seed classes for more natural compositions
    nouns = list(found_nouns & base_nouns) or list(base_nouns)
    adjs = list(found_adjs | base_adjs)
    return adjs, nouns


def build_dataset(adjs: list[str], nouns: list[str], model, max_examples: int) -> List[Dict[str, str]]:
    import random
    rng = random.Random(13)
    exs: List[Dict[str, str]] = []
    attempts, max_attempts = 0, max_examples * 10
    while len(exs) < max_examples and attempts < max_attempts:
        attempts += 1
        if len(adjs) < 2 or len(nouns) < 2:
            break
        a1, a2 = rng.sample(adjs, 2)
        n1, n2 = rng.sample(nouns, 2)
        use_plural = rng.random() < 0.3
        n1p = pluralize(n1) if use_plural else n1
        n2p = pluralize(n2) if use_plural else n2
        # Compose
        if rng.random() < 0.5:
            clean = f"A {a1} {n1p} and a {a2} {n2p} were nearby. The {a1}"
        else:
            clean = f"A {a2} {n2p} and a {a1} {n1p} were parked. The {a1}"
        target = f" {n1p}"
        foil = f" {n2p}"
        # Filter for single-token target/foil
        try:
            if model.to_single_token(target) is None or model.to_single_token(foil) is None:
                continue
        except AssertionError:
            continue
        exs.append({"clean": clean, "target": target, "foil": foil})
    return exs


def run() -> None:
    args = parse_args()
    model = load_model.load_transformerlens({"name": args.model_name, "dtype": args.dtype}, device=args.device)
    model.eval()

    adjs, nouns = harvest_pools()
    if args.noun_classes:
        classes = {c.strip().lower() for c in args.noun_classes.split(",") if c.strip()}
        vehicles = {"car", "truck", "boat", "bike", "train"}
        furniture = {"chair", "table", "sofa", "bed", "lamp", "couch"}
        class_map = {"vehicles": vehicles, "furniture": furniture}
        allowed = set()
        for c in classes:
            allowed |= class_map.get(c, set())
        if allowed:
            nouns = [n for n in nouns if n in allowed]
    dset = build_dataset(adjs, nouns, model, args.max_examples)
    if len(dset) < 512:
        raise SystemExit("Too few corpus-derived examples; refine pools or increase corpus")

    # Baseline (batched to avoid large MPS graphs)
    total = 0
    sum_ld = 0.0
    sum_acc = 0.0
    toks_batches: list[torch.Tensor] = []
    t_ids_batches: list[torch.Tensor] = []
    for i in range(0, len(dset), args.batch_size):
        chunk = dset[i : i + args.batch_size]
        toks = model.to_tokens([ex["clean"] for ex in chunk])
        with torch.no_grad():
            logits = model(toks)
        t_ids, f_ids = M._first_token_ids(model, chunk, {"dataset": {"target_field": "target", "foil_field": "foil"}})
        ld = M.logit_diff_first_token(logits, t_ids, f_ids)
        acc = (torch.softmax(logits[:, -1, :], dim=-1).argmax(dim=-1) == t_ids).float().mean().item()
        n = logits.size(0)
        total += n
        sum_ld += float(ld) * n
        sum_acc += float(acc) * n
        toks_batches.append(toks)
        t_ids_batches.append(t_ids)
    base_ld = sum_ld / max(1, total)
    base_acc = sum_acc / max(1, total)

    results = {
        "model": args.model_name,
        "dtype": args.dtype,
        "device": args.device,
        "layer_range": [args.layer_start, args.layer_end],
        "n_examples": len(dset),
        "baseline": {"logit_diff": float(base_ld), "acc": float(base_acc)},
        "rows": [],
    }

    # Sweep heads (batched)
    n_heads = model.cfg.n_heads
    for layer in range(args.layer_start, min(args.layer_end, model.cfg.n_layers)):
        for head in range(n_heads):
            node, fn = head_zero_hook(layer, head)
            total = 0
            sum_ld_a = 0.0
            sum_acc_a = 0.0
            for bi, i0 in enumerate(range(0, len(dset), args.batch_size)):
                toks = toks_batches[bi]
                t_ids = t_ids_batches[bi]
                with torch.no_grad():
                    logits_a = model.run_with_hooks(toks, fwd_hooks=[(node, fn)])
                # Recompute f_ids for this chunk (same chunk boundaries as baseline)
                chunk = dset[i0 : i0 + args.batch_size]
                _, f_ids = M._first_token_ids(model, chunk, {"dataset": {"target_field": "target", "foil_field": "foil"}})
                ld_a = M.logit_diff_first_token(logits_a, t_ids, f_ids)
                acc_a = (torch.softmax(logits_a[:, -1, :], dim=-1).argmax(dim=-1) == t_ids).float().mean().item()
                n = logits_a.size(0)
                total += n
                sum_ld_a += float(ld_a) * n
                sum_acc_a += float(acc_a) * n
            ld_a = sum_ld_a / max(1, total)
            acc_a = sum_acc_a / max(1, total)
            results["rows"].append({
                "layer": layer,
                "head": head,
                "ld_abl": float(ld_a),
                "ld_base": float(base_ld),
                "d_ld": float(ld_a - base_ld),
                "acc_abl": float(acc_a),
                "acc_base": float(base_acc),
                "d_acc": float(acc_a - base_acc),
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    try:
        import pandas as pd
        pd.DataFrame(results["rows"]).to_csv(args.output.with_suffix(".csv"), index=False)
    except Exception:
        pass
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    run()
