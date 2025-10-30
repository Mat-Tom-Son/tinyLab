"""Ablate multiple heads then patch one of them back in to measure its causal role."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..src.components import datasets, load_model, metrics
from ..src.utils import determinism, io


def build_cfg(config_path: Path, tag: str) -> dict:
    cfg = io.load_json(config_path)
    shared = cfg["shared"]
    for cond in cfg["conditions"]:
        if cond["tag"] == tag:
            dataset_cfg = {**shared["dataset"], **cond["dataset"]}
            break
    else:
        raise ValueError(f"Condition tag '{tag}' not found.")
    return {
        "model": shared["model"],
        "device": shared.get("device", "auto"),
        "dataset": dataset_cfg,
        "seed": shared.get("seeds", [0])[0],
        "metrics": cfg["shared"]["metrics"],
        "batch_size": shared["batch_size"],
    }


def zero_heads(node_layer: int, heads):
    def hook(z, hook):
        z = z.clone()
        for head in heads:
            z[:, :, head, :] = 0.0
        return z

    return hook


def patch_head(model, cache, layer: int, head: int):
    node = f"blocks.{layer}.attn.hook_z"

    def hook(z, hook):
        z = z.clone()
        z[:, :, head, :] = cache[node][:, :, head, :]
        return z

    return hook


def run(config: Path, tag: str, head: int, layer: int, heads_to_zero, samples: int, seed: int):
    cfg = build_cfg(config, tag)
    determinism.set_seed(seed)
    rows, _, _ = datasets.load_split(cfg["dataset"])
    rows = rows[:samples]

    device = cfg["device"]
    model = load_model.load_transformerlens(cfg["model"], device=device)
    model.eval()

    clean_texts = [ex[cfg["dataset"]["clean_field"]] for ex in rows]
    tokens = model.to_tokens(clean_texts)

    with torch.no_grad():
        logits_clean, cache = model.run_with_cache(tokens)

    baseline_summary, _ = metrics.evaluate_outputs(model, logits_clean, logits_clean, rows, {"dataset": cfg["dataset"], "seed": seed})

    zero_hook = zero_heads(layer, heads_to_zero)
    node = f"blocks.{layer}.attn.hook_z"
    with torch.no_grad():
        logits_zero = model.run_with_hooks(tokens, fwd_hooks=[(node, zero_hook)])
    zero_summary, _ = metrics.evaluate_outputs(model, logits_clean, logits_zero, rows, {"dataset": cfg["dataset"], "seed": seed})

    patch_hook = patch_head(model, cache, layer, head)
    with torch.no_grad():
        logits_half = model.run_with_hooks(
            tokens, fwd_hooks=[(node, zero_hook), (node, patch_hook)]
        )
    patch_summary, _ = metrics.evaluate_outputs(model, logits_clean, logits_half, rows, {"dataset": cfg["dataset"], "seed": seed})

    return {
        "baseline": baseline_summary,
        "zero": zero_summary,
        "patched": patch_summary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--zero-heads", nargs="+", type=int, required=True)
    parser.add_argument("--patch-head", type=int, required=True)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = run(
        config=args.config,
        tag=args.tag,
        head=args.patch_head,
        layer=args.layer,
        heads_to_zero=args.zero_heads,
        samples=args.samples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

