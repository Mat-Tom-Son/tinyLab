"""Compute OV (output) contributions of specified attention heads to target/foil logits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from ..src.components import datasets, load_model
from ..src.utils import determinism, io


def parse_head(arg: str) -> Tuple[int, int]:
    try:
        layer_str, head_str = arg.split(":")
        return int(layer_str), int(head_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid head spec '{arg}'. Use L:H.") from exc


def build_child_cfg(config_path: Path, tag: str) -> Dict:
    cfg = io.load_json(config_path)
    shared = cfg["shared"]
    dataset_cfg = shared["dataset"].copy()
    for cond in cfg["conditions"]:
        if cond["tag"] == tag:
            dataset_cfg.update(cond["dataset"])
            break
    else:
        raise ValueError(f"Tag '{tag}' not found in config {config_path}")
    return {
        "model": shared["model"],
        "device": shared.get("device", "auto"),
        "dataset": dataset_cfg,
        "seed": shared.get("seeds", [0])[0],
    }


def load_batch(child_cfg: Dict, samples: int):
    rows, split, data_hash = datasets.load_split(child_cfg["dataset"])
    rows = rows[:samples]
    return rows, split, data_hash


def head_contribution(
    model,
    cache,
    layer: int,
    head: int,
    target_ids: torch.Tensor,
    foil_ids: torch.Tensor,
) -> Dict[str, float]:
    # hook_z gives attention head outputs (batch, seq, heads, d_head)
    z = cache["z", layer][:, :, head, :]  # [batch, seq, d_head]
    seq_len = z.shape[1]
    final_pos = seq_len - 1
    z_final = z[:, final_pos, :]  # [batch, d_head]

    W_O = model.W_O[layer, head]  # [d_head, d_model]
    W_U = model.W_U  # [d_model, vocab]

    resid = torch.matmul(z_final, W_O)  # [batch, d_model]
    logits = torch.matmul(resid, W_U)  # [batch, vocab]

    batch_indices = torch.arange(logits.size(0), device=logits.device)
    target_contrib = logits[batch_indices, target_ids]
    foil_contrib = logits[batch_indices, foil_ids]

    delta = target_contrib - foil_contrib

    return {
        "mean_target": float(target_contrib.mean().item()),
        "mean_foil": float(foil_contrib.mean().item()),
        "mean_delta": float(delta.mean().item()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--heads", nargs="+", type=parse_head, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    child_cfg = build_child_cfg(args.config, args.tag)
    determinism.set_seed(args.seed)
    rows, split, data_hash = load_batch(child_cfg, args.samples)

    device = child_cfg["device"]
    model = load_model.load_transformerlens(child_cfg["model"], device=device)
    model.eval()

    texts = [ex[child_cfg["dataset"]["clean_field"]] for ex in rows]
    tokens = model.to_tokens(texts)

    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)

    target_ids = []
    foil_ids = []
    for ex in rows:
        t_id = model.to_single_token(ex[child_cfg["dataset"]["target_field"]])
        f_id = model.to_single_token(ex[child_cfg["dataset"]["foil_field"]])
        if t_id is None or f_id is None:
            raise ValueError("All examples must have single-token target/foil for OV analysis.")
        target_ids.append(t_id)
        foil_ids.append(f_id)

    target_ids_t = torch.tensor(target_ids, device=logits.device)
    foil_ids_t = torch.tensor(foil_ids, device=logits.device)

    results = []
    for layer, head in args.heads:
        contrib = head_contribution(model, cache, layer, head, target_ids_t, foil_ids_t)
        contrib.update({"layer": layer, "head": head})
        results.append(contrib)

    summary = {
        "config": str(args.config),
        "tag": args.tag,
        "samples": len(rows),
        "data_hash": data_hash,
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"Wrote OV contributions to {args.output}")


if __name__ == "__main__":
    main()
