"""Generate detailed OV projection reports for specified attention heads.

Outputs top/bottom-k tokens with logit contributions, plus mean stats per head.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from ..src.components import datasets, load_model
from ..src.utils import determinism, io


def parse_head(arg: str) -> Tuple[int, int]:
    layer_str, head_str = arg.split(":")
    return int(layer_str), int(head_str)


def build_child_cfg(config_path: Path, tag: str) -> Dict:
    cfg = io.load_json(config_path)
    shared = cfg["shared"]
    dataset_cfg = shared["dataset"].copy()
    for cond in cfg["conditions"]:
        if cond["tag"] == tag:
            dataset_cfg.update(cond["dataset"])
            break
    else:
        raise ValueError(f"Tag '{tag}' not found in {config_path}")
    return {
        "model": shared["model"],
        "device": shared.get("device", "auto"),
        "dataset": dataset_cfg,
        "seed": shared.get("seeds", [0])[0],
    }


def load_rows(child_cfg: Dict, samples: int):
    rows, split, data_hash = datasets.load_split(child_cfg["dataset"])
    rows = rows[:samples]
    return rows, data_hash


def head_vector(model, cache, layer: int, head: int) -> torch.Tensor:
    z = cache["z", layer][:, :, head, :]
    last = z[:, z.shape[1] - 1, :]
    resid = last @ model.W_O[layer, head]
    return resid.mean(dim=0)


def project_tokens(model, vec: torch.Tensor, top_k: int) -> Dict[str, List[Dict]]:
    logits = vec @ model.W_U  # [vocab]
    values_top, idx_top = torch.topk(logits, k=top_k)
    values_bot, idx_bot = torch.topk(-logits, k=top_k)
    tokenizer = model.tokenizer
    top = [
        {"token_id": int(idx), "token": tokenizer.decode(idx), "logit": float(val)}
        for val, idx in zip(values_top.tolist(), idx_top.tolist())
    ]
    bottom = [
        {"token_id": int(idx), "token": tokenizer.decode(idx), "logit": float(-val)}
        for val, idx in zip(values_bot.tolist(), idx_bot.tolist())
    ]
    return {"top": top, "bottom": bottom}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--heads", nargs="+", type=parse_head, required=True)
    parser.add_argument("--samples", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    child_cfg = build_child_cfg(args.config, args.tag)
    determinism.set_seed(args.seed)
    rows, data_hash = load_rows(child_cfg, args.samples)

    device = child_cfg["device"]
    model = load_model.load_transformerlens(child_cfg["model"], device=device)
    model.eval()

    texts = [ex[child_cfg["dataset"]["clean_field"]] for ex in rows]
    tokens = model.to_tokens(texts)

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    report = {
        "config": str(args.config),
        "tag": args.tag,
        "samples": len(rows),
        "data_hash": data_hash,
        "top_k": args.top_k,
        "heads": [],
    }

    for layer, head in args.heads:
        vec = head_vector(model, cache, layer, head)
        proj = project_tokens(model, vec, args.top_k)
        norm = float(vec.norm().item())
        report["heads"].append(
            {
                "layer": layer,
                "head": head,
                "vector_norm": norm,
                "tokens": proj,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"Wrote OV report to {args.output}")


if __name__ == "__main__":
    main()

