"""Inspect per-head attention behavior before and after ablations.

Example:
    python -m lab.analysis.inspect_heads \\
        --config lab/configs/run_h1_cross_condition_balanced.json \\
        --tag facts \\
        --heads 0:2 0:4 0:7 \\
        --samples 5 \\
        --output reports/inspect_heads_facts.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from ..src.components import datasets, load_model, metrics
from ..src.utils import io


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

    child_cfg = {
        "run_name": f"inspect_heads_{tag}",
        "device": shared.get("device", "auto"),
        "model": shared["model"],
        "dataset": dataset_cfg,
        "seed": shared.get("seeds", [0])[0],
    }
    return child_cfg


def load_subset(child_cfg: Dict, sample_count: int):
    rows, split, data_hash = datasets.load_split(child_cfg["dataset"])
    if sample_count < len(rows):
        rows = rows[:sample_count]
    return rows, split, data_hash


def collect_attention(
    model,
    tokens: torch.Tensor,
    str_tokens: List[List[str]],
    cache,
    layer: int,
    head: int,
) -> List[Dict]:
    pattern = cache["pattern", layer]  # [batch, head, q_pos, k_pos]
    results = []
    for i in range(tokens.shape[0]):
        seq_len = len(str_tokens[i])
        q_pos = seq_len - 1
        attn = pattern[i, head, q_pos, :seq_len].detach()
        top_vals, top_idx = torch.topk(attn, k=min(3, seq_len))
        decoded = [
            {
                "position": int(idx),
                "token": str_tokens[i][idx],
                "weight": float(val),
            }
            for idx, val in zip(top_idx.tolist(), top_vals.tolist())
        ]
        results.append(
            {
                "sequence_index": i,
                "query_position": q_pos,
                "top_attention": decoded,
            }
        )
    return results


def run_analysis(config: Path, tag: str, heads: List[Tuple[int, int]], samples: int, output: Path):
    child_cfg = build_child_cfg(config, tag)
    rows, split, data_hash = load_subset(child_cfg, samples)

    device = child_cfg["device"]
    model = load_model.load_transformerlens(child_cfg["model"], device=device)
    model.eval()

    texts = [ex[child_cfg["dataset"]["clean_field"]] for ex in rows]
    tokens = model.to_tokens(texts)
    str_tokens = [model.to_str_tokens(text) for text in texts]

    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)

    baseline_summary, baseline_per = metrics.evaluate_outputs(
        model, logits, logits, rows, child_cfg
    )

    target_field = child_cfg["dataset"]["target_field"]
    foil_field = child_cfg["dataset"]["foil_field"]
    target_ids = []
    foil_ids = []
    for ex in rows:
        t_id = model.to_single_token(ex[target_field])
        f_id = model.to_single_token(ex[foil_field])
        if t_id is None or f_id is None:
            raise ValueError("inspect_heads requires single-token target/foil entries.")
        target_ids.append(t_id)
        foil_ids.append(f_id)
    target_ids_t = torch.tensor(target_ids, device=logits.device)
    foil_ids_t = torch.tensor(foil_ids, device=logits.device)

    head_reports = []

    for layer, head in heads:
        attn_info = collect_attention(model, tokens, str_tokens, cache, layer, head)

        node = f"blocks.{layer}.attn.hook_z"

        def zero_fn(z, hook, h=head):
            z = z.clone()
            z[:, :, h, :] = 0.0
            return z

        with torch.no_grad():
            logits_zero = model.run_with_hooks(tokens, fwd_hooks=[(node, zero_fn)])

        zero_summary, zero_per = metrics.evaluate_outputs(
            model, logits, logits_zero, rows, child_cfg
        )

        delta_logits = logits_zero[:, -1, :] - logits[:, -1, :]
        target_delta = delta_logits[torch.arange(delta_logits.size(0)), target_ids_t]
        foil_delta = delta_logits[torch.arange(delta_logits.size(0)), foil_ids_t]

        per_example_deltas = []
        for i in range(delta_logits.size(0)):
            per_example_deltas.append(
                {
                    "index": i,
                    "target_id": int(target_ids[i]),
                    "foil_id": int(foil_ids[i]),
                    "target_delta": float(target_delta[i]),
                    "foil_delta": float(foil_delta[i]),
                }
            )

        head_reports.append(
            {
                "layer": layer,
                "head": head,
                "baseline_logit_diff": float(baseline_summary["logit_diff"]),
                "ablated_logit_diff": float(zero_summary["logit_diff"]),
                "delta": float(zero_summary["logit_diff"] - baseline_summary["logit_diff"]),
                "mean_target_delta": float(target_delta.mean().item()),
                "mean_foil_delta": float(foil_delta.mean().item()),
                "attention": attn_info,
                "logit_deltas": per_example_deltas,
            }
        )

    report = {
        "config": str(config),
        "tag": tag,
        "dataset_id": child_cfg["dataset"]["id"],
        "split": child_cfg["dataset"]["split"],
        "sample_count": len(rows),
        "data_hash": data_hash,
        "baseline_logit_diff": float(baseline_summary["logit_diff"]),
        "heads": head_reports,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(f"Wrote head inspection to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Cross-condition config JSON.")
    parser.add_argument("--tag", required=True, help="Condition tag inside the config.")
    parser.add_argument(
        "--heads", nargs="+", type=parse_head, required=True, help="Heads to inspect (L:H)."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of examples to inspect (default: 5).",
    )
    parser.add_argument("--output", type=Path, required=True, help="Where to write the JSON report.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_analysis(args.config, args.tag, args.heads, args.samples, args.output)


if __name__ == "__main__":
    main()
