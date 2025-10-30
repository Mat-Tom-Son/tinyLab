#!/usr/bin/env python3
"""Generate head pair configs from H1 (heads_zero) results."""
import json
import argparse
from pathlib import Path
import pandas as pd


def make_pairs(h1_run_dir, top_k=5, output_path=None):
    """Create pairs config from H1 results.

    Args:
        h1_run_dir: Path to H1 run directory
        top_k: Number of top critical heads to combine
        output_path: Where to save pairs config (default: lab/configs/battery_h5_pairs.json)
    """
    h1_dir = Path(h1_run_dir)

    # Load per-example data (has all head-level data)
    per_ex_path = h1_dir / "metrics" / "per_example.parquet"
    if not per_ex_path.exists():
        raise FileNotFoundError(f"Per-example data not found: {per_ex_path}")

    df = pd.read_parquet(per_ex_path)

    # Aggregate by (layer, head) - compute mean impact across examples and seeds
    # Assuming "logit_diff" is the primary metric (lower = more critical)
    grouped = df.groupby(["layer", "head"])["logit_diff"].mean().reset_index()

    # Sort by impact (ascending = most negative = biggest harm)
    # If your metric is inverted (higher = more critical), use descending
    grouped = grouped.sort_values("logit_diff", ascending=True)

    # Take top K most critical heads
    top_heads = grouped.head(top_k)

    print(f"Top {top_k} critical heads:")
    print(top_heads)

    # Generate all pairs from top K
    pairs = []
    heads_list = top_heads[["layer", "head"]].values.tolist()

    for i in range(len(heads_list)):
        for j in range(i + 1, len(heads_list)):
            layer1, head1 = heads_list[i]
            layer2, head2 = heads_list[j]

            # Only pair heads in the same layer for now
            if layer1 == layer2:
                pairs.append({"layer": int(layer1), "h1": int(head1), "h2": int(head2)})

    battery_cfg = {"type": "heads_pair_zero", "pairs": pairs}

    # Save
    if output_path is None:
        output_path = Path("lab/configs/battery_h5_pairs.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(battery_cfg, f, indent=2)

    print(f"\nCreated {len(pairs)} pairs in: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate head pairs from H1 results")
    parser.add_argument("h1_run_dir", help="Path to H1 run directory")
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of top heads to combine"
    )
    parser.add_argument("--output", help="Output path for battery config")

    args = parser.parse_args()
    make_pairs(args.h1_run_dir, args.top_k, args.output)
