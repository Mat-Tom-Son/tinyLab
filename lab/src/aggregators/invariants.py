"""Invariants detector for finding conserved components across conditions (v1.1)."""
import sys
import argparse
from pathlib import Path
import pandas as pd
from rich import print

from ..utils import io


def find_invariant_heads(df, k=10, metric="logit_diff", descending=True):
    """Find heads that are top-k across all conditions.

    Args:
        df: DataFrame with columns [condition, layer, head, metric, value]
        k: Top-k threshold
        metric: Metric to rank by
        descending: If True, rank descending (higher is more important)

    Returns:
        List of (layer, head) tuples that are top-k in all conditions
    """
    # Filter to target metric
    df_metric = df[df["metric"] == metric].copy()

    conditions = df_metric["condition"].unique()
    if len(conditions) == 0:
        return []

    # For each condition, find top-k heads
    condition_topk = {}
    for cond in conditions:
        cond_df = df_metric[df_metric["condition"] == cond]

        # Aggregate over seeds if present
        if "seed" in cond_df.columns:
            cond_agg = cond_df.groupby(["layer", "head"])["value"].mean().reset_index()
        else:
            cond_agg = cond_df

        # Sort and take top-k
        cond_sorted = cond_agg.sort_values("value", ascending=not descending)
        top_k_heads = cond_sorted.head(k)[["layer", "head"]]
        condition_topk[cond] = set(zip(top_k_heads["layer"], top_k_heads["head"]))

    # Find intersection
    invariant_set = set.intersection(*condition_topk.values())

    return sorted(list(invariant_set))


def find_invariant_layers(df, k=10, metric="logit_diff", descending=True):
    """Find layers that are top-k across all conditions.

    Args:
        df: DataFrame with columns [condition, layer, metric, value]
        k: Top-k threshold
        metric: Metric to rank by
        descending: If True, rank descending (higher is more important)

    Returns:
        List of layer indices that are top-k in all conditions
    """
    # Filter to target metric
    df_metric = df[df["metric"] == metric].copy()

    conditions = df_metric["condition"].unique()
    if len(conditions) == 0:
        return []

    # For each condition, find top-k layers
    condition_topk = {}
    for cond in conditions:
        cond_df = df_metric[df_metric["condition"] == cond]

        # Aggregate over seeds if present
        if "seed" in cond_df.columns:
            cond_agg = cond_df.groupby("layer")["value"].mean().reset_index()
        else:
            cond_agg = cond_df

        # Sort and take top-k
        cond_sorted = cond_agg.sort_values("value", ascending=not descending)
        top_k_layers = set(cond_sorted.head(k)["layer"])
        condition_topk[cond] = top_k_layers

    # Find intersection
    invariant_set = set.intersection(*condition_topk.values())

    return sorted(list(invariant_set))


def main(cross_condition_dir, k=10, metrics=None):
    """Find invariants across conditions.

    Args:
        cross_condition_dir: Path to artifacts/cross_condition directory
        k: Top-k threshold
        metrics: List of metrics to analyze (default: ["logit_diff"])
    """
    cross_dir = Path(cross_condition_dir)

    if not cross_dir.exists():
        print(f"[red]Directory not found: {cross_dir}[/red]")
        sys.exit(1)

    if metrics is None:
        metrics = ["logit_diff"]

    print(f"[blue]Finding invariants in: {cross_dir}[/blue]")
    print(f"[blue]Top-k: {k}, Metrics: {metrics}[/blue]")

    invariants = {
        "k": k,
        "metrics": metrics,
        "heads": {},
        "layers": {},
    }

    # Check for head matrix
    head_matrix_path = cross_dir / "head_matrix.parquet"
    if head_matrix_path.exists():
        print("[cyan]Analyzing head_matrix.parquet...[/cyan]")
        df_heads = pd.read_parquet(head_matrix_path)

        for metric in metrics:
            invariant_heads = find_invariant_heads(df_heads, k=k, metric=metric)
            invariants["heads"][metric] = [
                {"layer": int(layer), "head": int(head)} for layer, head in invariant_heads
            ]
            print(f"  {metric}: {len(invariant_heads)} invariant heads")

    # Check for layer matrix
    layer_matrix_path = cross_dir / "layer_matrix.parquet"
    if layer_matrix_path.exists():
        print("[cyan]Analyzing layer_matrix.parquet...[/cyan]")
        df_layers = pd.read_parquet(layer_matrix_path)

        for metric in metrics:
            invariant_layers = find_invariant_layers(df_layers, k=k, metric=metric)
            invariants["layers"][metric] = [int(layer) for layer in invariant_layers]
            print(f"  {metric}: {len(invariant_layers)} invariant layers")

    # Save results
    output_path = cross_dir / "invariants.json"
    io.save_json(invariants, output_path)
    print(f"[green]Saved invariants to: {output_path}[/green]")

    return invariants


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find invariant components across conditions")
    parser.add_argument("cross_condition_dir", help="Path to cross_condition directory")
    parser.add_argument("--k", type=int, default=10, help="Top-k threshold")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["logit_diff"],
        help="Metrics to analyze",
    )

    args = parser.parse_args()
    main(args.cross_condition_dir, k=args.k, metrics=args.metrics)
