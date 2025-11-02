"""Export head-impact rankings to CSV from a run's head_impact.parquet.

Usage:
  python -m lab.analysis.export_head_rankings \
    --run neg:lab/runs/h1_mistral_neg_fullstack_3seed_*/metrics/head_impact.parquet \
    --run cf:lab/runs/h1_mistral_cf_fullstack_3seed_*/metrics/head_impact.parquet \
    --run logic:lab/runs/h1_mistral_logic_fullstack_3seed_*/metrics/head_impact.parquet \
    --only-layer -1 \
    --metric logit_diff \
    --top-k 10 --bottom-k 5 \
    --outdir reports

Notes:
- Rankings sort ascending by metric value (lower logit_diff => stronger suppressor under zeroing).
- Set --only-layer 0 to restrict to layer 0 only; -1 keeps all layers.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import glob
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run",
        action="append",
        required=True,
        help="Tagged run input as tag:path or tag:glob to head_impact.parquet.",
    )
    p.add_argument("--metric", default="logit_diff", help="Metric column to rank.")
    p.add_argument(
        "--only-layer",
        type=int,
        default=-1,
        help="Restrict to this layer (e.g., 0). Use -1 for all layers.",
    )
    p.add_argument("--top-k", type=int, default=10, help="Export top-K worst heads.")
    p.add_argument("--bottom-k", type=int, default=5, help="Export bottom-K best heads.")
    p.add_argument("--outdir", type=Path, default=Path("reports"), help="Output dir.")
    p.add_argument("--prefix", type=str, default="mistral", help="Filename prefix (model tag).")
    return p.parse_args()


def resolve_path(pat: str) -> Path:
    matches = sorted(glob.glob(pat))
    if not matches:
        raise FileNotFoundError(f"No file matched: {pat}")
    # Use the first match by recency order if possible
    return Path(matches[0])


def load_head_impact(path: Path, metric: str, only_layer: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[df["metric"] == metric].copy()
    df = df[df["scale"] == 0.0]
    if only_layer >= 0:
        df = df[df["layer"] == only_layer]
    return df


def rank_heads(df: pd.DataFrame) -> pd.DataFrame:
    # Mean across seeds per (layer, head)
    agg = (
        df.groupby(["layer", "head"])['value']
        .mean()
        .reset_index()
        .rename(columns={"value": "mean_value"})
    )
    agg = agg.sort_values("mean_value", ascending=True, kind="mergesort").reset_index(drop=True)
    agg["rank"] = agg.index + 1
    return agg


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    for item in args.run:
        if ":" not in item:
            raise ValueError(f"--run must be tag:path, got: {item}")
        tag, pat = item.split(":", 1)
        path = resolve_path(pat)
        df = load_head_impact(path, args.metric, args.only_layer)
        if df.empty:
            print(f"[warn] No rows after filtering for {tag} ({path}).")
            continue
        ranking = rank_heads(df)

        # Save full ranking
        full_csv = args.outdir / f"{args.prefix}_{tag}_head_ranking.csv"
        ranking.to_csv(full_csv, index=False)

        # Save slices
        top_csv = args.outdir / f"{args.prefix}_{tag}_top{args.top_k}.csv"
        bot_csv = args.outdir / f"{args.prefix}_{tag}_bottom{args.bottom_k}.csv"
        ranking.head(args.top_k).to_csv(top_csv, index=False)
        ranking.tail(args.bottom_k).to_csv(bot_csv, index=False)

        print(f"Wrote: {full_csv}, {top_csv}, {bot_csv}")


if __name__ == "__main__":
    main()
