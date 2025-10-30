"""Utilities for computing head-ranking agreement statistics across conditions.

This script consumes one or more ``head_matrix.parquet`` artifacts produced by
the cross-condition orchestrator and reports layer-wise Spearman correlations,
bootstrap confidence intervals, and permutation p-values.

Example usage::

    python -m lab.analysis.head_rank_stats \\
        lab/runs/h1_cross_condition_physics_probe_*/artifacts/cross_condition/head_matrix.parquet \\
        --metric logit_diff \\
        --output reports/h1_head_rank_stats.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


DEFAULT_METRIC = "logit_diff"


@dataclass
class PairwiseStats:
    condition_a: str
    condition_b: str
    layer: int
    n_heads: int
    rho: float
    bootstrap_ci_low: float | None
    bootstrap_ci_high: float | None
    bootstrap_samples: int
    permutation_p: float | None
    permutation_samples: int
    permutation_null_mean: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "matrices",
        nargs="+",
        help="Paths or glob patterns for head_matrix.parquet files.",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help=f"Metric name to analyse (default: {DEFAULT_METRIC}).",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap samples for confidence intervals.",
    )
    parser.add_argument(
        "--n-permutation",
        type=int,
        default=1000,
        help="Number of permutations for null distribution.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for bootstrap/permutation sampling.",
    )
    parser.add_argument(
        "--min-heads",
        type=int,
        default=4,
        help="Minimum overlapping heads required to compute a statistic.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path (JSON). If omitted, prints to stdout.",
    )
    return parser.parse_args()


def expand_paths(patterns: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for pat in patterns:
        matches = list(Path().glob(pat))
        if not matches:
            raise FileNotFoundError(f"No files matched pattern: {pat}")
        for match in matches:
            if not match.exists():
                raise FileNotFoundError(match)
            paths.append(match)
    return paths


def load_head_matrix(paths: Iterable[Path], metric: str) -> pd.DataFrame:
    frames = []
    for path in paths:
        if path.suffix != ".parquet":
            raise ValueError(f"Expected parquet file, got {path}")
        frames.append(pd.read_parquet(path))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["metric"] == metric].copy()
    df = df.dropna(subset=["value"])
    if df.empty:
        raise ValueError(f"No rows found for metric '{metric}'.")
    return df


def make_seed_head_table(sub_df: pd.DataFrame) -> pd.DataFrame:
    """Return a table indexed by seed with columns=head numbers."""
    table = (
        sub_df.groupby(["seed", "head"])["value"]
        .mean()
        .unstack("head")
        .sort_index(axis=1)
    )
    return table


def zscore(series: pd.Series) -> np.ndarray:
    arr = series.to_numpy(dtype=float)
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if math.isclose(std, 0.0) or np.isnan(std):
        return np.zeros_like(arr)
    return (arr - mean) / std


def observed_stats(table_a: pd.DataFrame, table_b: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    # Align heads present in both tables.
    common_heads = sorted(set(table_a.columns).intersection(table_b.columns))
    if not common_heads:
        return np.array([]), np.array([])
    mean_a = table_a[common_heads].mean(axis=0)
    mean_b = table_b[common_heads].mean(axis=0)
    mask = ~(mean_a.isna() | mean_b.isna())
    mean_a = mean_a[mask]
    mean_b = mean_b[mask]
    z_a = zscore(mean_a)
    z_b = zscore(mean_b)
    return z_a, z_b


def compute_pair_stats(
    cond_a: str,
    cond_b: str,
    layer: int,
    table_a: pd.DataFrame,
    table_b: pd.DataFrame,
    min_heads: int,
    n_bootstrap: int,
    n_perm: int,
    rng: np.random.Generator,
) -> PairwiseStats | None:
    z_a, z_b = observed_stats(table_a, table_b)
    n_heads = len(z_a)
    if n_heads < min_heads:
        return None

    rho_obs, _ = spearmanr(z_a, z_b)
    if np.isnan(rho_obs):
        rho_obs = 0.0

    seeds = sorted(set(table_a.index).intersection(table_b.index))
    bootstrap_rhos: List[float] = []
    if n_bootstrap > 0 and seeds:
        for _ in range(n_bootstrap):
            sampled = rng.choice(seeds, size=len(seeds), replace=True)
            boot_a = table_a.loc[sampled]
            boot_b = table_b.loc[sampled]
            z_boot_a, z_boot_b = observed_stats(boot_a, boot_b)
            if len(z_boot_a) < min_heads:
                continue
            rho_boot, _ = spearmanr(z_boot_a, z_boot_b)
            if np.isnan(rho_boot):
                continue
            bootstrap_rhos.append(float(rho_boot))
    ci_low = ci_high = None
    if bootstrap_rhos:
        ci_low, ci_high = np.percentile(bootstrap_rhos, [2.5, 97.5])

    perm_p = None
    perm_mean = None
    if n_perm > 0:
        perm_rhos = []
        for _ in range(n_perm):
            shuffled = rng.permutation(z_b)
            rho_perm, _ = spearmanr(z_a, shuffled)
            if np.isnan(rho_perm):
                continue
            perm_rhos.append(float(rho_perm))
        if perm_rhos:
            perm_mean = float(np.mean(perm_rhos))
            greater = sum(abs(r) >= abs(rho_obs) for r in perm_rhos)
            perm_p = (greater + 1) / (len(perm_rhos) + 1)

    return PairwiseStats(
        condition_a=cond_a,
        condition_b=cond_b,
        layer=int(layer),
        n_heads=int(n_heads),
        rho=float(rho_obs),
        bootstrap_ci_low=ci_low if ci_low is not None else None,
        bootstrap_ci_high=ci_high if ci_high is not None else None,
        bootstrap_samples=len(bootstrap_rhos),
        permutation_p=perm_p,
        permutation_samples=n_perm,
        permutation_null_mean=perm_mean,
    )


def aggregate_tables(df: pd.DataFrame) -> Dict[Tuple[str, int], pd.DataFrame]:
    tables: Dict[Tuple[str, int], pd.DataFrame] = {}
    for (condition, layer), sub_df in df.groupby(["condition", "layer"]):
        tables[(condition, layer)] = make_seed_head_table(sub_df)
    return tables


def summarise(stats: List[PairwiseStats]) -> Dict:
    if not stats:
        return {}
    summary = {
        "pairs": [
            {
                "condition_a": s.condition_a,
                "condition_b": s.condition_b,
                "layer": s.layer,
                "n_heads": s.n_heads,
                "rho": s.rho,
                "bootstrap_ci_low": s.bootstrap_ci_low,
                "bootstrap_ci_high": s.bootstrap_ci_high,
                "bootstrap_samples": s.bootstrap_samples,
                "permutation_p": s.permutation_p,
                "permutation_samples": s.permutation_samples,
                "permutation_null_mean": s.permutation_null_mean,
            }
            for s in stats
        ]
    }
    # Aggregate helper stats
    rho_vals = [s.rho for s in stats]
    summary["rho_median"] = float(np.median(rho_vals))
    summary["rho_mean"] = float(np.mean(rho_vals))
    summary["n_entries"] = len(stats)
    return summary


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    paths = expand_paths(args.matrices)
    df = load_head_matrix(paths, args.metric)

    conditions = [str(c) for c in sorted(df["condition"].unique())]
    layers = [int(l) for l in sorted(df["layer"].unique())]

    tables = aggregate_tables(df)

    stats: List[PairwiseStats] = []
    for cond_a, cond_b in combinations(conditions, 2):
        for layer in layers:
            key_a = (cond_a, layer)
            key_b = (cond_b, layer)
            if key_a not in tables or key_b not in tables:
                continue
            stat = compute_pair_stats(
                cond_a,
                cond_b,
                layer,
                tables[key_a],
                tables[key_b],
                min_heads=args.min_heads,
                n_bootstrap=args.n_bootstrap,
                n_perm=args.n_permutation,
                rng=rng,
            )
            if stat:
                stats.append(stat)

    result = {
        "metric": args.metric,
        "conditions": conditions,
        "layers": layers,
        "min_heads": args.min_heads,
        "n_bootstrap": args.n_bootstrap,
        "n_permutation": args.n_permutation,
        "seed": args.seed,
        "summary": summarise(stats),
    }
    result["summary"]["details"] = result["summary"].pop("pairs", [])

    serialized = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    else:
        print(serialized)


if __name__ == "__main__":
    main()
