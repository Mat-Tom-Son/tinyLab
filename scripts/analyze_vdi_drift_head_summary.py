#!/usr/bin/env python3
"""Summarise VDI + drift metrics per head for a given run dir.

Given:
    - A run directory from scripts/pythia_layer0_vdi_drift.py
    - A VDI CSV (layer-specific) with vdi_effect per head

This script produces a per-head summary table with:
    - vdi_effect
    - delta_drift_final (within-trajectory, ablated - base)
    - delta_entropy_logits (within-trajectory)
    - mean_between_mid (paired base vs ablated drift at mid layer)
    - mean_between_final (paired drift at final layer)
    - mean_between_kl (paired KL divergence at logits)

Usage example:
    python3 scripts/analyze_vdi_drift_head_summary.py \
        --run-dir reports/pythia_layer0_vdi_drift_phase1b \
        --vdi-csv reports/pythia_layer0_vdi_drift_phase1b/pythia_layer0_vdi_sigma_0.050.csv

    # For a layer-4 drift run, but using a layer-4 VDI sweep:
    python3 scripts/analyze_vdi_drift_head_summary.py \
        --run-dir reports/pythia_layer4_vdi_drift_phase1b \
        --vdi-csv reports/layer_sweep_pythia2.8b/pythia_layer4_vdi_sigma_0.050.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def load_manifest(run_dir: Path) -> dict:
    """Load the manifest JSON from the run directory."""
    manifest_path = run_dir / "pythia_layer0_vdi_drift_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text())


def build_head_summary(
    drift_csv: Path,
    between_csv: Path,
    vdi_csv: Path,
    heads: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Build per-head summary DataFrame.

    Args:
        drift_csv: Path to drift trajectories CSV
        between_csv: Path to between-condition CSV
        vdi_csv: Path to VDI CSV (must contain columns ['head','vdi_effect'])
        heads: Optional list of head indices to include (default: all heads in drift_csv)
    """
    drift = pd.read_csv(drift_csv)
    between = pd.read_csv(between_csv)
    vdi = pd.read_csv(vdi_csv)

    # Base stats (head = -1, condition = base)
    base_final = drift[
        (drift["head"] == -1)
        & (drift["condition"] == "base")
        & (drift["layer"] == "resid_final")
    ]
    base_logits = drift[
        (drift["head"] == -1)
        & (drift["condition"] == "base")
        & (drift["layer"] == "logits")
    ]
    base_drift_final = (
        float(base_final["mean_drift"].mean()) if not base_final.empty else np.nan
    )
    base_entropy_logits = (
        float(base_logits["mean_entropy"].mean()) if not base_logits.empty else np.nan
    )

    # Within-trajectory per-head stats
    within_rows = []
    for h in sorted(drift["head"].unique()):
        if h < 0:
            continue
        if heads is not None and h not in heads:
            continue
        abl_final = drift[
            (drift["head"] == h)
            & (drift["condition"] == "ablated")
            & (drift["layer"] == "resid_final")
        ]
        abl_logits = drift[
            (drift["head"] == h)
            & (drift["condition"] == "ablated")
            & (drift["layer"] == "logits")
        ]
        if abl_final.empty or abl_logits.empty:
            continue
        drift_h = float(abl_final["mean_drift"].mean())
        ent_h = float(abl_logits["mean_entropy"].mean())
        within_rows.append(
            {
                "head_idx": int(h),
                "drift_final_ablated": drift_h,
                "entropy_logits_ablated": ent_h,
                "delta_drift_final": drift_h - base_drift_final,
                "delta_entropy_logits": ent_h - base_entropy_logits,
            }
        )

    within_df = pd.DataFrame(within_rows)

    # Between-condition per-head stats
    between_head = (
        between.groupby("head")
        .agg(
            mean_between_mid=("mean_drift_between_mid", "mean"),
            mean_between_final=("mean_drift_between_final", "mean"),
            mean_between_kl=("mean_kl_between", "mean"),
        )
        .reset_index()
        .rename(columns={"head": "head_idx"})
    )
    if heads is not None:
        between_head = between_head[between_head["head_idx"].isin(heads)]

    # VDI stats
    if "head" in vdi.columns:
        vdi_df = vdi.rename(columns={"head": "head_idx"})
    else:
        vdi_df = vdi.copy()
    vdi_df = vdi_df[["head_idx", "vdi_effect"]].drop_duplicates()
    if heads is not None:
        vdi_df = vdi_df[vdi_df["head_idx"].isin(heads)]

    # Combine
    summary = (
        within_df.merge(between_head, on="head_idx", how="inner")
        .merge(vdi_df, on="head_idx", how="left")
        .sort_values("head_idx")
        .reset_index(drop=True)
    )

    return summary


def compute_simple_correlations(summary: pd.DataFrame) -> dict:
    """Compute simple Pearson correlations between vdi_effect and key metrics."""
    out = {}
    x = summary["vdi_effect"].to_numpy()

    def corr(ycol: str) -> Optional[float]:
        y = summary[ycol].to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            return None
        return float(np.corrcoef(x[mask], y[mask])[0, 1])

    for col in [
        "delta_drift_final",
        "delta_entropy_logits",
        "mean_between_mid",
        "mean_between_final",
        "mean_between_kl",
    ]:
        out[col] = corr(col)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise VDI + drift metrics per head for a given run directory."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Run directory produced by scripts/pythia_layer0_vdi_drift.py",
    )
    parser.add_argument(
        "--vdi-csv",
        type=str,
        required=True,
        help="Path to VDI CSV to use for vdi_effect values (e.g. layer-specific sweep).",
    )
    parser.add_argument(
        "--heads",
        type=str,
        default="",
        help="Optional comma-separated list of head indices to include (default: all heads in drift data).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="head_summary.csv",
        help="Output CSV filename (relative to run-dir).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir)
    vdi_csv = Path(args.vdi_csv)
    if not vdi_csv.exists():
        raise FileNotFoundError(f"VDI CSV not found: {vdi_csv}")

    manifest = load_manifest(run_dir)

    drift_csv = run_dir / Path(manifest["drift_csv"]).name
    between_csv = run_dir / Path(manifest["between_csv"]).name

    if not drift_csv.exists():
        raise FileNotFoundError(f"Drift CSV not found: {drift_csv}")
    if not between_csv.exists():
        raise FileNotFoundError(f"Between-condition CSV not found: {between_csv}")

    if args.heads:
        heads = [int(h.strip()) for h in args.heads.split(",") if h.strip()]
    else:
        heads = None

    summary = build_head_summary(drift_csv, between_csv, vdi_csv, heads=heads)

    # Save summary
    out_path = run_dir / args.output
    summary.to_csv(out_path, index=False)
    print(f"Wrote per-head summary to {out_path}")

    # Print to stdout
    print("\nPer-head summary:")
    print(summary.to_string(index=False))

    # Simple correlations with vdi_effect
    corrs = compute_simple_correlations(summary)
    print("\nCorrelations with vdi_effect (Pearson, no p-values):")
    for k, v in corrs.items():
        if v is None:
            print(f"  {k}: n<2 or non-finite")
        else:
            print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()
