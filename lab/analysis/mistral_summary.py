"""Consolidate Mistral H1 results into summary CSV and trio percentiles.

Outputs:
- reports/mistral_summary_table.csv
- reports/mistral_trio_percentiles.json

Assumes the following runs exist (update paths via args if needed):
- Neg:   lab/runs/h1_mistral_neg_fullstack_3seed_*/metrics/head_impact.parquet
- CF:    lab/runs/h1_mistral_cf_fullstack_3seed_*/metrics/head_impact.parquet
- Logic: lab/runs/h1_mistral_logic_fullstack_3seed_*/metrics/head_impact.parquet
- Facts: lab/runs/h1_cross_condition_balanced_mistral_facts_*/metrics/head_impact.parquet
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


DEFAULT_SUMMARY = Path("reports/mistral_h1_logit_diff_summary.json")


def most_recent(pattern: str) -> Path:
    files = sorted(glob.glob(pattern), key=lambda p: Path(p).stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No files match: {pattern}")
    return Path(files[0])


def load_head_impact(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    # normalise expected columns
    return df


def percentile(value_series: pd.Series, val: float) -> float:
    # Percentile: proportion of values <= val
    return float((value_series <= val).sum() / max(1, len(value_series)) * 100.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--outdir", type=Path, default=Path("reports"))
    ap.add_argument("--neg", type=str, default="lab/runs/h1_mistral_neg_fullstack_3seed_*/metrics/head_impact.parquet")
    ap.add_argument("--cf", type=str, default="lab/runs/h1_mistral_cf_fullstack_3seed_*/metrics/head_impact.parquet")
    ap.add_argument("--logic", type=str, default="lab/runs/h1_mistral_logic_fullstack_3seed_*/metrics/head_impact.parquet")
    ap.add_argument("--facts", type=str, default="lab/runs/h1_cross_condition_balanced_mistral_facts_*/metrics/head_impact.parquet")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Load multi-seed summaries (logit_diff means)
    summaries = json.loads(args.summary.read_text())

    # Resolve parquet files
    neg_p = most_recent(args.neg)
    cf_p = most_recent(args.cf)
    logic_p = most_recent(args.logic)
    facts_p = most_recent(args.facts)

    runs: Dict[str, Path] = {
        "Negation": neg_p,
        "Counterfactual": cf_p,
        "Logic": logic_p,
        "Facts": facts_p,
    }

    # Build summary CSV
    rows = []
    for probe, data in (
        ("Facts", summaries["facts"]),
        ("Negation", summaries["neg_full"]),
        ("Counterfactual", summaries["cf_full"]),
        ("Logic", summaries["logic_full"]),
    ):
        rows.append(
            {
                "Probe": probe,
                "LogitDiff_Mean": data.get("mean"),
                "Seeds": 5 if probe == "Facts" else 3,
                "Scope": "Full H1",
            }
        )
    summary_df = pd.DataFrame(rows)
    summary_csv = args.outdir / "mistral_summary_table.csv"
    summary_df.to_csv(summary_csv, index=False)

    # Compute percentiles for suppressor trio across probes (layer 0 only)
    trio = [(0, 21), (0, 22), (0, 23)]
    trio_out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for label, path in runs.items():
        df = load_head_impact(path)
        df = df[(df["metric"] == "logit_diff") & (df["scale"] == 0.0) & (df["layer"] == 0)]
        vals = df["value"].astype(float)
        per_probe: Dict[str, Dict[str, float]] = {}
        for layer, head in trio:
            sel = df[(df["layer"] == layer) & (df["head"] == head)]["value"]
            if sel.empty:
                continue
            v = float(sel.mean())
            per_probe[f"{layer}:{head}"] = {
                "mean_value": v,
                "percentile": percentile(vals, v),
            }
        trio_out[label] = per_probe

    trio_json = args.outdir / "mistral_trio_percentiles.json"
    trio_json.write_text(json.dumps(trio_out, indent=2))

    print(f"Wrote {summary_csv}")
    print(f"Wrote {trio_json}")


if __name__ == "__main__":
    main()

