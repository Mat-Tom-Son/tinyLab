"""Generic model summary and trio percentiles from run metrics.

Example:
  python -m lab.analysis.model_summary \
    --label gpt2_medium \
    --facts 'lab/runs/h1_cross_condition_physics_balanced_facts_*/metrics/summary.json' \
    --neg   'lab/runs/h1_cross_condition_physics_balanced_neg_*/metrics/summary.json' \
    --cf    'lab/runs/h1_cross_condition_physics_balanced_cf_*/metrics/summary.json' \
    --logic 'lab/runs/h1_cross_condition_physics_balanced_logic_*/metrics/summary.json' \
    --head-neg   'lab/runs/h1_cross_condition_physics_balanced_neg_*/metrics/head_impact.parquet' \
    --head-cf    'lab/runs/h1_cross_condition_physics_balanced_cf_*/metrics/head_impact.parquet' \
    --head-logic 'lab/runs/h1_cross_condition_physics_balanced_logic_*/metrics/head_impact.parquet' \
    --head-facts 'lab/runs/h1_cross_condition_physics_balanced_facts_*/metrics/head_impact.parquet'
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict

import pandas as pd


def latest(pattern: str) -> Path:
    matches = sorted(glob.glob(pattern), key=lambda p: Path(p).stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No files for pattern: {pattern}")
    return Path(matches[0])


def load_summary(path: Path) -> Dict:
    return json.loads(path.read_text())


def percentile(series: pd.Series, val: float) -> float:
    return float((series <= val).sum() / max(1, len(series)) * 100.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--facts', required=True)
    ap.add_argument('--neg', required=True)
    ap.add_argument('--cf', required=True)
    ap.add_argument('--logic', required=True)
    ap.add_argument('--head-facts', required=True)
    ap.add_argument('--head-neg', required=True)
    ap.add_argument('--head-cf', required=True)
    ap.add_argument('--head-logic', required=True)
    ap.add_argument('--outdir', type=Path, default=Path('reports'))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Load summaries
    s = {
        'Facts': load_summary(latest(args.facts)).get('logit_diff', {}),
        'Negation': load_summary(latest(args.neg)).get('logit_diff', {}),
        'Counterfactual': load_summary(latest(args.cf)).get('logit_diff', {}),
        'Logic': load_summary(latest(args.logic)).get('logit_diff', {}),
    }

    # Summary CSV
    rows = []
    for probe, data in s.items():
        rows.append({
            'Probe': probe,
            'LogitDiff_Mean': data.get('mean'),
            'Seeds': len(data.get('values', [])),
            'Scope': 'Full H1'
        })
    df = pd.DataFrame(rows)
    out_csv = args.outdir / f"{args.label}_summary_table.csv"
    df.to_csv(out_csv, index=False)

    # Trio percentiles at layer 0
    trio = [(0, 2), (0, 4), (0, 7)]  # GPT-2 default suppressor trio
    head_paths = {
        'Facts': latest(args.head_facts),
        'Negation': latest(args.head_neg),
        'Counterfactual': latest(args.head_cf),
        'Logic': latest(args.head_logic),
    }
    trio_out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for probe, p in head_paths.items():
        imp = pd.read_parquet(p)
        imp = imp[(imp['metric']=='logit_diff') & (imp['scale']==0.0) & (imp['layer']==0)]
        vals = imp['value'].astype(float)
        per = {}
        for layer, head in trio:
            sel = imp[(imp['layer']==layer) & (imp['head']==head)]['value']
            if sel.empty:
                continue
            v = float(sel.mean())
            per[f'{layer}:{head}'] = {'mean_value': v, 'percentile': percentile(vals, v)}
        trio_out[probe] = per
    out_json = args.outdir / f"{args.label}_trio_percentiles.json"
    out_json.write_text(json.dumps(trio_out, indent=2))

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")


if __name__ == '__main__':
    main()

