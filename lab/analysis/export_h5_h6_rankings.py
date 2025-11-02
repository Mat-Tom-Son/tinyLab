"""Export standardized rankings for H5 (pairs/triplets) and H6 (reverse patch).

Scans lab/runs for matching experiments and writes consolidated CSVs under reports/.

Outputs (created if data exists):
- reports/h5_pairs_ranking.csv              (columns: run_id, layer, head_pair, mean_value)
- reports/h5_triplets_ranking.csv           (columns: run_id, layer, heads, label, mean_value)
- reports/h6_layer_ranking.csv              (columns: run_id, layer, granularity, mean_value)
"""

from __future__ import annotations

import os
from pathlib import Path
import pandas as pd


def _find_files(root: str, filename: str, prefix_filter: str) -> list[str]:
    hits = []
    for dirpath, _dirs, files in os.walk(root):
        if prefix_filter and prefix_filter not in dirpath:
            continue
        if filename in files:
            hits.append(str(Path(dirpath) / filename))
    return sorted(hits)


def export_h5_pairs(outdir: Path) -> Path | None:
    paths = _find_files('lab/runs', 'head_impact.parquet', 'h5_layer0_pairs_')
    if not paths:
        return None
    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        df = df[(df['metric']=='logit_diff') & (df['scale']==0.0)]
        if df.empty:
            continue
        # group by run, layer, pair label (df['head'] like 'h1-h2')
        g = df.groupby(['run_id','layer','head'])['value'].mean().reset_index()
        g = g.rename(columns={'head':'head_pair','value':'mean_value'})
        frames.append(g)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True).sort_values(['run_id','layer','mean_value'])
    out_path = outdir / 'h5_pairs_ranking.csv'
    out.to_csv(out_path, index=False)
    return out_path


def export_h5_triplets(outdir: Path) -> Path | None:
    paths = _find_files('lab/runs', 'layer_impact.parquet', 'h5_layer0_triplet_')
    if not paths:
        return None
    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        # heads_subset_zero stored as layer_impact with 'heads' string and 'label'
        if not {'heads','label','metric','value','layer','run_id'}.issubset(df.columns):
            continue
        df = df[df['metric']=='logit_diff']
        if df.empty:
            continue
        g = df.groupby(['run_id','layer','heads','label'])['value'].mean().reset_index()
        g = g.rename(columns={'value':'mean_value'})
        frames.append(g)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True).sort_values(['run_id','layer','mean_value'])
    out_path = outdir / 'h5_triplets_ranking.csv'
    out.to_csv(out_path, index=False)
    return out_path


def export_h6_layers(outdir: Path) -> Path | None:
    paths = _find_files('lab/runs', 'layer_impact.parquet', 'h6_')
    if not paths:
        return None
    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        if not {'granularity','metric','value','layer','run_id'}.issubset(df.columns):
            continue
        df = df[df['metric']=='logit_diff']
        if df.empty:
            continue
        g = df.groupby(['run_id','granularity','layer'])['value'].mean().reset_index()
        g = g.rename(columns={'value':'mean_value'})
        frames.append(g)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True).sort_values(['run_id','granularity','layer'])
    out_path = outdir / 'h6_layer_ranking.csv'
    out.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    outdir = Path('reports')
    outdir.mkdir(parents=True, exist_ok=True)
    p1 = export_h5_pairs(outdir)
    p2 = export_h5_triplets(outdir)
    p3 = export_h6_layers(outdir)
    print('Wrote:', p1, p2, p3)


if __name__ == '__main__':
    main()
