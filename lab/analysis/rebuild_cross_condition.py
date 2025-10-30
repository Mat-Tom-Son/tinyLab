"""Rebuild cross-condition artifacts from completed child runs.

Usage:
    python -m lab.analysis.rebuild_cross_condition <parent_config.json>

This is helpful when a cross-condition orchestrator run completes all child
executions but fails before writing aggregated artifacts (e.g., due to timeout).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..src.utils import io, hashing
from ..src.orchestrators.conditions import deep_merge


def collect(parent_cfg_path: Path) -> None:
    cfg = io.load_json(parent_cfg_path)
    run_name = cfg["run_name"]
    battery_path = cfg["battery"]
    shared = cfg["shared"]
    conditions = cfg["conditions"]

    parent_run_id = hashing.sha256_json(cfg)[:12]
    parent_run_dir = Path("lab/runs") / f"{run_name}_{parent_run_id}"
    cross_dir = parent_run_dir / "artifacts" / "cross_condition"
    cross_dir.mkdir(parents=True, exist_ok=True)

    all_head_impacts = []
    all_layer_impacts = []
    child_runs = []
    condition_summaries = {}

    for cond in conditions:
        tag = cond["tag"]
        child_cfg = deep_merge(shared, cond)
        child_cfg["run_name"] = f"{run_name}_{tag}"
        child_cfg["battery"] = battery_path

        child_run_id = hashing.sha256_json(child_cfg)[:12]
        child_run_dir = Path("lab/runs") / f"{child_cfg['run_name']}_{child_run_id}"
        if not child_run_dir.exists():
            candidates = sorted(
                Path("lab/runs").glob(f"{child_cfg['run_name']}_*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                child_run_dir = candidates[0]
                print(f"[yellow]Using latest run dir for {tag}: {child_run_dir.name}[/yellow]")
            else:
                print(f"[yellow]Missing child run dir for {tag}: {child_run_dir}[/yellow]")
                continue
        child_runs.append({"tag": tag, "run_dir": str(child_run_dir)})

        manifest_path = child_run_dir / "manifest.json"
        if manifest_path.exists():
            manifest = io.load_json(manifest_path)
            head_rel = manifest.get("head_impact", "metrics/head_impact.parquet")
            layer_rel = manifest.get("layer_impact", "metrics/layer_impact.parquet")
        else:
            print(f"[yellow]Missing manifest for {tag}: {manifest_path}[/yellow]")
            head_rel = "metrics/head_impact.parquet"
            layer_rel = "metrics/layer_impact.parquet"

        head_path = child_run_dir / head_rel
        if head_path.exists():
            df = pd.read_parquet(head_path)
            df["condition"] = tag
            all_head_impacts.append(df)

        layer_path = child_run_dir / layer_rel
        if layer_path.exists():
            df = pd.read_parquet(layer_path)
            df["condition"] = tag
            all_layer_impacts.append(df)

        summary_path = child_run_dir / "metrics" / "summary.json"
        if summary_path.exists():
            condition_summaries[tag] = io.load_json(summary_path)

    head_matrix_path = None
    layer_matrix_path = None

    if all_head_impacts:
        head_matrix = pd.concat(all_head_impacts, ignore_index=True)
        head_matrix_path = cross_dir / "head_matrix.parquet"
        head_matrix.to_parquet(head_matrix_path, index=False)
        print(f"[green]Saved head_matrix.parquet ({len(head_matrix)} rows)[/green]")

    if all_layer_impacts:
        layer_matrix = pd.concat(all_layer_impacts, ignore_index=True)
        layer_matrix_path = cross_dir / "layer_matrix.parquet"
        layer_matrix.to_parquet(layer_matrix_path, index=False)
        print(f"[green]Saved layer_matrix.parquet ({len(layer_matrix)} rows)[/green]")

    summary = {
        "run_name": run_name,
        "n_conditions": len(conditions),
        "conditions": [c["tag"] for c in conditions],
        "child_runs": child_runs,
        "summaries": condition_summaries,
    }
    summary_path = cross_dir / "summary.json"
    io.save_json(summary, summary_path)

    manifest = {
        "run_dir": str(parent_run_dir),
        "cross_condition_summary": str(summary_path.relative_to(parent_run_dir)),
        "child_runs": child_runs,
    }
    if head_matrix_path:
        manifest["head_matrix"] = str(head_matrix_path.relative_to(parent_run_dir))
    if layer_matrix_path:
        manifest["layer_matrix"] = str(layer_matrix_path.relative_to(parent_run_dir))

    io.save_json(manifest, parent_run_dir / "manifest.json")

    print(f"[green]Rebuilt cross-condition artifacts in {parent_run_dir}[/green]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent_config", help="Path to parent config.json (from run dir)")
    args = parser.parse_args()
    collect(Path(args.parent_config))


if __name__ == "__main__":
    main()
