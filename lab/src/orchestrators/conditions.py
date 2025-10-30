"""Cross-condition orchestrator for multi-condition experiments (v1.1)."""
import json
import sys
from pathlib import Path
import pandas as pd
from rich import print

from ..utils import io, hashing
from .. import harness


def deep_merge(base, overlay):
    """Deep merge two dicts, with overlay taking precedence."""
    result = base.copy()
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def main(config_path):
    """Run cross-condition experiment.

    Args:
        config_path: Path to cross-condition config JSON
    """
    cfg = io.load_json(config_path)

    run_name = cfg["run_name"]
    battery_path = cfg["battery"]
    shared = cfg["shared"]
    conditions = cfg["conditions"]

    print(f"[blue]Starting cross-condition run: {run_name}[/blue]")
    print(f"[blue]Conditions: {len(conditions)}[/blue]")

    # Create parent run directory
    run_id = hashing.sha256_json(cfg)[:12]
    parent_run_dir = Path("lab/runs") / f"{run_name}_{run_id}"
    parent_run_dir.mkdir(parents=True, exist_ok=True)

    cross_cond_dir = parent_run_dir / "artifacts" / "cross_condition"
    cross_cond_dir.mkdir(parents=True, exist_ok=True)

    # Save parent config
    io.save_json(cfg, parent_run_dir / "config.json")

    # Run each condition
    child_runs = []
    all_head_impacts = []
    all_layer_impacts = []
    condition_summaries = {}

    for cond in conditions:
        tag = cond["tag"]
        print(f"\n[cyan]Running condition: {tag}[/cyan]")

        # Build per-condition config
        child_cfg = deep_merge(shared, {k: v for k, v in cond.items() if k != "battery_overrides"})
        child_cfg["run_name"] = f"{run_name}_{tag}"
        child_cfg["battery"] = cond.get("battery", battery_path)
        battery_cfg = io.load_json(child_cfg["battery"])
        if "battery_overrides" in cond:
            battery_cfg = deep_merge(battery_cfg, cond["battery_overrides"])

        # Write temporary config
        temp_cfg_path = parent_run_dir / f"temp_{tag}.json"
        temp_battery_path = parent_run_dir / f"temp_{tag}_battery.json"
        io.save_json(battery_cfg, temp_battery_path)
        child_cfg["battery"] = str(temp_battery_path)
        io.save_json(child_cfg, temp_cfg_path)

        # Run via harness
        try:
            harness.main(str(temp_cfg_path))
        except Exception as e:
            print(f"[red]Condition {tag} failed: {e}[/red]")
            continue

        # Find child run directory
        child_run_id = hashing.sha256_json(child_cfg)[:12]
        child_run_dir = Path("lab/runs") / f"{child_cfg['run_name']}_{child_run_id}"

        if not child_run_dir.exists():
            print(f"[yellow]Warning: Child run dir not found for {tag}[/yellow]")
            continue

        child_runs.append({"tag": tag, "run_dir": str(child_run_dir)})

        # Load manifest
        manifest_path = child_run_dir / "manifest.json"
        if manifest_path.exists():
            manifest = io.load_json(manifest_path)

            # Collect head impact tables
            if "head_impact" in manifest:
                head_impact_path = child_run_dir / manifest["head_impact"]
                if head_impact_path.exists():
                    df = pd.read_parquet(head_impact_path)
                    df["condition"] = tag
                    all_head_impacts.append(df)

            # Collect layer impact tables
            if "layer_impact" in manifest:
                layer_impact_path = child_run_dir / manifest["layer_impact"]
                if layer_impact_path.exists():
                    df = pd.read_parquet(layer_impact_path)
                    df["condition"] = tag
                    all_layer_impacts.append(df)

            # Collect summary
            summary_path = child_run_dir / "metrics" / "summary.json"
            if summary_path.exists():
                summary = io.load_json(summary_path)
                condition_summaries[tag] = summary

        # Clean up temp config
        temp_cfg_path.unlink()
        if temp_battery_path.exists():
            temp_battery_path.unlink()

    # Concatenate impact tables
    if all_head_impacts:
        head_matrix = pd.concat(all_head_impacts, ignore_index=True)
        head_matrix_path = cross_cond_dir / "head_matrix.parquet"
        head_matrix.to_parquet(head_matrix_path, index=False)
        print(f"[green]Saved head_matrix.parquet ({len(head_matrix)} rows)[/green]")

    if all_layer_impacts:
        layer_matrix = pd.concat(all_layer_impacts, ignore_index=True)
        layer_matrix_path = cross_cond_dir / "layer_matrix.parquet"
        layer_matrix.to_parquet(layer_matrix_path, index=False)
        print(f"[green]Saved layer_matrix.parquet ({len(layer_matrix)} rows)[/green]")

    # Save cross-condition summary
    cross_summary = {
        "run_name": run_name,
        "n_conditions": len(conditions),
        "conditions": [c["tag"] for c in conditions],
        "child_runs": child_runs,
        "summaries": condition_summaries,
    }
    summary_path = cross_cond_dir / "summary.json"
    io.save_json(cross_summary, summary_path)

    # Save parent manifest
    parent_manifest = {
        "run_dir": str(parent_run_dir),
        "cross_condition_summary": str(summary_path.relative_to(parent_run_dir)),
        "child_runs": child_runs,
    }

    if all_head_impacts:
        parent_manifest["head_matrix"] = str(head_matrix_path.relative_to(parent_run_dir))

    if all_layer_impacts:
        parent_manifest["layer_matrix"] = str(layer_matrix_path.relative_to(parent_run_dir))

    io.save_json(parent_manifest, parent_run_dir / "manifest.json")

    print(f"\n[green]Cross-condition run complete![/green]")
    print(f"[green]Parent run dir: {parent_run_dir}[/green]")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m lab.src.orchestrators.conditions <config.json>")
        sys.exit(1)
    main(sys.argv[1])
