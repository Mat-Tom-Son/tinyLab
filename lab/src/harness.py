"""Main experiment harness for running ablation batteries."""
import json
import os
import time
import subprocess
import uuid
import sys
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from rich import print

from .components import tracking, datasets, load_model, metrics, profiling
from .utils import hashing, determinism, io, stats


def flatten_dict(d, parent_key="", sep="."):
    """Flatten nested dict for MLflow logging."""
    items = []
    for k, v in d.items():
        new_k = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_k, sep).items())
        elif isinstance(v, list):
            items.append((new_k, json.dumps(v)))  # Log lists as JSON strings
        else:
            items.append((new_k, v))
    return dict(items)


def main(cfg_path):
    """Run an experiment from a config file."""
    cfg = io.load_json(cfg_path)
    base_seed = cfg.get("seed", 0)
    determinism.set_seed(base_seed)

    device_cfg = cfg.get("device", "auto")
    if device_cfg == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = device_cfg
        if device == "mps" and not torch.backends.mps.is_available():
            print("[yellow]Requested MPS but backend is unavailable. Falling back to CPU.[/yellow]")
            device = "cpu"

    # Create run dir
    run_id = hashing.sha256_json(cfg)[:12]
    run_dir = Path("lab/runs") / f"{cfg['run_name']}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    out_dir = run_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Persist reproducibility info
    io.save_json(cfg, run_dir / "config.json")
    (run_dir / "config_hash.txt").write_text(hashing.sha256_json(cfg))
    git_commit = subprocess.getoutput("git rev-parse HEAD")
    (run_dir / "git_commit.txt").write_text(git_commit or "unknown")
    env_freeze = subprocess.getoutput("python -m pip freeze")
    (run_dir / "env.txt").write_text(env_freeze)

    # Tracking
    mlf = tracking.MLFlowTracker(
        experiment="tiny-ablation-lab", run_name=cfg["run_name"]
    )
    mlf.log_params(flatten_dict(cfg))

    # Load data
    dset, split_info, data_hash = datasets.load_split(cfg["dataset"])
    (run_dir / "data_hash.txt").write_text(data_hash)
    mlf.log_param("data_hash", data_hash)

    # Load model
    model = load_model.load_transformerlens(cfg["model"], device=device)

    # Profiling hooks
    prof_cfg = cfg.get("profiling", {})
    prof = profiling.Profiler(
        interval_s=prof_cfg.get("interval_s", 5.0),
        use_powermetrics=prof_cfg.get("powermetrics", False),
    )
    prof.start()

    # Run battery over N seeds
    seeds = cfg.get("seeds") or [base_seed]
    seed_summaries, per_ex_all = [], []
    impact_accum = None
    impact_tables = []  # v1.1: collect impact tables
    battery_cfg = io.load_json(cfg["battery"])

    print(
        f"[blue]Starting battery '{battery_cfg['type']}' for N={len(seeds)} seeds...[/blue]"
    )

    # v1.1: CPU verify slice (run before main battery)
    verify_cfg = cfg.get("verify_slice")
    verify_results = None
    if verify_cfg and device != "cpu":
        print(f"[yellow]Running CPU verify slice ({verify_cfg.get('n_examples', 20)} examples)...[/yellow]")
        verify_results = run_verify_slice(model, dset, cfg, battery_cfg, verify_cfg, device, seeds)
    try:
        for s in seeds:
            print(f"[cyan]Running seed {s}...[/cyan]")
            mlf.log_param(f"seed_{s}", s)
            determinism.set_seed(s)

            results = run_battery(model, dset, {**cfg, "seed": s}, battery_cfg, device)

            seed_summaries.append(results["summary"])
            per_ex_all.append(results["per_example"])
            if "impact_matrix" in results and results["impact_matrix"] is not None:
                impact_accum = (
                    results["impact_matrix"]
                    if impact_accum is None
                    else impact_accum.add(results["impact_matrix"], fill_value=0)
                )

            # v1.1: Collect impact tables
            if "head_impact_table" in results:
                impact_tables.append(results["head_impact_table"])
            if "layer_impact_table" in results:
                impact_tables.append(results["layer_impact_table"])

    except KeyboardInterrupt:
        print("[red]Interrupted by user.[/red]")
        io.save_json(
            {"error": "interrupted", "completed_seeds": len(seed_summaries)},
            out_dir / "partial_summary.json",
        )
    except Exception as e:
        print(f"[red]Error during battery run: {e}[/red]")
        io.save_json(
            {
                "error": str(e),
                "error_type": type(e).__name__,
                "completed_seeds": len(seed_summaries),
            },
            out_dir / "partial_summary.json",
        )
        raise
    finally:
        # Stop profiling
        prof.stop()
        prof.dump(out_dir / "profile.json")
        mlf.log_artifact(str(out_dir / "profile.json"))

    if not seed_summaries:
        print("[red]No seeds completed. Exiting.[/red]")
        mlf.end_run()
        return

    # Aggregate seeds
    agg_summary = {}
    for k in seed_summaries[0].keys():
        vals = [ss[k] for ss in seed_summaries]
        m, ci = stats.mean_ci(vals)
        agg_summary[k] = {"mean": m, "ci95": ci, "values": vals}

    io.save_json(agg_summary, out_dir / "summary.json")

    # Log aggregated metrics to MLflow
    for k, v in agg_summary.items():
        mlf.log_metric(f"{k}_mean", v["mean"])
        if v["ci95"]:
            mlf.log_metric(f"{k}_ci_lower", v["ci95"][0])
            mlf.log_metric(f"{k}_ci_upper", v["ci95"][1])

    # Save per-example results
    per_example = pd.concat(per_ex_all, ignore_index=True)
    per_example.to_parquet(out_dir / "per_example.parquet", index=False)

    # v1.1: Save impact tables
    if impact_tables:
        combined_impact = pd.concat(impact_tables, ignore_index=True)
        # Determine table type based on columns
        if "head" in combined_impact.columns:
            impact_path = out_dir / "head_impact.parquet"
            manifest_key = "head_impact"
        elif "layer" in combined_impact.columns:
            impact_path = out_dir / "layer_impact.parquet"
            manifest_key = "layer_impact"
        else:
            impact_path = out_dir / "impact.parquet"
            manifest_key = "impact"

        combined_impact.to_parquet(impact_path, index=False)
        mlf.log_artifact(str(impact_path))

    # Save artifacts (plots)
    if impact_accum is not None:
        impact_matrix = impact_accum / len(seeds)
        heatmap_path_png = artifact_dir / "impact_heatmap.png"

        from .viz import heatmap

        heatmap.save_heatmap(impact_matrix, heatmap_path_png)
        mlf.log_artifact(str(heatmap_path_png))
        html_path = heatmap_path_png.with_suffix(".html")
        if html_path.exists():
            mlf.log_artifact(str(html_path))

    # Finalize
    manifest = {
        "run_dir": str(run_dir),
        "metrics": "metrics/summary.json",
        "per_example": "metrics/per_example.parquet",
        "heatmap": "artifacts/impact_heatmap.png",
    }

    # v1.1: Add impact tables to manifest
    if impact_tables:
        manifest[manifest_key] = str(impact_path.relative_to(run_dir))

    # v1.1: Add verify results to manifest
    if verify_results:
        verify_path = out_dir / "verify.json"
        io.save_json(verify_results, verify_path)
        manifest["verify"] = str(verify_path.relative_to(run_dir))
        mlf.log_artifact(str(verify_path))

    io.save_json(manifest, run_dir / "manifest.json")
    mlf.end_run()
    print(f"[green]Done[/green]. Run dir: {run_dir}")


def run_verify_slice(model, dset, cfg, battery_cfg, verify_cfg, main_device, seeds):
    """Run CPU verification slice for MPS drift checking (v1.1).

    Args:
        model: Loaded model (will be moved to CPU)
        dset: Full dataset
        cfg: Main config
        battery_cfg: Battery config
        verify_cfg: Verify slice config
        main_device: Device used for main run
        seeds: List of seeds

    Returns:
        Dict with verification results
    """
    n_examples = verify_cfg.get("n_examples", 20)
    verify_device = verify_cfg.get("device", "cpu")

    # Take last n_examples (deterministic)
    verify_dset = dset[-n_examples:] if len(dset) > n_examples else dset

    # Run on main device first
    print(f"[dim]  Main device ({main_device}) pass...[/dim]")
    model.to(main_device)
    main_summaries = []
    for s in seeds:
        determinism.set_seed(s)
        results = run_battery(model, verify_dset, {**cfg, "seed": s}, battery_cfg, main_device)
        main_summaries.append(results["summary"])

    # Aggregate main results
    main_agg = {}
    for k in main_summaries[0].keys():
        vals = [ss[k] for ss in main_summaries]
        main_agg[k] = sum(vals) / len(vals)

    # Run on verify device
    print(f"[dim]  Verify device ({verify_device}) pass...[/dim]")
    model.to(verify_device)
    verify_summaries = []
    for s in seeds:
        determinism.set_seed(s)
        results = run_battery(model, verify_dset, {**cfg, "seed": s}, battery_cfg, verify_device)
        verify_summaries.append(results["summary"])

    # Aggregate verify results
    verify_agg = {}
    for k in verify_summaries[0].keys():
        vals = [ss[k] for ss in verify_summaries]
        verify_agg[k] = sum(vals) / len(vals)

    # Move model back to main device
    model.to(main_device)

    # Compare
    comparison = {
        "device_main": main_device,
        "device_verify": verify_device,
        "n_examples": len(verify_dset),
        "n_seeds": len(seeds),
        "metrics": {}
    }

    for metric in main_agg.keys():
        comparison["metrics"][metric] = {
            "main": main_agg[metric],
            "verify": verify_agg[metric],
            "abs_diff": abs(main_agg[metric] - verify_agg[metric])
        }

    return comparison


def run_battery(model, dset, cfg, battery_cfg, device):
    """Dispatch to the appropriate ablation module."""
    t = battery_cfg["type"]
    if t == "activation_patch":
        from .ablations import activation_patch

        return activation_patch.run(model, dset, cfg, battery_cfg, device)
    elif t == "heads_zero":
        from .ablations import heads_zero

        return heads_zero.run(model, dset, cfg, battery_cfg, device)
    elif t == "heads_pair_zero":
        from .ablations import heads_pair_zero

        return heads_pair_zero.run(model, dset, cfg, battery_cfg, device)
    elif t == "heads_subset_zero":
        from .ablations import heads_subset_zero

        return heads_subset_zero.run(model, dset, cfg, battery_cfg, device)
    elif t == "sae_toggle":
        from .ablations import sae_train, sae_toggle

        # This assumes SAE is pre-trained or loaded by sae_train
        sae = sae_train.train_or_load(model, dset, cfg, battery_cfg, device)
        return sae_toggle.run(model, dset, cfg, battery_cfg, sae, device)
    raise ValueError(f"Unknown battery type {t}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m lab.src.harness <config_path.json>")
        sys.exit(1)
    main(sys.argv[1])
