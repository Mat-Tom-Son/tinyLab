#!/usr/bin/env python3
"""Import historical tinyLab results into Aim.

Usage:
    python scripts/import_to_aim.py
    python scripts/import_to_aim.py --reports-dir reports/
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from aim import Run
import pandas as pd


def parse_filename(path: Path) -> dict[str, str]:
    """Extract simple metadata from a results filename."""
    stem = path.stem
    metadata: dict[str, str] = {}

    # Model
    if "gpt2m" in stem:
        metadata["model"] = "gpt2-medium"
    elif "gpt2l" in stem:
        metadata["model"] = "gpt2-large"
    elif "gpt2" in stem:
        metadata["model"] = "gpt2"
    elif "mistral" in stem:
        metadata["model"] = "mistral-7b"
    elif "pythia" in stem:
        metadata["model"] = "pythia"

    # Condition
    for cond in [
        "facts",
        "cf",
        "logic",
        "neg",
        "counterfactual",
        "negation",
        "logical",
    ]:
        if cond in stem:
            metadata["condition"] = cond
            break

    # Hypothesis marker (H1, H5, etc.)
    match = re.search(r"h(\d+)", stem)
    if match:
        metadata["hypothesis"] = f"H{match.group(1)}"

    # Artifact type (used for Aim experiment naming)
    lower = stem.lower()
    if "summary_table" in lower:
        metadata["artifact"] = "summary"
    elif "head_ranking" in lower:
        metadata["artifact"] = "head_ranking"
    elif "top12" in lower or "bottom6" in lower:
        metadata["artifact"] = "head_slice"
    elif "binder_sweep" in lower:
        metadata["artifact"] = "binder_sweep"
    elif "layer_entropy_scan" in lower:
        metadata["artifact"] = "entropy_scan"
    elif "layer_pca_rank" in lower:
        metadata["artifact"] = "pca_rank"
    elif "drift_trajectories" in lower:
        metadata["artifact"] = "drift_trajectories"
    elif "cross_model_summary" in lower:
        metadata["artifact"] = "cross_model_summary"
    else:
        metadata.setdefault("artifact", "metrics")

    return metadata


def import_head_rankings(csv_path: Path, repo_path: str | None = None) -> None:
    """Import a head-ranking CSV into an Aim run."""
    meta = parse_filename(csv_path)

    hyp = (meta.get("hypothesis") or "").lower()
    artifact = meta.get("artifact", "metrics")
    if hyp:
        experiment = f"{hyp}_{artifact}"
    else:
        experiment = f"imported_{artifact}"

    run = Run(
        repo=repo_path,
        experiment=experiment,
    )

    for tag in ["imported", "historical", *meta.values()]:
        try:
            run.add_tag(str(tag))
        except Exception:
            pass

    run["source_file"] = str(csv_path)
    run["imported"] = True
    for k, v in meta.items():
        run[k] = v

    df = pd.read_csv(csv_path)

    for idx, row in df.iterrows():
        raw_layer = row.get("layer", row.get("Layer", 0))
        raw_head = row.get("head", row.get("Head", idx))

        context: dict[str, object] = {}

        # Some tables (e.g., Pythia drift trajectories) use string labels
        # like "resid0" / "resid_mid" instead of numeric layer indices.
        def _maybe_int(value):
            s = str(value)
            return int(s) if s.lstrip("-").isdigit() else None

        layer = _maybe_int(raw_layer)
        head = _maybe_int(raw_head)

        if layer is not None:
            context["layer"] = layer
        else:
            context["layer_label"] = str(raw_layer)

        if head is not None:
            context["head"] = head
        else:
            context["head_label"] = str(raw_head)

        for col in df.columns:
            if col.lower() in {"layer", "head", "rank"}:
                continue
            try:
                value = float(row[col])
            except Exception:
                continue
            run.track(value, name=col.lower(), step=0, context=context)

    run.close()
    print(f"✓ Imported {csv_path.name}")


def import_json_metrics(json_path: Path, repo_path: str | None = None) -> None:
    """Import a generic JSON metrics file into an Aim run."""
    meta = parse_filename(json_path)

    hyp = (meta.get("hypothesis") or "").lower()
    artifact = meta.get("artifact", "metrics")
    if hyp:
        experiment = f"{hyp}_{artifact}"
    else:
        experiment = f"imported_{artifact}"

    run = Run(
        repo=repo_path,
        experiment=experiment,
    )

    for tag in ["imported", "historical", *meta.values()]:
        try:
            run.add_tag(str(tag))
        except Exception:
            pass

    run["source_file"] = str(json_path)
    run["imported"] = True
    for k, v in meta.items():
        run[k] = v

    with json_path.open() as f:
        data = json.load(f)

    def log_nested(obj, prefix: str = "") -> None:
        """Recursively log nested dictionaries/lists."""
        from aim import Distribution

        if isinstance(obj, dict):
            for key, val in obj.items():
                new_prefix = f"{prefix}/{key}" if prefix else key
                log_nested(val, new_prefix)
        elif isinstance(obj, (int, float)):
            run.track(float(obj), name=prefix, step=0)
        elif isinstance(obj, list) and all(isinstance(x, (int, float)) for x in obj):
            run.track(Distribution(obj), name=prefix, step=0)

    log_nested(data)
    run.close()
    print(f"✓ Imported {json_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import tinyLab results into Aim.")
    parser.add_argument(
        "--reports-dir",
        default="reports/",
        help="Path to reports directory (default: reports/)",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="Optional path to .aim directory (defaults to project root)",
    )
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)

    if not reports_dir.exists():
        raise SystemExit(f"Reports directory not found: {reports_dir}")

    print("Importing CSV files...")
    for csv_file in reports_dir.glob("**/*.csv"):
        try:
            import_head_rankings(csv_file, repo_path=args.repo)
        except Exception as exc:
            print(f"✗ Failed to import {csv_file.name}: {exc}")

    print("\nImporting JSON files...")
    for json_file in reports_dir.glob("**/*.json"):
        if "manifest" in json_file.name.lower():
            continue
        try:
            import_json_metrics(json_file, repo_path=args.repo)
        except Exception as exc:
            print(f"✗ Failed to import {json_file.name}: {exc}")

    print("\n✓ Import complete! Launch UI with: aim up")


if __name__ == "__main__":
    main()
