"""Aim-powered helper utilities for tinyLab visualizations.

These helpers query the local Aim repo and materialize small, static
summaries (Pandas tables + PNGs) that mirror the interactive dashboards:

- Suppressor atlas: head-ranking heatmaps (H1/H5/H6).
- Geometry panel: entropy Δ vs. layer/head.
- Pythia Stage‑1A drift trajectories.

All functions are optional and safe to skip if Aim is not installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd

try:  # Aim is an optional dependency
    from aim import Repo
except ImportError:  # pragma: no cover - optional
    Repo = None  # type: ignore[assignment]


def _require_repo() -> "Repo":
    if Repo is None:  # pragma: no cover - optional
        raise RuntimeError(
            "Aim is not installed. Install it and run scripts/setup_aim.sh first."
        )
    return Repo(".")


def _parse_numeric_label(value: Any) -> Any:
    """Best-effort conversion of labels like '6.0' → 6, else leave as string."""
    if value is None:
        return None
    s = str(value)
    # Allow simple integers (with optional decimal .0)
    try:
        f = float(s)
        if f.is_integer():
            return int(f)
        return f
    except ValueError:
        return s


def load_head_ranking_df(experiment: str = "imported_head_ranking") -> pd.DataFrame:
    """Return a tidy DataFrame of head-ranking metrics from Aim.

    Columns include:
        experiment, run_hash, model, condition, hypothesis, artifact,
        layer, head, mean_value
    """
    repo = _require_repo()
    rows: List[Dict[str, Any]] = []

    for run in repo.iter_runs():
        if run.experiment != experiment:
            continue
        meta = {
            "experiment": run.experiment,
            "run_hash": run.hash,
            "model": run.get("model", None),
            "condition": run.get("condition", None),
            "hypothesis": run.get("hypothesis", None),
            "artifact": run.get("artifact", None),
        }
        coll = run.metrics()
        for metric in coll.iter():
            if metric.name != "mean_value":
                continue
            ctx = metric.context.to_dict()
            layer_raw = ctx.get("layer", ctx.get("layer_label"))
            head_raw = ctx.get("head", ctx.get("head_label"))
            if layer_raw is None or head_raw is None:
                continue
            step, value = metric.values.last()
            rows.append(
                {
                    **meta,
                    "step": step,
                    "layer": _parse_numeric_label(layer_raw),
                    "head": _parse_numeric_label(head_raw),
                    "mean_value": float(value),
                }
            )

    return pd.DataFrame.from_records(rows)


def load_entropy_scan_df(experiment: str = "imported_entropy_scan") -> pd.DataFrame:
    """Return a tidy DataFrame of entropy scan metrics from Aim.

    Focuses on d_entropy_final (ablated − baseline output entropy).
    """
    repo = _require_repo()
    rows: List[Dict[str, Any]] = []

    for run in repo.iter_runs():
        if run.experiment != experiment:
            continue
        meta = {
            "experiment": run.experiment,
            "run_hash": run.hash,
            "model": run.get("model", None),
            "condition": run.get("condition", None),
            "hypothesis": run.get("hypothesis", None),
            "artifact": run.get("artifact", None),
        }
        coll = run.metrics()
        for metric in coll.iter():
            if metric.name != "d_entropy_final":
                continue
            ctx = metric.context.to_dict()
            layer_raw = ctx.get("layer", ctx.get("layer_label"))
            head_raw = ctx.get("head", ctx.get("head_label"))
            step, value = metric.values.last()
            rows.append(
                {
                    **meta,
                    "step": step,
                    "layer": _parse_numeric_label(layer_raw),
                    "head": _parse_numeric_label(head_raw),
                    "d_entropy_final": float(value),
                }
            )

    return pd.DataFrame.from_records(rows)


def load_drift_trajectories_df(
    experiment: str = "imported_drift_trajectories",
) -> pd.DataFrame:
    """Return Pythia drift trajectories from Aim."""
    repo = _require_repo()
    rows: List[Dict[str, Any]] = []

    for run in repo.iter_runs():
        if run.experiment != experiment:
            continue
        meta = {
            "experiment": run.experiment,
            "run_hash": run.hash,
            "model": run.get("model", None),
            "condition": run.get("condition", None),
            "hypothesis": run.get("hypothesis", None),
            "artifact": run.get("artifact", None),
        }
        coll = run.metrics()
        # drift trajectories use mean_drift / mean_entropy with layer_label + step
        for metric in coll.iter():
            if metric.name not in {"mean_drift", "mean_entropy"}:
                continue
            ctx = metric.context.to_dict()
            layer_label = ctx.get("layer_label", ctx.get("layer"))
            step, value = metric.values.last()
            rows.append(
                {
                    **meta,
                    "layer_label": layer_label,
                    "metric": metric.name,
                    "step": step,
                    "value": float(value),
                }
            )

    return pd.DataFrame.from_records(rows)


def plot_head_ranking_heatmaps(
    output_dir: str | Path = "figs/aim",
) -> None:
    """Export head-ranking heatmaps grouped by model/condition/hypothesis."""
    df = load_head_ranking_df()
    if df.empty:
        return

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    for (model, cond, hyp), sub in df.groupby(["model", "condition", "hypothesis"]):
        if sub.empty:
            continue
        pivot = sub.pivot_table(
            index="layer", columns="head", values="mean_value", aggfunc="mean"
        )
        plt.figure(figsize=(8, 6))
        im = plt.imshow(pivot.values, aspect="auto", cmap="viridis")
        plt.colorbar(im, label="mean_value")
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=90)
        plt.xlabel("Head")
        plt.ylabel("Layer")
        title_parts = [str(x) for x in (model, cond, hyp) if x]
        plt.title("Head ranking: " + " / ".join(title_parts))
        fname = "_".join(str(x) for x in ("head_ranking", model, cond, hyp) if x)
        safe_name = fname.replace("/", "_").replace(" ", "-")
        plt.tight_layout()
        plt.savefig(outdir / f"{safe_name}.png", dpi=200)
        plt.close()


def plot_entropy_delta_heatmaps(
    output_dir: str | Path = "figs/aim",
) -> None:
    """Export Δ entropy heatmaps (d_entropy_final) grouped by model/condition."""
    df = load_entropy_scan_df()
    if df.empty:
        return

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    for (model, cond), sub in df.groupby(["model", "condition"]):
        if sub.empty:
            continue
        pivot = sub.pivot_table(
            index="layer", columns="head", values="d_entropy_final", aggfunc="mean"
        )
        plt.figure(figsize=(8, 6))
        im = plt.imshow(
            pivot.values, aspect="auto", cmap="coolwarm", vmin=None, vmax=None
        )
        plt.colorbar(im, label="d_entropy_final (ablated − baseline)")
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=90)
        plt.xlabel("Head")
        plt.ylabel("Layer")
        title_parts = [str(x) for x in (model, cond) if x]
        plt.title("Δ entropy (final): " + " / ".join(title_parts))
        fname = "_".join(str(x) for x in ("entropy_scan", model, cond) if x)
        safe_name = fname.replace("/", "_").replace(" ", "-")
        plt.tight_layout()
        plt.savefig(outdir / f"{safe_name}.png", dpi=200)
        plt.close()


def plot_pythia_drift_trajectories(
    output_dir: str | Path = "figs/aim",
) -> None:
    """Export Pythia drift trajectories (mean_drift vs step per layer_label)."""
    df = load_drift_trajectories_df()
    if df.empty:
        return

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    for model, sub_model in df.groupby("model"):
        if sub_model.empty:
            continue
        plt.figure(figsize=(8, 6))
        for layer_label, sub_layer in sub_model[
            sub_model["metric"] == "mean_drift"
        ].groupby("layer_label"):
            sub_layer_sorted = sub_layer.sort_values("step")
            plt.plot(
                sub_layer_sorted["step"],
                sub_layer_sorted["value"],
                label=str(layer_label),
            )
        plt.xlabel("step")
        plt.ylabel("mean_drift")
        plt.title(f"Pythia drift trajectories (model={model})")
        plt.legend(title="layer")
        safe_name = f"pythia_drift_{model}".replace("/", "_").replace(" ", "-")
        plt.tight_layout()
        plt.savefig(outdir / f"{safe_name}.png", dpi=200)
        plt.close()
