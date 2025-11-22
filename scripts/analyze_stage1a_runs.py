#!/usr/bin/env python3
"""
Quick analysis of Stage-1A toy training runs.

Reads:
  reports/pilot_stage1a/train/stage1a_*/config.json
  reports/pilot_stage1a/train/stage1a_*/metrics.jsonl

Computes per-run summary:
  - final loss / accuracy
  - approximate emergence step for accuracy (first step where acc >= 0.8 * final plateau)

Writes:
  reports/pilot_stage1a/stage1a_summary.csv
and prints a simple table to stdout.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Dict


BASE_DIR = Path("reports/pilot_stage1a/train")
OUT_PATH = Path("reports/pilot_stage1a/stage1a_summary.csv")


def load_metrics(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    if not BASE_DIR.exists():
        print(f"No training runs found under {BASE_DIR}")
        return

    summaries: List[Dict] = []

    for run_dir in sorted(BASE_DIR.glob("stage1a_*")):
        cfg_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.jsonl"
        if not cfg_path.exists() or not metrics_path.exists():
            continue

        cfg = json.loads(cfg_path.read_text())
        args = cfg.get("args", {})

        cond = args.get("cond")
        seed = int(args.get("seed", 0))
        omega = float(args.get("omega", 1.0))
        head = int(args.get("head", -1))
        head_kind = args.get("head_kind", cond)

        metrics = load_metrics(metrics_path)
        if not metrics:
            continue

        # Final values
        final = metrics[-1]
        final_loss = float(final["loss"])
        final_acc = float(final["accuracy"])

        # IH score (if logged)
        ih_vals = [m.get("ih_score_head") for m in metrics if "ih_score_head" in m]
        final_ih = float(ih_vals[-1]) if ih_vals else None

        # Define plateau as mean accuracy over last 10% of steps (at least 10 points)
        n = len(metrics)
        tail_k = max(10, n // 10)
        tail = metrics[-tail_k:]
        plateau_acc = sum(m["accuracy"] for m in tail) / len(tail)
        threshold = 0.8 * plateau_acc

        # Emergence step (accuracy): first step where accuracy >= threshold
        emerge_step = None
        for m in metrics:
            if m["accuracy"] >= threshold:
                emerge_step = int(m["step"])
                break

        # Emergence step (IH score): analogous rule if ih_score_head present
        emerge_step_ih = None
        if ih_vals:
            tail_ih = ih_vals[-tail_k:]
            plateau_ih = sum(tail_ih) / len(tail_ih)
            threshold_ih = 0.8 * plateau_ih
            for m in metrics:
                val = m.get("ih_score_head")
                if val is not None and val >= threshold_ih:
                    emerge_step_ih = int(m["step"])
                    break
        else:
            plateau_ih = None
            threshold_ih = None

        summaries.append(
            {
                "run_dir": str(run_dir),
                "cond": cond,
                "head_kind": head_kind,
                "head": head,
                "omega": omega,
                "seed": seed,
                "final_loss": final_loss,
                "final_acc": final_acc,
                 "final_ih_score": final_ih,
                "plateau_acc": plateau_acc,
                "threshold_acc": threshold,
                "emerge_step": emerge_step,
                "plateau_ih": plateau_ih,
                "threshold_ih": threshold_ih,
                "emerge_step_ih": emerge_step_ih,
                "n_steps": int(metrics[-1]["step"]),
            }
        )

    if not summaries:
        print("No completed runs with metrics found.")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cond",
        "head_kind",
        "head",
        "omega",
        "seed",
        "final_loss",
        "final_acc",
        "final_ih_score",
        "plateau_acc",
        "threshold_acc",
        "emerge_step",
        "plateau_ih",
        "threshold_ih",
        "emerge_step_ih",
        "n_steps",
        "run_dir",
    ]
    with OUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)

    print(f"Wrote summary for {len(summaries)} runs to {OUT_PATH}\n")

    # Also print a compact table
    summaries.sort(key=lambda r: (r["cond"], r["omega"], r["seed"]))
    for row in summaries:
        ih_part = (
            f", ih_final={row['final_ih_score']:.3f} ih_step={row['emerge_step_ih']}"
            if row["final_ih_score"] is not None
            else ""
        )
        print(
            f"{row['cond']:10s} omega={row['omega']:3.1f} seed={row['seed']:d} "
            f"final_acc={row['final_acc']:.3f} emerge_step={row['emerge_step']}{ih_part}"
        )


if __name__ == "__main__":
    main()
