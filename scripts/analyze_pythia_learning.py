#!/usr/bin/env python3
"""Aggregate head-level effects across Pythia checkpoints.

Searches lab/runs for child runs matching h1_pythia_checkpoint_XXXXXX_<tag>_*
and concatenates their head_impact.parquet into a single long-form table with
checkpoint_step and condition columns, saving to reports/.
"""
from __future__ import annotations

import re
from pathlib import Path
import pandas as pd


RUNS = Path("lab/runs")
OUT = Path("reports/pythia_head_trajectory.parquet")


def parse_step_and_tag(run_dir: Path) -> tuple[int, str] | None:
    # Expected child run name: h1_pythia_checkpoint_000000_<tag>_<hash>
    m = re.match(r"h1_pythia_checkpoint_(\d{6})_([a-z]+)_", run_dir.name)
    if not m:
        return None
    step = int(m.group(1))
    tag = m.group(2)
    return step, tag


def main() -> None:
    records = []
    for child in RUNS.iterdir():
        parsed = parse_step_and_tag(child)
        if not parsed:
            continue
        step, tag = parsed
        parquet = child / "metrics" / "head_impact.parquet"
        if not parquet.exists():
            continue
        df = pd.read_parquet(parquet)
        df = df.assign(checkpoint_step=step, condition=tag)
        records.append(df)

    if not records:
        print("No Pythia child runs with head_impact.parquet found.")
        return

    out = pd.concat(records, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"Wrote {OUT} ({len(out)} rows)")


if __name__ == "__main__":
    main()

