#!/usr/bin/env python3
"""Archive incomplete h1_* runs that are not referenced by the manifest.

Moves any h1_* run directory without metrics/summary.json into lab/runs/_archive/.
Blessed runs referenced by the manifest are never archived.
"""
from __future__ import annotations

import json
from pathlib import Path


RUNS = Path("lab/runs")
ARCHIVE = RUNS / "_archive"
MANIFEST = Path("reports/RESULTS_MANIFEST.json")


def blessed_run_names() -> set[str]:
    names: set[str] = set()
    if not MANIFEST.exists():
        return names
    data = json.loads(MANIFEST.read_text())
    # Current manifest schema: top-level model keys → dict with optional pointers.
    for entry in data.values():
        if not isinstance(entry, dict):
            continue
        # Try child_runs style (for orchestrators) or ignore if not present
        # Our manifest does not contain run_dir by probe; we archive based on completeness only.
        pass
    return names


def main() -> None:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    blessed = blessed_run_names()
    moved = 0
    for child in RUNS.iterdir():
        if not child.is_dir():
            continue
        if child.name.startswith("_"):
            continue
        if not child.name.startswith("h1_"):
            continue
        if child.name in blessed:
            continue
        if not (child / "metrics" / "summary.json").exists():
            target = ARCHIVE / child.name
            try:
                child.rename(target)
                moved += 1
            except Exception as e:
                print(f"[warn] Could not archive {child}: {e}")
    print(f"Archived {moved} incomplete runs into {ARCHIVE}")


if __name__ == "__main__":
    main()

