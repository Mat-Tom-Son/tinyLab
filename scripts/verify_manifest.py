"""Verify RESULTS_MANIFEST.json integrity and referenced artifacts.

Usage:
  python3 scripts/verify_manifest.py              # Verify existing manifest
  python3 scripts/verify_manifest.py --check-only # Non-fatal check
  python3 scripts/verify_manifest.py --fix        # Print suggestions to fix
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


MANIFEST = Path("reports/RESULTS_MANIFEST.json")
RUNS_ROOT = Path("lab/runs")


def load_manifest() -> dict:
    if not MANIFEST.exists():
        return {}
    return json.loads(MANIFEST.read_text())


def check_manifest_structure(data: dict) -> list[str]:
    issues: list[str] = []
    if not data:
        issues.append(f"missing manifest: {MANIFEST}")
        return issues
    # Expect top-level model keys (mistral, gpt2_medium, ...)
    for model, entry in data.items():
        if not isinstance(entry, dict):
            issues.append(f"model '{model}' entry is not an object")
            continue
        # Check summary + trio
        for key in ("summary_table", "trio_percentiles"):
            path = entry.get(key)
            if path and not Path(path).exists():
                issues.append(f"missing {model}/{key}: {path}")
        # Check head rankings
        hr = entry.get("head_rankings", {})
        for probe, paths in hr.items():
            for subkey, p in (paths or {}).items():
                if p and not Path(p).exists():
                    issues.append(f"missing {model}/{probe}/{subkey}: {p}")
        # OV report list
        for p in entry.get("ov_reports", []) or []:
            if p and not Path(p).exists():
                issues.append(f"missing {model}/ov_report: {p}")
    return issues


def check_runs() -> list[str]:
    issues: list[str] = []
    for run_dir in RUNS_ROOT.iterdir():
        if not run_dir.is_dir():
            continue
        if "_archive" in run_dir.parts or run_dir.name.startswith("_"):
            continue
        if not run_dir.name.startswith("h1_"):
            continue
        # Required artifacts per run
        required = (
            run_dir / "metrics" / "summary.json",
            run_dir / "config.json",
            run_dir / "config_hash.txt",
            run_dir / "provenance.json",
        )
        for p in required:
            if not p.exists():
                issues.append(f"missing in {run_dir}: {p.relative_to(run_dir)}")
    return issues


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true", help="Return non-zero on issues, no fixes.")
    ap.add_argument("--fix", action="store_true", help="Suggest make targets to regenerate outputs.")
    args = ap.parse_args()

    print("== Manifest Verification ==")
    data = load_manifest()
    issues = check_manifest_structure(data)
    run_issues = check_runs()
    issues.extend(run_issues)

    if issues:
        print("Found issues:")
        for i in issues:
            print(" -", i)
        if args.fix:
            print("\nSuggestions:")
            print(" - make postprocess")
            print(" - python3 -m lab.analysis.build_manifest")
        return 1 if args.check_only else 0
    else:
        print("OK: manifest and run artifacts look consistent.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
