#!/usr/bin/env python3
"""Collate cross-model binder and sharpener top heads into a single CSV.

Scans reports/ for:
  - binder_sweep_*.json (and binder_sweep_corpus_*.json)
  - layer_entropy_scan_*.json

Outputs reports/cross_model_summary.csv with columns:
  model, binder_top (LxHy;...), sharpener_top (LxHy;...)
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import pandas as pd


def model_from_path(p: Path) -> str:
    m = re.search(r"_(gpt2[^_]*|mistral[^_.]*)", p.stem)
    return m.group(1) if m else p.stem


def top_binder(p: Path, k: int = 5) -> str:
    d = json.loads(p.read_text())
    rows = sorted(d.get("rows", []), key=lambda r: float(r.get("d_ld", 0.0)))[:k]
    return ";".join([f"L{int(r['layer'])}H{int(r['head'])}" for r in rows])


def top_sharpener(p: Path, k: int = 2) -> str:
    d = json.loads(p.read_text())
    rows = sorted(d.get("sharpener_scan", []), key=lambda r: float(r.get("d_entropy_final", 0.0)), reverse=True)[:k]
    return ";".join([f"L{int(r['layer'])}H{int(r['head'])}" for r in rows])


def main() -> None:
    reports = Path("reports")
    binder_files = list(reports.glob("binder_sweep_*.json")) + list(reports.glob("binder_sweep_corpus_*.json"))
    sharp_files = list(reports.glob("layer_entropy_scan_*.json"))

    rows = {}
    for p in binder_files:
        model = model_from_path(p)
        rows.setdefault(model, {})["binder_top"] = top_binder(p)
    for p in sharp_files:
        model = model_from_path(p)
        rows.setdefault(model, {})["sharpener_top"] = top_sharpener(p)

    data = []
    for model, vals in sorted(rows.items()):
        data.append({"model": model, **vals})
    df = pd.DataFrame(data)
    out = reports / "cross_model_summary.csv"
    out.write_text(df.to_csv(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

