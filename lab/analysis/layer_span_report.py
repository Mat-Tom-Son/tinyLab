"""Generate layer-span summaries comparing primary runs against controls.

This script consumes one or more ``layer_matrix.parquet`` files and aggregates
per-condition statistics such as the top-impact layer and the minimal contiguous
span of layers required to explain 70% of the total impact. The output is
written to ``reports/`` by default as both CSV and JSON summaries so that the
results can be versioned alongside other artifacts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_METRIC = "logit_diff"
DEFAULT_THRESHOLD = 0.7


@dataclass
class MatrixSpec:
    tag: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        action="append",
        metavar="TAG=PATH",
        required=True,
        help="Associate a label with a layer_matrix parquet file (e.g. primary=.../layer_matrix.parquet).",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help=f"Metric to analyse (default: {DEFAULT_METRIC}).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Cumulative impact threshold for span calculation (default: {DEFAULT_THRESHOLD:.2f}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for summary outputs.",
    )
    return parser.parse_args()


def parse_matrix_specs(values: Iterable[str]) -> List[MatrixSpec]:
    specs: List[MatrixSpec] = []
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected TAG=PATH, got '{item}'")
        tag, path = item.split("=", 1)
        specs.append(MatrixSpec(tag=tag.strip(), path=Path(path.strip())))
    return specs


def load_matrices(specs: Iterable[MatrixSpec], metric: str) -> Dict[str, pd.DataFrame]:
    loaded: Dict[str, pd.DataFrame] = {}
    for spec in specs:
        path_str = str(spec.path)
        if any(ch in path_str for ch in ["*", "?", "["]):
            matches = sorted(
                Path().glob(path_str),
                key=lambda p: p.stat().st_mtime,
            )
            if not matches:
                raise FileNotFoundError(f"No files matched pattern '{path_str}' for tag '{spec.tag}'")
            path = matches[-1]
        else:
            path = spec.path
        if path.suffix != ".parquet":
            raise ValueError(f"Expected parquet file for tag '{spec.tag}', got {path}")
        df = pd.read_parquet(path)
        df = df[df["metric"] == metric].dropna(subset=["value"]).copy()
        if df.empty:
            raise ValueError(f"No data for metric '{metric}' in {spec.path}")
        loaded[spec.tag] = df
    return loaded


def minimal_span(values: np.ndarray, threshold: float) -> Tuple[int, int, int]:
    """Return (span_length, start_idx, end_idx) for minimal contiguous window meeting threshold."""
    positives = np.maximum(values, 0.0)
    total = positives.sum()
    if total <= 0:
        return 0, 0, 0
    target = threshold * total
    n = len(values)
    best = (n + 1, 0, 0)
    for start in range(n):
        acc = 0.0
        for end in range(start, n):
            acc += positives[end]
            if acc >= target:
                length = end - start + 1
                if length < best[0]:
                    best = (length, start, end)
                break
    if best[0] == n + 1:
        return n, 0, n - 1
    return best


def build_summary(
    tables: Dict[str, pd.DataFrame],
    metric: str,
    threshold: float,
) -> pd.DataFrame:
    records: List[Dict] = []
    for tag, df in tables.items():
        for condition, cond_df in df.groupby("condition"):
            layers = sorted(cond_df["layer"].unique())
            mean_by_layer = (
                cond_df.groupby("layer")["value"]
                .mean()
                .reindex(layers)
                .fillna(0.0)
            )
            values = mean_by_layer.to_numpy(dtype=float)
            if values.size == 0:
                continue
            top_layer = int(mean_by_layer.idxmax())
            span_len, start_idx, end_idx = minimal_span(values, threshold)
            n_layers = len(values)
            records.append(
                {
                    "tag": tag,
                    "condition": condition,
                    "metric": metric,
                    "top_layer": top_layer,
                    "span_layers": int(span_len),
                    "span_fraction": float(span_len / n_layers if n_layers else 0.0),
                    "span_start_layer": int(layers[start_idx]),
                    "span_end_layer": int(layers[end_idx]),
                    "n_layers": n_layers,
                    "total_impact": float(np.maximum(values, 0.0).sum()),
                }
            )
    return pd.DataFrame.from_records(records)


def write_outputs(df: pd.DataFrame, output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "layer_span_summary.csv"
    json_path = output_dir / "layer_span_summary.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(df.to_json(orient="records", indent=2))
    return {"csv": str(csv_path), "json": str(json_path)}


def main() -> None:
    args = parse_args()
    specs = parse_matrix_specs(args.matrix)
    tables = load_matrices(specs, args.metric)
    summary = build_summary(tables, args.metric, args.threshold)
    artifacts = write_outputs(summary, args.output_dir)
    info = {
        "metric": args.metric,
        "threshold": args.threshold,
        "records": len(summary),
        "artifacts": artifacts,
    }
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
