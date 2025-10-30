"""Token frequency analysis and balancing for Tiny Ablation Lab corpora.

Given one or more JSONL corpora this script computes token frequency
distributions (over the ``target`` field by default) using a Hugging Face
tokenizer and can optionally emit a frequency-balanced subset for downstream
invariance checks.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from transformers import AutoTokenizer


@dataclass
class DatasetSummary:
    path: Path
    counts: Counter
    total: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="+",
        help="Paths to corpus JSONL files.",
    )
    parser.add_argument(
        "--field",
        default="target",
        help="JSON field to analyse (default: target).",
    )
    parser.add_argument(
        "--tokenizer",
        default="gpt2",
        help="Tokenizer name or path (default: gpt2).",
    )
    parser.add_argument(
        "--max-per-token",
        type=int,
        help="Optional cap on the number of examples per token when rebalancing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="If set, only emit token frequency statistics; no balanced files are created.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lab/data/corpora"),
        help="Directory to write balanced corpora (default: lab/data/corpora).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for frequency reports (default: reports).",
    )
    return parser.parse_args()


def load_dataset(path: Path, field: str) -> Tuple[List[Dict], List[str]]:
    records: List[Dict] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            if field not in obj:
                raise KeyError(f"Field '{field}' missing in record from {path}")
            records.append(obj)
    return records, [rec[field] for rec in records]


def compute_counts(
    dataset_paths: Iterable[Path],
    field: str,
    tokenizer,
) -> List[DatasetSummary]:
    summaries: List[DatasetSummary] = []
    for path in dataset_paths:
        records, values = load_dataset(path, field)
        counts = Counter()
        for value in values:
            token_ids = tokenizer.encode(value, add_special_tokens=False)
            if not token_ids:
                raise ValueError(f"Value '{value}' from {path} produced zero tokens.")
            if len(token_ids) != 1:
                raise ValueError(
                    f"Non single-token value encountered in {path}: '{value}' -> {token_ids}"
                )
            counts[token_ids[0]] += 1
        summaries.append(DatasetSummary(path=path, counts=counts, total=len(values)))
    return summaries


def write_report(
    summaries: Iterable[DatasetSummary],
    tokenizer,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Dict] = {}
    for summary in summaries:
        entries = []
        for token_id, count in summary.counts.most_common():
            entries.append(
                {
                    "token_id": int(token_id),
                    "token": tokenizer.decode([token_id]),
                    "count": int(count),
                }
            )
        report[summary.path.name] = {
            "total": summary.total,
            "distinct_tokens": len(summary.counts),
            "counts": entries,
        }
    path = output_dir / "token_frequency_summary.json"
    path.write_text(json.dumps(report, indent=2))
    return path


def rebalance_corpus(
    records: List[Dict],
    field: str,
    tokenizer,
    max_per_token: int | None,
    seed: int,
) -> List[Dict]:
    token_to_examples: Dict[int, List[Dict]] = defaultdict(list)
    for record in records:
        token_ids = tokenizer.encode(record[field], add_special_tokens=False)
        token_id = token_ids[0]
        token_to_examples[token_id].append(record)

    rng = random.Random(seed)
    balanced: List[Dict] = []
    per_token_cap = max_per_token or min(len(v) for v in token_to_examples.values())
    for token_id, examples in token_to_examples.items():
        if len(examples) > per_token_cap:
            rng.shuffle(examples)
        balanced.extend(examples[:per_token_cap])
    rng.shuffle(balanced)
    return balanced


def write_balanced(records: List[Dict], prototype: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{prototype.stem}_balanced.jsonl"
    with output_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return output_path


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    dataset_paths = [Path(p) for p in args.datasets]

    records_cache: Dict[Path, List[Dict]] = {}
    summaries = compute_counts(dataset_paths, args.field, tokenizer)
    report_path = write_report(summaries, tokenizer, args.report_dir)

    outputs: Dict[str, str] = {"report": str(report_path)}
    if not args.stats_only:
        for summary in summaries:
            records, _ = load_dataset(summary.path, args.field)
            records_cache[summary.path] = records
            balanced = rebalance_corpus(
                records, args.field, tokenizer, args.max_per_token, args.seed
            )
            out_path = write_balanced(balanced, summary.path, args.output_dir)
            outputs[f"balanced::{summary.path.name}"] = str(out_path)

    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()

