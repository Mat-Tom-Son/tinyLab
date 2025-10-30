"""Cluster top/bottom OV tokens by token string to summarise head semantics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def gather_tokens(report_path: Path, limit: int | None = None):
    data = json.loads(report_path.read_text())
    head_entries = data["heads"]
    tokens = []
    for head in head_entries:
        for entry in head["tokens"]["top"][:limit]:
            tokens.append(("top", entry["token"], entry["logit"]))
        for entry in head["tokens"]["bottom"][:limit]:
            tokens.append(("bottom", entry["token"], entry["logit"]))
    return tokens


def summarise(report_paths: list[Path], limit: int | None = None, top_n: int = 40):
    to_counter = Counter()
    bottom_counter = Counter()
    for path in report_paths:
        tokens = gather_tokens(path, limit)
        for pos, tok, _ in tokens:
            if pos == "top":
                to_counter[tok] += 1
            else:
                bottom_counter[tok] += 1
    top_summary = to_counter.most_common(top_n)
    bottom_summary = bottom_counter.most_common(top_n)
    return {
        "top": [
            {"token": tok, "count": count} for tok, count in top_summary
        ],
        "bottom": [
            {"token": tok, "count": count} for tok, count in bottom_summary
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", help="Path(s) to ov_report json files.")
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--limit", type=int, help="Only use the first LIMIT tokens per head list.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report_paths = [Path(p) for p in args.reports]
    summary = summarise(report_paths, limit=args.limit, top_n=args.top_n)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
