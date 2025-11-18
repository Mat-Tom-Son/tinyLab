#!/usr/bin/env python3
"""Generate Task B (weekday modular addition) data for Stage 1A.

Task definition (from the preregistration):

    [Day1] [+] [Offset] [=] -> predict [Day2]

Example:

    "Monday + 3 days = Thursday"

This script writes a JSONL file with fields:

    - input:  the prefix up to and including '=' (e.g. "Monday + 3 days =")
    - target: the correct day token to predict (e.g. " Thursday")
    - day1:   the starting day as a string (e.g. "Monday")
    - offset: the integer offset in days (e.g. 3)
    - day2:   the ground-truth resulting day (e.g. "Thursday")

The generated file can be used directly for training or as a source for
building train/val/test splits in a downstream script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np


DAYS: List[str] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def weekday_add(day: str, offset: int) -> str:
    """Compute Day2 = Day1 + offset (mod 7)."""
    if day not in DAYS:
        raise ValueError(f"Unknown day: {day}")
    idx = DAYS.index(day)
    return DAYS[(idx + offset) % len(DAYS)]


def generate_examples(
    n_examples: int,
    min_offset: int,
    max_offset: int,
    seed: int,
) -> List[dict]:
    """Generate Task B examples as dicts."""
    if min_offset < 0 or max_offset < 0:
        raise ValueError("Offsets must be non-negative integers.")
    if max_offset < min_offset:
        raise ValueError("max_offset must be >= min_offset.")

    rng = np.random.default_rng(seed)
    examples: List[dict] = []

    for _ in range(n_examples):
        day1 = str(rng.choice(DAYS))
        offset = int(rng.integers(min_offset, max_offset + 1))
        day2 = weekday_add(day1, offset)

        # Simple English phrasing; ensure consistent spacing so the target
        # appears as a separate token under typical BPEs.
        day_word = "day" if offset == 1 else "days"
        prefix = f"{day1} + {offset} {day_word} ="
        target = f" {day2}"

        examples.append(
            {
                "input": prefix,
                "target": target,
                "day1": day1,
                "offset": offset,
                "day2": day2,
            }
        )

    return examples


def write_jsonl(examples: List[dict], out_path: Path) -> None:
    """Write examples to a JSONL file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(examples)} examples to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Task B (weekday modular addition) JSONL data."
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=10000,
        help="Number of examples to generate.",
    )
    parser.add_argument(
        "--min-offset",
        type=int,
        default=0,
        help="Minimum offset in days (inclusive).",
    )
    parser.add_argument(
        "--max-offset",
        type=int,
        default=6,
        help="Maximum offset in days (inclusive).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="lab/data/task_b_weekdays.jsonl",
        help="Output JSONL path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    examples = generate_examples(
        n_examples=args.n_examples,
        min_offset=args.min_offset,
        max_offset=args.max_offset,
        seed=args.seed,
    )
    write_jsonl(examples, Path(args.out_path))


if __name__ == "__main__":
    main()
