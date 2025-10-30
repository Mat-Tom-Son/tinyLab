#!/usr/bin/env python3
"""Filter a JSONL corpus to entries whose target/foil are single tokens.

This script is intended for lab maintainers who want reproducible, single-token
classification datasets for activation patching metrics. It loads an input
JSONL file, validates each example with a tokenizer (defaults to GPT-2), keeps
only those whose target and foil fields each decode to exactly one token, and
optionally writes a new split file with configurable fractions.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from transformers import AutoTokenizer


def load_tokenizer(name: str):
    """Load a tokenizer with sensible defaults for decoder-only LMs."""
    # trust_remote_code to support models like Mistral
    tokenizer = AutoTokenizer.from_pretrained(
        name,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def filter_single_token_rows(
    rows: Sequence[dict],
    tokenizer_name: str,
    target_field: str = "target",
    foil_field: str = "foil",
) -> Tuple[List[dict], List[Tuple[int, str]]]:
    """Return rows whose target/foil are single tokens for the tokenizer."""

    tokenizer = load_tokenizer(tokenizer_name)

    def is_single_token(text: str) -> bool:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens) == 1

    kept, dropped = [], []
    for idx, row in enumerate(rows):
        tgt = row.get(target_field)
        foil = row.get(foil_field)
        if tgt is None or foil is None:
            dropped.append((idx, "missing target/foil"))
            continue
        if not (is_single_token(tgt) and is_single_token(foil)):
            dropped.append((idx, f"multi-token pair ({tgt!r}, {foil!r})"))
            continue
        kept.append(row)
    return kept, dropped


def hash_file(path: Path) -> str:
    """Compute sha256 hash for reproducibility metadata."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> List[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter dataset to single-token targets/foils.")
    parser.add_argument("input", type=Path, help="Input JSONL corpus.")
    parser.add_argument("output", type=Path, help="Output JSONL path for filtered corpus.")
    parser.add_argument(
        "--target-field", default="target", help="Field containing the correct token (default: target)."
    )
    parser.add_argument(
        "--foil-field", default="foil", help="Field containing the foil token (default: foil)."
    )
    parser.add_argument(
        "--tokenizer",
        default="gpt2",
        help="Tokenizer name (HF hub id) used for validation (default: gpt2).",
    )
    parser.add_argument(
        "--split-output",
        type=Path,
        help="Optional output path for a split json (train/val/test indices).",
    )
    parser.add_argument("--train-frac", type=float, default=0.7, help="Train fraction (default: 0.7).")
    parser.add_argument("--val-frac", type=float, default=0.15, help="Validation fraction (default: 0.15).")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for split shuffling.")

    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if not rows:
        raise SystemExit(f"No examples found in {args.input}")

    filtered, dropped = filter_single_token_rows(
        rows,
        tokenizer_name=args.tokenizer,
        target_field=args.target_field,
        foil_field=args.foil_field,
    )

    if not filtered:
        raise SystemExit("All rows were filtered out. Provide more single-token examples.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(args.output, filtered)
    print(f"Kept {len(filtered)} / {len(rows)} examples -> {args.output}")
    if dropped:
        print("Dropped examples:")
        for idx, reason in dropped:
            print(f"  - row {idx}: {reason}")

    # Write split file if requested
    if args.split_output:
        total = len(filtered)
        indices = list(range(total))
        random.seed(args.seed)
        random.shuffle(indices)

        n_train = int(total * args.train_frac)
        n_val = int(total * args.val_frac)
        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        data_hash = hash_file(args.output)
        split_data = {
            "seed": args.seed,
            "data_hash": data_hash,
            "train": sorted(train_idx),
            "val": sorted(val_idx),
            "test": sorted(test_idx),
        }
        args.split_output.parent.mkdir(parents=True, exist_ok=True)
        args.split_output.write_text(json.dumps(split_data, indent=2))
        print(
            f"Wrote split to {args.split_output} "
            f"(train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})"
        )


if __name__ == "__main__":
    main()
