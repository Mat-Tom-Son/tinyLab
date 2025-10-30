#!/usr/bin/env python3
"""Batch-build tokenizer-specific single-token corpora and splits."""

import argparse
import json
from pathlib import Path
from typing import List

from curate_single_token_dataset import (
    filter_single_token_rows,
    hash_file,
    load_jsonl,
    save_jsonl,
)


def write_split(indices: List[int], output_path: Path, data_hash: str, seed: int, train_frac: float, val_frac: float):
    import random

    random.seed(seed)
    shuffled = indices[:]
    random.shuffle(shuffled)

    total = len(shuffled)
    n_train = int(total * train_frac)
    n_val = int(total * val_frac)
    train_idx = shuffled[:n_train]
    val_idx = shuffled[n_train : n_train + n_val]
    test_idx = shuffled[n_train + n_val :]

    split = {
        "seed": seed,
        "data_hash": data_hash,
        "train": sorted(train_idx),
        "val": sorted(val_idx),
        "test": sorted(test_idx),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(split, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tokenizer-specific corpora variants.")
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Tokenizer name (HF repo id).",
    )
    parser.add_argument(
        "--suffix",
        required=True,
        help="Suffix appended to dataset ids (e.g., mistral).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "facts_single_token_v1",
            "negation_single_token_v1",
            "counterfactual_single_token_v1",
            "logical_single_token_v1",
        ],
        help="Base dataset IDs to process (without .jsonl).",
    )
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=13)

    args = parser.parse_args()

    base_dir = Path("lab/data/corpora")
    split_dir = Path("lab/data/splits")

    for dataset_id in args.datasets:
        base_path = base_dir / f"{dataset_id}.jsonl"
        if not base_path.exists():
            print(f"[warn] skipping {dataset_id}: {base_path} not found")
            continue

        rows = load_jsonl(base_path)
        filtered, dropped = filter_single_token_rows(
            rows,
            tokenizer_name=args.tokenizer,
        )

        new_id = f"{dataset_id}_{args.suffix}"
        out_path = base_dir / f"{new_id}.jsonl"
        save_jsonl(out_path, filtered)
        print(f"[info] {dataset_id}: kept {len(filtered)}/{len(rows)} → {out_path}")
        if dropped:
            print(f"       dropped {len(dropped)} examples (first 5 shown):")
            for idx, reason in dropped[:5]:
                print(f"         - row {idx}: {reason}")

        data_hash = hash_file(out_path)
        indices = list(range(len(filtered)))
        split_path = split_dir / f"{new_id}.split.json"
        write_split(indices, split_path, data_hash, args.seed, args.train_frac, args.val_frac)
        print(f"       wrote split → {split_path}")


if __name__ == "__main__":
    main()

