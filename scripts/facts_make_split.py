#!/usr/bin/env python3
"""Generate train/val/test splits for a dataset with hash validation."""
import json
import hashlib
import argparse
from pathlib import Path
import random


def hash_file(path):
    """Hash a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_split(corpus_id, train_frac=0.7, val_frac=0.15, seed=13):
    """Create a stratified split file.

    Args:
        corpus_id: Dataset name (e.g., "facts_v1")
        train_frac: Fraction for training
        val_frac: Fraction for validation
        seed: Random seed
    """
    root = Path("lab/data")
    corpus_path = root / "corpora" / f"{corpus_id}.jsonl"

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    # Load all examples
    with open(corpus_path) as f:
        lines = [line.strip() for line in f if line.strip()]

    n = len(lines)
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    # Split indices
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    # Hash corpus
    data_hash = hash_file(corpus_path)

    split_data = {
        "seed": seed,
        "data_hash": data_hash,
        "train": sorted(train_idx),
        "val": sorted(val_idx),
        "test": sorted(test_idx),
    }

    # Save split
    split_path = root / "splits" / f"{corpus_id}.split.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as f:
        json.dump(split_data, f, indent=2)

    print(f"Created split: {split_path}")
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print(f"  Data hash: {data_hash[:12]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset splits")
    parser.add_argument("corpus_id", help="Dataset ID (e.g., facts_v1)")
    parser.add_argument("--train", type=float, default=0.7, help="Train fraction")
    parser.add_argument("--val", type=float, default=0.15, help="Val fraction")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")

    args = parser.parse_args()
    make_split(args.corpus_id, args.train, args.val, args.seed)
