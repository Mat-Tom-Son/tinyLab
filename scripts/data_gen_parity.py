#!/usr/bin/env python3
"""
Generate parity/FSA dataset for grokking experiments.

Task: Given a binary string, determine if it has odd or even number of 1s.

This is a compositional task that requires:
1. Temporal reasoning (can't solve by lookup)
2. State tracking (must count/flip across tokens)
3. Circuit formation (information must flow token-to-token)

This is the PERFECT task for testing developmental control because:
- Cannot be memorized (exponential combinations)
- Has clear phase transition (when counting circuit emerges)
- Requires actual circuit assembly (not just embedding lookup)

Usage:
    python scripts/data_gen_parity.py --output-dir data_parity
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict


def generate_parity_dataset(
    min_length: int = 4,
    max_length: int = 16,
    n_train: int = 10000,
    n_test: int = 2000,
    seed: int = 42,
) -> tuple[List[Dict], List[Dict]]:
    """
    Generate parity checking dataset.

    Task: Count 1s in binary string, output ODD or EVEN

    Args:
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        n_train: Number of training examples
        n_test: Number of test examples
        seed: Random seed

    Returns:
        (train_data, test_data) where each item is:
            {
                'sequence': str (e.g., "1011001"),
                'target': str ("ODD" or "EVEN"),
                'length': int,
                'n_ones': int
            }
    """
    random.seed(seed)

    def count_ones(seq: str) -> int:
        return seq.count('1')

    def is_odd(n: int) -> bool:
        return n % 2 == 1

    def generate_example() -> Dict:
        """Generate a single random example."""
        length = random.randint(min_length, max_length)
        sequence = ''.join(random.choice('01') for _ in range(length))
        n_ones = count_ones(sequence)
        parity = "ODD" if is_odd(n_ones) else "EVEN"

        return {
            'sequence': sequence,
            'target': parity,
            'length': length,
            'n_ones': n_ones,
        }

    # Generate training data
    train_data = []
    seen_train = set()

    while len(train_data) < n_train:
        example = generate_example()
        seq = example['sequence']

        # Avoid duplicates
        if seq not in seen_train:
            seen_train.add(seq)
            train_data.append(example)

    # Generate test data (ensuring no overlap with train)
    test_data = []
    seen_test = set()

    while len(test_data) < n_test:
        example = generate_example()
        seq = example['sequence']

        # Ensure unique and not in training set
        if seq not in seen_train and seq not in seen_test:
            seen_test.add(seq)
            test_data.append(example)

    return train_data, test_data


def save_jsonl(data: List[Dict], path: Path) -> None:
    """Save data in JSONL format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f"Saved {len(data)} examples to {path}")


def print_stats(data: List[Dict], label: str):
    """Print dataset statistics."""
    lengths = [ex['length'] for ex in data]
    odd_count = sum(1 for ex in data if ex['target'] == 'ODD')
    even_count = len(data) - odd_count

    print(f"\n{label} Statistics:")
    print(f"  Total examples: {len(data)}")
    print(f"  ODD: {odd_count} ({100*odd_count/len(data):.1f}%)")
    print(f"  EVEN: {even_count} ({100*even_count/len(data):.1f}%)")
    print(f"  Length range: {min(lengths)}-{max(lengths)}")
    print(f"  Mean length: {sum(lengths)/len(lengths):.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate parity checking dataset for grokking experiments"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=4,
        help="Minimum sequence length (default: 4)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=16,
        help="Maximum sequence length (default: 16)",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=10000,
        help="Number of training examples (default: 10000)",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=2000,
        help="Number of test examples (default: 2000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data_parity",
        help="Output directory (default: data_parity/)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Generating Parity Dataset")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Sequence length: {args.min_length}-{args.max_length}")
    print(f"  Training examples: {args.n_train}")
    print(f"  Test examples: {args.n_test}")
    print(f"  Random seed: {args.seed}")

    # Generate data
    train_data, test_data = generate_parity_dataset(
        min_length=args.min_length,
        max_length=args.max_length,
        n_train=args.n_train,
        n_test=args.n_test,
        seed=args.seed,
    )

    # Print statistics
    print_stats(train_data, "Training")
    print_stats(test_data, "Test")

    # Save
    output_dir = Path(args.output_dir)
    train_path = output_dir / "parity_train.jsonl"
    test_path = output_dir / "parity_test.jsonl"

    save_jsonl(train_data, train_path)
    save_jsonl(test_data, test_path)

    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)

    # Show examples
    print("\nExample sequences:")
    for i in range(min(5, len(train_data))):
        ex = train_data[i]
        print(f"  '{ex['sequence']}' -> {ex['target']} (length={ex['length']}, ones={ex['n_ones']})")


if __name__ == "__main__":
    main()
