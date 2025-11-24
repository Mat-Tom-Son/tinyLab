#!/usr/bin/env python3
"""
Generate modular arithmetic dataset for grokking experiments.

Generates (a + b) mod p = c examples where p=113 (prime).
This task is known to exhibit grokking behavior (Power et al. 2022).

Usage:
    python scripts/data_gen_modular.py --output data/modular_p113.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict


def generate_modular_arithmetic(
    modulus: int = 113,
    train_fraction: float = 0.9,
    operator: str = "plus",
    seed: int = 42,
) -> tuple[List[Dict], List[Dict]]:
    """
    Generate exhaustive modular arithmetic dataset.

    Args:
        modulus: Prime modulus (p=113 is standard in grokking literature)
        train_fraction: Fraction of pairs to use for training
        operator: 'plus' or 'minus'
        seed: Random seed for reproducibility

    Returns:
        (train_data, test_data) where each item is:
            {
                'prompt': str (e.g., "42 + 17 ="),
                'target': str (e.g., "59"),
                'a': int,
                'b': int,
                'result': int
            }
    """
    random.seed(seed)

    # Generate all possible (a, b) pairs
    all_pairs = [(a, b) for a in range(modulus) for b in range(modulus)]
    random.shuffle(all_pairs)

    data = []
    for a, b in all_pairs:
        if operator == "plus":
            result = (a + b) % modulus
            format_templates = [
                "{a} + {b} =",
                "{a} plus {b} equals",
                "sum of {a} and {b} is",
                "what is {a} + {b}?",
            ]
            template = random.choice(format_templates)
            prompt = template.format(a=a, b=b)

        elif operator == "minus":
            result = (a - b) % modulus
            format_templates = [
                "{a} - {b} =",
                "{a} minus {b} equals",
                "what is {a} - {b}?",
            ]
            template = random.choice(format_templates)
            prompt = template.format(a=a, b=b)

        elif operator == "quadratic":
            # THE NONLINEAR TASK: a^2 + b^2 mod p
            result = (a**2 + b**2) % modulus
            format_templates = [
                "{a} squared plus {b} squared equals",
                "{a}^2 + {b}^2 =",
                "what is {a}^2 + {b}^2?",
            ]
            template = random.choice(format_templates)
            prompt = template.format(a=a, b=b)

        else:
            raise ValueError(f"Unknown operator: {operator}")

        data.append({
            'prompt': prompt,
            'target': str(result),
            'a': a,
            'b': b,
            'result': result,
        })

    # Split train/test
    split_idx = int(len(data) * train_fraction)
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    return train_data, test_data


def save_jsonl(data: List[Dict], path: Path) -> None:
    """Save data in JSONL format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f"Saved {len(data)} examples to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate modular arithmetic dataset for grokking experiments"
    )
    parser.add_argument(
        "--modulus",
        type=int,
        default=113,
        help="Prime modulus (default: 113, standard in grokking literature)",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.9,
        help="Fraction of pairs for training (default: 0.9)",
    )
    parser.add_argument(
        "--operator",
        type=str,
        choices=["plus", "minus", "quadratic"],
        default="plus",
        help="Arithmetic operator: plus (a+b), minus (a-b), quadratic (a^2+b^2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory (default: data/)",
    )

    args = parser.parse_args()

    print(f"Generating modular arithmetic dataset:")
    print(f"  modulus p={args.modulus}")
    print(f"  operator={args.operator}")
    print(f"  total pairs={args.modulus ** 2}")
    print(f"  train fraction={args.train_fraction}")
    print(f"  seed={args.seed}")

    train_data, test_data = generate_modular_arithmetic(
        modulus=args.modulus,
        train_fraction=args.train_fraction,
        operator=args.operator,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    train_path = output_dir / f"modular_p{args.modulus}_train.jsonl"
    test_path = output_dir / f"modular_p{args.modulus}_test.jsonl"

    save_jsonl(train_data, train_path)
    save_jsonl(test_data, test_path)

    print(f"\nDataset generation complete!")
    print(f"  Train: {len(train_data)} examples")
    print(f"  Test: {len(test_data)} examples")


if __name__ == "__main__":
    main()
