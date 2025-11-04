#!/usr/bin/env python3
"""Generate Pythia checkpoint H1 configs for the cross-condition orchestrator.

Creates one JSON per checkpoint using the existing heads_zero battery and
the Pythia-specific tokenizer corpora (suffix: _pythia).
"""

import json
from pathlib import Path

# Edit this list if you need to adjust checkpoints
CHECKPOINTS = [
    0,
    100,
    500,
    1000,
    2000,
    4000,
    8000,
    16000,
    32000,
    64000,
    128000,
    256000,
    282956,
]

CONFIG_DIR = Path("lab/configs")


def make_config(step: int) -> dict:
    run_name = f"h1_pythia_checkpoint_{step:06d}"
    # Pythia tokenizer variants; build them with scripts/build_tokenizer_variants.py
    shared = {
        "seeds": [0],
        "device": "mps",
        "model": {
            "family": "pythia",
            "name": "pythia-160m",
            "revision": f"step{step}",
            "dtype": "float16",
        },
        "dataset": {
            "id": "facts_single_token_v1_pythia",
            "split": "train",
            "clean_field": "clean",
            "corrupt_field": "corrupt",
            "target_field": "target",
            "foil_field": "foil",
        },
        "metrics": ["logit_diff", "acc_flip_rate"],
        "batch_size": 8,
        "max_seq_len": 256,
        "metric_span": "first_token",
    }

    conditions = [
        {
            "tag": "facts",
            "dataset": {"id": "facts_single_token_v1_pythia", "split": "train"},
        },
        {
            "tag": "neg",
            "dataset": {"id": "negation_single_token_v1_pythia", "split": "train"},
        },
        {
            "tag": "cf",
            "dataset": {
                "id": "counterfactual_single_token_v1_pythia",
                "split": "train",
            },
        },
        {
            "tag": "logic",
            "dataset": {"id": "logical_single_token_v1_pythia", "split": "train"},
        },
    ]

    return {
        "run_name": run_name,
        "battery": "lab/configs/battery_h1_heads_zero.json",
        "shared": shared,
        "conditions": conditions,
    }


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for step in CHECKPOINTS:
        cfg = make_config(step)
        out_path = CONFIG_DIR / f"run_h1_pythia_checkpoint_{step:06d}.json"
        out_path.write_text(json.dumps(cfg, indent=2))
        print(f"Created {out_path}")


if __name__ == "__main__":
    main()
