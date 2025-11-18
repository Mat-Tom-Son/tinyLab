#!/usr/bin/env python3
"""Measure circularity of weekday representations (Task B) for a given model.

Implements the CircularityScore metric defined in the Stage 1A preregistration:

    1. Collect activations for day tokens in the modular arithmetic task
       at a fixed layer and position.
    2. Project activations to 2D via PCA.
    3. Compute:
         - Angular order correlation between observed angles and ideal
           weekday angles on the unit circle.
         - Radial consistency: 1 - (sigma_radius / mu_radius).
    4. Define CircularityScore = AngleCorrelation * RadialConsistency.

This script computes these quantities for a single checkpoint/model and writes
summary metrics (and optionally per-point coordinates) to disk.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

import sys
from pathlib import Path as _Path

REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.src.components import load_model


DAYS: List[str] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


@dataclass
class CircularityConfig:
    layer_index: int
    position_index: int
    max_examples: int
    seed: int


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_task_b_examples(path: Path, max_examples: int, seed: int) -> List[Dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    examples: List[Dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        examples.append(json.loads(line))
    if not examples:
        raise RuntimeError(f"No examples found in {path}")
    n = min(max_examples, len(examples)) if max_examples > 0 else len(examples)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(examples))[:n]
    return [examples[int(i)] for i in indices]


def tokens_and_labels(
    model,
    examples: List[Dict],
) -> Tuple[torch.Tensor, np.ndarray]:
    prompts = [ex["input"] for ex in examples]
    labels = [DAYS.index(ex["day2"]) for ex in examples]
    tokens = model.to_tokens(prompts)
    return tokens, np.array(labels, dtype=np.int64)


def collect_activations(
    model,
    tokens: torch.Tensor,
    cfg: CircularityConfig,
    device: str,
) -> np.ndarray:
    model.to(device)
    model.eval()
    tokens = tokens.to(device)

    cache: Dict[str, torch.Tensor] = {}
    resid_name = f"blocks.{cfg.layer_index}.hook_resid_post"

    def hook_resid(resid, hook):
        cache["resid"] = resid.detach()
        return resid

    with torch.no_grad():
        model.run_with_hooks(
            tokens,
            fwd_hooks=[(resid_name, hook_resid)],
        )

    resid = cache["resid"].cpu()  # [batch, seq_len, d_model]
    batch_size, seq_len, d_model = resid.shape
    pos = cfg.position_index if cfg.position_index >= 0 else seq_len + cfg.position_index
    if pos < 0 or pos >= seq_len:
        raise ValueError(f"Position index {cfg.position_index} is out of range for seq_len={seq_len}")
    acts = resid[:, pos, :]  # [batch, d_model]
    return acts.numpy()


def pca_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0, keepdims=True)
    x_centered = x - mean
    u, s, vt = np.linalg.svd(x_centered, full_matrices=False)
    components = vt[:2]
    z = x_centered @ components.T
    return z


def circularity_metrics(z: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    x = z[:, 0]
    y = z[:, 1]
    angles = np.arctan2(y, x)
    ideal_angles = 2.0 * np.pi * labels.astype(np.float64) / len(DAYS)

    if np.std(angles) == 0 or np.std(ideal_angles) == 0:
        angle_corr = 0.0
    else:
        angle_corr = float(np.corrcoef(angles, ideal_angles)[0, 1])

    radii = np.sqrt(x**2 + y**2)
    mu_r = float(np.mean(radii))
    sigma_r = float(np.std(radii, ddof=0))
    if mu_r <= 0:
        radial_consistency = 0.0
    else:
        radial_consistency = float(1.0 - (sigma_r / mu_r))

    score = angle_corr * radial_consistency
    return {
        "angle_correlation": angle_corr,
        "radial_consistency": radial_consistency,
        "circularity_score": score,
        "mean_radius": mu_r,
        "std_radius": sigma_r,
    }


def save_summary(
    out_path: Path,
    model_name: str,
    layer_index: int,
    position_index: int,
    n_examples: int,
    metrics: Dict[str, float],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "model": model_name,
        "layer": layer_index,
        "position_index": position_index,
        "n_examples": n_examples,
        **metrics,
    }
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def save_points(
    out_path: Path,
    z: np.ndarray,
    labels: np.ndarray,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, (coord, label_idx) in enumerate(zip(z, labels)):
        rows.append(
            {
                "index": idx,
                "x": float(coord[0]),
                "y": float(coord[1]),
                "day_index": int(label_idx),
                "day_name": DAYS[int(label_idx)],
            }
        )
    df = pd.DataFrame.from_records(rows)
    df.to_csv(out_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure circularity of weekday representations (Task B)."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt2-small",
        help="TransformerLens model name (e.g., 'gpt2-small').",
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default=None,
        help="Optional HuggingFace repo for weights.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype for loading.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use ('auto', 'cpu', 'cuda', 'mps', etc.).",
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        default=0,
        help="Layer index (block) to probe.",
    )
    parser.add_argument(
        "--position-index",
        type=int,
        default=-1,
        help="Token position index to use (default: -1 for last token).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="lab/data/task_b_weekdays.jsonl",
        help="Path to Task B JSONL data.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=512,
        help="Maximum number of examples to use (0 = all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for subsampling.",
    )
    parser.add_argument(
        "--summary-out",
        type=str,
        default="reports/task_b_circularity_summary.json",
        help="Output JSON path for summary metrics.",
    )
    parser.add_argument(
        "--points-out",
        type=str,
        default="reports/task_b_circularity_points.csv",
        help="Output CSV path for 2D coordinates (optional).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = auto_device() if args.device == "auto" else args.device
    print(f"Using device: {device}")

    model_cfg = {
        "name": args.model_name,
        "dtype": args.dtype,
    }
    if args.hf_model:
        model_cfg["hf_model"] = args.hf_model

    model = load_model.load_transformerlens(model_cfg, device=device)

    cfg = CircularityConfig(
        layer_index=args.layer_index,
        position_index=args.position_index,
        max_examples=args.max_examples,
        seed=args.seed,
    )

    examples = load_task_b_examples(Path(args.data_path), cfg.max_examples, cfg.seed)
    tokens, labels = tokens_and_labels(model, examples)
    acts = collect_activations(model, tokens, cfg, device)

    z = pca_2d(acts)
    metrics = circularity_metrics(z, labels)

    print(
        f"Circularity (layer={cfg.layer_index}, pos={cfg.position_index}, "
        f"n={len(labels)}): score={metrics['circularity_score']:.4f}, "
        f"angle_corr={metrics['angle_correlation']:.4f}, "
        f"radial_consistency={metrics['radial_consistency']:.4f}"
    )

    save_summary(
        out_path=Path(args.summary_out),
        model_name=args.model_name,
        layer_index=cfg.layer_index,
        position_index=cfg.position_index,
        n_examples=len(labels),
        metrics=metrics,
    )
    save_points(Path(args.points_out), z, labels)


if __name__ == "__main__":
    main()
