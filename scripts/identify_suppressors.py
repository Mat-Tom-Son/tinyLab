#!/usr/bin/env python3
"""Identify candidate suppressor heads via VDI on a small model.

This script is a lightweight front-end around the variance dampening index (VDI)
probe used in the Pythia experiments. It is designed for the Stage 1A pilot:

    - Load a (typically small) TransformerLens model.
    - Sample or load a set of prompts.
    - Run a noise-based VDI probe on a chosen layer (default: layer 0).
    - Export per-head VDI statistics to CSV and print the top dampeners.

The Stage 1A preregistration uses these VDI scores across checkpoints and seeds
to pick a suppressor head (largest positive vdi_effect) and a random-head
control (median vdi_effect). This script provides the per-checkpoint VDI
metrics; aggregation and head selection can be built on top.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch

import sys
from pathlib import Path as _Path

REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.src.components import load_model


@dataclass
class VDIConfig:
    n_prompts: int = 256
    k_noise: int = 8
    sigma: float = 0.05
    seed: int = 0


def auto_device() -> str:
    """Pick a reasonable default device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_prompts_from_file(path: Path, n_prompts: int, seed: int) -> List[str]:
    """Load prompts from a newline-delimited text file."""
    text = path.read_text(encoding="utf-8").splitlines()
    # Filter out empty lines
    prompts = [line for line in text if line.strip()]
    if not prompts:
        raise RuntimeError(f"No non-empty prompts found in {path}")
    rng = np.random.default_rng(seed)
    rng.shuffle(prompts)
    if 0 < n_prompts < len(prompts):
        prompts = prompts[:n_prompts]
    return prompts


def make_synthetic_prompts(n_prompts: int, seed: int) -> List[str]:
    """Generate simple synthetic prompts if no file is provided.

    These are plain-text strings so they can be tokenised by any TransformerLens
    model with a standard tokenizer. For Stage 1A tasks, you will typically
    replace this with task-specific prompts or a dataset-backed loader.
    """
    rng = np.random.default_rng(seed)
    prompts: List[str] = []
    for i in range(n_prompts):
        # Simple synthetic pattern with a small alphabet of tokens
        a = int(rng.integers(0, 10))
        b = int(rng.integers(0, 10))
        prompts.append(f"A={a}, B={b}, sequence {i}")
    return prompts


def compute_mean_embed_norm(model, tokens: torch.Tensor, device: str) -> float:
    """Compute mean token embedding norm for scaling noise."""
    cache: Dict[str, torch.Tensor] = {}

    def hook_embed(emb, hook):
        cache["emb"] = emb.detach()
        return emb

    with torch.no_grad():
        model.run_with_hooks(
            tokens.to(device),
            fwd_hooks=[("hook_embed", hook_embed)],
        )
    emb = cache["emb"]
    return emb.norm(dim=-1).mean().item()


def compute_vdi_for_sigma(
    model,
    tokens: torch.Tensor,
    sigma: float,
    k_noise: int,
    seed: int,
    device: str,
    layer_index: int = 0,
) -> Dict:
    """Compute VDI statistics at a given layer for a noise scale.

    This follows the plan:
        - r0 = resid_pre before the target block attention
        - r1 = resid_post after that block
        - Inject Gaussian noise at the embedding layer
        - Measure Var_before from r0 (delta vs. clean)
        - Measure Var_after_full from r1 (delta vs. clean)
        - For each head h, zero its z output and recompute Var_after_minus[h]

    Returns a dict with scalar stats and per-head records.
    """
    model.to(device)
    model.eval()
    tokens = tokens.to(device)

    # Baseline residuals (no noise, all heads active) at the target layer
    base_cache: Dict[str, torch.Tensor] = {}

    resid_pre_name = f"blocks.{layer_index}.hook_resid_pre"
    resid_post_name = f"blocks.{layer_index}.hook_resid_post"
    attn_z_name = f"blocks.{layer_index}.attn.hook_z"

    def hook_r0(resid, hook):
        base_cache["r0"] = resid.detach()
        return resid

    def hook_r1(resid, hook):
        base_cache["r1"] = resid.detach()
        return resid

    with torch.no_grad():
        model.run_with_hooks(
            tokens,
            fwd_hooks=[
                (resid_pre_name, hook_r0),
                (resid_post_name, hook_r1),
            ],
        )

    r0_base = base_cache["r0"]
    r1_base = base_cache["r1"]

    # Noise scale from embedding geometry
    mean_embed_norm = compute_mean_embed_norm(model, tokens, device=device)
    noise_std = sigma * mean_embed_norm

    batch_size, seq_len, d_model = r0_base.shape
    n_positions = batch_size * seq_len * k_noise

    # Use global RNG seeded for reproducibility across runs.
    torch.manual_seed(seed)

    sum_sq_delta0 = 0.0
    sum_sq_delta1_full = 0.0

    for _ in range(k_noise):
        cache: Dict[str, torch.Tensor] = {}

        def hook_embed_noise(emb, hook):
            noise = torch.randn_like(emb) * noise_std
            cache["noise"] = noise
            return emb + noise

        def hook_r0_noisy(resid, hook):
            cache["r0_noisy"] = resid.detach()
            return resid

        def hook_r1_noisy(resid, hook):
            cache["r1_noisy"] = resid.detach()
            return resid

        with torch.no_grad():
            model.run_with_hooks(
                tokens,
                fwd_hooks=[
                    ("hook_embed", hook_embed_noise),
                    (resid_pre_name, hook_r0_noisy),
                    (resid_post_name, hook_r1_noisy),
                ],
            )

        delta0 = (cache["r0_noisy"] - r0_base).float()
        delta1 = (cache["r1_noisy"] - r1_base).float()

        # Mask non-finite values to avoid infinities propagating into the variance
        delta0 = torch.where(torch.isfinite(delta0), delta0, torch.zeros_like(delta0))
        delta1 = torch.where(torch.isfinite(delta1), delta1, torch.zeros_like(delta1))

        # Squared norm per-position, then sum over positions
        sum_sq_delta0 += delta0.pow(2).sum(dim=-1).sum().item()
        sum_sq_delta1_full += delta1.pow(2).sum(dim=-1).sum().item()

    var_before = sum_sq_delta0 / float(n_positions)
    var_after_full = sum_sq_delta1_full / float(n_positions)
    vdi_full = (var_before - var_after_full) / var_before if var_before > 0 else float("nan")

    # Per-head ablation: measure how much Var_after grows when a head is removed.
    n_heads = model.cfg.n_heads
    var_after_minus: List[float] = []
    vdi_minus: List[float] = []
    vdi_effect: List[float] = []

    for head in range(n_heads):
        # Baseline with this head zeroed (no noise)
        head_base_cache: Dict[str, torch.Tensor] = {}

        def zero_head(z, hook, head_idx=head):
            z = z.clone()
            z[:, :, head_idx, :] = 0.0
            return z

        def hook_r1_head_base(resid, hook):
            head_base_cache["r1_base_minus"] = resid.detach()
            return resid

        with torch.no_grad():
            model.run_with_hooks(
                tokens,
                fwd_hooks=[
                    (attn_z_name, zero_head),
                    (resid_post_name, hook_r1_head_base),
                ],
            )

        r1_base_minus = head_base_cache["r1_base_minus"]

        # Noise runs with this head zeroed
        torch.manual_seed(seed + head)
        sum_sq_delta1_minus = 0.0

        for _ in range(k_noise):
            cache: Dict[str, torch.Tensor] = {}

            def hook_embed_noise_head(emb, hook):
                noise = torch.randn_like(emb) * noise_std
                return emb + noise

            def hook_r1_head_noisy(resid, hook):
                cache["r1_noisy_minus"] = resid.detach()
                return resid

            with torch.no_grad():
                model.run_with_hooks(
                    tokens,
                    fwd_hooks=[
                        ("hook_embed", hook_embed_noise_head),
                        (attn_z_name, zero_head),
                        (resid_post_name, hook_r1_head_noisy),
                    ],
                )

            delta1_minus = (cache["r1_noisy_minus"] - r1_base_minus).float()
            delta1_minus = torch.where(
                torch.isfinite(delta1_minus),
                delta1_minus,
                torch.zeros_like(delta1_minus),
            )
            sum_sq_delta1_minus += delta1_minus.pow(2).sum(dim=-1).sum().item()

        v_after_minus = sum_sq_delta1_minus / float(n_positions)
        var_after_minus.append(v_after_minus)
        vdi_m = (var_before - v_after_minus) / var_before if var_before > 0 else float("nan")
        vdi_minus.append(vdi_m)
        vdi_eff = (v_after_minus - var_after_full) / var_before if var_before > 0 else float("nan")
        vdi_effect.append(vdi_eff)

    return {
        "sigma": sigma,
        "layer_index": layer_index,
        "var_before": var_before,
        "var_after_full": var_after_full,
        "vdi_full": vdi_full,
        "per_head": [
            {
                "head": int(h),
                "var_after_minus": float(var_after_minus[h]),
                "vdi_minus": float(vdi_minus[h]),
                "vdi_effect": float(vdi_effect[h]),
            }
            for h in range(n_heads)
        ],
    }


def save_vdi_csv(
    out_path: Path,
    model_name: str,
    layer_index: int,
    vdi_result: Dict,
) -> None:
    """Save per-head VDI stats to a CSV."""
    records: List[Dict] = []
    for row in vdi_result["per_head"]:
        rec = {
            "model": model_name,
            "layer": layer_index,
            "head": row["head"],
            "var_before": vdi_result["var_before"],
            "var_after_full": vdi_result["var_after_full"],
            "vdi_full": vdi_result["vdi_full"],
            "var_after_minus": row["var_after_minus"],
            "vdi_minus": row["vdi_minus"],
            "vdi_effect": row["vdi_effect"],
        }
        records.append(rec)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame.from_records(records)
    df.to_csv(out_path, index=False)
    print(f"[VDI] Wrote {out_path} ({len(df)} rows)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-head VDI statistics for a small TransformerLens model."
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
        help="Optional HuggingFace repository override for the model weights.",
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
        help="Transformer block index to probe (default: 0).",
    )
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=256,
        help="Number of prompts for the VDI experiment.",
    )
    parser.add_argument(
        "--k-noise",
        type=int,
        default=8,
        help="Number of noise samples per prompt.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.05,
        help="Noise scale as a fraction of mean embedding norm.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for prompt construction and noise.",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Optional path to a text file with one prompt per line. If omitted, synthetic prompts are generated.",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="reports/vdi_small_model_layer0.csv",
        help="Output CSV path for per-head VDI statistics.",
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
    print(f"Loaded model with {model.cfg.n_layers} layers and {model.cfg.n_heads} heads per layer.")

    vdi_cfg = VDIConfig(
        n_prompts=args.n_prompts,
        k_noise=args.k_noise,
        sigma=args.sigma,
        seed=args.seed,
    )
    print(
        f"[VDI] n_prompts={vdi_cfg.n_prompts}, k_noise={vdi_cfg.k_noise}, "
        f"sigma={vdi_cfg.sigma}, layer_index={args.layer_index}"
    )

    # Prompts and tokens
    if args.prompt_file:
        prompts = load_prompts_from_file(Path(args.prompt_file), vdi_cfg.n_prompts, vdi_cfg.seed)
    else:
        prompts = make_synthetic_prompts(vdi_cfg.n_prompts, vdi_cfg.seed)

    tokens = model.to_tokens(prompts)

    # Compute VDI
    vdi_result = compute_vdi_for_sigma(
        model=model,
        tokens=tokens,
        sigma=vdi_cfg.sigma,
        k_noise=vdi_cfg.k_noise,
        seed=vdi_cfg.seed,
        device=device,
        layer_index=args.layer_index,
    )

    out_path = Path(args.out_path)
    save_vdi_csv(out_path, args.model_name, args.layer_index, vdi_result)

    # Print a quick summary of candidate dampeners
    per_head = sorted(
        vdi_result["per_head"],
        key=lambda r: r["vdi_effect"],
        reverse=True,
    )
    print("[VDI] Top heads by vdi_effect:")
    for row in per_head[:5]:
        print(
            f"  head={row['head']}: vdi_effect={row['vdi_effect']:.4f}, "
            f"vdi_minus={row['vdi_minus']:.4f}"
        )


if __name__ == "__main__":
    main()
