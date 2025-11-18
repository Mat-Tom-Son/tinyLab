#!/usr/bin/env python3
"""Layer-0 VDI and drift experiments for Pythia-2.8B.

Phase 1 pipeline:
    - Compute a noise-based Variance Dampening Index (VDI) focused on layer 0.
    - Identify candidate suppressor heads from VDI.
    - Run drift / entropy trajectories with and without ablations for those heads.

This script is intentionally self-contained and does not depend on the main
TinyLab harness. It reuses the existing model-loading and dataset utilities.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from lab.src.components import load_model
from lab.src.components import datasets as tiny_datasets


@dataclass
class VDIConfig:
    n_prompts: int = 600
    k_noise: int = 8
    sigma: float = 0.05
    sigma_alt: float = 0.02
    seed: int = 0


@dataclass
class DriftConfig:
    n_prompts: int = 64
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    n_samples: int = 4
    seed: int = 0
    heads_per_drift: int = 4


def auto_device() -> str:
    """Pick a reasonable default device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pythia_prompts(n_prompts: int, seed: int = 0) -> List[str]:
    """Load mixed natural language prompts from the Pythia single-token corpora."""
    corpus_ids = [
        "facts_single_token_v1_pythia",
        "negation_single_token_v1_pythia",
        "counterfactual_single_token_v1_pythia",
        "logical_single_token_v1_pythia",
    ]
    texts: List[str] = []
    for ds_id in corpus_ids:
        rows, _, _ = tiny_datasets.load_split({"id": ds_id, "split": "train"})
        for row in rows:
            clean = row.get("clean")
            if isinstance(clean, str):
                texts.append(clean)

    if not texts:
        raise RuntimeError("No prompts loaded from Pythia corpora.")

    rng = random.Random(seed)
    rng.shuffle(texts)
    if n_prompts <= 0 or n_prompts >= len(texts):
        return texts
    return texts[:n_prompts]


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
    """Compute VDI statistics at layer 0 for a given noise scale.

    This follows the plan:
        - r0 = resid_pre before block 0 attention
        - r1 = resid_post after block 0
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
    vdi_full = (
        (var_before - var_after_full) / var_before if var_before > 0 else float("nan")
    )

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
        vdi_m = (
            (var_before - v_after_minus) / var_before
            if var_before > 0
            else float("nan")
        )
        vdi_minus.append(vdi_m)
        vdi_eff = (
            (v_after_minus - var_after_full) / var_before
            if var_before > 0
            else float("nan")
        )
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


def cosine_drift(sequence: torch.Tensor) -> torch.Tensor:
    """Compute 1 - cosine similarity between consecutive states.

    Args:
        sequence: [T, d_model] tensor
    Returns:
        [T-1] tensor of drift values
    """
    if sequence.shape[0] < 2:
        return torch.zeros(0, dtype=sequence.dtype, device=sequence.device)
    a = sequence[:-1]
    b = sequence[1:]
    cos = F.cosine_similarity(a, b, dim=-1)
    # Flatten in case there is a singleton batch dimension
    cos = cos.view(-1)
    return 1.0 - cos


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Compute token-wise entropy from logits.

    Args:
        logits: [T, vocab_size]
    Returns:
        [T] entropy values
    """
    # Work in float32 for numerical stability
    logits_f32 = logits.float()
    probs = F.softmax(logits_f32, dim=-1)
    # Add epsilon to avoid log(0)
    entropy = -(probs * (probs + 1e-9).log()).sum(dim=-1)
    # Mask any non-finite values
    entropy = torch.where(torch.isfinite(entropy), entropy, torch.zeros_like(entropy))
    # Flatten in case of a singleton batch dimension
    entropy = entropy.view(-1)
    return entropy


def sample_next_token(
    logits_last: torch.Tensor, temperature: float, top_p: float
) -> torch.Tensor:
    """Sample the next token using temperature + top-p sampling.

    Args:
        logits_last: [1, vocab_size]
    Returns:
        next_token: [1, 1] int tensor
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    logits = logits_last / temperature
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = probs.cumsum(dim=-1)

    # Create mask for tokens to truncate
    cutoff = cumulative_probs > top_p
    # Always keep at least the most likely token
    cutoff[..., 0] = False
    # Use a large negative value within float16 range
    sorted_logits[cutoff] = -1e4

    filtered_probs = F.softmax(sorted_logits, dim=-1)
    next_idx_in_sorted = torch.multinomial(filtered_probs, num_samples=1)
    next_token = sorted_indices.gather(-1, next_idx_in_sorted)
    return next_token


def run_drift_trajectories(
    model,
    prompts: Sequence[str],
    cfg: DriftConfig,
    device: str,
    ablate_head: Optional[int] = None,
    abl_layer_index: int = 0,
) -> Dict[str, np.ndarray]:
    """Run open-ended generation and track drift / entropy trajectories.

    Returns mean trajectories across prompts × samples for:
        - layer 0 resid_pre
        - mid-layer resid_post
        - final-layer resid_post
        - logits entropy
    """
    model.to(device)
    model.eval()

    n_layers = model.cfg.n_layers
    mid_layer = n_layers // 2
    attn_z_name = f"blocks.{abl_layer_index}.attn.hook_z"
    max_new = cfg.max_new_tokens

    sum_drift_layer0 = torch.zeros(max_new - 1, dtype=torch.float64)
    sum_drift_mid = torch.zeros(max_new - 1, dtype=torch.float64)
    sum_drift_final = torch.zeros(max_new - 1, dtype=torch.float64)
    sum_entropy = torch.zeros(max_new, dtype=torch.float64)
    n_runs = 0

    for prompt_idx, prompt in enumerate(prompts):
        base_tokens = model.to_tokens(prompt, prepend_bos=True).to(device)

        for sample_idx in range(cfg.n_samples):
            # Seed to make sampling reproducible
            torch.manual_seed(cfg.seed + prompt_idx * cfg.n_samples + sample_idx)
            tokens = base_tokens.clone()

            resid0_seq: List[torch.Tensor] = []
            resid_mid_seq: List[torch.Tensor] = []
            resid_final_seq: List[torch.Tensor] = []
            logits_seq: List[torch.Tensor] = []

            for _ in range(max_new):
                cache: Dict[str, torch.Tensor] = {}

                def hook_r0(resid, hook):
                    cache["r0_last"] = resid[:, -1, :].detach()
                    return resid

                def hook_r_mid(resid, hook):
                    cache["r_mid_last"] = resid[:, -1, :].detach()
                    return resid

                def hook_r_final(resid, hook):
                    cache["r_final_last"] = resid[:, -1, :].detach()
                    return resid

                fwd_hooks: List[Tuple[str, callable]] = [
                    ("blocks.0.hook_resid_pre", hook_r0),
                    (f"blocks.{mid_layer}.hook_resid_post", hook_r_mid),
                    (f"blocks.{n_layers-1}.hook_resid_post", hook_r_final),
                ]

                if ablate_head is not None:

                    def zero_head(z, hook, head_idx=ablate_head):
                        z = z.clone()
                        z[:, :, head_idx, :] = 0.0
                        return z

                    fwd_hooks.append((attn_z_name, zero_head))

                with torch.no_grad():
                    logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

                logits_last = logits[:, -1, :]
                resid0_seq.append(cache["r0_last"])
                resid_mid_seq.append(cache["r_mid_last"])
                resid_final_seq.append(cache["r_final_last"])
                logits_seq.append(logits_last.detach())

                next_token = sample_next_token(
                    logits_last, temperature=cfg.temperature, top_p=cfg.top_p
                )
                tokens = torch.cat([tokens, next_token], dim=1)

            # Stack into [T, d_model] / [T, vocab]
            resid0 = torch.stack(resid0_seq, dim=0)
            resid_mid = torch.stack(resid_mid_seq, dim=0)
            resid_final = torch.stack(resid_final_seq, dim=0)
            logits_all = torch.stack(logits_seq, dim=0)

            drift0 = cosine_drift(resid0).cpu().to(torch.float64)
            drift_mid = cosine_drift(resid_mid).cpu().to(torch.float64)
            drift_final = cosine_drift(resid_final).cpu().to(torch.float64)
            entropy = entropy_from_logits(logits_all).cpu().to(torch.float64)

            sum_drift_layer0 += drift0
            sum_drift_mid += drift_mid
            sum_drift_final += drift_final
            sum_entropy += entropy
            n_runs += 1

    if n_runs == 0:
        raise RuntimeError("No runs completed in drift trajectories.")

    mean_drift_layer0 = (sum_drift_layer0 / n_runs).numpy()
    mean_drift_mid = (sum_drift_mid / n_runs).numpy()
    mean_drift_final = (sum_drift_final / n_runs).numpy()
    mean_entropy = (sum_entropy / n_runs).numpy()

    return {
        "mean_drift_layer0": mean_drift_layer0,
        "mean_drift_mid": mean_drift_mid,
        "mean_drift_final": mean_drift_final,
        "mean_entropy": mean_entropy,
        "n_runs": n_runs,
    }


def save_vdi_results(
    out_dir: Path,
    model_name: str,
    sigma_result: Dict,
) -> Path:
    """Save per-head VDI records for a given sigma to CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sigma = sigma_result["sigma"]
    layer_index = int(sigma_result.get("layer_index", 0))
    records = []
    for row in sigma_result["per_head"]:
        records.append(
            {
                "model": model_name,
                "sigma": sigma,
                "head": row["head"],
                "layer": layer_index,
                "var_before": sigma_result["var_before"],
                "var_after_full": sigma_result["var_after_full"],
                "vdi_full": sigma_result["vdi_full"],
                "var_after_minus": row["var_after_minus"],
                "vdi_minus": row["vdi_minus"],
                "vdi_effect": row["vdi_effect"],
            }
        )

    df = pd.DataFrame.from_records(records)
    out_path = out_dir / f"pythia_layer{layer_index}_vdi_sigma_{sigma:.3f}.csv"
    df.to_csv(out_path, index=False)
    print(f"[VDI] Wrote {out_path} ({len(df)} rows)")
    return out_path


def save_drift_results(
    out_dir: Path,
    model_name: str,
    cfg: DriftConfig,
    base_stats: Dict[str, np.ndarray],
    head_stats: Dict[int, Dict[str, np.ndarray]],
) -> Path:
    """Save drift / entropy trajectories to a long-form CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict] = []
    max_new = cfg.max_new_tokens

    # Base (no ablation) trajectories
    for t in range(max_new - 1):
        records.append(
            {
                "model": model_name,
                "head": -1,
                "condition": "base",
                "layer": "resid0",
                "step": t,
                "mean_drift": float(base_stats["mean_drift_layer0"][t]),
                "mean_entropy": float(base_stats["mean_entropy"][t]),
            }
        )
        records.append(
            {
                "model": model_name,
                "head": -1,
                "condition": "base",
                "layer": "resid_mid",
                "step": t,
                "mean_drift": float(base_stats["mean_drift_mid"][t]),
                "mean_entropy": float(base_stats["mean_entropy"][t]),
            }
        )
        records.append(
            {
                "model": model_name,
                "head": -1,
                "condition": "base",
                "layer": "resid_final",
                "step": t,
                "mean_drift": float(base_stats["mean_drift_final"][t]),
                "mean_entropy": float(base_stats["mean_entropy"][t]),
            }
        )

    # Extra row for entropy at the last step
    last_step_entropy = float(base_stats["mean_entropy"][max_new - 1])
    records.append(
        {
            "model": model_name,
            "head": -1,
            "condition": "base",
            "layer": "logits",
            "step": max_new - 1,
            "mean_drift": float("nan"),
            "mean_entropy": last_step_entropy,
        }
    )

    # Ablated heads
    for head, stats in head_stats.items():
        for t in range(max_new - 1):
            records.append(
                {
                    "model": model_name,
                    "head": head,
                    "condition": "ablated",
                    "layer": "resid0",
                    "step": t,
                    "mean_drift": float(stats["mean_drift_layer0"][t]),
                    "mean_entropy": float(stats["mean_entropy"][t]),
                }
            )
            records.append(
                {
                    "model": model_name,
                    "head": head,
                    "condition": "ablated",
                    "layer": "resid_mid",
                    "step": t,
                    "mean_drift": float(stats["mean_drift_mid"][t]),
                    "mean_entropy": float(stats["mean_entropy"][t]),
                }
            )
            records.append(
                {
                    "model": model_name,
                    "head": head,
                    "condition": "ablated",
                    "layer": "resid_final",
                    "step": t,
                    "mean_drift": float(stats["mean_drift_final"][t]),
                    "mean_entropy": float(stats["mean_entropy"][t]),
                }
            )

        last_entropy = float(stats["mean_entropy"][max_new - 1])
        records.append(
            {
                "model": model_name,
                "head": head,
                "condition": "ablated",
                "layer": "logits",
                "step": max_new - 1,
                "mean_drift": float("nan"),
                "mean_entropy": last_entropy,
            }
        )

    df = pd.DataFrame.from_records(records)
    out_path = out_dir / "pythia_layer0_drift_trajectories.csv"
    df.to_csv(out_path, index=False)
    print(f"[Drift] Wrote {out_path} ({len(df)} rows)")
    return out_path


def run_between_condition_metrics(
    model,
    prompts: Sequence[str],
    cfg: DriftConfig,
    device: str,
    heads: Sequence[int],
    abl_layer_index: int = 0,
) -> Tuple[Dict[int, Dict[str, float]], Path, List[Dict]]:
    """Run paired base vs ablated trajectories and compute divergence metrics.

    For each head h and each (prompt, sample), we:
        - Run a base trajectory
        - Run an ablated trajectory with head h zeroed
        - Compute mean 1 - cos similarity between base and ablated residuals at mid/final layers
        - Compute mean KL(p_base || p_ablated) at logits

    Returns:
        per_head_summary: dict mapping head -> {mean_drift_between_mid, mean_drift_between_final, mean_kl_between}
        csv_path: path to the between-condition CSV
        rows: raw per-run rows (for logging)
    """
    model.to(device)
    model.eval()

    n_layers = model.cfg.n_layers
    mid_layer = n_layers // 2
    attn_z_name = f"blocks.{abl_layer_index}.attn.hook_z"
    max_new = cfg.max_new_tokens

    per_head_summary: Dict[int, Dict[str, float]] = {}
    rows: List[Dict] = []

    for head in heads:
        sum_mid = 0.0
        sum_final = 0.0
        sum_kl = 0.0
        n_runs = 0

        print(f"[Between] Running paired trajectories for head {head}...")

        for prompt_idx, prompt in enumerate(prompts):
            base_tokens = model.to_tokens(prompt, prepend_bos=True).to(device)

            for sample_idx in range(cfg.n_samples):
                seed = cfg.seed + prompt_idx * cfg.n_samples + sample_idx

                # Base trajectory
                torch.manual_seed(seed)
                tokens_base = base_tokens.clone()
                resid_mid_base_seq: List[torch.Tensor] = []
                resid_final_base_seq: List[torch.Tensor] = []
                logits_base_seq: List[torch.Tensor] = []

                for _ in range(max_new):
                    cache: Dict[str, torch.Tensor] = {}

                    def hook_r_mid_base(resid, hook):
                        cache["r_mid_last"] = resid[:, -1, :].detach()
                        return resid

                    def hook_r_final_base(resid, hook):
                        cache["r_final_last"] = resid[:, -1, :].detach()
                        return resid

                    with torch.no_grad():
                        logits_base = model.run_with_hooks(
                            tokens_base,
                            fwd_hooks=[
                                (
                                    f"blocks.{mid_layer}.hook_resid_post",
                                    hook_r_mid_base,
                                ),
                                (
                                    f"blocks.{n_layers-1}.hook_resid_post",
                                    hook_r_final_base,
                                ),
                            ],
                        )

                    logits_last_base = logits_base[:, -1, :]
                    resid_mid_base_seq.append(cache["r_mid_last"])
                    resid_final_base_seq.append(cache["r_final_last"])
                    logits_base_seq.append(logits_last_base.detach())

                    next_token_base = sample_next_token(
                        logits_last_base,
                        temperature=cfg.temperature,
                        top_p=cfg.top_p,
                    )
                    tokens_base = torch.cat([tokens_base, next_token_base], dim=1)

                # Ablated trajectory for this head
                torch.manual_seed(seed)
                tokens_ablate = base_tokens.clone()
                resid_mid_ablate_seq: List[torch.Tensor] = []
                resid_final_ablate_seq: List[torch.Tensor] = []
                logits_ablate_seq: List[torch.Tensor] = []

                for _ in range(max_new):
                    cache: Dict[str, torch.Tensor] = {}

                    def hook_r_mid_ablate(resid, hook):
                        cache["r_mid_last"] = resid[:, -1, :].detach()
                        return resid

                    def hook_r_final_ablate(resid, hook):
                        cache["r_final_last"] = resid[:, -1, :].detach()
                        return resid

                    def zero_head(z, hook, head_idx=head):
                        z = z.clone()
                        z[:, :, head_idx, :] = 0.0
                        return z

                    with torch.no_grad():
                        logits_ablate = model.run_with_hooks(
                            tokens_ablate,
                            fwd_hooks=[
                                (attn_z_name, zero_head),
                                (
                                    f"blocks.{mid_layer}.hook_resid_post",
                                    hook_r_mid_ablate,
                                ),
                                (
                                    f"blocks.{n_layers-1}.hook_resid_post",
                                    hook_r_final_ablate,
                                ),
                            ],
                        )

                    logits_last_ablate = logits_ablate[:, -1, :]
                    resid_mid_ablate_seq.append(cache["r_mid_last"])
                    resid_final_ablate_seq.append(cache["r_final_last"])
                    logits_ablate_seq.append(logits_last_ablate.detach())

                    next_token_ablate = sample_next_token(
                        logits_last_ablate,
                        temperature=cfg.temperature,
                        top_p=cfg.top_p,
                    )
                    tokens_ablate = torch.cat([tokens_ablate, next_token_ablate], dim=1)

                # Stack sequences: [T, d_model] / [T, vocab]
                resid_mid_base = torch.stack(resid_mid_base_seq, dim=0)
                resid_final_base = torch.stack(resid_final_base_seq, dim=0)
                resid_mid_ablate = torch.stack(resid_mid_ablate_seq, dim=0)
                resid_final_ablate = torch.stack(resid_final_ablate_seq, dim=0)
                logits_base_all = torch.stack(logits_base_seq, dim=0).float()
                logits_ablate_all = torch.stack(logits_ablate_seq, dim=0).float()

                # Cosine divergence between base and ablated
                cos_mid = F.cosine_similarity(
                    resid_mid_base, resid_mid_ablate, dim=-1
                ).view(-1)
                cos_final = F.cosine_similarity(
                    resid_final_base, resid_final_ablate, dim=-1
                ).view(-1)
                drift_between_mid = 1.0 - cos_mid
                drift_between_final = 1.0 - cos_final

                # KL divergence between base and ablated logits
                log_probs_base = F.log_softmax(logits_base_all, dim=-1)
                log_probs_ablate = F.log_softmax(logits_ablate_all, dim=-1)
                probs_base = log_probs_base.exp()
                kl_per_t = (probs_base * (log_probs_base - log_probs_ablate)).sum(
                    dim=-1
                )
                # Mask any non-finite values
                kl_per_t = torch.where(
                    torch.isfinite(kl_per_t),
                    kl_per_t,
                    torch.zeros_like(kl_per_t),
                )

                mean_mid = float(drift_between_mid.mean().cpu())
                mean_final = float(drift_between_final.mean().cpu())
                mean_kl = float(kl_per_t.mean().cpu())

                sum_mid += mean_mid
                sum_final += mean_final
                sum_kl += mean_kl
                n_runs += 1

                rows.append(
                    {
                        "model": (
                            model.cfg.model_name
                            if hasattr(model.cfg, "model_name")
                            else "unknown"
                        ),
                        "head": head,
                        "prompt_index": prompt_idx,
                        "sample_index": sample_idx,
                        "mean_drift_between_mid": mean_mid,
                        "mean_drift_between_final": mean_final,
                        "mean_kl_between": mean_kl,
                    }
                )

        if n_runs == 0:
            per_head_summary[head] = {
                "mean_drift_between_mid": float("nan"),
                "mean_drift_between_final": float("nan"),
                "mean_kl_between": float("nan"),
            }
        else:
            per_head_summary[head] = {
                "mean_drift_between_mid": sum_mid / n_runs,
                "mean_drift_between_final": sum_final / n_runs,
                "mean_kl_between": sum_kl / n_runs,
            }

    return per_head_summary, Path(""), rows


def save_between_results(
    out_dir: Path,
    model_name: str,
    rows: List[Dict],
) -> Path:
    """Save between-condition divergence metrics to CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame.from_records(rows)
    if "model" not in df.columns:
        df.insert(0, "model", model_name)
    out_path = out_dir / "pythia_layer0_drift_between.csv"
    df.to_csv(out_path, index=False)
    print(f"[Between] Wrote {out_path} ({len(df)} rows)")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Layer-0 VDI + drift experiments for Pythia-2.8B (TransformerLens)."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="pythia-2.8b",
        help="TransformerLens model name (e.g., 'pythia-2.8b').",
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
        default="float16",
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
        "--vdi-prompts",
        type=int,
        default=600,
        help="Number of prompts for the VDI experiment.",
    )
    parser.add_argument(
        "--vdi-k-noise",
        type=int,
        default=8,
        help="Number of noise samples per prompt.",
    )
    parser.add_argument(
        "--vdi-sigma",
        type=float,
        default=0.05,
        help="Main noise scale as a fraction of mean embedding norm.",
    )
    parser.add_argument(
        "--vdi-sigma-alt",
        type=float,
        default=0.02,
        help="Alternate noise scale for a sanity check.",
    )
    parser.add_argument(
        "--drift-prompts",
        type=int,
        default=64,
        help="Number of prompts for the drift experiment.",
    )
    parser.add_argument(
        "--drift-max-new",
        type=int,
        default=256,
        help="Number of new tokens to generate in drift experiment.",
    )
    parser.add_argument(
        "--drift-temperature",
        type=float,
        default=0.7,
        help="Generation temperature.",
    )
    parser.add_argument(
        "--drift-top-p",
        type=float,
        default=0.9,
        help="Generation top-p threshold.",
    )
    parser.add_argument(
        "--drift-samples",
        type=int,
        default=4,
        help="Number of stochastic samples per prompt for drift.",
    )
    parser.add_argument(
        "--drift-heads",
        type=int,
        default=4,
        help="Number of top VDI heads to run drift ablations for.",
    )
    parser.add_argument(
        "--drift-heads-list",
        type=str,
        default="",
        help="Optional comma-separated list of head indices to run drift ablations for. "
        "If provided, this overrides --drift-heads.",
    )
    parser.add_argument(
        "--abl-layer-index",
        type=int,
        default=0,
        help="Layer index at which to ablate attention heads (default: 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for prompt selection and noise.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="reports/pythia_layer0_vdi_drift",
        help="Output directory for CSVs.",
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
    n_layers = model.cfg.n_layers
    mid_layer_index = n_layers // 2
    last_layer_index = n_layers - 1
    print(
        f"Model has {n_layers} layers (mid={mid_layer_index}, last={last_layer_index})"
    )

    # Phase 1a: VDI on layer 0
    vdi_cfg = VDIConfig(
        n_prompts=args.vdi_prompts,
        k_noise=args.vdi_k_noise,
        sigma=args.vdi_sigma,
        sigma_alt=args.vdi_sigma_alt,
        seed=args.seed,
    )
    print(
        f"[VDI] n_prompts={vdi_cfg.n_prompts}, k_noise={vdi_cfg.k_noise}, "
        f"sigma={vdi_cfg.sigma}, sigma_alt={vdi_cfg.sigma_alt}"
    )

    prompts_vdi = load_pythia_prompts(vdi_cfg.n_prompts, seed=vdi_cfg.seed)
    tokens_vdi = model.to_tokens(prompts_vdi)

    out_dir = Path(args.out_dir)

    # Main sigma
    vdi_main = compute_vdi_for_sigma(
        model=model,
        tokens=tokens_vdi,
        sigma=vdi_cfg.sigma,
        k_noise=vdi_cfg.k_noise,
        seed=vdi_cfg.seed,
        device=device,
    )
    vdi_main_path = save_vdi_results(out_dir, args.model_name, vdi_main)

    # Alternate sigma (sanity check)
    if vdi_cfg.sigma_alt > 0:
        vdi_alt = compute_vdi_for_sigma(
            model=model,
            tokens=tokens_vdi,
            sigma=vdi_cfg.sigma_alt,
            k_noise=vdi_cfg.k_noise,
            seed=vdi_cfg.seed,
            device=device,
        )
        save_vdi_results(out_dir, args.model_name, vdi_alt)
    else:
        vdi_alt = None

    # Determine heads to use for drift experiment
    if args.drift_heads_list:
        head_list = [
            int(h.strip()) for h in args.drift_heads_list.split(",") if h.strip()
        ]
        drift_heads = head_list
        print(f"[Drift] Using fixed head list for drift: {drift_heads}")
    else:
        per_head = vdi_main["per_head"]
        per_head_sorted = sorted(per_head, key=lambda r: r["vdi_effect"], reverse=True)
        drift_heads = [row["head"] for row in per_head_sorted[: args.drift_heads]]
        print(f"[Drift] Top heads by VDI_effect (sigma={vdi_cfg.sigma}): {drift_heads}")

    # Phase 1b: Drift experiment on a separate prompt set
    drift_cfg = DriftConfig(
        n_prompts=args.drift_prompts,
        max_new_tokens=args.drift_max_new,
        temperature=args.drift_temperature,
        top_p=args.drift_top_p,
        n_samples=args.drift_samples,
        seed=args.seed,
        heads_per_drift=args.drift_heads,
    )

    prompts_drift = load_pythia_prompts(drift_cfg.n_prompts, seed=drift_cfg.seed + 1)
    # To keep runtime bounded, we can truncate if necessary
    prompts_drift = prompts_drift[: drift_cfg.n_prompts]
    print(
        f"[Drift] prompts={len(prompts_drift)}, "
        f"max_new_tokens={drift_cfg.max_new_tokens}, "
        f"n_samples={drift_cfg.n_samples}"
    )

    # Base trajectories (no ablation)
    base_stats = run_drift_trajectories(
        model=model,
        prompts=prompts_drift,
        cfg=drift_cfg,
        device=device,
        ablate_head=None,
        abl_layer_index=args.abl_layer_index,
    )

    head_stats: Dict[int, Dict[str, np.ndarray]] = {}
    for head in drift_heads:
        print(f"[Drift] Running ablation trajectories for head {head}...")
        stats = run_drift_trajectories(
            model=model,
            prompts=prompts_drift,
            cfg=drift_cfg,
            device=device,
            ablate_head=head,
            abl_layer_index=args.abl_layer_index,
        )
        head_stats[head] = stats

    drift_path = save_drift_results(
        out_dir=out_dir,
        model_name=args.model_name,
        cfg=drift_cfg,
        base_stats=base_stats,
        head_stats=head_stats,
    )

    # Phase 1c: Between-condition divergence metrics for the same heads
    between_summary, _, between_rows = run_between_condition_metrics(
        model=model,
        prompts=prompts_drift,
        cfg=drift_cfg,
        device=device,
        heads=drift_heads,
        abl_layer_index=args.abl_layer_index,
    )
    between_path = save_between_results(
        out_dir=out_dir,
        model_name=args.model_name,
        rows=between_rows,
    )

    # Minimal manifest
    manifest = {
        "model_name": args.model_name,
        "device": device,
        "vdi_main_csv": str(vdi_main_path),
        "drift_csv": str(drift_path),
        "between_csv": str(between_path),
        "vdi_sigma": vdi_cfg.sigma,
        "vdi_sigma_alt": vdi_cfg.sigma_alt,
        "top_heads": drift_heads,
        "n_layers": int(n_layers),
        "drift_mid_layer_index": int(mid_layer_index),
        "drift_last_layer_index": int(last_layer_index),
    }
    manifest_path = out_dir / "pythia_layer0_vdi_drift_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[Manifest] Wrote {manifest_path}")


if __name__ == "__main__":
    main()
