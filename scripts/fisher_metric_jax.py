
import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
from jax import jacobian
import torch
import torch.nn.functional as F
import numpy as np

# Add local directory to path to import train_stage1b_grokking
sys.path.append(os.getcwd())
# Try-except block to handle potential import errors if running from different pwd
try:
    from scripts.train_stage1b_grokking import GrokkingTransformer, GrokkingConfig, prepare_batch, load_modular_data
except ImportError:
    # Fallback if running directly from scripts/
    sys.path.append(str(Path(os.getcwd()).parent))
    from scripts.train_stage1b_grokking import GrokkingTransformer, GrokkingConfig, prepare_batch, load_modular_data

def compute_entropy(probs):
    """Compute Shannon entropy in nats."""
    # H = - sum p log p
    # Add eps for stability
    return -jnp.sum(probs * jnp.log(probs + 1e-10))

def compute_edi(probs):
    """Compute Entropy Dispersion Index (EDI)."""
    # EDI = H / H_max
    n = probs.shape[-1]
    h_max = jnp.log(n)
    h = compute_entropy(probs)
    return h / h_max

def fisher_metric(params):
    """
    Compute Fisher Information Metric for categorical distribution w.r.t logits.
    F_ij = sum_c p_c * (d log p_c / d theta_i) * (d log p_c / d theta_j)
    
    For softmax, Jacobian J_c = d p_c / d theta = diag(p) - p p^T  (w.r.t logits? No)
    d log p_c / d theta_i = (d p_c / d theta_i) / p_c
                          = (p_c (delta_ci - p_i)) / p_c
                          = delta_ci - p_i
    
    So grad_log_p_c = e_c - p
    
    F = sum_c p_c (e_c - p)(e_c - p)^T
      = sum_c p_c (e_c e_c^T - e_c p^T - p e_c^T + p p^T)
      = sum_c p_c e_c e_c^T - sum_c p_c e_c p^T - sum_c p_c p e_c^T + sum_c p_c p p^T
      = diag(p) - p p^T - p p^T + p p^T (since sum p_c = 1)
      = diag(p) - p p^T
      
    This is the standard result for Fisher w.r.t natural parameters (logits).
    """
    
    # We can compute this analytically for speed and precision
    p = jax.nn.softmax(params)
    
    # F = diag(p) - p p^T
    fisher = jnp.diag(p) - jnp.outer(p, p)
    
    return fisher

def compute_spectrum(fisher_matrix):
    """Compute eigenvalues, sorted descending."""
    evals = jnp.linalg.eigvalsh(fisher_matrix)
    # Sort descending
    return jnp.flip(evals)

def log_det_metric(evals, eps_list=[1e-8, 1e-6, 1e-4, 1e-3, 1e-2]):
    """Compute pseudo-log-determinant for various epsilons."""
    results = {}
    for eps in eps_list:
        # Pseudo-determinant: sum log of eigenvalues > eps
        valid = evals[evals > eps]
        if len(valid) > 0:
            results[f"logdet_eps_{eps}"] = float(jnp.sum(jnp.log(valid)))
            results[f"rank_eps_{eps}"] = int(len(valid))
        else:
            results[f"logdet_eps_{eps}"] = -float('inf')
            results[f"rank_eps_{eps}"] = 0
            
    # Trace is sum of eigenvalues (independent of eps cutoff usually)
    results["trace"] = float(jnp.sum(evals))
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()
    
    rng = jax.random.PRNGKey(args.seed)

    # Load checkpoint
    print(f"Loading checkpoint {args.ckpt}...")
    try:
        ckpt = torch.load(args.ckpt, map_location="cpu")
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {args.ckpt}")
        return

    # Reconstruct config (assuming it's in checkpoint or we infer standard)
    ckpt_path = Path(args.ckpt)
    config_path = ckpt_path.parent.parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg_dict = json.load(f)["config"]
        valid_keys = GrokkingConfig.__annotations__.keys()
        cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid_keys}
        cfg = GrokkingConfig(**cfg_dict)
    else:
        cfg = GrokkingConfig()

    model = GrokkingTransformer(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Load Data (Test set)
    test_data = load_modular_data(Path(cfg.data_path_test))
    batch = test_data[:64]
    input_ids, targets = prepare_batch(batch, cfg, torch.device("cpu"))

    # Forward pass
    print("Running model...")
    with torch.no_grad():
        logits, layer_scores = model(input_ids, return_attention_scores=True)
    
    # Analyze all heads
    # layer_scores[0] shape: [B, H, T, T]
    scores_l0 = layer_scores[0] 
    n_heads = scores_l0.shape[1]
    
    for head_idx in range(n_heads):
        print(f"\n=== Analyzing Head {head_idx} ===")
        scores_h = scores_l0[:, head_idx, 5, :6] # [B, 6]
        attn_logits_jax = jnp.array(scores_h.numpy())
        
        results_list = []
        
        for i in range(attn_logits_jax.shape[0]):
            logits_i = attn_logits_jax[i]
            
            # 1. Baseline Metrics
            p_i = jax.nn.softmax(logits_i)
            entropy = float(compute_entropy(p_i))
            edi = float(compute_edi(p_i))
            
            fisher_base = fisher_metric(logits_i)
            evals_base = compute_spectrum(fisher_base)
            metrics_base = log_det_metric(evals_base)
            
            rng, subkey = jax.random.split(rng)
            logits_shuffled = jax.random.permutation(subkey, logits_i)
            fisher_shuffled = fisher_metric(logits_shuffled)
            evals_shuffled = compute_spectrum(fisher_shuffled)
            metrics_shuffled = log_det_metric(evals_shuffled)
            
            row = {
                "sample_idx": i,
                "entropy": entropy,
                "edi": edi,
                "eigenvalues_base": [float(x) for x in evals_base[:10]], 
                **{f"base_{k}": v for k, v in metrics_base.items()},
                **{f"shuff_{k}": v for k, v in metrics_shuffled.items()}
            }
            results_list.append(row)

        # Aggregation
        avg_edi = np.mean([r["edi"] for r in results_list])
        avg_logdet_1e_4 = np.mean([r["base_logdet_eps_0.0001"] for r in results_list])
        avg_trace = np.mean([r["base_trace"] for r in results_list])
        
        # Verify Shuffling Identity
        diff_trace = np.mean([abs(r["base_trace"] - r["shuff_trace"]) for r in results_list])
        
        print(f"Avg EDI: {avg_edi:.4f}")
        print(f"Avg Trace: {avg_trace:.4f}")
        print(f"Avg LogDet (eps=1e-4): {avg_logdet_1e_4:.4f}")
        print(f"Shuffling Delta Trace: {diff_trace:.2e}")
        
        # Explicitly calculate correlation
        edis = [r["edi"] for r in results_list]
        logdets = [r["base_logdet_eps_0.0001"] for r in results_list]
        traces = [r["base_trace"] for r in results_list]
        
        if np.std(edis) > 1e-6:
            corr_edi_logdet = np.corrcoef(edis, logdets)[0, 1]
            corr_edi_trace = np.corrcoef(edis, traces)[0, 1]
            print(f"Corr(EDI, LogDet): {corr_edi_logdet:.4f}")
        else:
            print(f"Corr(EDI, LogDet): Undefined (Var=0)")

if __name__ == "__main__":
    main()
