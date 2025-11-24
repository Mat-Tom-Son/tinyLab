#!/usr/bin/env python3
"""
Measure VDI (Value-Distribution Imbalance) across all heads.

VDI measures whether a head acts as a suppressor (flattens distribution)
or amplifier (sharpens distribution).

When we perturb head-0 with omega, do other heads compensate?
- If omega=0.5 (weaken head-0), do heads 1,2,3 become more suppressive?
- If omega=1.5 (strengthen head-0), do heads 1,2,3 become less suppressive?
"""

import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from train_parity import ParityTransformer, ParityConfig, load_parity_data, prepare_batch


def compute_attention_entropy(attn_weights):
    """
    Compute entropy of attention distribution.

    High entropy = flattened distribution (suppressor)
    Low entropy = peaked distribution (amplifier)

    Args:
        attn_weights: [batch, heads, seq, seq] attention weights

    Returns:
        entropy per head: [heads]
    """
    # attn_weights: [batch, heads, seq_query, seq_key]
    # We want to measure how "flat" vs "peaked" the distribution is

    # Compute entropy over keys for each query position
    # H = -sum(p * log(p))
    eps = 1e-10
    log_attn = torch.log(attn_weights + eps)
    entropy = -(attn_weights * log_attn).sum(dim=-1)  # [batch, heads, seq_query]

    # Average over batch and query positions
    mean_entropy = entropy.mean(dim=[0, 2])  # [heads]

    return mean_entropy.cpu().numpy()


def compute_attention_variance(attn_weights):
    """
    Compute variance of attention distribution.

    High variance = peaked (some positions get high attention)
    Low variance = flat (attention spread evenly)

    Args:
        attn_weights: [batch, heads, seq, seq]

    Returns:
        variance per head: [heads]
    """
    # Variance across attention targets
    var = attn_weights.var(dim=-1)  # [batch, heads, seq_query]
    mean_var = var.mean(dim=[0, 2])  # [heads]

    return mean_var.cpu().numpy()


def compute_vdi_proxy(attn_weights):
    """
    VDI proxy: ratio of entropy to max possible entropy.

    VDI = H / H_max
    - VDI ≈ 1.0: suppressor (flat distribution)
    - VDI ≈ 0.0: amplifier (peaked distribution)

    Args:
        attn_weights: [batch, heads, seq, seq]

    Returns:
        VDI per head: [heads]
    """
    entropy = compute_attention_entropy(attn_weights)

    # Max entropy = log(seq_len)
    seq_len = attn_weights.shape[-1]
    max_entropy = np.log(seq_len)

    vdi = entropy / max_entropy

    return vdi


def analyze_head_compensation():
    """
    Analyze VDI across omega values to detect compensation.
    """

    print("="*70)
    print("VDI Compensation Analysis")
    print("="*70)
    print()
    print("Measuring attention patterns in ALL heads across omega sweep")
    print("to detect Le Chatelier compensation effects.")
    print()

    omega_values = [0.5, 0.7, 1.0, 1.3, 1.5]
    device = torch.device("cpu")

    # Load test data (same for all runs)
    cfg = ParityConfig(data_path_train="data_parity_medium/parity_train.jsonl",
                       data_path_test="data_parity_medium/parity_test.jsonl")
    test_data = load_parity_data(Path(cfg.data_path_test))

    results = {}

    for omega in omega_values:
        print(f"Analyzing omega={omega}...")

        # Load final checkpoint
        model_dir = Path(f"reports/parity/train/parity_head0_omega{omega}_seed0")
        final_model = model_dir / "final_model.pt"

        if not final_model.exists():
            print(f"  ⚠ Model not found, skipping")
            continue

        # Load model
        model = ParityTransformer(cfg).to(device)
        model.load_state_dict(torch.load(final_model, map_location=device))
        model.eval()

        # Collect attention patterns on test set
        all_entropies = []
        all_variances = []
        all_vdis = []

        with torch.no_grad():
            # Use first 100 test examples for speed
            batch = test_data[:100]
            input_ids, targets = prepare_batch(batch, cfg, device)

            # Forward pass through first layer to get attention
            bsz, seq_len = input_ids.shape
            pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, seq_len)
            x = model.token_emb(input_ids) + model.pos_emb(pos)

            # First transformer block (where we perturbed head-0)
            block = model.blocks[0]
            h = block.ln1(x)

            # Compute attention weights
            q = block.W_q(h).view(bsz, seq_len, block.n_heads, block.d_head)
            k = block.W_k(h).view(bsz, seq_len, block.n_heads, block.d_head)

            attn_scores = torch.einsum("bthd,bshd->bhts", q, k) / np.sqrt(block.d_head)
            attn_mask = model.attn_mask[:, :, :seq_len, :seq_len]
            attn_scores = attn_scores + attn_mask
            attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)

            # Measure VDI for each head
            entropy = compute_attention_entropy(attn_weights)
            variance = compute_attention_variance(attn_weights)
            vdi = compute_vdi_proxy(attn_weights)

            all_entropies.append(entropy)
            all_variances.append(variance)
            all_vdis.append(vdi)

        # Average across batches
        mean_entropy = np.mean(all_entropies, axis=0)
        mean_variance = np.mean(all_variances, axis=0)
        mean_vdi = np.mean(all_vdis, axis=0)

        results[omega] = {
            'entropy': mean_entropy,
            'variance': mean_variance,
            'vdi': mean_vdi,
        }

        print(f"  ✓ Measured {block.n_heads} heads")

    # Print results
    print()
    print("="*70)
    print("Results: Attention Entropy by Head")
    print("="*70)
    print()
    print("(Higher entropy = more suppressive/flattened distribution)")
    print()
    print(f"{'Omega':<10} {'Head-0':<12} {'Head-1':<12} {'Head-2':<12} {'Head-3':<12}")
    print("-"*70)

    baseline_vdi = results[1.0]['vdi']

    for omega in omega_values:
        r = results[omega]
        entropy = r['entropy']
        print(f"{omega:<10} ", end="")
        for h in range(len(entropy)):
            print(f"{entropy[h]:.4f}       ", end="")
        print()

    print()
    print("="*70)
    print("Results: VDI (Value-Distribution Imbalance)")
    print("="*70)
    print()
    print("VDI = Entropy / MaxEntropy (1.0 = flat/suppressive, 0.0 = peaked/amplifying)")
    print()
    print(f"{'Omega':<10} {'Head-0':<12} {'Head-1':<12} {'Head-2':<12} {'Head-3':<12}")
    print("-"*70)

    for omega in omega_values:
        r = results[omega]
        vdi = r['vdi']
        print(f"{omega:<10} ", end="")
        for h in range(len(vdi)):
            delta = vdi[h] - baseline_vdi[h]
            sign = "+" if delta > 0 else ""
            print(f"{vdi[h]:.4f} ({sign}{delta:.3f}) ", end="")
        print()

    print()
    print("="*70)
    print("Compensation Analysis")
    print("="*70)
    print()

    # Check for inverse correlation
    print("HEAD-0 PERTURBATION vs OTHER HEADS RESPONSE:")
    print()

    # Compare extreme perturbations to baseline
    if 0.5 in results and 1.5 in results and 1.0 in results:
        vdi_weak = results[0.5]['vdi']
        vdi_strong = results[1.5]['vdi']
        vdi_baseline = results[1.0]['vdi']

        print("When HEAD-0 is WEAKENED (ω=0.5):")
        print(f"  Head-0 VDI: {vdi_weak[0]:.4f} (baseline: {vdi_baseline[0]:.4f}, Δ={vdi_weak[0]-vdi_baseline[0]:+.3f})")
        print("  Other heads should COMPENSATE (become more suppressive):")
        for h in range(1, len(vdi_weak)):
            delta = vdi_weak[h] - vdi_baseline[h]
            comp = "✓ COMPENSATING" if delta > 0.01 else "- no change" if abs(delta) < 0.01 else "✗ ANTI-COMPENSATING"
            print(f"    Head-{h}: {vdi_weak[h]:.4f} (Δ={delta:+.3f}) {comp}")
        print()

        print("When HEAD-0 is STRENGTHENED (ω=1.5):")
        print(f"  Head-0 VDI: {vdi_strong[0]:.4f} (baseline: {vdi_baseline[0]:.4f}, Δ={vdi_strong[0]-vdi_baseline[0]:+.3f})")
        print("  Other heads should COMPENSATE (become less suppressive):")
        for h in range(1, len(vdi_strong)):
            delta = vdi_strong[h] - vdi_baseline[h]
            comp = "✓ COMPENSATING" if delta < -0.01 else "- no change" if abs(delta) < 0.01 else "✗ ANTI-COMPENSATING"
            print(f"    Head-{h}: {vdi_strong[h]:.4f} (Δ={delta:+.3f}) {comp}")
        print()

        # Summary
        print("="*70)
        print("COMPENSATION VERDICT:")
        print("="*70)
        print()

        # Check if other heads show inverse correlation with head-0 perturbation
        weak_comp_count = sum(1 for h in range(1, len(vdi_weak)) if (vdi_weak[h] - vdi_baseline[h]) > 0.01)
        strong_comp_count = sum(1 for h in range(1, len(vdi_strong)) if (vdi_strong[h] - vdi_baseline[h]) < -0.01)

        if weak_comp_count >= 2 or strong_comp_count >= 2:
            print("✓ COMPENSATION DETECTED")
            print()
            print("Evidence:")
            print(f"  - {weak_comp_count}/3 heads compensate when head-0 weakened")
            print(f"  - {strong_comp_count}/3 heads compensate when head-0 strengthened")
            print()
            print("This supports the Le Chatelier hypothesis:")
            print("  When head-0 is perturbed, other heads attempt to restore equilibrium")
            print("  by adjusting their suppression/amplification in the opposite direction.")
        else:
            print("⚠ LIMITED OR NO COMPENSATION DETECTED")
            print()
            print(f"  - Only {weak_comp_count}/3 heads compensate when head-0 weakened")
            print(f"  - Only {strong_comp_count}/3 heads compensate when head-0 strengthened")
            print()
            print("This suggests:")
            print("  - Compensation may be weak or operate through different mechanisms")
            print("  - The stability basin effect may arise from other dynamics")
            print("  - Further investigation needed (gradient flow, other layers)")

    print()
    print("="*70)

    return results


if __name__ == "__main__":
    analyze_head_compensation()
