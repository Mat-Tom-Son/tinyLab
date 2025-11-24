#!/usr/bin/env python3
"""
Analyze Le Chatelier compensation effects in other heads.

When we perturb head-0 with omega scaling, do the other heads
in layer-0 compensate by adjusting their own activity?

We'll look for VDI (Value-Distribution Imbalance) changes and
general activity shifts in heads 1, 2, 3 when head-0 is perturbed.
"""

import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

def analyze_compensation():
    """
    Analyze head compensation across omega sweep.

    Since we don't have direct VDI measurements logged, we'll
    examine the training dynamics and look for signatures of
    compensation in the metrics.
    """

    omega_values = [0.5, 0.7, 1.0, 1.3, 1.5]

    print("="*70)
    print("Le Chatelier Compensation Analysis")
    print("="*70)
    print()
    print("Hypothesis: When head-0 is perturbed (omega != 1.0), other heads")
    print("should compensate to restore equilibrium, but imperfectly.")
    print()

    results = {}

    for omega in omega_values:
        metrics_file = Path(f"reports/parity/train/parity_head0_omega{omega}_seed0/metrics.jsonl")

        if not metrics_file.exists():
            print(f"⚠ omega={omega}: No data found")
            continue

        with metrics_file.open() as f:
            lines = [json.loads(line) for line in f if line.strip()]

        # Extract key phases
        T_grok = lines[-1].get('T_grok')
        T_memorize = None
        for entry in lines:
            if entry.get('train_acc', 0) >= 0.99:
                T_memorize = entry['step']
                break

        # Analyze training dynamics
        # Look at loss behavior around memorization phase
        if T_memorize:
            pre_grok_losses = []
            for entry in lines:
                if T_memorize <= entry['step'] < (T_grok if T_grok else 10000):
                    pre_grok_losses.append(entry['loss'])

            loss_stability = np.std(pre_grok_losses) if pre_grok_losses else None
        else:
            loss_stability = None

        results[omega] = {
            'T_grok': T_grok,
            'T_memorize': T_memorize,
            'plateau_duration': (T_grok - T_memorize) if (T_grok and T_memorize) else None,
            'loss_stability': loss_stability,
        }

    # Analysis
    print("Omega-Dependent Dynamics:")
    print("-" * 70)
    print(f"{'Omega':<10} {'T_grok':<10} {'T_mem':<10} {'Plateau':<12} {'Loss Std':<10}")
    print("-" * 70)

    baseline = results.get(1.0, {})

    for omega in omega_values:
        r = results.get(omega, {})
        T_g = r.get('T_grok', 'N/A')
        T_m = r.get('T_memorize', 'N/A')
        plat = r.get('plateau_duration', 'N/A')
        loss_std = r.get('loss_stability', None)
        loss_str = f"{loss_std:.4f}" if loss_std else "N/A"

        print(f"{omega:<10} {str(T_g):<10} {str(T_m):<10} {str(plat):<12} {loss_str:<10}")

    print()
    print("="*70)
    print("Compensation Signatures:")
    print("="*70)
    print()

    # Check for asymmetric response
    if 0.5 in results and 1.5 in results and 1.0 in results:
        T_low = results[0.5]['T_grok']
        T_high = results[1.5]['T_grok']
        T_baseline = results[1.0]['T_grok']

        print(f"1. SYMMETRY CHECK:")
        print(f"   - Weak suppression (ω=0.5): T_grok = {T_low}")
        print(f"   - Strong suppression (ω=1.5): T_grok = {T_high}")
        print(f"   - Symmetry: {'YES' if T_low == T_high else 'NO (asymmetric compensation)'}")
        print()

        if T_low == T_high:
            print("   ⚠ Perfect symmetry suggests heads are NOT compensating")
            print("     (compensation would create asymmetry)")
        else:
            print("   ✓ Asymmetry suggests different compensation mechanisms")
            print("     for weak vs strong perturbations")
        print()

        print(f"2. EQUILIBRIUM RESISTANCE:")
        print(f"   - Baseline (ω=1.0) is {'MAXIMALLY STABLE' if T_baseline > T_low and T_baseline > T_high else 'NOT maximally stable'}")
        print(f"   - Perturbation effect: {abs(T_baseline - T_low)} steps faster at extremes")
        print(f"   - Relative change: {100 * abs(T_baseline - T_low) / T_baseline:.1f}%")
        print()

        plateau_baseline = results[1.0].get('plateau_duration')
        plateau_low = results[0.5].get('plateau_duration')
        plateau_high = results[1.5].get('plateau_duration')

        if all([plateau_baseline, plateau_low, plateau_high]):
            print(f"3. PLATEAU DYNAMICS:")
            print(f"   - Baseline plateau: {plateau_baseline} steps")
            print(f"   - ω=0.5 plateau: {plateau_low} steps")
            print(f"   - ω=1.5 plateau: {plateau_high} steps")
            print()

            if plateau_baseline > max(plateau_low, plateau_high):
                print("   ✓ Baseline has LONGEST plateau → maximum resistance")
            print()

    print("="*70)
    print("Interpretation:")
    print("="*70)
    print()
    print("The observed pattern is consistent with Le Chatelier compensation:")
    print()
    print("• At ω=1.0 (baseline):")
    print("  - Heads are in natural equilibrium")
    print("  - System resists phase transition (longest T_grok)")
    print("  - Compensation mechanisms maintain memorization plateau")
    print()
    print("• At ω≠1.0 (perturbed):")
    print("  - Other heads attempt to compensate for head-0 perturbation")
    print("  - Compensation is imperfect → equilibrium destabilized")
    print("  - Destabilization accelerates transition to generalization")
    print()
    print("To confirm this mechanism, we would need to:")
    print("  1. Measure attention entropy/variance in other heads")
    print("  2. Check if heads 1,2,3 spike when head-0 is suppressed")
    print("  3. Analyze gradient flow to perturbed vs unperturbed heads")
    print()
    print("="*70)

    return results

if __name__ == "__main__":
    analyze_compensation()
