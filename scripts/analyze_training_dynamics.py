#!/usr/bin/env python3
"""
Analyze training dynamics to detect compensation effects.

The final model states show identical VDI, but compensation
might occur DURING training and then equilibrate.

Let's look for signatures of compensation in the training curves themselves.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def analyze_dynamics_compensation():
    """
    Analyze training dynamics for compensation signatures.
    """

    print("="*70)
    print("Training Dynamics Analysis for Compensation")
    print("="*70)
    print()

    omega_values = [0.5, 0.7, 1.0, 1.3, 1.5]

    results = {}

    for omega in omega_values:
        metrics_file = Path(f"reports/parity/train/parity_head0_omega{omega}_seed0/metrics.jsonl")

        if not metrics_file.exists():
            continue

        with metrics_file.open() as f:
            metrics = [json.loads(line) for line in f if line.strip()]

        # Extract key trajectories
        steps = [m['step'] for m in metrics]
        losses = [m['loss'] for m in metrics]
        train_accs = [m['train_acc'] for m in metrics]
        test_accs = [m['test_acc'] for m in metrics]

        # Analyze loss trajectory shape
        # Hypothesis: Compensation creates different loss dynamics

        # 1. Loss descent rate (early training)
        early_steps = [i for i, s in enumerate(steps) if s <= 1000]
        if len(early_steps) > 2:
            early_losses = [losses[i] for i in early_steps]
            early_descent_rate = (early_losses[0] - early_losses[-1]) / len(early_steps)
        else:
            early_descent_rate = 0

        # 2. Loss stability in plateau (between memorization and grokking)
        T_grok = metrics[-1].get('T_grok')
        if T_grok:
            plateau_steps = [i for i, s in enumerate(steps) if 1000 <= s < T_grok]
            if plateau_steps:
                plateau_losses = [losses[i] for i in plateau_steps]
                plateau_volatility = np.std(plateau_losses)
            else:
                plateau_volatility = 0
        else:
            plateau_volatility = 0

        # 3. Grokking sharpness (how fast does test acc jump?)
        if T_grok:
            # Find test acc before and after grok
            grok_idx = next((i for i, s in enumerate(steps) if s == T_grok), None)
            if grok_idx and grok_idx > 0:
                pre_grok_acc = test_accs[grok_idx - 1] if grok_idx > 0 else test_accs[0]
                post_grok_steps = [i for i, s in enumerate(steps) if s >= T_grok + 500]
                if post_grok_steps:
                    post_grok_acc = test_accs[post_grok_steps[0]]
                    grok_sharpness = (post_grok_acc - pre_grok_acc) / 500  # acc gain per step
                else:
                    grok_sharpness = 0
            else:
                grok_sharpness = 0
        else:
            grok_sharpness = 0

        results[omega] = {
            'T_grok': T_grok,
            'early_descent_rate': early_descent_rate,
            'plateau_volatility': plateau_volatility,
            'grok_sharpness': grok_sharpness,
            'final_loss': losses[-1],
            'steps': steps,
            'losses': losses,
            'test_accs': test_accs,
        }

    # Print analysis
    print("TRAINING DYNAMICS SIGNATURES:")
    print("-"*70)
    print(f"{'Omega':<10} {'T_grok':<10} {'Early Descent':<15} {'Plateau Vol':<15} {'Grok Sharp':<12}")
    print("-"*70)

    for omega in omega_values:
        r = results[omega]
        print(f"{omega:<10} {str(r['T_grok']):<10} {r['early_descent_rate']:<15.6f} "
              f"{r['plateau_volatility']:<15.6f} {r['grok_sharpness']:<12.6f}")

    print()
    print("="*70)
    print("INTERPRETATION:")
    print("="*70)
    print()

    # Check for patterns
    baseline = results[1.0]

    print("Early Descent Rate (first 1000 steps):")
    for omega in omega_values:
        delta = results[omega]['early_descent_rate'] - baseline['early_descent_rate']
        if abs(delta) > 0.0001:
            direction = "FASTER" if delta > 0 else "SLOWER"
            print(f"  ω={omega}: {direction} descent (Δ={delta:+.6f})")
    print()

    print("Plateau Volatility (loss variance during memorization):")
    for omega in omega_values:
        delta = results[omega]['plateau_volatility'] - baseline['plateau_volatility']
        if abs(delta) > 0.001:
            direction = "MORE VOLATILE" if delta > 0 else "MORE STABLE"
            print(f"  ω={omega}: {direction} (Δ={delta:+.6f})")
    print()

    print("Grokking Sharpness (how sudden is the transition):")
    for omega in omega_values:
        r = results[omega]
        if r['grok_sharpness'] > 0:
            rel = r['grok_sharpness'] / baseline['grok_sharpness'] if baseline['grok_sharpness'] > 0 else 1.0
            print(f"  ω={omega}: {r['grok_sharpness']:.6f} ({rel:.2f}x baseline)")
    print()

    # Create visualization
    print("="*70)
    print("Generating training curves comparison...")
    print("="*70)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Loss curves
    ax1 = axes[0]
    colors = {'0.5': '#E63946', '0.7': '#F77F00', '1.0': '#06A77D',
              '1.3': '#4361EE', '1.5': '#7209B7'}

    for omega in omega_values:
        r = results[omega]
        ax1.plot(r['steps'], r['losses'], label=f'ω={omega}',
                linewidth=2, alpha=0.8, color=colors.get(str(omega), 'gray'))

    ax1.set_xlabel('Training Step', fontsize=12, weight='bold')
    ax1.set_ylabel('Loss', fontsize=12, weight='bold')
    ax1.set_title('Loss Trajectories Across Omega Values', fontsize=14, weight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 10000)

    # Plot 2: Test accuracy curves
    ax2 = axes[1]

    for omega in omega_values:
        r = results[omega]
        T_grok = r['T_grok']
        ax2.plot(r['steps'], r['test_accs'], label=f'ω={omega} (T={T_grok})',
                linewidth=2, alpha=0.8, color=colors.get(str(omega), 'gray'))

        # Mark grokking point
        if T_grok:
            grok_idx = next((i for i, s in enumerate(r['steps']) if s == T_grok), None)
            if grok_idx:
                ax2.plot(T_grok, r['test_accs'][grok_idx], 'o',
                        markersize=10, color=colors.get(str(omega), 'gray'))

    ax2.set_xlabel('Training Step', fontsize=12, weight='bold')
    ax2.set_ylabel('Test Accuracy', fontsize=12, weight='bold')
    ax2.set_title('Grokking Dynamics: Test Accuracy Over Time', fontsize=14, weight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='Grok threshold')
    ax2.set_xlim(0, 10000)
    ax2.set_ylim(0.4, 1.0)

    plt.tight_layout()

    output_path = Path("reports/training_dynamics_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved training dynamics plot to {output_path}")

    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"✓ Saved PDF version")

    print()
    print("="*70)
    print("COMPENSATION HYPOTHESIS ASSESSMENT:")
    print("="*70)
    print()
    print("The identical final VDI values across all omega suggest:")
    print()
    print("1. FINAL STATE CONVERGENCE:")
    print("   - All configurations converge to similar attention patterns")
    print("   - After grokking, the 'solution' looks the same")
    print("   - VDI compensation (if it exists) happens DURING training, not at end")
    print()
    print("2. ALTERNATIVE COMPENSATION MECHANISMS:")
    print("   - Compensation might be in MLP layers (not just attention)")
    print("   - Compensation might be in Layer-1 (we only checked Layer-0)")
    print("   - Compensation might be in gradient flow (not weight patterns)")
    print()
    print("3. STABILITY BASIN MECHANISM:")
    print("   - The stability basin may NOT arise from head compensation")
    print("   - Instead, it may arise from:")
    print("     a) Optimization dynamics (loss landscape geometry)")
    print("     b) Weight decay interaction with omega perturbation")
    print("     c) Critical slowing down near natural equilibrium")
    print()
    print("RECOMMENDATION:")
    print("  - Analyze gradients during training (not just final weights)")
    print("  - Check Layer-1 heads for compensation")
    print("  - Examine MLP activations across omega values")
    print()

    return results


if __name__ == "__main__":
    analyze_dynamics_compensation()
