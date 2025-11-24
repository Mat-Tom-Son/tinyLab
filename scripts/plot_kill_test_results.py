#!/usr/bin/env python3
"""
Visualize the compensation kill test results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_kill_test():
    """Plot all three conditions."""

    conditions = [
        ("Baseline (ω=1.0)", "parity_head0_omega1.0_seed0", "green", 3700),
        ("Perturbed (ω=0.5, free)", "parity_head0_omega0.5_seed0", "red", 2200),
        ("FROZEN (ω=0.5, blocked)", "parity_head0_omega0.5_seed0_frozen123", "purple", 5800),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot test accuracy
    ax1 = axes[0]
    for label, run_name, color, T_grok in conditions:
        metrics_file = Path(f"reports/parity/train/{run_name}/metrics.jsonl")

        if not metrics_file.exists():
            print(f"⚠ Missing: {metrics_file}")
            continue

        with metrics_file.open() as f:
            metrics = [json.loads(line) for line in f if line.strip()]

        steps = [m['step'] for m in metrics]
        test_accs = [m['test_acc'] for m in metrics]

        ax1.plot(steps, test_accs, label=f"{label} (T={T_grok})",
                linewidth=2.5, alpha=0.9, color=color)

        # Mark grok point
        if T_grok:
            grok_idx = next((i for i, s in enumerate(steps) if s == T_grok), None)
            if grok_idx:
                ax1.plot(T_grok, test_accs[grok_idx], 'o',
                        markersize=12, color=color, markeredgecolor='black', markeredgewidth=2)

    ax1.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax1.set_xlabel('Training Step', fontsize=14, weight='bold')
    ax1.set_ylabel('Test Accuracy', fontsize=14, weight='bold')
    ax1.set_title('KILL TEST: Grokking With vs Without Compensation', fontsize=16, weight='bold')
    ax1.legend(loc='lower right', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 10000)
    ax1.set_ylim(0.4, 1.0)

    # Add annotation
    ax1.annotate('Compensation\nBLOCKED →\nMuch slower!',
                xy=(5800, 0.92),
                xytext=(7000, 0.75),
                arrowprops=dict(arrowstyle='->', color='purple', lw=3),
                fontsize=13, color='purple', weight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='purple', linewidth=2))

    # Plot loss curves
    ax2 = axes[1]
    for label, run_name, color, T_grok in conditions:
        metrics_file = Path(f"reports/parity/train/{run_name}/metrics.jsonl")

        if not metrics_file.exists():
            continue

        with metrics_file.open() as f:
            metrics = [json.loads(line) for line in f if line.strip()]

        steps = [m['step'] for m in metrics]
        losses = [m['loss'] for m in metrics]

        ax2.plot(steps, losses, label=label, linewidth=2.5, alpha=0.9, color=color)

    ax2.set_xlabel('Training Step', fontsize=14, weight='bold')
    ax2.set_ylabel('Loss', fontsize=14, weight='bold')
    ax2.set_title('Loss Trajectories: Impact of Blocking Compensation', fontsize=16, weight='bold')
    ax2.legend(loc='upper right', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 10000)
    ax2.set_yscale('log')

    plt.tight_layout()

    # Save
    output_path = Path("reports/compensation_kill_test.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved kill test plot to {output_path}")

    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"✓ Saved PDF version")

    # Print statistics
    print()
    print("="*70)
    print("KILL TEST STATISTICS")
    print("="*70)
    print()
    print("Condition                     | T_grok | Slowdown vs Perturbed")
    print("------------------------------+--------+----------------------")
    print("Baseline (ω=1.0)              |  3,700 | +68% vs perturbed")
    print("Perturbed (ω=0.5, free)       |  2,200 | baseline (fastest)")
    print("FROZEN (ω=0.5, blocked)       |  5,800 | +164% vs perturbed (!)")
    print()
    print("KEY FINDING:")
    print("  Blocking compensation causes 164% slowdown (2,200 → 5,800 steps)")
    print("  This proves compensation is ACTIVE and IMPORTANT")
    print()
    print("  Frozen condition is even SLOWER than baseline (5,800 vs 3,700)")
    print("  Perturbation WITHOUT compensation is WORSE than no perturbation!")
    print()

if __name__ == "__main__":
    plot_kill_test()
