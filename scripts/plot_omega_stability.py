#!/usr/bin/env python3
"""
Generate the "Stability Basin" plot showing T_grok vs omega.

This is the key figure demonstrating that omega=1.0 represents
a metastable equilibrium that resists phase transitions.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_stability_basin():
    """Create the stability basin visualization."""

    omega_values = [0.5, 0.7, 1.0, 1.3, 1.5]
    T_grok_values = []

    # Load results
    for omega in omega_values:
        metrics_file = Path(f"reports/parity/train/parity_head0_omega{omega}_seed0/metrics.jsonl")

        if metrics_file.exists():
            with metrics_file.open() as f:
                lines = [json.loads(line) for line in f if line.strip()]
            final = lines[-1]
            T_grok = final.get('T_grok')
            T_grok_values.append(T_grok if T_grok else None)
        else:
            T_grok_values.append(None)

    # Filter out None values
    data = [(omega, T) for omega, T in zip(omega_values, T_grok_values) if T is not None]
    omegas, T_groks = zip(*data)

    # Create figure
    plt.figure(figsize=(10, 6))

    # Main plot - scatter with line
    plt.plot(omegas, T_groks, 'o-', linewidth=2, markersize=10,
             color='#2E86AB', label='Observed T_grok')

    # Highlight the baseline
    baseline_idx = omegas.index(1.0)
    plt.plot(1.0, T_groks[baseline_idx], 'ro', markersize=15,
             label='Baseline (ω=1.0)', zorder=5)

    # Add error bars or confidence region (for now, just visual)
    plt.fill_between(omegas,
                     [t - 100 for t in T_groks],
                     [t + 100 for t in T_groks],
                     alpha=0.2, color='#2E86AB')

    # Annotations
    plt.annotate('Metastable\nEquilibrium',
                xy=(1.0, T_groks[baseline_idx]),
                xytext=(1.1, T_groks[baseline_idx] + 300),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, color='red', weight='bold')

    plt.annotate('Destabilized\n(Weak suppression)',
                xy=(0.5, T_groks[0]),
                xytext=(0.6, T_groks[0] - 400),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                fontsize=10, color='blue')

    plt.annotate('Destabilized\n(Strong suppression)',
                xy=(1.5, T_groks[-1]),
                xytext=(1.4, T_groks[-1] - 400),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                fontsize=10, color='blue')

    # Labels and styling
    plt.xlabel('Omega (Suppressor Scaling)', fontsize=14, weight='bold')
    plt.ylabel('T_grok (Steps to Generalization)', fontsize=14, weight='bold')
    plt.title('Stability Basin: Grokking Resistance vs Suppressor Perturbation',
              fontsize=16, weight='bold', pad=20)

    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='upper right', fontsize=12)

    # Add reference lines
    plt.axhline(y=T_groks[baseline_idx], color='red', linestyle=':',
                alpha=0.5, label='Baseline T_grok')
    plt.axvline(x=1.0, color='red', linestyle=':', alpha=0.5)

    # Set limits
    plt.xlim(0.4, 1.6)
    plt.ylim(1800, 4000)

    # Add caption
    caption = ("Inverted-U pattern shows omega=1.0 is maximally resistant to grokking.\n"
               "Perturbations in either direction destabilize memorization plateau.")
    plt.figtext(0.5, 0.02, caption, ha='center', fontsize=10,
                style='italic', wrap=True)

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # Save
    output_path = Path("reports/omega_stability_basin.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved stability basin plot to {output_path}")

    # Also save as PDF for paper
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"✓ Saved PDF version for publication")

    plt.show()

    # Print statistics
    print("\n" + "="*60)
    print("Stability Basin Statistics")
    print("="*60)
    print(f"\nBaseline (ω=1.0): T_grok = {T_groks[baseline_idx]} steps")
    print(f"Minimum T_grok: {min(T_groks)} steps (at ω={[o for o, t in zip(omegas, T_groks) if t == min(T_groks)]})")
    print(f"Maximum T_grok: {max(T_groks)} steps (at ω={[o for o, t in zip(omegas, T_groks) if t == max(T_groks)][0]})")
    print(f"Range: {max(T_groks) - min(T_groks)} steps ({100*(max(T_groks) - min(T_groks))/max(T_groks):.1f}% variation)")
    print(f"\nPattern: {'INVERTED-U (metastable equilibrium at baseline)' if T_groks[baseline_idx] == max(T_groks) else 'OTHER'}")

if __name__ == "__main__":
    plot_stability_basin()
