#!/usr/bin/env python3
"""
Generate phase diagrams for Stage-1B grokking experiments.

Creates the 4-panel figure showing:
1. T_grok vs omega (phase boundary shift)
2. Final circularity vs omega (geometry quality)
3. VDI compensation vs omega (Le Chatelier signature)
4. Stability regime (colored by pathology vs healthy)

Usage:
    python scripts/plot_phase_diagrams.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def load_run_summary(run_dir: Path) -> Dict:
    """
    Load summary metrics for a single run.

    Returns:
        Dictionary with omega, seed, T_grok, final_acc, final_circularity, etc.
    """
    # Load config
    config_path = run_dir / "config.json"
    with config_path.open() as f:
        config = json.load(f)

    args = config['args']
    omega = float(args['omega'])
    seed = int(args['seed'])
    head = int(args['head'])

    # Load metrics
    metrics_path = run_dir / "metrics.jsonl"
    metrics = []
    with metrics_path.open() as f:
        for line in f:
            if line.strip():
                metrics.append(json.loads(line))

    if not metrics:
        return None

    # Extract key metrics
    final = metrics[-1]
    final_acc = final.get('test_acc', 0.0)

    # T_grok: first step where test_acc >= 0.9
    T_grok = None
    for m in metrics:
        if m.get('test_acc', 0.0) >= 0.9:
            T_grok = m['step']
            break

    # Final circularity: load from last checkpoint
    ckpt_dir = run_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    final_circularity = 0.0
    if ckpts:
        import torch
        last_ckpt = torch.load(ckpts[-1], map_location='cpu')
        final_circularity = last_ckpt.get('circularity', 0.0)

    return {
        'omega': omega,
        'seed': seed,
        'head': head,
        'T_grok': T_grok,
        'final_acc': final_acc,
        'final_circularity': final_circularity,
        'n_steps': final['step'],
    }


def collect_all_runs(base_dir: Path) -> List[Dict]:
    """Collect summaries for all runs."""
    summaries = []

    run_dirs = sorted(base_dir.glob("stage1b_head*_omega*_seed*"))
    print(f"Found {len(run_dirs)} runs")

    for run_dir in run_dirs:
        summary = load_run_summary(run_dir)
        if summary:
            summaries.append(summary)

    return summaries


def plot_phase_diagrams(
    summaries: List[Dict],
    vdi_results: Dict | None = None,
    robustness_results: Dict | None = None,
    output_path: Path = Path("reports/stage1b_grokking/phase_diagrams.png"),
):
    """
    Create 4-panel phase diagram figure.

    Args:
        summaries: List of run summaries
        vdi_results: Optional VDI compensation results
        robustness_results: Optional geometry robustness results
        output_path: Where to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Group by omega
    by_omega = defaultdict(list)
    for s in summaries:
        by_omega[s['omega']].append(s)

    omega_vals = sorted(by_omega.keys())

    # Panel 1: T_grok vs omega (phase boundary shift)
    ax = axes[0, 0]
    T_grok_mean = []
    T_grok_std = []

    for omega in omega_vals:
        runs = by_omega[omega]
        T_groks = [r['T_grok'] for r in runs if r['T_grok'] is not None]

        if T_groks:
            T_grok_mean.append(np.mean(T_groks))
            T_grok_std.append(np.std(T_groks))
        else:
            T_grok_mean.append(None)
            T_grok_std.append(0)

    # Filter out None values
    valid_mask = [t is not None for t in T_grok_mean]
    omega_valid = [o for o, v in zip(omega_vals, valid_mask) if v]
    T_grok_valid = [t for t, v in zip(T_grok_mean, valid_mask) if v]
    T_grok_std_valid = [s for s, v in zip(T_grok_std, valid_mask) if v]

    ax.errorbar(omega_valid, T_grok_valid, yerr=T_grok_std_valid,
                fmt='o-', capsize=5, linewidth=2, markersize=8)
    ax.set_xlabel('ω (suppressor scaling)', fontsize=12)
    ax.set_ylabel('T_grok (steps to 90% accuracy)', fontsize=12)
    ax.set_title('Phase Boundary Shift', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Panel 2: Final circularity vs omega (geometry quality)
    ax = axes[0, 1]
    circ_mean = []
    circ_std = []

    for omega in omega_vals:
        runs = by_omega[omega]
        circs = [r['final_circularity'] for r in runs]
        circ_mean.append(np.mean(circs))
        circ_std.append(np.std(circs))

    ax.errorbar(omega_vals, circ_mean, yerr=circ_std,
                fmt='o-', capsize=5, linewidth=2, markersize=8, color='green')
    ax.set_xlabel('ω (suppressor scaling)', fontsize=12)
    ax.set_ylabel('Final circularity score', fontsize=12)
    ax.set_title('Geometry Quality', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Panel 3: VDI compensation vs omega (Le Chatelier signature)
    ax = axes[1, 0]

    if vdi_results and 'all_results' in vdi_results:
        vdi_by_omega = defaultdict(list)
        for r in vdi_results['all_results']:
            vdi_by_omega[r['omega']].append(r['total_compensation'])

        omega_vdi = sorted(vdi_by_omega.keys())
        comp_mean = [np.mean(vdi_by_omega[o]) for o in omega_vdi]
        comp_std = [np.std(vdi_by_omega[o]) for o in omega_vdi]

        ax.errorbar(omega_vdi, comp_mean, yerr=comp_std,
                    fmt='o-', capsize=5, linewidth=2, markersize=8, color='purple')
        ax.set_xlabel('ω (suppressor scaling)', fontsize=12)
        ax.set_ylabel('Σ|VDI| (other L0 heads)', fontsize=12)
        ax.set_title('Le Chatelier Compensation', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add correlation annotation
        corr = vdi_results.get('correlation_omega_compensation', 0.0)
        ax.text(0.05, 0.95, f'r = {corr:.3f}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax.text(0.5, 0.5, 'VDI data not available\nRun analyze_vdi_compensation.py',
                transform=ax.transAxes, ha='center', va='center', fontsize=10)
        ax.set_xlabel('ω (suppressor scaling)', fontsize=12)
        ax.set_ylabel('Σ|VDI| (other L0 heads)', fontsize=12)
        ax.set_title('Le Chatelier Compensation', fontsize=14, fontweight='bold')

    # Panel 4: Stability regime (colored by pathology vs healthy)
    ax = axes[1, 1]

    # Classify runs as healthy (test_acc > 0.7) or pathological
    for omega in omega_vals:
        runs = by_omega[omega]
        for run in runs:
            acc = run['final_acc']
            T_grok = run['T_grok']

            if T_grok is None:
                T_grok = run['n_steps']  # Didn't grok

            color = 'green' if acc > 0.7 else 'red'
            marker = 'o' if acc > 0.7 else 'x'
            ax.scatter(omega, T_grok, c=color, marker=marker, s=100, alpha=0.7)

    ax.set_xlabel('ω (suppressor scaling)', fontsize=12)
    ax.set_ylabel('T_grok (steps)', fontsize=12)
    ax.set_title('Stability Regime', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
               markersize=10, label='Healthy (acc > 0.7)'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='red',
               markersize=10, label='Pathological (acc ≤ 0.7)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPhase diagrams saved to {output_path}")


def print_summary_statistics(summaries: List[Dict]):
    """Print summary statistics."""
    by_omega = defaultdict(list)
    for s in summaries:
        by_omega[s['omega']].append(s)

    print("\nSummary Statistics by Omega:")
    print("=" * 70)
    print(f"{'Omega':<8} {'T_grok':<15} {'Circularity':<15} {'Acc':<10} {'N':<5}")
    print("-" * 70)

    for omega in sorted(by_omega.keys()):
        runs = by_omega[omega]
        T_groks = [r['T_grok'] for r in runs if r['T_grok'] is not None]
        circs = [r['final_circularity'] for r in runs]
        accs = [r['final_acc'] for r in runs]

        if T_groks:
            T_grok_str = f"{np.mean(T_groks):.0f} ± {np.std(T_groks):.0f}"
        else:
            T_grok_str = "N/A"

        circ_str = f"{np.mean(circs):.3f} ± {np.std(circs):.3f}"
        acc_str = f"{np.mean(accs):.3f}"

        print(f"{omega:<8.1f} {T_grok_str:<15} {circ_str:<15} {acc_str:<10} {len(runs):<5}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Generate phase diagrams")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="reports/stage1b_grokking/train",
        help="Base directory with training runs",
    )
    parser.add_argument(
        "--vdi-results",
        type=str,
        default="reports/stage1b_grokking/vdi_compensation.json",
        help="VDI compensation results (optional)",
    )
    parser.add_argument(
        "--robustness-results",
        type=str,
        default="reports/stage1b_grokking/geometry_robustness.json",
        help="Geometry robustness results (optional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/stage1b_grokking/phase_diagrams.png",
        help="Output path for figure",
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir)

    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist")
        print("Run the training sweep first: bash scripts/run_stage1b_sweep.sh")
        return

    # Collect run summaries
    summaries = collect_all_runs(base_dir)

    if not summaries:
        print("No completed runs found")
        return

    print(f"Collected {len(summaries)} run summaries")

    # Print statistics
    print_summary_statistics(summaries)

    # Load optional VDI results
    vdi_results = None
    vdi_path = Path(args.vdi_results)
    if vdi_path.exists():
        with vdi_path.open() as f:
            vdi_results = json.load(f)
        print(f"\nLoaded VDI results from {vdi_path}")

    # Load optional robustness results
    robustness_results = None
    rob_path = Path(args.robustness_results)
    if rob_path.exists():
        with rob_path.open() as f:
            robustness_results = json.load(f)
        print(f"Loaded robustness results from {rob_path}")

    # Generate phase diagrams
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_phase_diagrams(
        summaries=summaries,
        vdi_results=vdi_results,
        robustness_results=robustness_results,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
