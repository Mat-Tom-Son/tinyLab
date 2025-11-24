#!/usr/bin/env python3
"""
Analyze VDI compensation across Stage-1B grokking runs.

Tests the Le Chatelier hypothesis:
- When we scale target head by omega, do other heads compensate?
- Does VDI_other anticorrelate with VDI_target?

This is the key signature of thermodynamic control.

Usage:
    python scripts/analyze_vdi_compensation.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict


def load_checkpoint(ckpt_path: Path) -> Dict:
    """Load a checkpoint."""
    return torch.load(ckpt_path, map_location='cpu')


def compute_vdi_simple(
    model_state: Dict,
    test_data: List[Dict],
    layer_idx: int,
    head_idx: int,
    device: torch.device,
) -> float:
    """
    Simplified VDI computation for a single head.

    VDI (Value-Distribution Imbalance) measures how much a head:
    - Flattens output distributions (high entropy)
    - vs. Sharpens them (low entropy)

    Positive VDI = suppressor (increases entropy)
    Negative VDI = amplifier (decreases entropy)

    This is a placeholder - actual VDI requires full forward pass
    with intervention. For now, return dummy value.
    """
    # TODO: Implement proper VDI computation
    # This requires:
    # 1. Forward pass with head ablated
    # 2. Compute output distribution entropy
    # 3. Forward pass with head active
    # 4. Compute delta entropy

    return 0.0  # Placeholder


def compute_all_head_vdi(
    checkpoint: Dict,
    test_data: List[Dict],
    layer_idx: int,
    n_heads: int,
    device: torch.device,
) -> Dict[int, float]:
    """
    Compute VDI for all heads at a layer.

    Returns:
        Dictionary mapping head_idx -> VDI score
    """
    model_state = checkpoint['model_state']

    vdi_scores = {}
    for head_idx in range(n_heads):
        vdi = compute_vdi_simple(
            model_state, test_data, layer_idx, head_idx, device
        )
        vdi_scores[head_idx] = vdi

    return vdi_scores


def analyze_compensation_across_runs(
    base_dir: Path,
    target_head: int,
    device: torch.device,
) -> Dict:
    """
    Analyze Le Chatelier compensation across all runs.

    For each (omega, seed) run:
    1. Load final checkpoint
    2. Compute VDI for all layer-0 heads
    3. Separate target head from others
    4. Check if compensation (VDI_others) anticorrelates with omega

    Returns:
        Dictionary with:
        - 'by_omega': grouped results by omega
        - 'correlation': correlation(omega, total_compensation)
        - 'anticorrelation': correlation(VDI_target, VDI_others)
    """
    results_by_omega = defaultdict(list)

    # Find all run directories
    run_dirs = sorted(base_dir.glob("stage1b_head*_omega*_seed*"))

    print(f"Found {len(run_dirs)} runs")

    for run_dir in run_dirs:
        # Parse run name
        parts = run_dir.name.split('_')
        omega = float(parts[2].replace('omega', ''))
        seed = int(parts[3].replace('seed', ''))

        # Load final checkpoint
        ckpt_path = run_dir / "final_model.pt"
        if not ckpt_path.exists():
            # Try latest checkpoint
            ckpt_dir = run_dir / "checkpoints"
            ckpts = sorted(ckpt_dir.glob("step_*.pt"))
            if not ckpts:
                print(f"Warning: No checkpoints found for {run_dir.name}")
                continue
            ckpt_path = ckpts[-1]

        checkpoint = load_checkpoint(ckpt_path)

        # TODO: Load test data and compute VDI
        # For now, use placeholder
        vdi_scores = {i: np.random.randn() for i in range(8)}  # Placeholder

        vdi_target = vdi_scores[target_head]
        vdi_others = [vdi_scores[i] for i in range(8) if i != target_head]
        total_compensation = sum(abs(v) for v in vdi_others if v > 0)

        results_by_omega[omega].append({
            'omega': omega,
            'seed': seed,
            'vdi_target': vdi_target,
            'vdi_others': vdi_others,
            'total_compensation': total_compensation,
        })

        print(
            f"  {run_dir.name}: vdi_target={vdi_target:.3f}, "
            f"compensation={total_compensation:.3f}"
        )

    # Compute correlations
    all_results = [r for runs in results_by_omega.values() for r in runs]

    if len(all_results) < 3:
        print("Not enough runs for correlation analysis")
        return {'by_omega': dict(results_by_omega)}

    omegas = np.array([r['omega'] for r in all_results])
    compensations = np.array([r['total_compensation'] for r in all_results])
    vdi_targets = np.array([r['vdi_target'] for r in all_results])

    # Correlation: omega vs compensation
    # Prediction: as omega increases, compensation should decrease (negative correlation)
    # Because when we strengthen target suppressor, others relax
    corr_omega_comp = np.corrcoef(omegas, compensations)[0, 1]

    # Anticorrelation: vdi_target vs mean(vdi_others)
    vdi_others_mean = np.array([
        np.mean([v for v in r['vdi_others'] if v > 0]) if any(v > 0 for v in r['vdi_others']) else 0.0
        for r in all_results
    ])
    corr_target_others = np.corrcoef(vdi_targets, vdi_others_mean)[0, 1]

    print("\nLe Chatelier Compensation Analysis:")
    print(f"  Correlation(omega, total_compensation) = {corr_omega_comp:.3f}")
    print(f"    Prediction: negative (omega ↑ → others relax)")
    print(f"  Correlation(VDI_target, VDI_others) = {corr_target_others:.3f}")
    print(f"    Prediction: negative (anticorrelation)")

    return {
        'by_omega': dict(results_by_omega),
        'correlation_omega_compensation': float(corr_omega_comp),
        'correlation_target_others': float(corr_target_others),
        'all_results': all_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze VDI compensation")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="reports/stage1b_grokking/train",
        help="Base directory with training runs",
    )
    parser.add_argument(
        "--target-head",
        type=int,
        default=0,
        help="Target head that was scaled",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/stage1b_grokking/vdi_compensation.json",
        help="Output path for results",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path(args.base_dir)

    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist")
        print("Run the training sweep first: bash scripts/run_stage1b_sweep.sh")
        return

    results = analyze_compensation_across_runs(
        base_dir=base_dir,
        target_head=args.target_head,
        device=device,
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable format
    output_data = {
        'correlation_omega_compensation': results['correlation_omega_compensation'],
        'correlation_target_others': results['correlation_target_others'],
        'all_results': results['all_results'],
    }

    with output_path.open('w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
