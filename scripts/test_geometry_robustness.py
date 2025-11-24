#!/usr/bin/env python3
"""
Test geometry robustness across Stage-1B runs.

Hypothesis:
- Higher omega → delayed crystallization → wider basin → more robust geometry
- Lower omega → fast crystallization → sharper basin → brittler geometry

Test:
1. Inject Gaussian noise at layer-0
2. Measure circularity degradation
3. Compare AUC (area under curve) across omega values

Usage:
    python scripts/test_geometry_robustness.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict


def load_checkpoint_and_model(
    ckpt_path: Path,
    device: torch.device,
):
    """Load checkpoint and reconstruct model."""
    from scripts.train_stage1b_grokking import GrokkingTransformer, GrokkingConfig

    checkpoint = torch.load(ckpt_path, map_location=device)

    # Load config from run directory
    run_dir = ckpt_path.parent.parent
    config_path = run_dir / "config.json"
    with config_path.open() as f:
        config_data = json.load(f)

    cfg = GrokkingConfig(**config_data['config'])
    model = GrokkingTransformer(cfg).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    return model, cfg


def compute_circularity_with_noise(
    model,
    test_data: List[Dict],
    cfg,
    device: torch.device,
    layer_idx: int,
    noise_std: float,
) -> float:
    """
    Compute circularity after injecting noise at specified layer.

    Args:
        model: Trained model
        test_data: Test examples
        cfg: Model config
        device: torch device
        layer_idx: Layer to inject noise
        noise_std: Standard deviation of Gaussian noise

    Returns:
        Circularity score
    """
    from scripts.train_stage1b_grokking import prepare_batch
    from sklearn.decomposition import PCA

    model.eval()

    with torch.no_grad():
        # Get activations for test batch
        batch = test_data[:min(512, len(test_data))]
        input_ids, targets = prepare_batch(batch, cfg, device)

        # Forward with hooks to inject noise
        def noise_hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            noise = torch.randn_like(output) * noise_std
            return output + noise

        # Register hook on target layer
        handle = model.blocks[layer_idx].register_forward_hook(noise_hook)

        try:
            logits, layer_acts = model(
                input_ids,
                return_layer_activations=True
            )

            # Extract activations at position 3 (before prediction)
            acts = layer_acts[layer_idx][:, 3, :]  # [batch, d_model]
            acts_np = acts.cpu().numpy()

            # Compute circularity
            if acts_np.shape[0] < 3:
                return 0.0

            pca = PCA(n_components=2)
            acts_2d = pca.fit_transform(acts_np)

            center = acts_2d.mean(axis=0)
            centered = acts_2d - center
            radii = np.linalg.norm(centered, axis=1)

            mean_r = radii.mean()
            std_r = radii.std()

            if mean_r < 1e-6:
                return 0.0

            cv = std_r / mean_r
            circularity = max(0.0, 1.0 - cv)

            return float(circularity)

        finally:
            handle.remove()


def test_robustness_single_run(
    run_dir: Path,
    test_data: List[Dict],
    noise_levels: List[float],
    device: torch.device,
) -> Dict:
    """
    Test geometry robustness for a single run.

    Returns:
        Dictionary with noise_level -> circularity mapping + AUC
    """
    # Load final checkpoint
    ckpt_path = run_dir / "final_model.pt"
    if not ckpt_path.exists():
        # Try latest checkpoint
        ckpt_dir = run_dir / "checkpoints"
        ckpts = sorted(ckpt_dir.glob("step_*.pt"))
        if not ckpts:
            raise ValueError(f"No checkpoints found in {run_dir}")
        ckpt_path = ckpts[-1]

    model, cfg = load_checkpoint_and_model(ckpt_path, device)

    results = {}
    for noise_std in noise_levels:
        circularity = compute_circularity_with_noise(
            model=model,
            test_data=test_data,
            cfg=cfg,
            device=device,
            layer_idx=0,
            noise_std=noise_std,
        )
        results[f'noise_{noise_std}'] = circularity

    # Compute AUC (area under curve)
    circ_vals = [results[f'noise_{n}'] for n in noise_levels]
    auc = float(np.trapz(circ_vals, noise_levels))
    results['auc'] = auc

    return results


def analyze_robustness_across_runs(
    base_dir: Path,
    test_data: List[Dict],
    noise_levels: List[float],
    device: torch.device,
) -> Dict:
    """
    Analyze geometry robustness across all runs.

    Tests hypothesis:
    - Higher omega → higher AUC (more robust)
    - Lower omega → lower AUC (more brittle)
    """
    results_by_omega = defaultdict(list)

    # Find all run directories
    run_dirs = sorted(base_dir.glob("stage1b_head*_omega*_seed*"))

    print(f"Found {len(run_dirs)} runs")
    print(f"Testing with noise levels: {noise_levels}")

    for run_dir in run_dirs:
        # Parse run name
        parts = run_dir.name.split('_')
        omega = float(parts[2].replace('omega', ''))
        seed = int(parts[3].replace('seed', ''))

        print(f"  Testing {run_dir.name}...")

        try:
            robustness = test_robustness_single_run(
                run_dir=run_dir,
                test_data=test_data,
                noise_levels=noise_levels,
                device=device,
            )

            results_by_omega[omega].append({
                'omega': omega,
                'seed': seed,
                'robustness': robustness,
                'auc': robustness['auc'],
            })

            print(f"    AUC = {robustness['auc']:.3f}")

        except Exception as e:
            print(f"    Error: {e}")
            continue

    # Compute correlation: omega vs AUC
    all_results = [r for runs in results_by_omega.values() for r in runs]

    if len(all_results) < 3:
        print("Not enough runs for correlation analysis")
        return {'by_omega': dict(results_by_omega)}

    omegas = np.array([r['omega'] for r in all_results])
    aucs = np.array([r['auc'] for r in all_results])

    correlation = np.corrcoef(omegas, aucs)[0, 1]

    print("\nGeometry Robustness Analysis:")
    print(f"  Correlation(omega, AUC) = {correlation:.3f}")
    print(f"    Prediction: positive (omega ↑ → more robust geometry)")

    # Group statistics
    print("\nBy omega:")
    for omega in sorted(results_by_omega.keys()):
        runs = results_by_omega[omega]
        aucs_for_omega = [r['auc'] for r in runs]
        mean_auc = np.mean(aucs_for_omega)
        std_auc = np.std(aucs_for_omega)
        print(f"  omega={omega:.1f}: AUC={mean_auc:.3f} ± {std_auc:.3f} (n={len(runs)})")

    return {
        'by_omega': dict(results_by_omega),
        'correlation': float(correlation),
        'all_results': all_results,
    }


def plot_robustness_curves(
    results: Dict,
    noise_levels: List[float],
    output_path: Path,
):
    """
    Plot robustness curves (circularity vs noise) for each omega.
    """
    plt.figure(figsize=(10, 6))

    by_omega = results['by_omega']

    for omega in sorted(by_omega.keys()):
        runs = by_omega[omega]

        # Average across seeds
        circs_by_noise = defaultdict(list)
        for run in runs:
            for noise_std in noise_levels:
                key = f'noise_{noise_std}'
                if key in run['robustness']:
                    circs_by_noise[noise_std].append(run['robustness'][key])

        noise_vals = sorted(circs_by_noise.keys())
        mean_circs = [np.mean(circs_by_noise[n]) for n in noise_vals]
        std_circs = [np.std(circs_by_noise[n]) for n in noise_vals]

        plt.plot(noise_vals, mean_circs, 'o-', label=f'ω={omega:.1f}')
        plt.fill_between(
            noise_vals,
            np.array(mean_circs) - np.array(std_circs),
            np.array(mean_circs) + np.array(std_circs),
            alpha=0.2,
        )

    plt.xlabel('Noise std')
    plt.ylabel('Circularity score')
    plt.title('Geometry Robustness: Circularity under Noise Injection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved robustness curves to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test geometry robustness")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="reports/stage1b_grokking/train",
        help="Base directory with training runs",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/modular_p113_test.jsonl",
        help="Test data path",
    )
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs='+',
        default=[0.0, 0.1, 0.2, 0.5, 1.0],
        help="Noise standard deviations to test",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/stage1b_grokking/geometry_robustness.json",
        help="Output path for results",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path(args.base_dir)

    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist")
        return

    # Load test data
    test_data = []
    with Path(args.test_data).open() as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))

    print(f"Loaded {len(test_data)} test examples")

    results = analyze_robustness_across_runs(
        base_dir=base_dir,
        test_data=test_data,
        noise_levels=args.noise_levels,
        device=device,
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'correlation': results['correlation'],
        'all_results': results['all_results'],
    }

    with output_path.open('w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Plot robustness curves
    plot_path = output_path.parent / "robustness_curves.png"
    plot_robustness_curves(results, args.noise_levels, plot_path)


if __name__ == "__main__":
    main()
