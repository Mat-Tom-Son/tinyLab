#!/usr/bin/env python3
"""
Phase 2 Reanalysis v2: Use fine-grained phase2_metrics.jsonl data.

Addresses detection resolution issue: developmental_trajectory.json samples
every 500 steps, but crystallization windows can be as short as 100 steps.

phase2_metrics.jsonl has step-by-step VDI tracking (when available).
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_metrics_jsonl(metrics_path: Path) -> List[Dict]:
    """Load phase2_metrics.jsonl file."""
    metrics = []
    with open(metrics_path, 'r') as f:
        for line in f:
            metrics.append(json.loads(line))
    return metrics


def detect_crystallization_from_metrics(
    metrics: List[Dict],
    start_threshold: float = 0.001,
    end_threshold: float = 0.0001,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Detect crystallization window from phase2_metrics.jsonl.

    Returns:
        (start_step, end_step) or (None, None) if not detected
    """
    start_step = None
    end_step = None

    for entry in metrics:
        step = entry['step']
        vdi_std = entry.get('vdi_std')

        if vdi_std is None:
            continue  # Skip entries without VDI (baseline/dual_timescale before fix)

        # Crystallization start: first time VDI std drops below threshold
        if start_step is None and vdi_std < start_threshold:
            start_step = step

        # Crystallization end: first time VDI std drops below stricter threshold
        if start_step is not None and end_step is None and vdi_std < end_threshold:
            end_step = step

    return start_step, end_step


def compute_equilibrium_vdi(metrics: List[Dict], window: int = 10) -> Tuple[float, float]:
    """
    Compute final equilibrium VDI mean and std over last window steps.

    Returns:
        (mean_vdi, vdi_std_at_equilibrium)
    """
    # Get last window entries that have vdi_mean
    final_entries = [e for e in metrics[-window:] if 'vdi_mean' in e]

    if not final_entries:
        return None, None

    mean_vdi = np.mean([e['vdi_mean'] for e in final_entries])
    # VDI std at equilibrium (how crystallized are the heads)
    vdi_std = np.mean([e.get('vdi_std', 0.0) for e in final_entries])

    return mean_vdi, vdi_std


def analyze_condition_v2(
    condition: str,
    base_path: Path,
    phase1_baseline: Dict,
) -> Dict:
    """Analyze using phase2_metrics.jsonl."""
    condition_path = base_path / condition
    seeds_data = []

    for seed in [0, 1, 2]:
        seed_path = condition_path / f"seed{seed}"
        metrics_path = seed_path / "phase2_metrics.jsonl"

        if not metrics_path.exists():
            print(f"⚠️  Missing: {metrics_path}")
            continue

        metrics = load_metrics_jsonl(metrics_path)

        # Detect crystallization
        start, end = detect_crystallization_from_metrics(metrics)
        duration = (end - start) if (start and end) else None

        # Calculate speedup vs Phase 1 baseline
        speedup = None
        if duration is not None:
            phase1_duration = phase1_baseline['crystallization_mean']
            speedup = 100 * (1 - duration / phase1_duration)

        # Equilibrium VDI
        equilibrium_vdi, equilibrium_vdi_std = compute_equilibrium_vdi(metrics)

        seeds_data.append({
            'seed': seed,
            'crystallization_start': start,
            'crystallization_end': end,
            'duration': duration,
            'speedup': speedup,
            'equilibrium_vdi': equilibrium_vdi,
            'equilibrium_vdi_std': equilibrium_vdi_std,
            'total_steps': len(metrics),
        })

    # Condition statistics
    durations = [s['duration'] for s in seeds_data if s['duration'] is not None]
    speedups = [s['speedup'] for s in seeds_data if s['speedup'] is not None]
    equilibria = [s['equilibrium_vdi'] for s in seeds_data if s['equilibrium_vdi'] is not None]

    return {
        'condition': condition,
        'seeds': seeds_data,
        'mean_duration': np.mean(durations) if durations else None,
        'std_duration': np.std(durations) if len(durations) > 1 else None,
        'mean_speedup': np.mean(speedups) if speedups else None,
        'mean_equilibrium_vdi': np.mean(equilibria) if equilibria else None,
        'std_equilibrium_vdi': np.std(equilibria) if len(equilibria) > 1 else 0.0,
    }


def print_detailed_table(results: List[Dict], phase1_baseline: Dict):
    """Print detailed analysis table."""
    print("\n" + "="*110)
    print("PHASE 2 REANALYSIS v2: Fine-Grained Crystallization Detection (phase2_metrics.jsonl)")
    print("="*110)
    print()
    print(f"Phase 1 Baseline: {phase1_baseline['crystallization_mean']} ± {phase1_baseline['crystallization_std']} steps")
    print(f"Phase 1 VDI Equilibrium: {phase1_baseline['vdi_target']}")
    print()
    print("-"*110)
    print(f"{'Condition':<25} | {'Seed':<4} | {'Start':<7} | {'End':<7} | {'Duration':<8} | {'Speedup':<9} | {'VDI Eq':<8} | {'Total':<6}")
    print("-"*110)

    for result in results:
        condition = result['condition']

        for seed_data in result['seeds']:
            seed = seed_data['seed']
            start = seed_data['crystallization_start']
            end = seed_data['crystallization_end']
            duration = seed_data['duration']
            speedup = seed_data['speedup']
            vdi_eq = seed_data['equilibrium_vdi']
            total = seed_data['total_steps']

            start_str = f"{start}" if start is not None else "N/A"
            end_str = f"{end}" if end is not None else "N/A"
            duration_str = f"{duration}" if duration is not None else "N/A"
            speedup_str = f"{speedup:+6.1f}%" if speedup is not None else "N/A"
            vdi_str = f"{vdi_eq:.4f}" if vdi_eq is not None else "N/A"

            print(f"{condition:<25} | {seed:<4} | {start_str:<7} | {end_str:<7} | {duration_str:<8} | {speedup_str:<9} | {vdi_str:<8} | {total:<6}")

        # Condition summary
        mean_dur = result['mean_duration']
        std_dur = result['std_duration']
        mean_spd = result['mean_speedup']
        mean_vdi = result['mean_equilibrium_vdi']
        std_vdi = result['std_equilibrium_vdi']

        if mean_dur is not None:
            dur_str = f"{mean_dur:.0f}±{std_dur:.0f}" if std_dur else f"{mean_dur:.0f}"
            vdi_str = f"{mean_vdi:.4f}±{std_vdi:.4f}" if mean_vdi is not None else "N/A"
            print(f"{'':<25} | {'AVG':<4} | {'':<7} | {'':<7} | {dur_str:<8} | {mean_spd:+6.1f}% | {vdi_str:<16} |")
        else:
            vdi_str = f"{mean_vdi:.4f}±{std_vdi:.4f}" if mean_vdi is not None else "N/A"
            print(f"{'':<25} | {'AVG':<4} | {'':<7} | {'':<7} | {'UNSTABLE':<8} | {'N/A':<9} | {vdi_str:<16} |")
        print("-"*110)

    print()


def print_summary_v2(results: List[Dict], phase1_baseline: Dict):
    """Print summary with key findings."""
    print("\n" + "="*110)
    print("KEY FINDINGS")
    print("="*110)
    print()

    # Detection rate
    total = sum(len(r['seeds']) for r in results)
    complete = sum(1 for r in results for s in r['seeds'] if s['duration'] is not None)

    print(f"1. DETECTION SUCCESS:")
    print(f"   - Complete windows detected: {complete}/{total} runs ({100*complete/total:.1f}%)")
    print()

    # Acceleration
    print(f"2. CRYSTALLIZATION ACCELERATION:")
    print(f"   - Phase 1 baseline: {phase1_baseline['crystallization_mean']} steps")
    print()

    for result in results:
        if result['mean_duration'] is not None:
            print(f"   - {result['condition']:<25}: {result['mean_duration']:>6.0f} steps ({result['mean_speedup']:+.1f}%)")
        else:
            print(f"   - {result['condition']:<25}: UNSTABLE (no clean crystallization)")

    print()

    # Equilibrium shift
    print(f"3. EQUILIBRIUM SHIFT:")
    print(f"   - Phase 1 natural: VDI = {phase1_baseline['vdi_target']}")
    print()

    for result in results:
        if result['mean_equilibrium_vdi'] is not None:
            delta = result['mean_equilibrium_vdi'] - phase1_baseline['vdi_target']
            print(f"   - {result['condition']:<25}: {result['mean_equilibrium_vdi']:.4f} (Δ = {delta:+.4f})")

    print()
    print("="*110)


def main():
    project_root = Path(__file__).parent.parent
    phase2_path = project_root / "reports" / "phase2"
    output_path = project_root / "reports" / "phase2_reanalysis_v2.json"

    phase1_baseline = {
        'vdi_target': 0.611992,
        'crystallization_mean': 1500,  # From Phase 1 analysis
        'crystallization_std': 400,
    }

    conditions = [
        'baseline',
        'dual_timescale',
        'explicit_convergence',
        'intentional_vdi_target',
        'early_convergence',
    ]

    results = []
    for condition in conditions:
        print(f"Analyzing {condition}...")
        result = analyze_condition_v2(condition, phase2_path, phase1_baseline)
        results.append(result)

    print_detailed_table(results, phase1_baseline)
    print_summary_v2(results, phase1_baseline)

    # Save
    output_data = {
        'phase1_baseline': phase1_baseline,
        'conditions': results,
        'metadata': {
            'data_source': 'phase2_metrics.jsonl (100-step resolution)',
            'start_threshold': 0.001,
            'end_threshold': 0.0001,
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Saved to: {output_path}\n")


if __name__ == '__main__':
    main()
