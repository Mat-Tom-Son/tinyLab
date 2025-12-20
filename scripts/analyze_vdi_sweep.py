#!/usr/bin/env python3
"""
Analyze VDI Target Sweep Results

Critical analysis: Does final VDI track the target, or is there
a forced attractor at ~0.44?
"""

import json
from pathlib import Path
import numpy as np


def analyze_sweep():
    """Analyze VDI sweep results and determine the mechanism."""
    project_root = Path(__file__).parent.parent
    phase2_path = project_root / "reports" / "phase2"

    vdi_targets = [0.45, 0.50, 0.55, 0.60, 0.65]
    results = []

    print("\n" + "="*80)
    print("VDI TARGET SWEEP ANALYSIS")
    print("="*80)
    print()

    for target_vdi in vdi_targets:
        condition_name = f"vdi_sweep_{target_vdi}"
        condition_path = phase2_path / condition_name

        seeds_data = []

        for seed in [0, 1, 2]:
            seed_path = condition_path / f"seed{seed}"
            metrics_path = seed_path / "phase2_metrics.jsonl"

            if not metrics_path.exists():
                print(f"⚠️  Missing: {metrics_path}")
                continue

            # Read last line of metrics file to get final VDI
            with open(metrics_path, 'r') as f:
                lines = f.readlines()
                if not lines:
                    continue
                last_metrics = json.loads(lines[-1])

            final_vdi = last_metrics.get('vdi_mean', None)

            seeds_data.append({
                'seed': seed,
                'target_vdi': target_vdi,
                'final_vdi': final_vdi,
                'delta': final_vdi - target_vdi if final_vdi else None,
            })

        # Condition statistics
        final_vdis = [s['final_vdi'] for s in seeds_data if s['final_vdi'] is not None]

        if final_vdis:
            mean_final_vdi = np.mean(final_vdis)
            std_final_vdi = np.std(final_vdis) if len(final_vdis) > 1 else 0.0
            mean_delta = mean_final_vdi - target_vdi
        else:
            mean_final_vdi = None
            std_final_vdi = None
            mean_delta = None

        results.append({
            'target_vdi': target_vdi,
            'seeds': seeds_data,
            'mean_final_vdi': mean_final_vdi,
            'std_final_vdi': std_final_vdi,
            'mean_delta': mean_delta,
        })

    # Print results table
    print("-"*80)
    print(f"{'Target VDI':<15} | {'Seed':<4} | {'Final VDI':<15} | {'Delta':<15}")
    print("-"*80)

    for result in results:
        target = result['target_vdi']

        for seed_data in result['seeds']:
            seed = seed_data['seed']
            final = seed_data['final_vdi']
            delta = seed_data['delta']

            final_str = f"{final:.4f}" if final is not None else "N/A"
            delta_str = f"{delta:+.4f}" if delta is not None else "N/A"

            print(f"{target:<15} | {seed:<4} | {final_str:<15} | {delta_str:<15}")

        # Mean
        mean_final = result['mean_final_vdi']
        std_final = result['std_final_vdi']
        mean_delta = result['mean_delta']

        if mean_final is not None:
            print(f"{'':<15} | {'AVG':<4} | {mean_final:.4f}±{std_final:.4f}  | {mean_delta:+.4f}")
        else:
            print(f"{'':<15} | {'AVG':<4} | {'N/A':<15} | {'N/A':<15}")

        print("-"*80)

    print()

    # Determine the mechanism
    print("="*80)
    print("INTERPRETATION")
    print("="*80)
    print()

    # Check if final VDI tracks target
    valid_results = [r for r in results if r['mean_final_vdi'] is not None]

    if not valid_results:
        print("⚠️  No valid results yet. Experiments may still be running.")
        return

    if len(valid_results) < 3:
        print(f"⚠️  Only {len(valid_results)}/5 targets complete. Need at least 3 for classification.")
        print(f"   Current data:")
        for r in valid_results:
            print(f"   Target {r['target_vdi']:.2f} → Final {r['mean_final_vdi']:.4f} (Δ = {r['mean_delta']:+.4f})")
        print()
        print("   Waiting for more results...")
        return

    # Compute correlation between target and final VDI
    targets = [r['target_vdi'] for r in valid_results]
    finals = [r['mean_final_vdi'] for r in valid_results]

    # Check tracking vs forced attractor
    deltas = [r['mean_delta'] for r in valid_results]
    mean_abs_delta = np.mean([abs(d) for d in deltas])

    # Check if finals are all similar (forced attractor) or track targets
    final_std = np.std(finals)
    final_range = max(finals) - min(finals)

    print(f"Final VDI range: {final_range:.4f}")
    print(f"Final VDI std: {final_std:.4f}")
    print(f"Mean absolute delta from target: {mean_abs_delta:.4f}")
    print()

    # Extended diagnostics
    print("="*80)
    print("DETAILED DIAGNOSTICS")
    print("="*80)
    print()

    # 1. Saturation check: Does tracking degrade at high targets?
    low_targets = [r for r in valid_results if r['target_vdi'] <= 0.50]
    high_targets = [r for r in valid_results if r['target_vdi'] >= 0.60]

    saturation_detected = False
    if low_targets and high_targets:
        low_deltas = [abs(r['mean_delta']) for r in low_targets]
        high_deltas = [abs(r['mean_delta']) for r in high_targets]

        mean_low_delta = np.mean(low_deltas)
        mean_high_delta = np.mean(high_deltas)

        print(f"1. SATURATION CHECK:")
        print(f"   Low targets (≤0.50): mean |Δ| = {mean_low_delta:.4f}")
        print(f"   High targets (≥0.60): mean |Δ| = {mean_high_delta:.4f}")

        if mean_high_delta > 2 * mean_low_delta:
            print(f"   ⚠️  SATURATION DETECTED: High targets fail to track (ratio = {mean_high_delta/mean_low_delta:.2f}x)")
            saturation_detected = True
        else:
            print(f"   ✓ No saturation (ratio = {mean_high_delta/mean_low_delta:.2f}x)")
        print()

    # 2. Variance pattern: Does instability increase with target?
    variances = [(r['target_vdi'], r['std_final_vdi']) for r in valid_results if r['std_final_vdi'] is not None]

    instability_detected = False
    if len(variances) > 1:
        print(f"2. INSTABILITY CHECK (variance by target):")
        for target, var in sorted(variances):
            print(f"   Target {target:.2f}: σ = {var:.6f}")

        min_var = min(v[1] for v in variances)
        max_var = max(v[1] for v in variances)

        if max_var > 5 * min_var and max_var > 0.01:
            print(f"   ⚠️  INSTABILITY DETECTED: Variance increases with target (max/min = {max_var/min_var:.1f}x)")
            instability_detected = True
        else:
            print(f"   ✓ Variance stable across targets (max/min = {max_var/min_var:.1f}x)")
        print()

    # 3. Seed divergence: Do seeds agree less at extreme targets?
    print(f"3. SEED AGREEMENT (by target):")
    for result in valid_results:
        target = result['target_vdi']
        seed_vdis = [s['final_vdi'] for s in result['seeds'] if s['final_vdi'] is not None]

        if len(seed_vdis) > 1:
            seed_range = max(seed_vdis) - min(seed_vdis)
            print(f"   Target {target:.2f}: {seed_vdis} (range = {seed_range:.4f})")
        else:
            print(f"   Target {target:.2f}: {seed_vdis} (single seed)")
    print()

    print("="*80)
    print("OUTCOME CLASSIFICATION")
    print("="*80)
    print()

    # Decision tree with 5 outcomes
    if final_range < 0.02:  # All finals within 0.02 of each other
        print("OUTCOME 2: FORCED ATTRACTOR")
        print()
        print(f"All conditions converge to VDI ≈ {np.mean(finals):.4f} ± {final_std:.4f} regardless of target.")
        print(f"The system cannot maintain equilibria outside this narrow range under")
        print(f"dual-timescale + homeostatic pressure.")
        print()
        print("NARRATIVE: \"There's a forced attractor at VDI ≈ 0.44 under homeostatic")
        print("pressure. This suggests information-geometric constraints on the space of")
        print("feasible equilibria. Q is constrained, not fully designable.\"")
        print()
        print("PAPER STRENGTH: ⭐⭐⭐⭐ (Strong—points to deeper structure)")

    elif mean_abs_delta < 0.02 and not saturation_detected and not instability_detected:
        print("OUTCOME 1: PERFECT TRACKING")
        print()
        print(f"Final VDI tracks target within ±{mean_abs_delta:.4f}.")
        print(f"The system successfully locks onto whatever equilibrium you specify.")
        print()
        print("NARRATIVE: \"Q is fully designable via training regime. Set-point loss")
        print("successfully steers the equilibrium to any specified VDI target within")
        print("the tested range (0.45-0.65).\"")
        print()
        print("PAPER STRENGTH: ⭐⭐⭐⭐⭐ (Perfect, but raises question: why does Phase 1")
        print("naturally settle at 0.61 if any VDI is achievable?)")

    elif saturation_detected and mean_abs_delta < 0.05:
        print("OUTCOME 3: PARTIAL TRACKING WITH SATURATION")
        print()
        print(f"System tracks low targets accurately (Δ ≈ {mean_low_delta:.4f}) but fails at high targets (Δ ≈ {mean_high_delta:.4f}).")
        print(f"There's a feasible range for VDI equilibria, with ceiling around {max(finals):.2f}.")
        print()
        print("NARRATIVE: \"Q is designable within a feasible range (≈0.45-0.60). Beyond")
        print("this ceiling, information-geometric or capacity constraints prevent higher")
        print("equilibria. Q is partially constrained.\"")
        print()
        print("PAPER STRENGTH: ⭐⭐⭐⭐ (Interesting—reveals soft constraints)")

    elif instability_detected:
        print("OUTCOME 4: INSTABILITY AT EXTREME TARGETS")
        print()
        print(f"High variance ({max_var:.4f}) at extreme targets suggests training instability.")
        print(f"The system can reach different equilibria but requires careful tuning.")
        print()
        print("NARRATIVE: \"Q is designable in a stable range. Higher set-point targets")
        print("create training instability, suggesting fundamental constraints on dual-")
        print("timescale architectures with strong homeostatic pressure.\"")
        print()
        print("PAPER STRENGTH: ⭐⭐⭐ (Honest but needs explanation)")

    elif mean_abs_delta > 0.10:
        print("OUTCOME 5: WEAK/NO TRACKING")
        print()
        print(f"Large deviations (±{mean_abs_delta:.4f}) suggest set-point loss isn't working.")
        print(f"Possible causes: λ_setpoint too weak, loss contradiction, or incompatibility.")
        print()
        print("RECOMMENDATION: Debug set-point mechanism or increase λ_setpoint.")
        print()
        print("PAPER STRENGTH: ⭐ (Would need follow-up experiments)")

    else:
        print("OUTCOME 3-4 BOUNDARY: PARTIAL TRACKING")
        print()
        print(f"Final VDI partially tracks target (±{mean_abs_delta:.4f}) with moderate variance.")
        print(f"System reaches different equilibria but not with high precision.")
        print()
        print("NARRATIVE: \"Q is partially designable. The system can shift equilibria")
        print("but exhibits imperfect tracking, suggesting competing pressures between")
        print("task, homeostasis, and set-point objectives.\"")
        print()
        print("PAPER STRENGTH: ⭐⭐⭐ (Needs careful interpretation)")

    print()

    # Plot comparison
    print("="*80)
    print("TARGET vs FINAL VDI")
    print("="*80)
    print()

    for target, final in zip(targets, finals):
        bar_length = int(final * 80)
        target_pos = int(target * 80)

        bar = [' '] * 80
        for i in range(bar_length):
            bar[i] = '█'
        if target_pos < 80:
            bar[target_pos] = '|'  # Mark target

        print(f"{target:.2f} → {final:.4f}  {''.join(bar)}")

    print()
    print("Legend: █ = final VDI, | = target")
    print()


if __name__ == '__main__':
    analyze_sweep()
