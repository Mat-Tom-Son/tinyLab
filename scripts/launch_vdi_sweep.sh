#!/bin/bash
# VDI Target Sweep Experiment
#
# Critical experiment: Does final VDI track the target, or is there
# a forced attractor at ~0.44 under homeostatic pressure?

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "================================"
echo "VDI TARGET SWEEP EXPERIMENT"
echo "================================"
echo ""
echo "Testing 5 VDI targets × 3 seeds = 15 runs"
echo "Duration: ~2-3 hours on MPS"
echo ""

# VDI targets
VDI_TARGETS=(0.45 0.50 0.55 0.60 0.65)
SEEDS=(0 1 2)

echo "Starting sweep at $(date)"
echo ""

for target_vdi in "${VDI_TARGETS[@]}"; do
    condition_name="vdi_sweep_${target_vdi}"

    echo "========================================"
    echo "Launching: target_vdi = $target_vdi"
    echo "========================================"

    # Create output directories
    for seed in "${SEEDS[@]}"; do
        mkdir -p "reports/phase2/${condition_name}/seed${seed}"
    done

    # Launch 3 seeds in parallel
    for seed in "${SEEDS[@]}"; do
        echo "  Starting seed ${seed}..."

        .venv/bin/python scripts/train_phase2_dual_timescale.py \
            --condition "$condition_name" \
            --seed "$seed" \
            --steps 10000 \
            > "reports/phase2/${condition_name}/seed${seed}/training.log" 2>&1 &
    done

    # Wait for all 3 seeds to complete before next target
    echo "  Waiting for seeds 0, 1, 2 to complete..."
    wait

    echo "  ✓ target_vdi = $target_vdi complete"
    echo ""
done

echo "================================"
echo "VDI SWEEP COMPLETE"
echo "================================"
echo ""
echo "Completed at $(date)"
echo ""
echo "Results: reports/phase2/vdi_sweep_*/"
echo ""
echo "Next: Run analysis to check if VDI tracks target"
echo "  .venv/bin/python scripts/analyze_vdi_sweep.py"
echo ""
