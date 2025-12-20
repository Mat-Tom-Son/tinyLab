#!/bin/bash
# VDI Target Sweep Experiment
#
# Critical experiment to determine: Does final VDI track the target,
# or is there a forced attractor at ~0.44 under homeostatic pressure?
#
# Hypothesis 1: Final VDI ≈ target → "Q is designable"
# Hypothesis 2: Final VDI ≈ 0.44 always → "Forced attractor exists"

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

# VDI targets to sweep
VDI_TARGETS=(0.45 0.50 0.55 0.60 0.65)

# Seeds
SEEDS=(0 1 2)

# Create output directory
mkdir -p reports/vdi_sweep

# Condition template: intentional_vdi_target with varying target_vdi
LAMBDA_COMPENSATION=0.5
LAMBDA_CONVERGENCE=0.3
LAMBDA_SETPOINT=0.2
FAST_LR_SCALE=1.0
SLOW_LR_SCALE=0.1

echo "Starting sweep at $(date)"
echo ""

# Launch all conditions serially, seeds in parallel
for target_vdi in "${VDI_TARGETS[@]}"; do
    target_name="vdi_target_${target_vdi}"

    echo "========================================"
    echo "Launching: target_vdi = $target_vdi"
    echo "========================================"

    # Create output directories
    for seed in "${SEEDS[@]}"; do
        mkdir -p "reports/vdi_sweep/${target_name}/seed${seed}"
    done

    # Launch 3 seeds in parallel
    for seed in "${SEEDS[@]}"; do
        output_dir="reports/vdi_sweep/${target_name}/seed${seed}"

        echo "  Starting seed ${seed}..."

        .venv/bin/python scripts/train_vdi_sweep.py \
            --target_vdi "$target_vdi" \
            --seed "$seed" \
            --steps 10000 \
            --output_dir "$output_dir" \
            --lambda_compensation "$LAMBDA_COMPENSATION" \
            --lambda_convergence "$LAMBDA_CONVERGENCE" \
            --lambda_setpoint "$LAMBDA_SETPOINT" \
            --fast_lr_scale "$FAST_LR_SCALE" \
            --slow_lr_scale "$SLOW_LR_SCALE" \
            > "${output_dir}/training.log" 2>&1 &
    done

    # Wait for all 3 seeds to complete before moving to next target
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
echo "Results saved to: reports/vdi_sweep/"
echo ""
echo "Next steps:"
echo "  1. Run analysis: .venv/bin/python scripts/analyze_vdi_sweep.py"
echo "  2. Check if final VDI tracks target (Q designable) or forced to 0.44 (attractor)"
echo ""
