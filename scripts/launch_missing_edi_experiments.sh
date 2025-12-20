#!/bin/bash
# Launch Missing EDI Target Experiments
#
# Part 1: Missing EDI targets (0.50, 0.60)
# Part 2: Lambda setpoint sweep (target 0.65, lambda ∈ {0.1, 0.2, 0.5, 1.0})
#
# Total: 18 runs (~4-6 hours)

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "========================================"
echo "MISSING EDI EXPERIMENTS"
echo "========================================"
echo ""
echo "Part 1: Missing EDI targets (0.50, 0.60) - 6 runs"
echo "Part 2: Lambda setpoint sweep - 12 runs"
echo "Total: 18 runs (~4-6 hours on MPS)"
echo ""
echo "Starting at $(date)"
echo ""

# Configuration (matching existing VDI sweep)
LAMBDA_COMPENSATION=0.5
LAMBDA_CONVERGENCE=0.3
LAMBDA_SETPOINT=0.2
FAST_LR_SCALE=1.0
SLOW_LR_SCALE=0.1

SEEDS=(0 1 2)

# ========================================
# PART 1: Missing EDI Targets (0.50, 0.60)
# ========================================

echo "========================================"
echo "PART 1: Missing EDI Targets"
echo "========================================"
echo ""

MISSING_TARGETS=(0.50 0.60)

for target_vdi in "${MISSING_TARGETS[@]}"; do
    target_name="vdi_target_${target_vdi}"

    echo "----------------------------------------"
    echo "Launching: target_vdi = $target_vdi"
    echo "----------------------------------------"

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

    # Wait for all 3 seeds to complete
    echo "  Waiting for seeds to complete..."
    wait

    echo "  ✓ target_vdi = $target_vdi complete"
    echo ""
done

echo ""
echo "========================================"
echo "PART 2: Lambda Setpoint Sweep"
echo "========================================"
echo ""
echo "Testing λ_setpoint effect on high target (0.65)"
echo "Question: Is forced attractor geometric or just weak controller?"
echo ""

# Lambda values to test
LAMBDA_VALUES=(0.1 0.2 0.5 1.0)
TARGET_VDI=0.65

for lambda_set in "${LAMBDA_VALUES[@]}"; do
    lambda_name="lambda_${lambda_set}"

    echo "----------------------------------------"
    echo "Launching: λ_setpoint = $lambda_set"
    echo "----------------------------------------"

    # Create output directories
    for seed in "${SEEDS[@]}"; do
        mkdir -p "reports/lambda_sweep/target_0.65/${lambda_name}/seed${seed}"
    done

    # Launch 3 seeds in parallel
    for seed in "${SEEDS[@]}"; do
        output_dir="reports/lambda_sweep/target_0.65/${lambda_name}/seed${seed}"

        echo "  Starting seed ${seed}..."

        .venv/bin/python scripts/train_vdi_sweep.py \
            --target_vdi "$TARGET_VDI" \
            --seed "$seed" \
            --steps 10000 \
            --output_dir "$output_dir" \
            --lambda_compensation "$LAMBDA_COMPENSATION" \
            --lambda_convergence "$LAMBDA_CONVERGENCE" \
            --lambda_setpoint "$lambda_set" \
            --fast_lr_scale "$FAST_LR_SCALE" \
            --slow_lr_scale "$SLOW_LR_SCALE" \
            > "${output_dir}/training.log" 2>&1 &
    done

    # Wait for all 3 seeds to complete
    echo "  Waiting for seeds to complete..."
    wait

    echo "  ✓ λ_setpoint = $lambda_set complete"
    echo ""
done

echo ""
echo "========================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "========================================"
echo ""
echo "Completed at $(date)"
echo ""
echo "Results saved to:"
echo "  - reports/vdi_sweep/vdi_target_0.50/"
echo "  - reports/vdi_sweep/vdi_target_0.60/"
echo "  - reports/lambda_sweep/target_0.65/"
echo ""
echo "Next steps:"
echo "  1. Run VDI sweep analysis: .venv/bin/python scripts/analyze_vdi_sweep.py"
echo "  2. Analyze lambda sweep results manually"
echo "  3. Update Experiment 4 in paper with complete saturation curve"
echo ""
