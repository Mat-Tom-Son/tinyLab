#!/bin/bash
# Phase 2 Launch Script: All 5 Conditions × 3 Seeds
#
# Runs experiments serially (one condition at a time, 3 seeds in parallel)
# Total runtime: ~5-6 hours
#
# Usage: bash scripts/launch_phase2_battery.sh

set -e

echo "================================================================"
echo "Phase 2: Dual-Timescale Training - Full Battery"
echo "================================================================"
echo "Conditions: baseline, dual_timescale, explicit_convergence,"
echo "            intentional_vdi, early_convergence"
echo "Seeds per condition: 3"
echo "Total experiments: 15"
echo "================================================================"
echo ""

# Configuration
PYTHON=".venv/bin/python"
SCRIPT="scripts/train_phase2_dual_timescale.py"
STEPS=10000
MODULUS=113
DEVICE="auto"
MONITOR_INTERVAL=500

# Conditions to run
CONDITIONS=(
    "baseline"
    "dual_timescale"
    "explicit_convergence"
    "intentional_vdi_target"
    "early_convergence"
)

SEEDS=(0 1 2)

# Track start time
START_TIME=$(date +%s)

# Run each condition serially
for condition in "${CONDITIONS[@]}"; do
    echo ""
    echo "================================================================"
    echo "Launching condition: $condition"
    echo "================================================================"

    # Create output directory
    mkdir -p "reports/phase2/$condition"

    # Launch 3 seeds in parallel for this condition
    for seed in "${SEEDS[@]}"; do
        echo "  Starting seed $seed..."

        $PYTHON $SCRIPT \
            --condition "$condition" \
            --seed "$seed" \
            --steps $STEPS \
            --p $MODULUS \
            --monitor-interval $MONITOR_INTERVAL \
            --device "$DEVICE" \
            > "reports/phase2/$condition/seed${seed}/training.log" 2>&1 &

        # Store PID
        PIDS[$seed]=$!
        echo "    PID: ${PIDS[$seed]}"
    done

    # Wait for all 3 seeds of this condition to complete
    echo ""
    echo "  Waiting for seeds to complete..."
    for seed in "${SEEDS[@]}"; do
        wait ${PIDS[$seed]}
        echo "    ✓ Seed $seed complete"
    done

    echo ""
    echo "✓ Condition $condition complete"

    # Compute elapsed time
    ELAPSED=$(($(date +%s) - START_TIME))
    ELAPSED_MIN=$((ELAPSED / 60))
    echo "  Elapsed time: ${ELAPSED_MIN} minutes"
done

# Final summary
END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))
TOTAL_MIN=$((TOTAL_ELAPSED / 60))
TOTAL_HOURS=$((TOTAL_MIN / 60))
REMAINING_MIN=$((TOTAL_MIN % 60))

echo ""
echo "================================================================"
echo "Phase 2 Battery: COMPLETE"
echo "================================================================"
echo "Total experiments: 15"
echo "Total runtime: ${TOTAL_HOURS}h ${REMAINING_MIN}m"
echo "Results: reports/phase2/"
echo "================================================================"
echo ""
echo "Next steps:"
echo "  1. Run: python scripts/analyze_phase2_results.py"
echo "  2. Generate comparative figure"
echo "  3. Write Phase 2 summary"
echo ""
