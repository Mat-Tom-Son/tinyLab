#!/bin/bash
# Demonstration of developmental monitoring framework
#
# This runs a quick experiment showing:
# 1. Baseline training (ω=1.0) with monitoring
# 2. Perturbation experiments (ω=0.5, ω=1.5)
# 3. Visualization of developmental trajectory
#
# Usage:
#   bash scripts/run_developmental_demo.sh

set -e

echo "========================================================================"
echo "Developmental Monitoring Framework Demo"
echo "========================================================================"
echo ""
echo "This demonstration will:"
echo "  1. Train 3 models with omega sweep (0.5, 1.0, 1.5)"
echo "  2. Track VDI snap, Le Chatelier compensation, MI saturation"
echo "  3. Generate visualizations"
echo ""
echo "Expected runtime: ~30 minutes on CPU, ~10 minutes on MPS/CUDA"
echo ""

# Configuration
STEPS=${STEPS:-5000}
MONITOR_INTERVAL=${MONITOR_INTERVAL:-250}
KILL_TEST_FREQ=${KILL_TEST_FREQ:-2}
DEVICE=${DEVICE:-auto}
SEEDS=${SEEDS:-"0"}

echo "Configuration:"
echo "  Training steps: $STEPS"
echo "  Monitor interval: $MONITOR_INTERVAL"
echo "  Kill test frequency: every $KILL_TEST_FREQ checkpoints"
echo "  Device: $DEVICE"
echo "  Seeds: $SEEDS"
echo ""

read -p "Press Enter to continue or Ctrl-C to abort..."
echo ""

# Create output directory
OUTDIR="reports/developmental_monitoring"
mkdir -p "$OUTDIR"

# Omega sweep
OMEGAS="0.5 1.0 1.5"

echo "========================================================================"
echo "Phase 1: Training with Omega Sweep"
echo "========================================================================"
echo ""

for omega in $OMEGAS; do
    for seed in $SEEDS; do
        echo "Training: omega=$omega, seed=$seed"
        echo "----------------------------------------"

        python scripts/train_with_developmental_monitoring.py \
            --task parity \
            --omega "$omega" \
            --head 0 \
            --seed "$seed" \
            --steps "$STEPS" \
            --monitor-interval "$MONITOR_INTERVAL" \
            --kill-test-frequency "$KILL_TEST_FREQ" \
            --device "$DEVICE"

        echo ""
    done
done

echo "========================================================================"
echo "Phase 2: Visualization"
echo "========================================================================"
echo ""

for omega in $OMEGAS; do
    for seed in $SEEDS; do
        TRAJ_PATH="$OUTDIR/parity_omega${omega}_seed${seed}/developmental_trajectory.json"

        if [ -f "$TRAJ_PATH" ]; then
            echo "Visualizing: omega=$omega, seed=$seed"

            python scripts/visualize_developmental_trajectory.py "$TRAJ_PATH"

            echo "  ✓ Saved to: ${TRAJ_PATH%.json}.png"
            echo ""
        else
            echo "  ⚠ Trajectory not found: $TRAJ_PATH"
        fi
    done
done

echo "========================================================================"
echo "Phase 3: Summary"
echo "========================================================================"
echo ""

echo "Results saved to: $OUTDIR"
echo ""
echo "Key outputs:"
echo "  - developmental_trajectory.json: Raw monitoring data"
echo "  - developmental_trajectory.png: Diagnostic plots"
echo "  - training_metrics.jsonl: Training loss/accuracy"
echo ""

# Generate summary table
echo "Snap Detection Summary:"
echo ""
printf "%-10s %-10s %-12s %-15s\n" "Omega" "Seed" "Snap Step" "Confidence"
echo "--------------------------------------------------------------"

for omega in $OMEGAS; do
    for seed in $SEEDS; do
        TRAJ_PATH="$OUTDIR/parity_omega${omega}_seed${seed}/developmental_trajectory.json"

        if [ -f "$TRAJ_PATH" ]; then
            # Extract snap info using jq if available
            if command -v jq &> /dev/null; then
                SNAP_DETECTED=$(jq -r '.summary.snap_detected' "$TRAJ_PATH")
                if [ "$SNAP_DETECTED" = "true" ]; then
                    SNAP_STEP=$(jq -r '.summary.snap_step' "$TRAJ_PATH")
                    SNAP_CONF=$(jq -r '.summary.snap_confidence' "$TRAJ_PATH")
                    printf "%-10s %-10s %-12s %-15s\n" "$omega" "$seed" "$SNAP_STEP" "$SNAP_CONF"
                else
                    printf "%-10s %-10s %-12s %-15s\n" "$omega" "$seed" "Not detected" "-"
                fi
            else
                printf "%-10s %-10s %-12s %-15s\n" "$omega" "$seed" "(install jq)" "(install jq)"
            fi
        fi
    done
done

echo ""
echo "========================================================================"
echo "Demo Complete!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Review visualizations in $OUTDIR/*/developmental_trajectory.png"
echo "  2. Inspect raw data in developmental_trajectory.json"
echo "  3. See docs/DEVELOPMENTAL_MONITORING.md for integration guide"
echo ""
echo "Expected observations:"
echo "  - ω=0.5 (weakened): Delayed snap, increased compensation"
echo "  - ω=1.0 (baseline): Normal trajectory"
echo "  - ω=1.5 (strengthened): Accelerated snap, reduced compensation"
echo ""
