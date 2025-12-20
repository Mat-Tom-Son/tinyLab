#!/bin/bash
# Monitor Phase 2 Experiments Status
#
# Usage: bash scripts/monitor_phase2.sh

echo "================================================================"
echo "Phase 2: Experiment Status Monitor"
echo "================================================================"
echo ""

# Check active processes
ACTIVE=$(ps aux | grep "train_phase2_dual_timescale.py" | grep -v grep | wc -l | tr -d ' ')
echo "Active training processes: $ACTIVE"
echo ""

# Check each condition
CONDITIONS=("baseline" "dual_timescale" "explicit_convergence" "intentional_vdi_target" "early_convergence")

for condition in "${CONDITIONS[@]}"; do
    echo "--- $condition ---"

    for seed in 0 1 2; do
        dir="reports/phase2/$condition/seed$seed"

        # Check if results exist
        if [ -f "$dir/phase2_summary.json" ]; then
            # Extract results
            duration=$(cat "$dir/phase2_summary.json" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['results'].get('crystallization_duration', 'N/A'))" 2>/dev/null || echo "N/A")
            final_acc=$(cat "$dir/phase2_summary.json" | python3 -c "import sys, json; d=json.load(sys.stdin); print(f\"{d['results']['final_test_acc']:.3f}\")" 2>/dev/null || echo "N/A")
            echo "  Seed $seed: ✓ COMPLETE | Duration: $duration steps | Acc: $final_acc"
        elif [ -f "$dir/training.log" ] && [ -s "$dir/training.log" ]; then
            # Check log for latest step
            last_step=$(tail -20 "$dir/training.log" 2>/dev/null | grep "Step" | tail -1 | awk '{print $2}' | tr -d ' ')
            if [ ! -z "$last_step" ]; then
                echo "  Seed $seed: ⏳ RUNNING | Latest step: $last_step"
            else
                echo "  Seed $seed: ⏳ RUNNING | Initializing..."
            fi
        elif ps aux | grep "train_phase2.*--condition $condition.*--seed $seed" | grep -v grep > /dev/null 2>&1; then
            echo "  Seed $seed: 🔄 STARTING"
        else
            echo "  Seed $seed: ⭕ NOT STARTED"
        fi
    done
    echo ""
done

echo "================================================================"
echo "Quick Commands:"
echo "  Watch logs:      tail -f reports/phase2/baseline/seed0/training.log"
echo "  Check processes: ps aux | grep train_phase2 | grep -v grep"
echo "  Kill all:        pkill -f train_phase2_dual_timescale.py"
echo "  Rerun monitor:   bash scripts/monitor_phase2.sh"
echo "================================================================"
