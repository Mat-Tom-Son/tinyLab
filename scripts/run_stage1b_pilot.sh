#!/bin/bash
#
# Stage-1B Pilot Run (CPU-friendly)
#
# Runs a reduced sweep to validate the experiment:
# - 3 omega values: 0.5, 1.0, 1.5
# - 1 seed per omega
# - 5000 steps (enough to see if grokking happens)
#
# Expected runtime on CPU: ~8-12 hours total
#
# Usage:
#   bash scripts/run_stage1b_pilot.sh

set -e

# Configuration
OMEGA_VALUES=(0.5 1.0 1.5)
SEEDS=(0)
TARGET_HEAD=0
STEPS=5000

echo "[stage1b-pilot] Starting pilot sweep"
echo "  omega values: ${OMEGA_VALUES[@]}"
echo "  seeds: ${SEEDS[@]}"
echo "  steps: ${STEPS}"
echo "  target head: ${TARGET_HEAD}"
echo ""
echo "Expected runtime: ~8-12 hours on CPU"
echo ""

# Generate dataset if not exists
DATA_DIR="data"
DATA_FILE="${DATA_DIR}/modular_p113_train.jsonl"

if [ ! -f "${DATA_FILE}" ]; then
    echo "[stage1b-pilot] Generating modular arithmetic dataset..."
    python3 scripts/data_gen_modular.py \
        --modulus 113 \
        --train-fraction 0.9 \
        --seed 42 \
        --output-dir "${DATA_DIR}"
    echo ""
fi

# Run pilot sweep
TOTAL_RUNS=$((${#OMEGA_VALUES[@]} * ${#SEEDS[@]}))
CURRENT=0

START_TIME=$(date +%s)

for omega in "${OMEGA_VALUES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo ""
        echo "=========================================="
        echo "[${CURRENT}/${TOTAL_RUNS}] omega=${omega}, seed=${seed}"
        echo "=========================================="

        RUN_START=$(date +%s)

        python3 scripts/train_stage1b_grokking.py \
            --omega "${omega}" \
            --head "${TARGET_HEAD}" \
            --seed "${seed}" \
            --steps "${STEPS}"

        RUN_END=$(date +%s)
        RUN_ELAPSED=$((RUN_END - RUN_START))
        echo ""
        echo "Run completed in $(($RUN_ELAPSED / 60)) minutes"

        # Estimate remaining time
        if [ $CURRENT -lt $TOTAL_RUNS ]; then
            TOTAL_ELAPSED=$((RUN_END - START_TIME))
            AVG_TIME=$((TOTAL_ELAPSED / CURRENT))
            REMAINING_RUNS=$((TOTAL_RUNS - CURRENT))
            EST_REMAINING=$((AVG_TIME * REMAINING_RUNS))
            echo "Estimated time remaining: $(($EST_REMAINING / 60)) minutes"
        fi
        echo ""
    done
done

END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "[stage1b-pilot] Pilot sweep complete!"
echo "=========================================="
echo "Total time: $(($TOTAL_ELAPSED / 60)) minutes"
echo ""
echo "Results in: reports/stage1b_grokking/train/"
echo ""
echo "Quick check results:"
for omega in "${OMEGA_VALUES[@]}"; do
    RUN_DIR="reports/stage1b_grokking/train/stage1b_head0_omega${omega}_seed0"
    if [ -f "${RUN_DIR}/metrics.jsonl" ]; then
        echo ""
        echo "Omega = ${omega}:"
        tail -1 "${RUN_DIR}/metrics.jsonl" | python3 -c "import sys, json; d=json.load(sys.stdin); print(f\"  Final: step={d['step']}, test_acc={d['test_acc']:.3f}, T_grok={d['T_grok']}\")"
    fi
done

echo ""
echo "Next steps:"
echo "  1. Check if grokking happened (T_grok values)"
echo "  2. If yes: run full sweep with bash scripts/run_stage1b_sweep.sh"
echo "  3. If no: adjust parameters or task complexity"
echo "  4. Generate plots: python3 scripts/plot_phase_diagrams.py"
