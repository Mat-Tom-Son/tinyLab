#!/bin/bash
#
# Stage-1B omega sweep launcher
#
# Runs grokking experiments across omega grid with multiple seeds.
#
# Omega grid: [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.7]
# Seeds: [0, 1, 2]
# Total: 21 runs
#
# Usage:
#   bash scripts/run_stage1b_sweep.sh

set -e

# Configuration
OMEGA_VALUES=(0.3 0.5 0.7 1.0 1.3 1.5 1.7)
SEEDS=(0 1 2)
TARGET_HEAD=0  # Will be determined by VDI probe on baseline
STEPS=20000

echo "[stage1b-sweep] Starting omega sweep"
echo "  omega values: ${OMEGA_VALUES[@]}"
echo "  seeds: ${SEEDS[@]}"
echo "  target head: ${TARGET_HEAD}"
echo "  steps: ${STEPS}"
echo ""

# First, generate dataset if not exists
DATA_DIR="data"
DATA_FILE="${DATA_DIR}/modular_p113_train.jsonl"

if [ ! -f "${DATA_FILE}" ]; then
    echo "[stage1b-sweep] Generating modular arithmetic dataset..."
    python scripts/data_gen_modular.py \
        --modulus 113 \
        --train-fraction 0.9 \
        --seed 42 \
        --output-dir "${DATA_DIR}"
    echo ""
fi

# TODO: Add VDI probe to identify strongest suppressor head
# For now, using head 0 as placeholder

# Run sweep
TOTAL_RUNS=$((${#OMEGA_VALUES[@]} * ${#SEEDS[@]}))
CURRENT=0

for omega in "${OMEGA_VALUES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo "[${CURRENT}/${TOTAL_RUNS}] Running omega=${omega}, seed=${seed}"

        python scripts/train_stage1b_grokking.py \
            --omega "${omega}" \
            --head "${TARGET_HEAD}" \
            --seed "${seed}" \
            --steps "${STEPS}"

        echo ""
    done
done

echo "[stage1b-sweep] Sweep complete! Results in reports/stage1b_grokking/train/"
echo ""
echo "Next steps:"
echo "  1. Run VDI analysis on checkpoints"
echo "  2. Compute phase diagrams"
echo "  3. Test Le Chatelier compensation"
