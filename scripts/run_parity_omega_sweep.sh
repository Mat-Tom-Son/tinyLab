#!/bin/bash
#
# Parity Omega Sweep - Test Thermodynamic Hypothesis
#
# Baseline: T_grok = 3,700 steps at omega=1.0
#
# Hypothesis: Suppressor scaling (omega) controls phase transition timing
# - Low omega (weak suppression): Earlier grokking
# - High omega (strong suppression): Later grokking
#

set -e

echo "=========================================="
echo "Parity Omega Sweep"
echo "=========================================="
echo ""
echo "Baseline Result:"
echo "  omega=1.0 → T_grok=3,700 steps"
echo ""
echo "Testing omega values: 0.5, 0.7, 1.3, 1.5"
echo "  (1.0 already complete)"
echo ""
echo "Steps: 10,000 (sufficient to capture T_grok)"
echo "Dataset: Length 10-12, 1,000 train examples"
echo ""
echo "Predictions:"
echo "  omega=0.5: T_grok < 3,700 (weak suppression)"
echo "  omega=0.7: T_grok ≈ 3,500"
echo "  omega=1.0: T_grok = 3,700 ✓ (baseline)"
echo "  omega=1.3: T_grok ≈ 3,900"
echo "  omega=1.5: T_grok > 4,000 (strong suppression)"
echo ""
echo "=========================================="
echo ""

OMEGA_VALUES=(0.5 0.7 1.3 1.5)
SEED=0
HEAD=0
STEPS=10000

for omega in "${OMEGA_VALUES[@]}"; do
    echo "[$(date +%H:%M:%S)] Starting omega=${omega}..."

    .venv/bin/python scripts/train_parity.py \
        --omega ${omega} \
        --head ${HEAD} \
        --seed ${SEED} \
        --steps ${STEPS} \
        --device cpu \
        --data-dir data_parity_medium \
        > parity_omega${omega}_seed${SEED}.log 2>&1

    # Extract T_grok
    T_GROK=$(grep -o '"T_grok": [0-9]*' reports/parity/train/parity_head${HEAD}_omega${omega}_seed${SEED}/metrics.jsonl | tail -1 | grep -o '[0-9]*' || echo "null")

    echo "[$(date +%H:%M:%S)] omega=${omega} complete. T_grok=${T_GROK}"
    echo ""
done

echo "=========================================="
echo "Omega Sweep Complete!"
echo "=========================================="
echo ""
echo "Results Summary:"
echo "----------------"

for omega in 0.5 0.7 1.0 1.3 1.5; do
    METRICS_FILE="reports/parity/train/parity_head${HEAD}_omega${omega}_seed${SEED}/metrics.jsonl"
    if [ -f "$METRICS_FILE" ]; then
        T_GROK=$(grep -o '"T_grok": [0-9]*' "$METRICS_FILE" | tail -1 | grep -o '[0-9]*' || echo "null")
        FINAL_ACC=$(tail -1 "$METRICS_FILE" | grep -o '"test_acc": [0-9.]*' | grep -o '[0-9.]*')
        echo "  omega=${omega}: T_grok=${T_GROK}, final_test_acc=${FINAL_ACC}"
    else
        echo "  omega=${omega}: NOT RUN"
    fi
done

echo ""
echo "Detailed results in:"
echo "  reports/parity/train/parity_head${HEAD}_omega*_seed${SEED}/"
echo ""
