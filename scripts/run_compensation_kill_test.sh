#!/bin/bash
#
# COMPENSATION KILL TEST
#
# This experiment tests whether homeostatic compensation is ACTIVE or PASSIVE.
#
# Design:
# 1. Baseline: omega=1.0, no freezing (reference)
# 2. Perturbed: omega=0.5, no freezing (fast grokking, compensation allowed)
# 3. KILL TEST: omega=0.5, freeze heads 1,2,3 (compensation blocked)
#
# Predictions:
# - If compensation is ACTIVE/NECESSARY:
#   Test 3 should FAIL to grok (or grok much later than Test 2)
#   Because system cannot redistribute variance to restore equilibrium
#
# - If compensation is PASSIVE (just convergence):
#   Test 3 should grok similarly to Test 2
#   Because system doesn't need compensation, just finds same attractor
#
# This distinguishes:
#   Active homeostasis vs Passive optimization dynamics
#

set -e

echo "=========================================="
echo "COMPENSATION KILL TEST"
echo "=========================================="
echo ""
echo "Testing whether homeostatic compensation is active or passive."
echo ""
echo "Three conditions:"
echo "  1. BASELINE (ω=1.0, no freeze): Reference grokking"
echo "  2. PERTURBED (ω=0.5, no freeze): Accelerated grokking with compensation"
echo "  3. KILL TEST (ω=0.5, freeze 1,2,3): Compensation blocked"
echo ""
echo "Critical comparison: Does Test 3 fail to grok (or grok late)?"
echo "  YES → Compensation is ACTIVE (system needs it)"
echo "  NO → Compensation is PASSIVE (system doesn't need it)"
echo ""
echo "=========================================="
echo ""

# Shared parameters
SEED=0
HEAD=0
STEPS=10000
DATA_DIR="data_parity_medium"

echo "[$(date +%H:%M:%S)] TEST 1: BASELINE (ω=1.0, no freeze)"
echo "  Expected: Grok at ~3,700 steps (stable baseline)"
echo ""

.venv/bin/python scripts/train_parity.py \
    --omega 1.0 \
    --head ${HEAD} \
    --seed ${SEED} \
    --steps ${STEPS} \
    --device cpu \
    --data-dir ${DATA_DIR} \
    > killtest_baseline.log 2>&1

T_BASELINE=$(grep -o '"T_grok": [0-9]*' reports/parity/train/parity_head0_omega1.0_seed0/metrics.jsonl | tail -1 | grep -o '[0-9]*' || echo "null")
echo "[$(date +%H:%M:%S)] ✓ Baseline complete. T_grok = ${T_BASELINE}"
echo ""

echo "[$(date +%H:%M:%S)] TEST 2: PERTURBED (ω=0.5, no freeze)"
echo "  Expected: Grok at ~2,200 steps (fast with compensation)"
echo ""

.venv/bin/python scripts/train_parity.py \
    --omega 0.5 \
    --head ${HEAD} \
    --seed ${SEED} \
    --steps ${STEPS} \
    --device cpu \
    --data-dir ${DATA_DIR} \
    > killtest_perturbed.log 2>&1

T_PERTURBED=$(grep -o '"T_grok": [0-9]*' reports/parity/train/parity_head0_omega0.5_seed0/metrics.jsonl | tail -1 | grep -o '[0-9]*' || echo "null")
echo "[$(date +%H:%M:%S)] ✓ Perturbed complete. T_grok = ${T_PERTURBED}"
echo ""

echo "[$(date +%H:%M:%S)] TEST 3: KILL TEST (ω=0.5, freeze heads 1,2,3)"
echo "  CRITICAL TEST: Can system grok without compensation?"
echo ""

.venv/bin/python scripts/train_parity_frozen.py \
    --omega 0.5 \
    --head ${HEAD} \
    --seed ${SEED} \
    --steps ${STEPS} \
    --device cpu \
    --data-dir ${DATA_DIR} \
    --freeze-heads 1 2 3 \
    > killtest_frozen.log 2>&1

T_FROZEN=$(grep -o '"T_grok": [0-9]*' reports/parity/train/parity_head0_omega0.5_seed0_frozen123/metrics.jsonl | tail -1 | grep -o '[0-9]*' || echo "null")
echo "[$(date +%H:%M:%S)] ✓ Kill test complete. T_grok = ${T_FROZEN}"
echo ""

echo "=========================================="
echo "RESULTS SUMMARY"
echo "=========================================="
echo ""
echo "Condition                     | T_grok | Interpretation"
echo "------------------------------+--------+----------------------------------"
echo "1. Baseline (ω=1.0)           | ${T_BASELINE}  | Natural grokking (reference)"
echo "2. Perturbed (ω=0.5)          | ${T_PERTURBED}  | Fast grokking (compensation works)"
echo "3. FROZEN (ω=0.5, no comp)    | ${T_FROZEN}  | ??? (CRITICAL RESULT)"
echo ""
echo "=========================================="
echo "INTERPRETATION:"
echo "=========================================="
echo ""

if [ "$T_FROZEN" = "null" ]; then
    echo "✓ ACTIVE COMPENSATION CONFIRMED"
    echo ""
    echo "Kill test FAILED to grok within ${STEPS} steps."
    echo "When compensation was blocked, system could not find the circuit."
    echo ""
    echo "This proves:"
    echo "  - Compensation is NECESSARY for grokking"
    echo "  - System actively redistributes variance to restore equilibrium"
    echo "  - This is homeostatic resistance, not passive convergence"
    echo ""
    echo "Implications:"
    echo "  - The stability basin arises from active compensation mechanisms"
    echo "  - Networks have 'metabolism' that fights perturbations"
    echo "  - Cannot control developmental windows without understanding compensation"
elif [ "$T_FROZEN" -gt $((T_PERTURBED * 2)) ]; then
    echo "✓ COMPENSATION IS IMPORTANT (but not strictly necessary)"
    echo ""
    echo "Kill test grokked MUCH later (T=${T_FROZEN} vs T=${T_PERTURBED})."
    echo "Compensation accelerates grokking but system can eventually succeed without it."
    echo ""
    echo "This suggests:"
    echo "  - Compensation provides an efficient path"
    echo "  - But system can find alternative routes (slower, messier)"
    echo "  - Partial homeostatic effect"
else
    echo "✗ COMPENSATION IS PASSIVE (convergence effect only)"
    echo ""
    echo "Kill test grokked at similar time (T=${T_FROZEN} vs T=${T_PERTURBED})."
    echo "Blocking compensation did not significantly impact grokking."
    echo ""
    echo "This suggests:"
    echo "  - Compensation is not the mechanism"
    echo "  - Stability basin arises from optimization dynamics (loss landscape)"
    echo "  - System passively converges to same attractor regardless of path"
    echo ""
    echo "Revised hypothesis:"
    echo "  - Weight decay + omega interaction reshapes loss landscape"
    echo "  - Critical slowing down near natural equilibrium"
    echo "  - No active homeostasis required"
fi

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. Examine training curves for all three conditions"
echo "2. Compare loss volatility (is frozen more/less stable?)"
echo "3. Check if frozen condition finds different solution (different final VDI)"
echo "4. Run with other omega values (ω=1.5) to test symmetry"
echo ""
echo "Detailed logs:"
echo "  killtest_baseline.log"
echo "  killtest_perturbed.log"
echo "  killtest_frozen.log"
echo ""
