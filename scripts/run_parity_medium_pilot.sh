#!/bin/bash
#
# Parity Medium Difficulty Pilot (length 10-12)
#
# Tests the Goldilocks Zone:
# - Search space: ~1,000-4,096 possible strings
# - Training set: 1,000 examples
# - Model MUST generalize, cannot just memorize
#
# This should:
# 1. Hit 100% Train Accuracy (memorization works)
# 2. Stay at ~50% Test Accuracy (random guessing)
# 3. Then GROK: sudden jump to ~100% Test (finds circuit)
#

set -e

echo "=========================================="
echo "Parity Medium Difficulty Pilot"
echo "=========================================="
echo ""
echo "Dataset: Length 10-12 binary strings"
echo "  Search space: 2^10 to 2^12 (1,024-4,096 strings)"
echo "  Training: 1,000 examples"
echo "  Test: 500 examples"
echo ""
echo "Model: 2-layer, 4-head, 64-dim transformer"
echo "Steps: 10,000 (baseline)"
echo "Omega: 1.0 (no perturbation)"
echo ""
echo "What to watch for:"
echo "  Phase 1: Train acc → 100% (memorization)"
echo "  Phase 2: Test acc stays ~50% (stuck)"
echo "  Phase 3: GROK - Test acc jumps to ~100%"
echo ""

python3 scripts/train_parity.py \
    --omega 1.0 \
    --head 0 \
    --seed 0 \
    --steps 10000 \
    --device cpu \
    --data-dir data_parity_medium \
    2>&1 | tee parity_medium_pilot.log

echo ""
echo "=========================================="
echo "Pilot Complete!"
echo "=========================================="
echo ""
echo "Check results:"
echo "  grep 'step' parity_medium_pilot.log | tail -20"
echo "  cat reports/parity/train/parity_head0_omega1.0_seed0/metrics.jsonl | tail -10"
