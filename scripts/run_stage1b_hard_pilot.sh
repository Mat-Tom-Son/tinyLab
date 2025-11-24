#!/bin/bash
#
# Stage-1B Hard Mode Pilot (p=997)
#
# Forces genuine grokking by using p=997 with only 0.5% training data
# Search space: 997^2 ≈ 1M pairs
# Training set: ~5000 examples (0.5%)
# Model MUST learn the algorithm, cannot memorize
#

set -e

echo "=========================================="
echo "Stage-1B HARD MODE Pilot"
echo "=========================================="
echo ""

# 1. Generate Hard Data (p=997)
echo "[1/3] Generating Hard Dataset (p=997, 0.5% training data)..."
mkdir -p data_hard
python3 scripts/data_gen_modular.py \
    --modulus 997 \
    --train-fraction 0.005 \
    --seed 42 \
    --output-dir data_hard

echo ""
echo "[2/3] Updating config to use p=997..."

# Backup original data
if [ ! -L data/modular_p113_train.jsonl.bak ]; then
    if [ -f data/modular_p113_train.jsonl ]; then
        mv data/modular_p113_train.jsonl data/modular_p113_train.jsonl.bak
        mv data/modular_p113_test.jsonl data/modular_p113_test.jsonl.bak
    fi
fi

# Symlink to hard data
ln -sf ../data_hard/modular_p997_train.jsonl data/modular_p113_train.jsonl
ln -sf ../data_hard/modular_p997_test.jsonl data/modular_p113_test.jsonl

echo "  Training set size: $(wc -l < data_hard/modular_p997_train.jsonl) examples"
echo "  Test set size: $(wc -l < data_hard/modular_p997_test.jsonl) examples"
echo "  Search space: 994,009 total pairs"
echo "  Training coverage: ~0.5%"
echo ""

echo "[3/3] Starting Hard Mode Training..."
echo "  Model: 1-layer, 64-dim (tiny)"
echo "  Steps: 10,000"
echo "  Omega: 1.0 (baseline)"
echo ""
echo "What to watch for:"
echo "  - Train acc → 1.0 (model memorizes training set)"
echo "  - Test acc stays at ~0.001 (random guessing)"
echo "  - Then GROK: sudden jump in test acc"
echo ""

python3 scripts/train_stage1b_grokking.py \
    --omega 1.0 \
    --head 0 \
    --seed 0 \
    --steps 10000

echo ""
echo "=========================================="
echo "Hard Mode Pilot Complete!"
echo "=========================================="
echo ""
echo "Check results:"
echo "  tail -50 reports/stage1b_grokking/train/stage1b_head0_omega1.0_seed0/metrics.jsonl"
echo ""
echo "To restore original data:"
echo "  rm data/modular_p113_{train,test}.jsonl"
echo "  mv data/modular_p113_train.jsonl.bak data/modular_p113_train.jsonl"
echo "  mv data/modular_p113_test.jsonl.bak data/modular_p113_test.jsonl"
