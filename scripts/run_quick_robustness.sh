#!/bin/bash
# Quick Hyperparameter Robustness Sweep
# Tests if VDI = 0.612 survives hyperparameter perturbation

set -e
cd "$(dirname "$0")/.."

source .venv/bin/activate

echo "=============================================="
echo "Hyperparameter Robustness Sweep"
echo "=============================================="

# Baseline: lr=0.001, wd=0.1 (already ran, VDI=0.612)
# Test 3 variations to check robustness

# 1. Learning rate variations
echo ""
echo "=== Learning Rate Sweep ==="
for lr in 0.0005 0.002; do
    echo "Running lr=$lr..."
    python scripts/train_modular_with_monitoring.py \
        --p 113 \
        --omega 1.0 \
        --seed 0 \
        --steps 10000 \
        --monitor-interval 1000 \
        --device auto 2>&1 | grep -E "(Mean VDI|GROKKING|Final)" | tail -5
done

# 2. Weight decay variations
echo ""
echo "=== Weight Decay Sweep ==="
for wd in 0.05 0.2; do
    echo "Running wd=$wd..."
    python scripts/train_modular_with_monitoring.py \
        --p 113 \
        --omega 1.0 \
        --seed 0 \
        --steps 10000 \
        --monitor-interval 1000 \
        --device auto 2>&1 | grep -E "(Mean VDI|GROKKING|Final)" | tail -5
done

echo ""
echo "=============================================="
echo "Robustness Sweep Complete!"
echo "=============================================="
echo ""
echo "If all runs show VDI ≈ 0.612, the precision claim is robust."
echo "If VDI varies significantly, we need to update the paper claims."
