#!/bin/bash
# Run Missing VDI Sweep Experiments
# Fill in gaps for targets 0.50 and 0.60 that the reviewers identified

set -e
cd "$(dirname "$0")/.."

echo "=============================================="
echo "VDI Sweep: Missing Targets (0.50, 0.60)"
echo "=============================================="

# Check for data
if [ ! -f "data/modular_p113_train.jsonl" ]; then
    echo "Generating modular arithmetic data..."
    python scripts/data_gen_modular.py --p 113
fi

# Run missing target 0.50 (3 seeds)
echo ""
echo "Running VDI target = 0.50..."
for seed in 0 1 2; do
    echo "  Seed $seed..."
    python scripts/train_vdi_sweep.py \
        --target_vdi 0.50 \
        --seed $seed \
        --steps 10000 \
        --output_dir "reports/phase2/vdi_sweep_0.50/seed${seed}" \
        --lambda_compensation 0.5 \
        --lambda_convergence 0.3 \
        --lambda_setpoint 0.2 \
        --device auto
done

# Run missing target 0.60 (3 seeds)
echo ""
echo "Running VDI target = 0.60..."
for seed in 0 1 2; do
    echo "  Seed $seed..."
    python scripts/train_vdi_sweep.py \
        --target_vdi 0.60 \
        --seed $seed \
        --steps 10000 \
        --output_dir "reports/phase2/vdi_sweep_0.60/seed${seed}" \
        --lambda_compensation 0.5 \
        --lambda_convergence 0.3 \
        --lambda_setpoint 0.2 \
        --device auto
done

echo ""
echo "=============================================="
echo "VDI Sweep Complete! Analyzing results..."
echo "=============================================="
python scripts/analyze_vdi_sweep.py
