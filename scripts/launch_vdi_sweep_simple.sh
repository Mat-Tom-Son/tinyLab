#!/bin/bash
# VDI Target Sweep - Simplified approach
#
# Modifies EXPERIMENTAL_CONDITIONS in homeostasis_aware_loss.py to add sweep conditions,
# then runs the standard training script.

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "================================"
echo "VDI TARGET SWEEP EXPERIMENT"
echo "================================"
echo ""
echo "Testing 5 VDI targets × 3 seeds = 15 runs"
echo ""

# VDI targets to sweep
VDI_TARGETS=(0.45 0.50 0.55 0.60 0.65)
SEEDS=(0 1 2)

# Configuration (matching intentional_vdi_target but with varying target_vdi)
LAMBDA_COMPENSATION=0.5
LAMBDA_CONVERGENCE=0.3
LAMBDA_SETPOINT=0.2
FAST_LR_SCALE=1.0
SLOW_LR_SCALE=0.1

echo "Adding sweep conditions to homeostasis_aware_loss.py..."

# Temporarily add sweep conditions to EXPERIMENTAL_CONDITIONS
python3 << 'PYTHON_SCRIPT'
import sys
from pathlib import Path

# Read the file
loss_file = Path("lab/src/losses/homeostasis_aware_loss.py")
content = loss_file.read_text()

# Find the EXPERIMENTAL_CONDITIONS dict and add sweep conditions
vdi_targets = [0.45, 0.50, 0.55, 0.60, 0.65]

sweep_conditions = ""
for target_vdi in vdi_targets:
    sweep_conditions += f"""
    'vdi_sweep_{target_vdi}': {{
        'name': 'VDI Sweep (target={target_vdi})',
        'lambda_compensation': 0.5,
        'lambda_convergence': 0.3,
        'lambda_setpoint': 0.2,
        'target_vdi': {target_vdi},
        'fast_lr_scale': 1.0,
        'slow_lr_scale': 0.1,
        'description': f'Test if equilibrium tracks target_vdi={target_vdi}',
    }},
"""

# Insert before the closing brace of EXPERIMENTAL_CONDITIONS
insert_marker = "}\n\n\nif __name__ == '__main__':"
if insert_marker in content:
    parts = content.split(insert_marker)
    new_content = parts[0].rstrip()[:-1] + ",\n" + sweep_conditions + "}\n\n\nif __name__ == '__main__':" + parts[1]
    loss_file.write_text(new_content)
    print("✓ Added sweep conditions")
else:
    print("⚠ Warning: Could not find insertion point")
    sys.exit(1)
PYTHON_SCRIPT

echo ""
echo "Launching sweep experiments..."
echo ""

for target_vdi in "${VDI_TARGETS[@]}"; do
    condition_name="vdi_sweep_${target_vdi}"

    echo "========================================"
    echo "Target VDI: $target_vdi"
    echo "========================================"

    # Launch 3 seeds in parallel
    for seed in "${SEEDS[@]}"; do
        echo "  Starting seed ${seed}..."

        .venv/bin/python scripts/train_phase2_dual_timescale.py \
            --condition "$condition_name" \
            --seed "$seed" \
            --steps 10000 \
            > "reports/phase2/${condition_name}/seed${seed}/training.log" 2>&1 &
    done

    # Wait for all seeds to complete
    echo "  Waiting for seeds to complete..."
    wait

    echo "  ✓ Complete"
    echo ""
done

echo "================================"
echo "VDI SWEEP COMPLETE"
echo "================================"
echo ""
echo "Results in: reports/phase2/vdi_sweep_*/"
echo ""
echo "Next: Run analysis script to check if VDI tracks target"
echo ""

PYTHON_SCRIPT
