#!/bin/bash
# Timescale Ablation Experiment
# Test if the forced attractor location moves with different slow_lr values

set -e
cd "$(dirname "$0")/.."

echo "=============================================="
echo "Timescale Ablation: Does Attractor Move?"
echo "=============================================="

source .venv/bin/activate

# Test slow_lr values: 0.01, 0.05, 0.1 (default), 0.5
# Each relative to fast_lr = 1.0
# Run with highest target (0.65) to see if higher slow_lr allows reaching it

for slow_lr in 0.01 0.05 0.1 0.5; do
    echo ""
    echo "Running slow_lr = $slow_lr..."
    for seed in 0 1 2; do
        python scripts/train_vdi_sweep.py \
            --target_vdi 0.65 \
            --seed $seed \
            --steps 10000 \
            --output_dir "reports/timescale_ablation/slow_lr_${slow_lr}/seed${seed}" \
            --lambda_compensation 0.5 \
            --lambda_convergence 0.3 \
            --lambda_setpoint 0.2 \
            --slow_lr_scale $slow_lr \
            --device auto
    done
done

echo ""
echo "=============================================="
echo "Timescale Ablation Complete!"
echo "=============================================="

# Analyze results
python -c "
import json
from pathlib import Path
import numpy as np

print('\\nRESULTS SUMMARY:')
print('='*60)
print(f'{\"slow_lr\":<10} | {\"Seed 0\":<10} | {\"Seed 1\":<10} | {\"Seed 2\":<10} | {\"Mean\":<10}')
print('-'*60)

for slow_lr in ['0.01', '0.05', '0.1', '0.5']:
    vdis = []
    for seed in [0, 1, 2]:
        summary_path = Path(f'reports/timescale_ablation/slow_lr_{slow_lr}/seed{seed}/sweep_summary.json')
        if summary_path.exists():
            with open(summary_path) as f:
                data = json.load(f)
                vdi = data['results']['final_vdi_mean']
                vdis.append(vdi)
        else:
            vdis.append(None)
    
    vdi_strs = [f'{v:.4f}' if v else 'N/A' for v in vdis]
    valid = [v for v in vdis if v]
    mean = np.mean(valid) if valid else None
    mean_str = f'{mean:.4f}' if mean else 'N/A'
    print(f'{slow_lr:<10} | {vdi_strs[0]:<10} | {vdi_strs[1]:<10} | {vdi_strs[2]:<10} | {mean_str:<10}')

print()
print('If mean VDI increases with slow_lr, the attractor is timescale-dependent.')
print('If mean VDI stays constant, the attractor is architectural.')
"
