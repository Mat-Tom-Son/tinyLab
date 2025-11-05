#!/usr/bin/env bash
set -euo pipefail

# Simple helper to loop over models for binder sweep + PCA rank.
# Usage:
#   bash scripts/multi_model_loop.sh gpt2-small gpt2-medium

MODELS=("$@")

if [ ${#MODELS[@]} -eq 0 ]; then
  echo "Usage: $0 <model1> [model2 ...]" >&2
  exit 1
fi

for M in "${MODELS[@]}"; do
  SAFE=${M//\//_}
  echo "== Model: $M =="
  # Binder sweep (mid-layers 8:12)
  python3 -m lab.analysis.binder_sweep \
    --model-name "$M" --device mps \
    --layer-start 8 --layer-end 12 \
    --max-examples 5000 --batch-size 64 \
    --output "reports/binder_sweep_${SAFE}.json" || true

  # PCA rank (facts)
  python3 -m lab.analysis.layer_pca_rank \
    --config lab/configs/run_h1_cross_condition_balanced.json \
    --tag facts --model-name "$M" --device mps \
    --samples 1024 --var-frac 0.90 \
    --output "reports/layer_pca_rank_${SAFE}_facts.json" || true

  # Sharpeners (facts, last-k=3) + overlay
  python3 -m lab.analysis.layer_entropy_and_sharpener_scan \
    --config lab/configs/run_h1_cross_condition_balanced.json \
    --tag facts --model-name "$M" --device mps \
    --samples 512 --last-k 3 \
    --output "reports/layer_entropy_scan_${SAFE}_facts.json" || true
  python3 paper/scripts/entropy_sharpener_plot.py \
    --input "reports/layer_entropy_scan_${SAFE}_facts.json" \
    --out "figs/entropy_overlay_${SAFE}" || true
done

echo "Done. Consider: make figures"
