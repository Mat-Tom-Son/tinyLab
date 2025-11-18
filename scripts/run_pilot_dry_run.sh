#!/usr/bin/env bash
set -euo pipefail

# Simple Stage 1A dry run:
#   Generation (Task B) -> Baseline Measurement (Circularity) ->
#   Structural Probe (VDI) -> TODO: Training + Scaling Hooks integration.
#
# This script is meant to validate that the new utilities and data flows work
# end-to-end on a small pretrained model (e.g., gpt2-small). It does NOT train
# the 2-layer pilot model; you will plug your training loop into the marked
# section below.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "[pilot] Using ROOT_DIR=${ROOT_DIR}"

REPORTS_DIR="reports/pilot_stage1a"
mkdir -p "${REPORTS_DIR}"

########################################
# 1. Generate Task B (Weekday) data
########################################

TASK_B_PATH="lab/data/task_b_weekdays.jsonl"
echo "[pilot] Generating Task B data at ${TASK_B_PATH}"
python3 scripts/generate_task_b.py \
  --n-examples 2000 \
  --min-offset 0 \
  --max-offset 6 \
  --seed 0 \
  --out-path "${TASK_B_PATH}"

########################################
# 2. Baseline circularity on a pretrained model
########################################

echo "[pilot] Measuring baseline circularity on gpt2-small (no training, sanity check)"
python3 scripts/measure_circularity.py \
  --model-name gpt2-small \
  --dtype float32 \
  --device auto \
  --layer-index 0 \
  --position-index -1 \
  --data-path "${TASK_B_PATH}" \
  --max-examples 512 \
  --seed 0 \
  --summary-out "${REPORTS_DIR}/baseline_circularity_gpt2small.json" \
  --points-out "${REPORTS_DIR}/baseline_circularity_gpt2small_points.csv"

########################################
# 3. VDI probe on a small pretrained model
########################################

echo "[pilot] Running VDI probe on gpt2-small (layer 0) for structural sanity check"
python3 scripts/identify_suppressors.py \
  --model-name gpt2-small \
  --dtype float32 \
  --device auto \
  --layer-index 0 \
  --n-prompts 256 \
  --k-noise 4 \
  --sigma 0.05 \
  --seed 0 \
  --out-path "${REPORTS_DIR}/vdi_layer0_gpt2small.csv"

echo "[pilot] VDI CSV written to ${REPORTS_DIR}/vdi_layer0_gpt2small.csv"
echo "[pilot] You can now apply the prereg selection logic to pick suppressor/random heads."

########################################
# 4. TODO: Plug in 2-layer training + scaling hooks
########################################

cat << 'EOF'
[pilot] NOTE:
  This dry run uses a fixed pretrained model (gpt2-small) to validate:
    - Task B data generation
    - CircularityScore measurement
    - VDI computation on layer 0

  To execute the full Stage 1A protocol you still need to:
    - Train the 2-layer pilot model on Task A/B from scratch.
    - At checkpoints 5/10/15, run identify_suppressors.py to compute VDI.
    - Select suppressor and random-head controls from those VDI CSVs.
    - Wrap your training loop with scaling hooks from lab/src/utils/head_scaling.py
      so that alpha-scaling applies at every training step from step 0 onward.
EOF

echo "[pilot] Dry run complete."
