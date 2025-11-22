#!/usr/bin/env bash
set -euo pipefail

# Run a minimal end-to-end Stage-1A pilot suite:
# - Generate Task B weekday data
# - Baseline circularity on gpt2-small (multi-seed)
# - VDI probe on layer-0 (multi-seed)
# - Optional dry-run summary script
#
# Environment toggles:
#   SEEDS="0 1"          # space-separated seeds
#   REPORTS_DIR="reports/pilot_stage1a/suite"
#   RUN_CIRCULARITY=1    # set 0 to skip
#   RUN_VDI=1            # set 0 to skip
#   RUN_DRY_RUN=1        # set 0 to skip the small dry run
#   TASK_B_SIZE=2000     # number of Task B examples to generate
#   DEVICE="auto"        # passed to scripts (auto/cuda/cpu)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SEEDS=(${SEEDS:-0 1})
REPORTS_DIR="${REPORTS_DIR:-reports/pilot_stage1a/suite}"
RUN_CIRCULARITY="${RUN_CIRCULARITY:-1}"
RUN_VDI="${RUN_VDI:-1}"
RUN_DRY_RUN="${RUN_DRY_RUN:-1}"
TASK_B_SIZE="${TASK_B_SIZE:-2000}"
DEVICE="${DEVICE:-auto}"
TASK_B_PATH="lab/data/task_b_weekdays.jsonl"
PYBIN="${PYBIN:-python3}"

echo "[stage1a] ROOT=${ROOT_DIR}"
mkdir -p "${REPORTS_DIR}"

if ! ${PYBIN} - <<'PY' >/dev/null 2>&1
import torch, sys
print(torch.__version__)
sys.exit(0)
PY
then
  echo "[stage1a] Python or torch not available; did you activate the venv?"
  exit 1
fi

echo "[stage1a] Checking CUDA availability..."
${PYBIN} - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
PY

echo "[stage1a] Generating Task B data -> ${TASK_B_PATH}"
${PYBIN} scripts/generate_task_b.py \
  --n-examples "${TASK_B_SIZE}" \
  --min-offset 0 \
  --max-offset 6 \
  --seed 0 \
  --out-path "${TASK_B_PATH}"

if [[ "${RUN_CIRCULARITY}" == "1" ]]; then
  echo "[stage1a] Running baseline circularity (gpt2-small) for seeds: ${SEEDS[*]}"
  for s in "${SEEDS[@]}"; do
    out_json="${REPORTS_DIR}/baseline_circularity_gpt2small_seed${s}.json"
    out_points="${REPORTS_DIR}/baseline_circularity_gpt2small_seed${s}_points.csv"
    ${PYBIN} scripts/measure_circularity.py \
      --model-name gpt2-small \
      --dtype float16 \
      --device "${DEVICE}" \
      --layer-index 0 \
      --position-index -1 \
      --data-path "${TASK_B_PATH}" \
      --max-examples 512 \
      --seed "${s}" \
      --summary-out "${out_json}" \
      --points-out "${out_points}"
  done
fi

if [[ "${RUN_VDI}" == "1" ]]; then
  echo "[stage1a] Running VDI probe (layer 0) for seeds: ${SEEDS[*]}"
  for s in "${SEEDS[@]}"; do
    out_csv="${REPORTS_DIR}/vdi_layer0_gpt2small_seed${s}.csv"
    ${PYBIN} scripts/identify_suppressors.py \
      --model-name gpt2-small \
      --dtype float16 \
      --device "${DEVICE}" \
      --layer-index 0 \
      --n-prompts 256 \
      --k-noise 4 \
      --sigma 0.05 \
      --seed "${s}" \
      --out-path "${out_csv}"
  done
fi

if [[ "${RUN_DRY_RUN}" == "1" ]]; then
  echo "[stage1a] Running pilot dry-run summary script..."
  bash scripts/run_pilot_dry_run.sh
fi

echo "[stage1a] Suite complete. Outputs in ${REPORTS_DIR}"
