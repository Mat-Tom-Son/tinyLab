#!/usr/bin/env bash
set -euo pipefail

# One-command Stage-1A prereg helper:
#   1) (Optional) Run the Stage-1A suite (Task B data, circularity, VDI, dry-run)
#   2) Aggregate VDI across seeds to pick suppressor + random heads
#   3) (Optional) Launch training jobs via a user-specified template
#
# Environment toggles:
#   SUITE_FIRST=1              # run the suite first (set 0 to skip)
#   SEEDS="0 1"                # seeds for VDI/circularity
#   TRAIN_SEEDS="0 1"          # seeds for training jobs
#   SUPPRESSOR_OMEGAS="0.5 1.5"
#   RANDOM_OMEGAS="0.5 1.5"
#   REPORTS_DIR="reports/pilot_stage1a/suite"
#   HEAD_SELECTION_JSON="reports/pilot_stage1a/head_selection.json"
#   DEVICE="auto"              # passed through to the suite
#   PYBIN="python3"            # python executable
#   RUN_BASELINE=1             # training toggles
#   RUN_SUPPRESSOR=1
#   RUN_RANDOM=1
#   TRAIN_CMD_TEMPLATE=""      # e.g. 'python train_stage1a.py --cond {cond} --seed {seed} --omega {omega} --head {head} --head-kind {head_kind}'
#
# Placeholders for TRAIN_CMD_TEMPLATE:
#   {cond}        -> baseline | suppressor | random
#   {seed}        -> seed value
#   {omega}       -> scaling factor (1.0 for baseline)
#   {head}        -> selected head index (suppressor or random)
#   {head_kind}   -> suppressor | random | baseline

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SUITE_FIRST="${SUITE_FIRST:-1}"
SEEDS=(${SEEDS:-0 1})
TRAIN_SEEDS=(${TRAIN_SEEDS:-0 1})
SUPPRESSOR_OMEGAS=(${SUPPRESSOR_OMEGAS:-0.5 1.5})
RANDOM_OMEGAS=(${RANDOM_OMEGAS:-0.5 1.5})
REPORTS_DIR="${REPORTS_DIR:-reports/pilot_stage1a/suite}"
HEAD_SELECTION_JSON="${HEAD_SELECTION_JSON:-reports/pilot_stage1a/head_selection.json}"
DEVICE="${DEVICE:-auto}"
PYBIN="${PYBIN:-python3}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_SUPPRESSOR="${RUN_SUPPRESSOR:-1}"
RUN_RANDOM="${RUN_RANDOM:-1}"
TRAIN_CMD_TEMPLATE="${TRAIN_CMD_TEMPLATE:-}"
SEEDS_STR="${SEEDS[*]}"

echo "[stage1a-prereg] ROOT=${ROOT_DIR}"

if [[ "${SUITE_FIRST}" == "1" ]]; then
  echo "[stage1a-prereg] Running Stage-1A suite (VDI + circularity)..."
  SEEDS="${SEEDS[*]}" DEVICE="${DEVICE}" PYBIN="${PYBIN}" \
    bash scripts/run_stage1a_suite.sh
fi

echo "[stage1a-prereg] Selecting suppressor/random heads from VDI CSVs..."
${PYBIN} - <<PY
import json
from pathlib import Path
import sys

import pandas as pd

reports_dir = Path("${REPORTS_DIR}")
seeds = "${SEEDS_STR}".split()
out_path = Path("${HEAD_SELECTION_JSON}")

paths = []
for s in seeds:
    p = reports_dir / f"vdi_layer0_gpt2small_seed{s}.csv"
    if p.exists():
        paths.append(p)
if not paths:
    print("[stage1a-prereg] No VDI CSVs found; expected vdi_layer0_gpt2small_seed*.csv in", reports_dir)
    sys.exit(1)

dfs = [pd.read_csv(p) for p in paths]
df = pd.concat(dfs, ignore_index=True)

group = df.groupby("head")["vdi_effect"].mean().reset_index()
if group.empty:
    print("[stage1a-prereg] VDI table is empty.")
    sys.exit(1)

suppressor_row = group.sort_values("vdi_effect", ascending=False).iloc[0]
median_val = group["vdi_effect"].median()
non_suppressor = group[group["head"] != suppressor_row["head"]]
if non_suppressor.empty:
    random_row = suppressor_row
else:
    random_row = non_suppressor.iloc[(non_suppressor["vdi_effect"] - median_val).abs().argsort().iloc[0]]

out = {
    "seeds": seeds,
    "source_csv": [str(p) for p in paths],
    "suppressor_head": int(suppressor_row["head"]),
    "suppressor_mean_vdi": float(suppressor_row["vdi_effect"]),
    "random_head": int(random_row["head"]),
    "random_mean_vdi": float(random_row["vdi_effect"]),
}

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(out, indent=2))

print("[stage1a-prereg] Selection:")
print(json.dumps(out, indent=2))
PY

if [[ -z "${TRAIN_CMD_TEMPLATE}" ]]; then
  echo "[stage1a-prereg] TRAIN_CMD_TEMPLATE not set; skipping training launches."
  exit 0
fi

SUPPRESSOR_HEAD=$( ${PYBIN} - <<PY
import json
from pathlib import Path
data = json.loads(Path("${HEAD_SELECTION_JSON}").read_text())
print(data["suppressor_head"])
PY
)

RANDOM_HEAD=$( ${PYBIN} - <<PY
import json
from pathlib import Path
data = json.loads(Path("${HEAD_SELECTION_JSON}").read_text())
print(data["random_head"])
PY
)

run_training() {
  local cond="$1"
  local seed="$2"
  local omega="$3"
  local head="$4"
  local head_kind="$5"

  local cmd
  cmd=$("${PYBIN}" - "$cond" "$seed" "$omega" "$head" "$head_kind" <<'PY'
import os, sys
tpl = os.environ["TRAIN_CMD_TEMPLATE"]
cond, seed, omega, head, head_kind = sys.argv[1:6]
print(tpl.format(cond=cond, seed=seed, omega=omega, head=head, head_kind=head_kind))
PY
)

  echo "[stage1a-prereg] Running: ${cmd}"
  bash -lc "${cmd}"
}

if [[ "${RUN_BASELINE}" == "1" ]]; then
  for s in "${TRAIN_SEEDS[@]}"; do
    run_training "baseline" "${s}" "1.0" "${SUPPRESSOR_HEAD}" "baseline"
  done
fi

if [[ "${RUN_SUPPRESSOR}" == "1" ]]; then
  for s in "${TRAIN_SEEDS[@]}"; do
    for omega in "${SUPPRESSOR_OMEGAS[@]}"; do
      run_training "suppressor" "${s}" "${omega}" "${SUPPRESSOR_HEAD}" "suppressor"
    done
  done
fi

if [[ "${RUN_RANDOM}" == "1" ]]; then
  for s in "${TRAIN_SEEDS[@]}"; do
    for omega in "${RANDOM_OMEGAS[@]}"; do
      run_training "random" "${s}" "${omega}" "${RANDOM_HEAD}" "random"
    done
  done
fi

echo "[stage1a-prereg] All requested training jobs launched."
