#!/usr/bin/env bash
set -euo pipefail

steps=(0 100 500 1000 2000 4000 8000 16000 32000 64000 128000 256000 282956)

for step in "${steps[@]}"; do
  cfg="lab/configs/run_h1_pythia_checkpoint_$(printf "%06d" "$step").json"
  echo "[info] Running $cfg"
  python3 -m lab.src.orchestrators.conditions "$cfg"
done

echo "[info] Aggregating head trajectories"
python3 scripts/analyze_pythia_learning.py

