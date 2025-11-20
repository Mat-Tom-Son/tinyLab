#!/usr/bin/env bash
set -euo pipefail

INTERVAL="${INTERVAL:-30}"
OUT_FILE="${OUT_FILE:-logs/gpu_usage.csv}"
SAMPLES="${SAMPLES:-0}" # 0 = infinite

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required to log GPU usage."
  exit 1
fi

mkdir -p "$(dirname "${OUT_FILE}")"
echo "timestamp,utilization.gpu (%),memory.used (MiB),memory.total (MiB),power.draw (W)" > "${OUT_FILE}"

echo "Logging GPU usage to ${OUT_FILE} every ${INTERVAL}s (samples=${SAMPLES:-infinite})"

count=0
while true; do
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv,noheader,nounits >> "${OUT_FILE}"
  count=$((count + 1))
  if [[ "${SAMPLES}" -gt 0 && "${count}" -ge "${SAMPLES}" ]]; then
    echo "Reached requested sample count (${SAMPLES}); exiting."
    exit 0
  fi
  sleep "${INTERVAL}"
done
