#!/usr/bin/env bash
set -euo pipefail

echo "🧭 Launching Aim UI for tinyLab..."

if [ ! -d ".venv" ]; then
  echo "Error: .venv not found. Run scripts/setup_env.sh first." >&2
  exit 1
fi

source .venv/bin/activate

if ! command -v aim >/dev/null 2>&1; then
  echo "Error: 'aim' CLI not found in current environment." >&2
  echo "Run: bash scripts/setup_aim.sh" >&2
  exit 1
fi

echo "Using Python: $(python --version 2>&1)"
echo "Aim version: $(aim version 2>&1)"
echo "Aim repo: $(python -c 'from aim import Repo; print(Repo(\".\").path)' 2>/dev/null || echo 'not initialized')"

echo "Starting Aim UI at http://localhost:43800 ..."
exec aim up
