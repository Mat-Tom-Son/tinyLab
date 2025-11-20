#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Mat-Tom-Son/tinyLab.git}"
BRANCH="${BRANCH:-main}"
TARGET_DIR="${TARGET_DIR:-tinyLab}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_DRY_RUN="${RUN_DRY_RUN:-1}"
SKIP_DVC="${SKIP_DVC:-0}"

echo "=== Tiny Ablation Lab: Linux CUDA installer ==="
echo "Repo: ${REPO_URL}"
echo "Branch: ${BRANCH}"
echo "Target directory: ${TARGET_DIR}"

SUDO_BIN="$(command -v sudo || true)"

if command -v apt-get >/dev/null 2>&1; then
  echo "Installing base tools (git, python, build-essential)..."
  ${SUDO_BIN:-} apt-get update -y
  ${SUDO_BIN:-} apt-get install -y --no-install-recommends \
    git python3 python3-venv python3-pip python3-dev \
    build-essential ca-certificates wget curl pkg-config libssl-dev
fi

if [[ ! -d "${TARGET_DIR}/.git" ]]; then
  echo "Cloning repository..."
  git clone --branch "${BRANCH}" --depth 1 "${REPO_URL}" "${TARGET_DIR}"
else
  echo "Existing checkout detected at ${TARGET_DIR}; skipping clone."
fi

cd "${TARGET_DIR}"

echo ""
echo "Running CUDA environment setup..."
bash scripts/setup_env_cuda.sh

echo ""
echo "Activating environment..."
source .venv/bin/activate

if [[ "${SKIP_DVC}" != "1" ]]; then
  echo ""
  echo "Pulling DVC-tracked data (optional)..."
  if ! dvc pull; then
    echo "⚠️  DVC pull failed or remote not configured. Set SKIP_DVC=1 to skip."
  fi
else
  echo "Skipping DVC pull (SKIP_DVC=1)."
fi

if [[ "${RUN_SMOKE}" == "1" ]]; then
  echo ""
  echo "Running CUDA smoke test..."
  python smoke_test_cuda.py || { echo "Smoke test failed"; exit 1; }
else
  echo "Skipping smoke test (RUN_SMOKE=0)."
fi

if [[ "${RUN_DRY_RUN}" == "1" ]]; then
  echo ""
  echo "Running Stage-1A pilot dry run..."
  bash scripts/run_pilot_dry_run.sh || { echo "Pilot dry run failed"; exit 1; }
else
  echo "Skipping pilot dry run (RUN_DRY_RUN=0)."
fi

cat <<'EOF'

All set! Common next steps:
  - Activate: source .venv/bin/activate
  - Launch your command, e.g.: bash scripts/run_pilot_dry_run.sh
  - Monitor GPU: watch -n 5 nvidia-smi
EOF
