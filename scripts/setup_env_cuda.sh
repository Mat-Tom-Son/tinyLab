#!/usr/bin/env bash
set -euo pipefail

echo "=== Tiny Ablation Lab Setup (Debian 12 + CUDA) ==="

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Default CUDA-compatible PyTorch versions for cu121 wheels
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.5.1}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_DVC="${INSTALL_DVC:-1}"
RESET_VENV="${RESET_VENV:-1}"

SUDO_BIN="$(command -v sudo || true)"

if command -v apt-get >/dev/null 2>&1; then
  echo "Installing system dependencies with apt..."
  ${SUDO_BIN:-} apt-get update -y
  ${SUDO_BIN:-} apt-get install -y --no-install-recommends \
    build-essential git wget curl ca-certificates \
    python3 python3-venv python3-pip python3-dev \
    pkg-config libssl-dev
fi

echo ""
echo "Checking CUDA visibility..."
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "✓ NVIDIA drivers detected"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
  echo "⚠️  nvidia-smi not found. GPU/driver may not be available."
fi

if command -v nvcc >/dev/null 2>&1; then
  echo "✓ nvcc detected: $(nvcc --version | grep release || true)"
else
  echo "⚠️  nvcc not found; CUDA toolkit may be missing from PATH"
fi

echo ""
echo "Creating virtual environment at ${VENV_DIR}..."
if [[ "${RESET_VENV}" == "1" && -d "${VENV_DIR}" ]]; then
  echo "Resetting existing virtual environment at ${VENV_DIR}..."
  rm -rf "${VENV_DIR}"
fi
if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
VENV_PY="${VENV_DIR}/bin/python"
if [[ ! -x "${VENV_PY}" ]]; then
  echo "Virtualenv missing or broken at ${VENV_DIR}; ensure python3-venv is installed."
  exit 1
fi
source "${VENV_DIR}/bin/activate"

"${VENV_PY}" -m pip install --upgrade pip wheel setuptools

echo ""
echo "Installing PyTorch (CUDA preferred)..."
if command -v nvidia-smi >/dev/null 2>&1; then
  TARGET_INDEX="${TORCH_INDEX}"
  echo "Using CUDA wheel index: ${TARGET_INDEX}"
else
  TARGET_INDEX="https://download.pytorch.org/whl/cpu"
  echo "GPU not detected; using CPU wheel index: ${TARGET_INDEX}"
fi

"${VENV_PY}" -m pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "${TARGET_INDEX}"

echo ""
echo "Installing core dependencies..."
"${VENV_PY}" -m pip install \
  transformer-lens==2.16.1 \
  transformers==4.57.1 \
  datasets==4.3.0 \
  mlflow==3.5.1 \
  umap-learn==0.5.9.post2 \
  plotly==6.3.1 \
  kaleido==1.1.0 \
  matplotlib==3.10.7 \
  pandas==2.3.3 \
  numpy==1.26.4 \
  psutil==7.1.2 \
  orjson==3.11.4 \
  rich==14.2.0 \
  pyarrow==21.0.0 \
  pydantic==2.12.3 \
  tqdm==4.67.1

echo ""
echo "Installing SAELens..."
"${VENV_PY}" -m pip install sae-lens==6.20.1

if [[ "${INSTALL_DVC}" == "1" ]]; then
  echo ""
  echo "Installing DVC with GCS support..."
  "${VENV_PY}" -m pip install "dvc[gcs]>=3.51,<4"
fi

echo ""
echo "Installing tiny-ablation-lab package for imports..."
"${VENV_PY}" -m pip install -e .

echo ""
echo "=== Validation ==="
${VENV_PY} - <<'PY'
import platform
import sys
import torch

print("--- Setup Validation ---")
print(f"Platform: {platform.system()} {platform.release()}")
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / (1024**3)
        print(f"  GPU {i}: {props.name}")
        print(f"    Compute capability: {props.major}.{props.minor}")
        print(f"    Total memory: {mem_gb:.2f} GB")
        print(f"    Multi-processors: {props.multi_processor_count}")

    try:
        x = torch.randn(1024, 1024, device="cuda")
        _ = x @ x.T
        print("✓ Basic CUDA tensor operations work")
    except Exception as e:
        print(f"✗ CUDA tensor test failed: {e}")
else:
    print("⚠️  CUDA not available. Using CPU wheels.")
print("------------------------")
PY

echo ""
echo "Setup complete! Activate with: source ${VENV_DIR}/bin/activate"
echo "Next steps:"
echo "  1. Run CUDA smoke test: python smoke_test_cuda.py"
echo "  2. Run pilot dry run: bash scripts/run_pilot_dry_run.sh"
echo "  3. For limited VRAM, use float16 and smaller batch sizes"
