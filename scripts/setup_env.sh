#!/bin/bash
set -e

echo "=== Tiny Ablation Lab Setup ==="

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip wheel setuptools

# PyTorch (MPS ships with the CPU wheel on macOS)
echo "Installing PyTorch..."
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cpu

# Core libs
echo "Installing core dependencies..."
pip install \
  transformer-lens==2.16.1 \
  transformers==4.57.1 \
  datasets==4.3.0 \
  mlflow==3.5.1 \
  umap-learn==0.5.9.post2 \
  plotly==6.3.1 \
  kaleido==1.1.0 \
  matplotlib==3.10.7 \
  pandas==2.3.3 \
  numpy==2.3.3 \
  psutil==7.1.2 \
  orjson==3.11.4 \
  rich==14.2.0 \
  pyarrow==21.0.0 \
  pydantic==2.12.3 \
  tqdm==4.67.1

# SAELens (pin to a stable commit or use main)
echo "Installing SAELens..."
pip install sae-lens==6.20.1

# Run validation
echo ""
echo "=== Validation ==="
python - <<'PY'
import torch, platform, sys
print("--- Setup Validation ---")
print("MPS available:", torch.backends.mps.is_available())
print(f"macOS: {platform.mac_ver()[0]}, Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print("------------------------")
PY

echo ""
echo "Setup complete! Activate with: source .venv/bin/activate"
