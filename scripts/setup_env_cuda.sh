#!/bin/bash
set -e

echo "=== Tiny Ablation Lab Setup (CUDA/NVIDIA) ==="

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  WARNING: nvidia-smi not found. Please install NVIDIA drivers first."
    echo "   Visit: https://www.nvidia.com/Download/index.aspx"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ NVIDIA drivers detected"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip wheel setuptools

# PyTorch with CUDA support
echo ""
echo "Installing PyTorch with CUDA 12.1 support..."
echo "Note: Adjust CUDA version if needed based on your system"
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu121

# Core libs (same versions as Apple Silicon setup for reproducibility)
echo ""
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
echo ""
echo "Installing SAELens..."
pip install sae-lens==6.20.1

# Run validation
echo ""
echo "=== Validation ==="
python - <<'PY'
import torch
import platform
import sys

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

    # Test tensor allocation
    try:
        x = torch.randn(1000, 1000, device='cuda')
        y = x @ x.T
        print("✓ Basic CUDA tensor operations work")
    except Exception as e:
        print(f"✗ CUDA tensor test failed: {e}")
else:
    print("⚠️  CUDA not available. Will use CPU.")
    print("   Check NVIDIA drivers and PyTorch CUDA installation.")
print("------------------------")
PY

echo ""
echo "Setup complete! Activate with: source .venv/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Run CUDA smoke test: python smoke_test_cuda.py"
echo "  2. Update configs to use 'cuda' device (see docs/CUDA_SETUP.md)"
echo "  3. For limited VRAM, use float16 and smaller batch sizes"
