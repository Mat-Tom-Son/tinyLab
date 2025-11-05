# CUDA Setup Guide for Tiny Ablation Lab

This guide helps you run the Tiny Ablation Lab on NVIDIA GPUs with CUDA support. The original codebase was designed for Apple Silicon (MPS), but this setup enables full compatibility with NVIDIA hardware.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Validation](#validation)
4. [Memory Considerations](#memory-considerations)
5. [Running Experiments](#running-experiments)
6. [Troubleshooting](#troubleshooting)
7. [Performance Tips](#performance-tips)

## Prerequisites

### Hardware Requirements

- NVIDIA GPU with CUDA support (Compute Capability 3.5+)
- Recommended: 12GB+ VRAM for GPT-2 Medium experiments
- For Mistral-7B: 16GB+ VRAM recommended

### Software Requirements

- Linux (Ubuntu 20.04+ recommended) or Windows with WSL2
- NVIDIA drivers (version 520.61.05+ for CUDA 12.1)
- Python 3.10+
- ~10GB disk space (for models + dependencies)

### Check Your GPU

```bash
# Verify NVIDIA driver installation
nvidia-smi

# Check CUDA version
nvidia-smi | grep "CUDA Version"
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA RTX 3090    Off  | 00000000:01:00.0  On |                  N/A |
| 30%   45C    P8    25W / 350W |    512MiB / 24576MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

## Installation

### 1. Clone and Setup Environment

```bash
git clone https://github.com/Mat-Tom-Son/tinyLab.git
cd tinyLab

# Run CUDA-specific setup script
bash scripts/setup_env_cuda.sh
```

The setup script will:
- Create a Python virtual environment
- Install PyTorch with CUDA 12.1 support
- Install all required dependencies (TransformerLens, transformers, etc.)
- Validate CUDA availability

**Note**: If you have a different CUDA version installed, edit `scripts/setup_env_cuda.sh` and change the PyTorch installation line:

```bash
# For CUDA 11.8
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu118

# For CPU-only (slow, not recommended)
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cpu
```

### 2. Activate Environment

```bash
source .venv/bin/activate
```

## Validation

### Run CUDA Smoke Test

```bash
python smoke_test_cuda.py
```

Expected output:
```
=== Tiny Ablation Lab CUDA Smoke Test ===

1. Checking CUDA availability...
   ✓ CUDA available
   CUDA version: 12.1
   cuDNN version: 8902
   Number of GPUs: 1
   GPU 0: NVIDIA GeForce RTX 3090 (24.00 GB)

2. CUDA memory baseline...
   Total: 24.00 GB
   Allocated: 0.00 GB
   Reserved: 0.00 GB
   Free: 24.00 GB

3. Loading GPT-2 small...
   Using dtype: torch.float16
   ✓ Model loaded (12 layers, 12 heads)
   CUDA memory after load: 0.23 GB allocated, 0.25 GB reserved

...

=== ✓ All checks passed! ===
```

### Verify Reproducibility Data

The original study data should be present:

```bash
# Check corpus files
ls -lh lab/data/corpora/

# Check split files
ls -lh lab/data/splits/

# If missing, the experiments will still run but won't exactly replicate paper results
```

## Memory Considerations

### VRAM Requirements by Model

| Model | Parameters | float32 | float16 | bfloat16 |
|-------|-----------|---------|---------|----------|
| GPT-2 Small | 124M | ~2GB | ~1GB | ~1GB |
| GPT-2 Medium | 355M | ~3GB | ~1.5GB | ~1.5GB |
| GPT-2 Large | 774M | ~6GB | ~3GB | ~3GB |
| Mistral-7B | 7.2B | ~28GB | ~14GB | ~14GB |

**Note**: These are base model sizes. Activations, gradients, and batch processing add overhead.

### Memory Budget for Experiments

For **GPT-2 Medium** experiments (recommended starting point):

| VRAM | Recommended Settings |
|------|---------------------|
| 8GB  | float16, batch_size=2, single seed |
| 12GB | float16, batch_size=4, multi-seed OK |
| 16GB+ | float16, batch_size=8, full experiments |

For **Mistral-7B** experiments:

| VRAM | Recommended Settings |
|------|---------------------|
| 12GB | Not recommended (use GPT-2 Medium) |
| 16GB | float16, batch_size=1, low_memory=true |
| 24GB+ | float16, batch_size=2-4, comfortable |

### Optimization Flags

The updated `load_model.py` supports a `low_memory` flag:

```json
{
  "model": {
    "name": "gpt2-medium",
    "dtype": "float16",
    "low_memory": true
  }
}
```

This enables:
- CUDA cache clearing before model load
- TF32 acceleration on Ampere+ GPUs (RTX 30xx, A100, etc.)
- Memory usage monitoring and reporting

## Running Experiments

### Option 1: Use Pre-Made CUDA Configs

We've created CUDA-optimized configs:

```bash
# Standard CUDA config (12-16GB VRAM)
python -m lab.src.harness lab/configs/run_h1_cross_condition_balanced_cuda.json

# Low-memory config (8-12GB VRAM)
python -m lab.src.harness lab/configs/run_h1_cross_condition_balanced_cuda_low_mem.json
```

### Option 2: Convert Existing Configs

Convert any MPS config to CUDA:

```bash
# Convert a single config
python scripts/convert_configs_to_cuda.py \
  lab/configs/run_h1_cross_condition_balanced.json

# This creates: run_h1_cross_condition_balanced_cuda.json

# Convert with low-memory optimizations
python scripts/convert_configs_to_cuda.py \
  lab/configs/run_h1_cross_condition_balanced.json \
  --low-memory \
  --batch-size 4

# Dry-run to preview changes
python scripts/convert_configs_to_cuda.py \
  lab/configs/*.json \
  --dry-run
```

### Option 3: Manual Config Editing

Edit any config and change:

```json
{
  "shared": {
    "device": "cuda",        // Change from "mps" to "cuda"
    "model": {
      "name": "gpt2-medium",
      "dtype": "float16",    // Use float16 instead of float32
      "low_memory": true     // Optional: enable for limited VRAM
    },
    "batch_size": 4          // Reduce if OOM errors occur
  }
}
```

### Monitor GPU Usage

```bash
# Watch GPU memory in real-time
watch -n 1 nvidia-smi

# Or use this more detailed view
watch -n 1 'nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free --format=csv'
```

## Reproducing Paper Results

The paper experiments were run on Apple Silicon. To validate findings on CUDA:

### H1: Layer-0 Suppressor Discovery (GPT-2 Medium)

```bash
# Full experiment (12+ GB VRAM)
python -m lab.src.harness lab/configs/run_h1_cross_condition_balanced_cuda.json

# Expected runtime: ~20-30 minutes on RTX 3090
# Memory usage: ~4-6 GB
```

### H5: Pair Ablations

```bash
python -m lab.src.harness lab/configs/run_h5_layer0_triplet_balanced.json
# (Update device to "cuda" first)
```

### Geometric Signature Analysis

```bash
# Run activation entropy analysis
python -m lab.analysis.activation_entropy \
  --config lab/configs/run_h1_cross_condition_balanced_cuda.json \
  --tag facts \
  --device cuda \
  --samples 64 \
  --heads 2 4 7 \
  --random-samples 50 \
  --entropy-methods subspace,diagonal,per_token \
  --output reports/activation_entropy_gpt2medium_facts_cuda.json

# Repeat for: --tag cf, neg, logic
```

Expected results should closely match paper (differences <5% are normal due to:
- Hardware differences (CUDA vs MPS)
- Numerical precision variations
- Random seed initialization)

## Troubleshooting

### "CUDA out of memory"

**Solutions**:
1. Reduce batch size: `"batch_size": 2` or `"batch_size": 1`
2. Use float16: `"dtype": "float16"`
3. Enable low-memory mode: `"low_memory": true`
4. Use a smaller model: `"name": "gpt2-small"`
5. Reduce sample size in analysis scripts: `--samples 32`

```bash
# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache()"
```

### "CUDA not available" after installation

```bash
# Check PyTorch CUDA installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu121
```

### Slow performance on CUDA

1. **Check GPU utilization**: `nvidia-smi` should show >80% GPU util during forward passes
2. **Verify float16**: Using float32 on GPU is slower
3. **Check batch size**: Too small batches underutilize GPU (try batch_size=8-16)
4. **Disable debugging**: Remove any `torch.autograd.set_detect_anomaly(True)`

### Model download fails

```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/large/disk

# Use HF mirror if needed
export HF_ENDPOINT=https://hf-mirror.com

# Download model manually first
python -c "from transformers import GPT2LMHeadModel; GPT2LMHeadModel.from_pretrained('gpt2-medium')"
```

### Different results from paper

Small numerical differences are expected when switching hardware (MPS → CUDA). To minimize:
1. Use the same dtype as paper: `"dtype": "float16"`
2. Verify same random seeds: `"seeds": [0, 1, 2]`
3. Check data hashes match: Look at `data_hash.txt` in run directories

If differences are >10%, there may be a configuration issue.

## Performance Tips

### 1. Use Mixed Precision (float16)

```json
{"dtype": "float16"}  // 2x faster, 1/2 memory
```

### 2. Optimize Batch Size

```bash
# Find optimal batch size for your GPU
for bs in 2 4 8 16; do
  echo "Testing batch_size=$bs"
  # Edit config, run quick test, measure time
done
```

General guidelines:
- **8-12 GB VRAM**: batch_size=4
- **16 GB VRAM**: batch_size=8
- **24 GB+ VRAM**: batch_size=16

### 3. Enable TF32 (Ampere+ GPUs)

Automatically enabled with `"low_memory": true` on RTX 30xx/40xx, A100, H100.

### 4. Pin Memory for DataLoaders

If you extend the codebase with custom data loaders:

```python
DataLoader(..., pin_memory=True, num_workers=4)
```

### 5. Use Gradient Checkpointing (for training/SAE work)

```python
model.gradient_checkpointing_enable()
```

### 6. Profile Your Code

```bash
# Use PyTorch profiler
python -m torch.utils.bottleneck your_script.py

# Or NVIDIA Nsight Systems
nsys profile -o profile python your_script.py
```

## Benchmarks

Approximate runtimes for key experiments on different GPUs:

| Experiment | GPU | VRAM Used | Runtime |
|-----------|-----|-----------|---------|
| H1 Cross-Condition (GPT-2 Med, 3 seeds) | RTX 3090 | ~5 GB | ~25 min |
| H1 Cross-Condition (GPT-2 Med, 3 seeds) | RTX 4090 | ~5 GB | ~18 min |
| H1 Cross-Condition (GPT-2 Med, 3 seeds) | A100 40GB | ~5 GB | ~15 min |
| Geometric Signature (64 samples) | RTX 3090 | ~4 GB | ~8 min |
| H5 Triplet Ablation | RTX 3090 | ~5 GB | ~12 min |

## Multi-GPU Support

The codebase currently runs on a single GPU. For multi-GPU setups:

```bash
# Specify GPU
CUDA_VISIBLE_DEVICES=0 python -m lab.src.harness config.json

# Or in config
{"device": "cuda:1"}  // Use second GPU
```

## Comparing MPS vs CUDA Results

To verify numerical consistency:

```bash
# Run same config on both platforms
# MPS (on Mac):
python -m lab.src.harness lab/configs/run_h1_cross_condition_balanced.json

# CUDA (on Linux):
python -m lab.src.harness lab/configs/run_h1_cross_condition_balanced_cuda.json

# Compare summary metrics
diff lab/runs/h1_*/metrics/summary.json
```

Expected: ΔLD differences <0.05, calibration differences <0.01

## Contributing CUDA Improvements

If you find optimizations or fixes for CUDA:
1. Test on multiple GPU types if possible
2. Verify reproducibility with paper results
3. Update this doc with your findings
4. Submit a PR!

## References

- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [TransformerLens Documentation](https://transformerlensorg.github.io/TransformerLens/)
- [Original Paper Results](paper/main.pdf)

## Support

For CUDA-specific issues:
- Check [GitHub Issues](https://github.com/Mat-Tom-Son/tinyLab/issues)
- Tag issues with `cuda` label
- Include `nvidia-smi` output and error messages

---

**Happy experimenting on CUDA!** 🚀
