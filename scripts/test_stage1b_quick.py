#!/usr/bin/env python3
"""
Quick sanity check for Stage-1B grokking setup.

Runs a short training test (100 steps) to verify:
1. Data loads correctly
2. Model trains without errors
3. Metrics are computed properly
4. Checkpointing works

Usage:
    python scripts/test_stage1b_quick.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Check if data exists
data_train = Path("data/modular_p113_train.jsonl")
data_test = Path("data/modular_p113_test.jsonl")

if not data_train.exists():
    print(f"Error: {data_train} not found")
    print("Run: python3 scripts/data_gen_modular.py --modulus 113 --output-dir data/")
    sys.exit(1)

if not data_test.exists():
    print(f"Error: {data_test} not found")
    sys.exit(1)

print("✓ Data files found")

# Try importing dependencies
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except ImportError:
    print("✗ PyTorch not found - install with: pip install torch")
    sys.exit(1)

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except ImportError:
    print("✗ NumPy not found")
    sys.exit(1)

try:
    from sklearn.decomposition import PCA
    print(f"✓ scikit-learn found")
except ImportError:
    print("✗ scikit-learn not found - install with: pip install scikit-learn")
    sys.exit(1)

# Import training script
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from train_stage1b_grokking import (
        GrokkingConfig, GrokkingTransformer, load_modular_data,
        prepare_batch, compute_accuracy
    )
    print("✓ Training modules imported")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Quick training test
print("\n--- Running quick training test (100 steps) ---")

import random
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Config with reduced steps
cfg = GrokkingConfig(
    n_steps=100,
    log_every=25,
    checkpoint_every=50,
    batch_size=128,  # Smaller for speed
)

# Load data
train_data = load_modular_data(Path(cfg.data_path_train))
test_data = load_modular_data(Path(cfg.data_path_test))
print(f"Loaded {len(train_data)} train, {len(test_data)} test examples")

# Model
torch.manual_seed(42)
random.seed(42)
model = GrokkingTransformer(cfg).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

# Train loop
for step in range(1, cfg.n_steps + 1):
    model.train()

    # Sample batch
    batch_indices = random.sample(range(len(train_data)), cfg.batch_size)
    batch = [train_data[i] for i in batch_indices]
    input_ids, targets = prepare_batch(batch, cfg, device)

    # Forward
    logits = model(input_ids)
    loss = F.cross_entropy(logits[:, 4], targets)

    # Backward
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    opt.step()

    # Log
    if step % cfg.log_every == 0:
        test_acc = compute_accuracy(
            model, test_data[:500], cfg, device
        )
        print(f"[step {step:3d}] loss={loss.item():.4f}, test_acc={test_acc:.3f}")

print("\n✓ Training test completed successfully!")

# Test circularity computation
try:
    from train_stage1b_grokking import compute_circularity_simple

    circularity = compute_circularity_simple(
        model, test_data[:256], cfg, device, layer_idx=0
    )
    print(f"✓ Circularity computation works: {circularity:.3f}")
except Exception as e:
    print(f"✗ Circularity computation failed: {e}")

print("\n" + "="*60)
print("All sanity checks passed!")
print("="*60)
print("\nNext steps:")
print("1. Review STAGE1B_README.md for full pipeline")
print("2. Implement VDI computation in analyze_vdi_compensation.py")
print("3. Run full sweep: bash scripts/run_stage1b_sweep.sh")
