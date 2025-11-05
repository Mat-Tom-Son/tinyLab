#!/usr/bin/env python3
"""CUDA-specific smoke test to validate the lab setup on NVIDIA GPUs."""
import sys
import torch
from transformer_lens import HookedTransformer
from pathlib import Path

print("=== Tiny Ablation Lab CUDA Smoke Test ===\n")

# 1. Check CUDA availability
print("1. Checking CUDA availability...")
if torch.cuda.is_available():
    print("   ✓ CUDA available")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   cuDNN version: {torch.backends.cudnn.version()}")
    print(f"   Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / (1024**3)
        print(f"   GPU {i}: {props.name} ({mem_gb:.2f} GB)")
    device = "cuda"
else:
    print("   ✗ CUDA not available")
    print("   Falling back to CPU (will be slow)")
    device = "cpu"

# 2. Memory baseline
if device == "cuda":
    print("\n2. CUDA memory baseline...")
    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    reserved = torch.cuda.memory_reserved(0) / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"   Total: {total:.2f} GB")
    print(f"   Allocated: {allocated:.2f} GB")
    print(f"   Reserved: {reserved:.2f} GB")
    print(f"   Free: {total - reserved:.2f} GB")

# 3. Load a tiny model
print("\n3. Loading GPT-2 small...")
try:
    # Use float16 for CUDA to save memory
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"   Using dtype: {dtype}")

    model = HookedTransformer.from_pretrained(
        "gpt2-small", device=device, dtype=dtype
    )
    print(f"   ✓ Model loaded ({model.cfg.n_layers} layers, {model.cfg.n_heads} heads)")

    # Memory after model load
    if device == "cuda":
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"   CUDA memory after load: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
except Exception as e:
    print(f"   ✗ Failed to load model: {e}")
    sys.exit(1)

# 4. Test forward pass
print("\n4. Testing forward pass...")
try:
    text = "The capital of France is"
    tokens = model.to_tokens(text)
    logits = model(tokens)
    print(f"   ✓ Forward pass successful (output shape: {logits.shape})")

    # Memory after forward
    if device == "cuda":
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"   CUDA memory after forward: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
except Exception as e:
    print(f"   ✗ Forward pass failed: {e}")
    sys.exit(1)

# 5. Test hooks
print("\n5. Testing activation hooks...")
try:
    cache = {}

    def hook_fn(act, hook):
        cache[hook.name] = act.detach()
        return act

    with torch.no_grad():
        model.run_with_hooks(
            tokens, fwd_hooks=[(f"blocks.{model.cfg.n_layers-1}.hook_resid_post", hook_fn)]
        )
    print(f"   ✓ Hooks work (captured {len(cache)} activations)")

    # Memory after hooks
    if device == "cuda":
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"   CUDA memory after hooks: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
except Exception as e:
    print(f"   ✗ Hooks failed: {e}")
    sys.exit(1)

# 6. Test data loading
print("\n6. Testing data loading...")
try:
    from lab.src.components import datasets

    if Path("lab/data/splits/facts_v1.split.json").exists():
        dset, split_info, data_hash = datasets.load_split(
            {"id": "facts_v1", "split": "test"}
        )
        print(f"   ✓ Loaded {len(dset)} examples (hash: {data_hash[:12]}...)")
    else:
        print("   ⚠ Sample data not found (run: python3 scripts/facts_make_split.py facts_v1)")
except Exception as e:
    print(f"   ✗ Data loading failed: {e}")

# 7. Test metrics
print("\n7. Testing metrics computation...")
try:
    from lab.src.components import metrics as M

    # Create dummy logits
    batch_size = 2
    seq_len = 5
    vocab_size = model.cfg.d_vocab
    clean_logits = torch.randn(batch_size, seq_len, vocab_size)
    ablated_logits = clean_logits + torch.randn_like(clean_logits) * 0.1

    t_ids = torch.tensor([100, 200])
    f_ids = torch.tensor([300, 400])

    ld = M.logit_diff_first_token(ablated_logits, t_ids, f_ids)
    print(f"   ✓ Metrics work (logit_diff: {ld:.3f})")
except Exception as e:
    print(f"   ✗ Metrics failed: {e}")
    sys.exit(1)

# 8. Clean up and final memory check
print("\n8. Cleanup and final memory check...")
try:
    del model
    del logits
    del cache
    if device == "cuda":
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"   CUDA memory after cleanup: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        print("   ✓ Memory cleanup successful")
except Exception as e:
    print(f"   ⚠ Cleanup warning: {e}")

print("\n=== ✓ All checks passed! ===")
print("\nYour CUDA setup is ready for experiments!")
print("\nRecommendations for your VRAM:")
if device == "cuda":
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"  - Total VRAM: {total_mem:.2f} GB")
    if total_mem < 16:
        print(f"  - ⚠ Limited VRAM detected. Use these settings:")
        print(f"    • GPT-2 Medium: float16, batch_size=4-8")
        print(f"    • Mistral-7B: float16, batch_size=1-2, gradient checkpointing if training")
        print(f"    • Enable 'low_memory': true in model config")
    else:
        print(f"  - ✓ Sufficient VRAM for most experiments")
        print(f"    • GPT-2 Medium: float16, batch_size=8-16")
        print(f"    • Mistral-7B: float16, batch_size=2-4")

print("\nNext steps:")
print("  1. Update configs to use 'cuda' device (or use scripts/convert_configs_to_cuda.py)")
print("  2. Run an experiment: python -m lab.src.harness lab/configs/run_h1_cross_condition_balanced.json")
print("  3. Monitor GPU usage: watch -n 1 nvidia-smi")
