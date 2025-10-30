#!/usr/bin/env python3
"""Quick smoke test to validate the lab setup."""
import sys
import torch
from transformer_lens import HookedTransformer
from pathlib import Path

print("=== Tiny Ablation Lab Smoke Test ===\n")

# 1. Check MPS
print("1. Checking MPS availability...")
if torch.backends.mps.is_available():
    print("   ✓ MPS available")
    device = "mps"
else:
    print("   ⚠ MPS not available, using CPU")
    device = "cpu"

# 2. Load a tiny model
print("\n2. Loading GPT-2 small...")
try:
    model = HookedTransformer.from_pretrained(
        "gpt2-small", device=device, dtype=torch.float32
    )
    print(f"   ✓ Model loaded ({model.cfg.n_layers} layers, {model.cfg.n_heads} heads)")
except Exception as e:
    print(f"   ✗ Failed to load model: {e}")
    sys.exit(1)

# 3. Test forward pass
print("\n3. Testing forward pass...")
try:
    text = "The capital of France is"
    tokens = model.to_tokens(text)
    logits = model(tokens)
    print(f"   ✓ Forward pass successful (output shape: {logits.shape})")
except Exception as e:
    print(f"   ✗ Forward pass failed: {e}")
    sys.exit(1)

# 4. Test hooks
print("\n4. Testing activation hooks...")
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
except Exception as e:
    print(f"   ✗ Hooks failed: {e}")
    sys.exit(1)

# 5. Test data loading
print("\n5. Testing data loading...")
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

# 6. Test metrics
print("\n6. Testing metrics computation...")
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

print("\n=== ✓ All checks passed! ===")
print("\nNext steps:")
print("  1. Ensure sample data exists: python3 scripts/facts_make_split.py facts_v1")
print("  2. Run an experiment: python3 -m lab.src.harness lab/configs/run_h2_layer_geom_c2x.json")
print("  3. View results: mlflow ui --backend-store-uri file://$(pwd)/mlruns")
