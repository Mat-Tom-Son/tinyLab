#!/usr/bin/env python3
"""Patch to integrate developmental monitoring into Stage-1B training.

This script demonstrates how to add monitoring to the existing
train_stage1b_grokking.py script without major refactoring.

The monitor will track:
1. VDI snap timing relative to grokking transition
2. Le Chatelier compensation across omega sweep
3. MI saturation boundaries

Usage:
    # Run original script with monitoring enabled
    python scripts/integrate_monitoring_stage1b.py \
        --omega 1.0 \
        --head 0 \
        --seed 0 \
        --monitor
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "lab"))
sys.path.insert(0, str(Path(__file__).parent))

from src.components.developmental_monitor import (
    DevelopmentalMonitor,
    detect_vdi_snap,
    check_saturation_boundary,
)


def patch_transformer_block_for_monitoring(block):
    """Monkey-patch TransformerBlock to cache attention weights."""
    original_forward = block.forward

    def forward_with_cache(x, layer_idx, layer_head_config=None, attn_mask=None):
        # Call original forward
        result = original_forward(x, layer_idx, layer_head_config, attn_mask)

        # Cache attention weights (computed inside forward)
        # This requires extracting them from the forward computation
        # For demonstration, we'll recompute them here (inefficient but safe)
        bsz, seq_len, _ = x.shape
        h = block.ln1(x)
        q = block.W_q(h).view(bsz, seq_len, block.n_heads, block.d_head)
        k = block.W_k(h).view(bsz, seq_len, block.n_heads, block.d_head)

        attn_scores = torch.einsum("bthd,bshd->bhts", q, k) / np.sqrt(block.d_head)
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask[:, :, :seq_len, :seq_len]
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)

        block._cached_attn_weights = attn_weights.detach()

        return result

    block.forward = forward_with_cache


def setup_monitoring_hooks(model, enable=True):
    """Add monitoring infrastructure to model."""
    if not enable:
        return None

    # Add attention cache
    model.attention_cache = [None] * len(model.blocks)

    # Patch each block to cache attention
    for block in model.blocks:
        patch_transformer_block_for_monitoring(block)

    # Add hook to collect cached weights
    def make_collection_hook(layer_idx):
        def hook(module, input, output):
            if hasattr(module, "_cached_attn_weights"):
                model.attention_cache[layer_idx] = module._cached_attn_weights

        return hook

    for layer_idx, block in enumerate(model.blocks):
        block.register_forward_hook(make_collection_hook(layer_idx))

    return model


def create_monitoring_wrapper(
    original_train_fn,
    model,
    target_head=(0, 0),
    monitor_interval=500,
    omega_sweep=None,
):
    """Wrap training function to add monitoring checkpoints.

    Args:
        original_train_fn: The original training function
        model: Transformer model
        target_head: (layer, head) to monitor
        monitor_interval: Steps between checkpoints
        omega_sweep: Omega values for kill testing

    Returns:
        Wrapped training function with monitoring
    """
    if omega_sweep is None:
        omega_sweep = [0.5, 1.0, 1.5]

    monitor = DevelopmentalMonitor(
        model=model,
        target_head=target_head,
        omega_sweep=omega_sweep,
        kill_test_frequency=2,
    )

    def wrapped_train_fn(*args, **kwargs):
        # Extract step counter and data from original function
        # This is a simplified example - adjust based on actual signature

        result = original_train_fn(*args, **kwargs)

        # Add monitoring hook here if needed
        return result, monitor

    return wrapped_train_fn, monitor


def main():
    """Demonstration of how to integrate monitoring."""
    parser = argparse.ArgumentParser(description="Integrate monitoring into Stage-1B")
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--monitor", action="store_true", help="Enable monitoring")
    parser.add_argument(
        "--monitor-interval", type=int, default=500, help="Steps between checkpoints"
    )
    args = parser.parse_args()

    print("\nStage-1B Integration Example")
    print("=" * 70)

    if not args.monitor:
        print("Monitoring disabled. Run with --monitor to enable.")
        print(
            "This would run the original train_stage1b_grokking.py without changes."
        )
        return

    print(f"Monitoring enabled:")
    print(f"  Target head: (0, {args.head})")
    print(f"  Omega: {args.omega}")
    print(f"  Monitor interval: {args.monitor_interval}")
    print()

    # Demonstration of the integration pattern:
    print("Integration Pattern:")
    print(
        """
    1. Import monitoring components:
       from lab.src.components.developmental_monitor import DevelopmentalMonitor

    2. Setup model with attention caching:
       setup_monitoring_hooks(model, enable=True)

    3. Initialize monitor:
       monitor = DevelopmentalMonitor(
           model=model,
           target_head=(0, args.head),
           omega_sweep=[0.5, 0.7, 1.0, 1.3, 1.5],
           kill_test_frequency=2
       )

    4. In training loop, add monitoring checkpoints:
       if step % monitor_interval == 0:
           # Get Layer 0 activations
           layer0_acts = model.blocks[0](embeddings, layer_idx=0)

           # Record checkpoint
           checkpoint = monitor.record_checkpoint(
               step=step,
               data_batch=input_ids,
               layer0_activations=layer0_acts
           )

           # Check for snap
           if not monitor.snap_detected:
               snap = detect_vdi_snap(monitor.vdi_history)
               if snap.snap_detected:
                   print(f"VDI SNAP at step {snap.snap_step}!")

           # Report kill test results
           if checkpoint.kill_test_result:
               kt = checkpoint.kill_test_result
               print(f"Le Chatelier: {kt.le_chatelier_detected}")
               print(f"Compensation: {kt.compensation_score:.4f}")

    5. After training, save trajectory:
       monitor.save_trajectory(output_dir / "developmental_trajectory.json")
    """
    )

    print("\nFor a complete working example, see:")
    print("  scripts/train_with_developmental_monitoring.py")
    print()
    print("To visualize results:")
    print("  python scripts/visualize_developmental_trajectory.py \\")
    print("      reports/developmental_monitoring/.../developmental_trajectory.json")
    print()


if __name__ == "__main__":
    main()
