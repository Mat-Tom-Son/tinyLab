#!/usr/bin/env python3
"""
Phase 2: Dual-Timescale Training Experiments

Engineering homeostatic crystallization via dual-timescale training.
Tests 5 experimental conditions to compress crystallization window.

Based on Phase 1 findings:
- VDI equilibrium: 0.611992 (exact across 5 seeds)
- Crystallization window: 3700 ± 400 steps
- Target: Compress window by 20-50%

Usage:
    python scripts/train_phase2_dual_timescale.py \
        --condition dual_timescale \
        --seed 0 \
        --steps 10000 \
        --p 113
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Import from existing codebase
from scripts.train_stage1b_grokking import (
    GrokkingConfig,
    GrokkingTransformer,
    load_modular_data,
    prepare_batch,
)

from lab.src.losses.homeostasis_aware_loss import (
    HomeostasisAwareLoss,
    EXPERIMENTAL_CONDITIONS,
)

from lab.src.wrappers.dual_timescale_wrapper import create_wrapper

from lab.src.components.developmental_monitor import DevelopmentalMonitor

# Phase 1 baseline (from statistical analysis)
PHASE1_BASELINE = {
    'vdi_target': 0.611992,
    'crystallization_mean': 3700,
    'crystallization_std': 400,
}


def add_attention_caching(model: nn.Module):
    """Add attention caching hooks to model (from train_modular_with_monitoring.py)."""
    model.attention_cache = []

    def make_attention_hook(layer_idx):
        def hook(module, input, output):
            if hasattr(module, "_cached_attn_weights"):
                while len(model.attention_cache) <= layer_idx:
                    model.attention_cache.append(None)
                model.attention_cache[layer_idx] = module._cached_attn_weights
        return hook

    for layer_idx, block in enumerate(model.blocks):
        block.register_forward_hook(make_attention_hook(layer_idx))


def modify_block_to_cache_attention(block: nn.Module):
    """Monkey-patch TransformerBlock to cache attention weights."""
    original_forward = block.forward

    def forward_with_cache(x, layer_idx, layer_head_config=None, attn_mask=None):
        # Recompute attention for caching
        bsz, seq_len, _ = x.shape
        h = block.ln1(x)
        q = block.W_q(h).view(bsz, seq_len, block.n_heads, block.d_head)
        k = block.W_k(h).view(bsz, seq_len, block.n_heads, block.d_head)

        attn_scores = torch.einsum("bthd,bshd->bhts", q, k) / np.sqrt(block.d_head)
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask[:, :, :seq_len, :seq_len]
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)

        # Cache for monitoring
        block._cached_attn_weights = attn_weights.detach()

        # Call original forward
        result = original_forward(x, layer_idx, layer_head_config, attn_mask)
        return result

    block.forward = forward_with_cache


def train_phase2_condition(
    config: GrokkingConfig,
    condition_name: str,
    output_dir: Path,
    monitor_interval: int = 500,
):
    """Training loop with Phase 2 dual-timescale control."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device setup
    if config.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(config.device)

    print(f"Using device: {device}")

    # Load data
    print("Loading modular arithmetic data...")
    train_data = load_modular_data(Path(config.data_path_train))
    test_data = load_modular_data(Path(config.data_path_test))
    print(f"Train: {len(train_data)} examples, Test: {len(test_data)} examples")

    # Detect modulus from data
    if train_data:
        config.modulus = max(ex["a"] for ex in train_data) + 1
        print(f"Detected modulus: {config.modulus}")

    # Initialize model
    model = GrokkingTransformer(config).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")

    # Add attention caching
    add_attention_caching(model)
    for block in model.blocks:
        modify_block_to_cache_attention(block)

    # Get experimental condition config
    condition_config = EXPERIMENTAL_CONDITIONS[condition_name]

    print(f"\n{'='*70}")
    print(f"Phase 2 Condition: {condition_config['name']}")
    print(f"{'='*70}")
    print(f"Description: {condition_config['description']}")
    print(f"Lambdas: comp={condition_config['lambda_compensation']}, "
          f"conv={condition_config['lambda_convergence']}, "
          f"setpoint={condition_config['lambda_setpoint']}")
    print(f"LR scales: fast={condition_config['fast_lr_scale']}, "
          f"slow={condition_config['slow_lr_scale']}")
    print(f"{'='*70}\n")

    # Initialize dual-timescale wrapper
    wrapper = create_wrapper(
        model,
        condition_config,  # Pass whole config
        base_lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # Initialize homeostatic loss
    loss_fn = HomeostasisAwareLoss(
        vocab_size=config.modulus,
        config=condition_config,
    )

    # Initialize developmental monitor
    target_head = (0, 0)  # Monitor Layer 0, Head 0
    monitor = DevelopmentalMonitor(
        model=model,
        target_head=target_head,
        omega_sweep=[0.5, 0.7, 1.0, 1.3, 1.5],
        kill_test_frequency=2,
    )

    # Training metrics
    metrics_log = []
    phase2_log = []

    # Tracking
    grokking_step = None
    crystallization_start = None
    crystallization_end = None

    # Training loop
    for step in range(config.n_steps):
        model.train()

        # Sample batch
        batch = np.random.choice(train_data, size=config.batch_size, replace=True)
        input_ids, targets = prepare_batch(batch, config, device)

        # Forward pass
        logits = model(input_ids)

        # Extract Layer 0 attention weights for VDI computation
        if len(model.attention_cache) > 0 and model.attention_cache[0] is not None:
            attention_layer0 = model.attention_cache[0]  # [batch, n_heads, seq, seq]
        else:
            # Fallback: create dummy attention (should not happen if caching works)
            attention_layer0 = None

        # Get layer outputs for compensation loss (optional)
        # For now, pass None - loss will skip compensation term if not available
        intermediates = None

        # Compute homeostatic loss
        # For modular arithmetic: predict at position 5
        loss, loss_dict = loss_fn(
            logits[:, 5],  # Position 5 prediction [batch, vocab]
            targets,  # [batch]
            attention_weights=attention_layer0,
            intermediates=intermediates,
        )

        # Backward pass (dual-timescale)
        grad_info = wrapper.training_step(loss, phase='both', grad_clip=config.grad_clip)

        # Logging
        if step % config.log_every == 0:
            model.eval()
            with torch.no_grad():
                # Evaluate on test set
                test_batch = test_data[: min(1000, len(test_data))]
                test_ids, test_targets = prepare_batch(test_batch, config, device)
                test_logits = model(test_ids)
                test_loss = F.cross_entropy(test_logits[:, 5], test_targets)
                test_acc = (
                    (test_logits[:, 5].argmax(dim=-1) == test_targets).float().mean().item()
                )

                # Train accuracy
                train_batch = train_data[: min(1000, len(train_data))]
                train_ids, train_targets = prepare_batch(train_batch, config, device)
                train_logits = model(train_ids)
                train_acc = (
                    (train_logits[:, 5].argmax(dim=-1) == train_targets)
                    .float()
                    .mean()
                    .item()
                )

            # Track grokking
            if grokking_step is None and test_acc > 0.95:
                grokking_step = step
                print(f"\n🎯 GROKKING at step {step}! Test acc: {test_acc:.3f}\n")

            # Track crystallization (based on VDI std from loss_dict)
            if 'vdi_std' in loss_dict:
                vdi_std = loss_dict['vdi_std']
                vdi_mean = loss_dict['vdi_mean']

                if grokking_step is not None and crystallization_start is None:
                    if vdi_std < 0.001:
                        crystallization_start = step
                        print(f"\n🔷 CRYSTALLIZATION START at step {step}! VDI std: {vdi_std:.6f}\n")

                if crystallization_start is not None and crystallization_end is None:
                    if vdi_std < 0.0001:
                        crystallization_end = step
                        print(f"\n✨ CRYSTALLIZATION END at step {step}! VDI: {vdi_mean:.6f}\n")

            print(
                f"Step {step:5d} | "
                f"Loss: {loss.item():.4f} | "
                f"Train Acc: {train_acc:.3f} | "
                f"Test Acc: {test_acc:.3f} | "
                f"VDI: {loss_dict.get('vdi_mean', 0):.4f} ± {loss_dict.get('vdi_std', 0):.6f}"
            )

            # Log Phase 2 metrics
            phase2_metrics = {
                "step": step,
                "condition": condition_name,
                "train_loss": loss.item(),
                "test_loss": test_loss.item(),
                "train_acc": train_acc,
                "test_acc": test_acc,
                **loss_dict,  # Includes task_loss, convergence_loss, etc.
                "fast_grad_norm": grad_info['fast_grad_norm'],
                "slow_grad_norm": grad_info['slow_grad_norm'],
                "is_grokking": grokking_step is not None,
                "is_crystallized": crystallization_end is not None,
            }
            phase2_log.append(phase2_metrics)

            metrics_log.append(
                {
                    "step": step,
                    "train_loss": loss.item(),
                    "test_loss": test_loss.item(),
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                }
            )

        # Developmental monitoring (every 500 steps)
        if step > 0 and step % monitor_interval == 0:
            print(f"\n{'='*70}")
            print(f"DEVELOPMENTAL CHECKPOINT @ Step {step}")
            print(f"{'='*70}")

            model.eval()
            with torch.no_grad():
                # Prepare monitoring batch
                monitor_batch_size = min(128, len(test_data))
                monitor_data = np.random.choice(
                    test_data, size=monitor_batch_size, replace=False
                )
                monitor_ids, _ = prepare_batch(monitor_data, config, device)

                # Get Layer 0 activations
                bsz, seq_len = monitor_ids.shape
                pos = (
                    torch.arange(seq_len, device=device)
                    .unsqueeze(0)
                    .expand(bsz, seq_len)
                )
                layer0_input = model.token_emb(monitor_ids) + model.pos_emb(pos)
                attn_mask_truncated = model.attn_mask[:, :, :seq_len, :seq_len]
                layer0_output = model.blocks[0](
                    layer0_input, layer_idx=0, attn_mask=attn_mask_truncated
                )

                # Record checkpoint
                checkpoint = monitor.record_checkpoint(
                    step=step,
                    data_batch=monitor_ids,
                    layer0_activations=layer0_output,
                )

                # Report VDI
                vdi = checkpoint.vdi_snapshot
                print(f"\nVDI Snapshot:")
                print(f"   Mean VDI: {vdi.mean_vdi:.6f} (std: {vdi.vdi_std:.6f})")
                if vdi.vdi_velocity is not None:
                    print(f"   Velocity: {vdi.vdi_velocity:.8f}")

                print(f"\n{'='*70}\n")

        # Early stopping if crystallized and stable
        if crystallization_end is not None and step > crystallization_end + 1000:
            print(f"\n✓ Crystallization complete and stable. Stopping early at step {step}.")
            break

    # Final summary
    print("\n" + "=" * 70)
    print(f"PHASE 2 CONDITION: {condition_name} - COMPLETE")
    print("=" * 70)

    if grokking_step is not None:
        print(f"Grokking step: {grokking_step}")

    if crystallization_start is not None and crystallization_end is not None:
        duration = crystallization_end - crystallization_start
        print(f"Crystallization window: {crystallization_start} → {crystallization_end}")
        print(f"Window duration: {duration} steps")

        # Compare to Phase 1 baseline
        baseline_duration = 1500  # From Phase 1 (seed 0: 3000-4500)
        speedup = (baseline_duration - duration) / baseline_duration
        print(f"Speedup vs baseline: {speedup:.1%}")

        if speedup > 0.3:
            print(f"✓ EXCELLENT SPEEDUP (>30%)")
        elif speedup > 0.1:
            print(f"✓ SUCCESSFUL (positive speedup)")
        elif speedup > 0:
            print(f"~ MODEST SPEEDUP")
        else:
            print(f"✗ NO SPEEDUP (slower than baseline)")

    final_metrics = metrics_log[-1] if metrics_log else {}
    print(f"\nFinal Performance:")
    print(f"  Train Acc: {final_metrics.get('train_acc', 0):.3f}")
    print(f"  Test Acc: {final_metrics.get('test_acc', 0):.3f}")

    # Save results
    monitor.save_trajectory(output_dir / "developmental_trajectory.json")

    with open(output_dir / "training_metrics.jsonl", "w") as f:
        for m in metrics_log:
            f.write(json.dumps(m) + "\n")

    with open(output_dir / "phase2_metrics.jsonl", "w") as f:
        for m in phase2_log:
            f.write(json.dumps(m) + "\n")

    # Save summary
    summary = {
        "condition": condition_name,
        "condition_config": condition_config,
        "phase1_baseline": PHASE1_BASELINE,
        "results": {
            "grokking_step": grokking_step,
            "crystallization_start": crystallization_start,
            "crystallization_end": crystallization_end,
            "crystallization_duration": (
                crystallization_end - crystallization_start
                if crystallization_start and crystallization_end
                else None
            ),
            "final_train_acc": final_metrics.get('train_acc', 0),
            "final_test_acc": final_metrics.get('test_acc', 0),
        },
    }

    with open(output_dir / "phase2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    print("=" * 70)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Dual-timescale training experiments"
    )
    parser.add_argument(
        "--condition",
        type=str,
        required=True,
        choices=list(EXPERIMENTAL_CONDITIONS.keys()),
        help="Experimental condition to run",
    )
    parser.add_argument("--p", type=int, default=113, help="Modulus")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--steps", type=int, default=10000, help="Training steps")
    parser.add_argument(
        "--monitor-interval",
        type=int,
        default=500,
        help="Steps between monitoring checkpoints",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (auto/cpu/cuda/mps)"
    )
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup config
    config = GrokkingConfig(
        device=args.device,
        n_steps=args.steps,
        modulus=args.p,
        data_path_train=f"data/modular_p{args.p}_train.jsonl",
        data_path_test=f"data/modular_p{args.p}_test.jsonl",
    )

    # Output directory
    output_dir = Path(
        f"reports/phase2/{args.condition}/seed{args.seed}"
    )

    print(f"\n{'='*70}")
    print(f"Phase 2: Dual-Timescale Training")
    print(f"{'='*70}")
    print(f"Condition: {args.condition}")
    print(f"Modulus: {args.p}")
    print(f"Seed: {args.seed}")
    print(f"Steps: {args.steps}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")

    # Run training
    summary = train_phase2_condition(
        config=config,
        condition_name=args.condition,
        output_dir=output_dir,
        monitor_interval=args.monitor_interval,
    )

    print("\n✓ Phase 2 experiment complete!")


if __name__ == "__main__":
    main()
