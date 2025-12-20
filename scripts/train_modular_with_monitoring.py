#!/usr/bin/env python3
"""
Modular arithmetic training with integrated developmental monitoring.

This combines Stage-1B grokking experiments with the developmental monitoring
framework to observe VDI snap during the grokking transition.

Phase 1: Observation
- Track VDI trajectory during modular arithmetic training
- Detect snap timing relative to grokking
- Measure Le Chatelier compensation dynamics
- Generate publication-quality figures

Usage:
    python scripts/train_modular_with_monitoring.py \
        --p 97 \
        --omega 1.0 \
        --seed 0 \
        --steps 10000 \
        --monitor-interval 500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "lab"))
sys.path.insert(0, str(Path(__file__).parent))

from src.components.developmental_monitor import (
    DevelopmentalMonitor,
    check_saturation_boundary,
    detect_vdi_snap,
)

# Import from Stage-1B script
from train_stage1b_grokking import (
    GrokkingConfig,
    GrokkingTransformer,
    load_modular_data,
    prepare_batch,
)


def add_attention_caching(model: nn.Module):
    """Add attention caching hooks to model."""
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


def train_with_monitoring(
    config: GrokkingConfig,
    output_dir: Path,
    monitor_interval: int = 500,
    kill_test_frequency: int = 2,
    omega_sweep: Optional[list] = None,
):
    """Training loop with developmental monitoring."""
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

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    # Initialize developmental monitor
    target_head = (0, 0)  # Monitor Layer 0, Head 0
    if omega_sweep is None:
        omega_sweep = [0.5, 0.7, 1.0, 1.3, 1.5]

    monitor = DevelopmentalMonitor(
        model=model,
        target_head=target_head,
        omega_sweep=omega_sweep,
        kill_test_frequency=kill_test_frequency,
    )

    print(f"\nDevelopmental Monitoring Configuration:")
    print(f"  Target head: {target_head}")
    print(f"  Omega sweep: {omega_sweep}")
    print(f"  Monitor interval: {monitor_interval} steps")
    print(f"  Kill test frequency: every {kill_test_frequency} checkpoints")
    print()

    # Training metrics
    metrics_log = []

    # Training loop
    for step in range(config.n_steps):
        model.train()

        # Sample batch
        batch = np.random.choice(train_data, size=config.batch_size, replace=True)
        input_ids, targets = prepare_batch(batch, config, device)

        # Forward pass
        logits = model(input_ids)
        # Loss: predict result at position 5 (last position in sequence)
        loss = F.cross_entropy(logits[:, 5], targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        # Logging
        if step % config.log_every == 0:
            model.eval()
            with torch.no_grad():
                # Evaluate on test set
                test_batch = test_data[: min(1000, len(test_data))]
                test_ids, test_targets = prepare_batch(
                    test_batch, config, device
                )
                test_logits = model(test_ids)
                test_loss = F.cross_entropy(test_logits[:, 5], test_targets)
                test_acc = (
                    (test_logits[:, 5].argmax(dim=-1) == test_targets).float().mean().item()
                )

                # Train accuracy
                train_batch = train_data[: min(1000, len(train_data))]
                train_ids, train_targets = prepare_batch(
                    train_batch, config, device
                )
                train_logits = model(train_ids)
                train_acc = (
                    (train_logits[:, 5].argmax(dim=-1) == train_targets)
                    .float()
                    .mean()
                    .item()
                )

            print(
                f"Step {step:5d} | "
                f"Loss: {loss.item():.4f} | "
                f"Train Acc: {train_acc:.3f} | "
                f"Test Acc: {test_acc:.3f}"
            )

            metrics_log.append(
                {
                    "step": step,
                    "train_loss": loss.item(),
                    "test_loss": test_loss.item(),
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                }
            )

        # Developmental monitoring
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

                # Get Layer 0 activations for MI estimation
                bsz, seq_len = monitor_ids.shape
                pos = (
                    torch.arange(seq_len, device=device)
                    .unsqueeze(0)
                    .expand(bsz, seq_len)
                )
                layer0_input = model.token_emb(monitor_ids) + model.pos_emb(pos)
                # Truncate attention mask to match sequence length
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
                print(f"\nA. VDI Snapshot:")
                print(f"   Mean VDI: {vdi.mean_vdi:.4f} (std: {vdi.vdi_std:.4f})")
                if vdi.vdi_velocity is not None:
                    print(f"   Velocity: {vdi.vdi_velocity:.6f}")
                if vdi.vdi_acceleration is not None:
                    print(f"   Acceleration: {vdi.vdi_acceleration:.6f}")

                # Check for snap
                if not monitor.snap_detected and len(monitor.vdi_history) >= 5:
                    snap_result = detect_vdi_snap(monitor.vdi_history)
                    if snap_result.snap_detected:
                        print(f"\n🔔 VDI SNAP DETECTED!")
                        print(f"   Snap step: {snap_result.snap_step}")
                        print(f"   Confidence: {snap_result.snap_confidence:.3f}")
                        print(
                            f"   Current test acc: {test_acc:.3f}"
                        )
                        print(
                            f"   This is the grokking transition!"
                        )
                        monitor.snap_detected = True
                        monitor.snap_result = snap_result

                # Report kill test
                if checkpoint.kill_test_result:
                    kt = checkpoint.kill_test_result
                    print(f"\nB. Homeostatic Kill Test:")
                    print(f"   Le Chatelier detected: {kt.le_chatelier_detected}")
                    print(f"   Compensation score: {kt.compensation_score:.4f}")
                    if kt.le_chatelier_detected:
                        print(f"   ✓ Compensation is active!")

                # Report MI saturation
                if checkpoint.mi_snapshot:
                    mi = checkpoint.mi_snapshot
                    is_saturated, diagnosis = check_saturation_boundary(mi)
                    print(f"\nC. Mutual Information:")
                    print(f"   {diagnosis}")

                print(f"\n{'='*70}\n")

    # Final analysis
    print("\n" + "=" * 70)
    print("FINAL DEVELOPMENTAL TRAJECTORY ANALYSIS")
    print("=" * 70 + "\n")

    summary = monitor.analyze_trajectory()

    print(f"Snap Detection:")
    print(f"  Detected: {summary['snap_detected']}")
    if summary["snap_detected"]:
        print(f"  Step: {summary['snap_step']}")
        print(f"  Confidence: {summary['snap_confidence']:.3f}")

        # Find test accuracy at snap
        snap_metrics = [m for m in metrics_log if m["step"] == summary["snap_step"]]
        if snap_metrics:
            print(f"  Test accuracy at snap: {snap_metrics[0]['test_acc']:.3f}")

    print(f"\nCompensation by Phase:")
    for phase, score in summary["compensation_by_phase"].items():
        print(f"  {phase}: {score:.4f}")

    print(f"\nLe Chatelier Confirmed: {summary['le_chatelier_confirmed']}")

    if summary["saturation_warning"]:
        print(f"\n⚠️  Saturation Warning:")
        print(f"  Saturated at steps: {summary['saturated_steps']}")

    print(f"\nTotal Checkpoints: {summary['total_checkpoints']}")

    # Final test accuracy
    final_metrics = metrics_log[-1] if metrics_log else {}
    print(f"\nFinal Performance:")
    print(f"  Train Acc: {final_metrics.get('train_acc', 0):.3f}")
    print(f"  Test Acc: {final_metrics.get('test_acc', 0):.3f}")
    if final_metrics.get('test_acc', 0) > 0.95:
        print(f"  ✓ GROKKING ACHIEVED!")

    # Save results
    monitor.save_trajectory(output_dir / "developmental_trajectory.json")

    with open(output_dir / "training_metrics.jsonl", "w") as f:
        for m in metrics_log:
            f.write(json.dumps(m) + "\n")

    with open(output_dir / "config.json", "w") as f:
        json.dump(
            {
                "config": {
                    "modulus": config.modulus,
                    "n_steps": config.n_steps,
                    "lr": config.lr,
                    "weight_decay": config.weight_decay,
                    "d_model": config.d_model,
                    "n_layers": config.n_layers,
                    "n_heads": config.n_heads,
                },
                "monitoring": {
                    "target_head": list(target_head),
                    "omega_sweep": omega_sweep,
                    "monitor_interval": monitor_interval,
                },
                "summary": summary,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to {output_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Train modular arithmetic with developmental monitoring"
    )
    parser.add_argument("--p", type=int, default=97, help="Modulus")
    parser.add_argument("--omega", type=float, default=1.0, help="Omega scaling")
    parser.add_argument("--head", type=int, default=0, help="Head to scale")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--steps", type=int, default=10000, help="Training steps")
    parser.add_argument(
        "--monitor-interval",
        type=int,
        default=500,
        help="Steps between monitoring checkpoints",
    )
    parser.add_argument(
        "--kill-test-frequency",
        type=int,
        default=2,
        help="Run kill test every N monitoring checkpoints",
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
        f"reports/developmental_monitoring/modular_p{args.p}_omega{args.omega}_seed{args.seed}"
    )

    print(f"\nPhase 1: Observation - Modular Arithmetic with Monitoring")
    print(f"Modulus: {args.p}")
    print(f"Omega: {args.omega}")
    print(f"Head: {args.head}")
    print(f"Seed: {args.seed}")
    print(f"Steps: {args.steps}")
    print(f"Output: {output_dir}\n")

    # Run training with monitoring
    train_with_monitoring(
        config=config,
        output_dir=output_dir,
        monitor_interval=args.monitor_interval,
        kill_test_frequency=args.kill_test_frequency,
    )


if __name__ == "__main__":
    main()
