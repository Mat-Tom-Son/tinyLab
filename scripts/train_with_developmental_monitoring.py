#!/usr/bin/env python3
"""Training script with integrated developmental monitoring.

This demonstrates how to use the DevelopmentalMonitor to track:
- VDI "snap" detection (crystallization of Layer-0 bottleneck)
- Homeostatic kill testing (Le Chatelier compensation)
- MI saturation boundaries

Based on the parity/grokking training scripts but with monitoring hooks.

Usage:
    python scripts/train_with_developmental_monitoring.py \
        --task parity \
        --omega 1.0 \
        --head 0 \
        --seed 0 \
        --steps 10000 \
        --monitor-interval 500 \
        --kill-test-frequency 2
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

# Add lab to path
sys.path.insert(0, str(Path(__file__).parent.parent / "lab"))

from src.components.developmental_monitor import (
    DevelopmentalMonitor,
    check_saturation_boundary,
    detect_vdi_snap,
)

# Import from existing training scripts
sys.path.insert(0, str(Path(__file__).parent))
try:
    from train_parity import (
        ParityConfig,
        ParityTransformer,
        load_parity_data,
        prepare_batch,
    )
except ImportError:
    print("Warning: Could not import train_parity. Make sure it exists.")
    ParityConfig = None
    ParityTransformer = None


def add_attention_caching(model: nn.Module):
    """Modify model to cache attention weights for VDI computation.

    This adds hooks to each TransformerBlock to cache attention weights
    in model.attention_cache.
    """
    model.attention_cache = []

    def make_attention_hook(layer_idx):
        def hook(module, input, output):
            # Assuming output contains attention weights
            # You may need to modify TransformerBlock to return attn_weights
            if hasattr(module, "_cached_attn_weights"):
                # Ensure we have space
                while len(model.attention_cache) <= layer_idx:
                    model.attention_cache.append(None)
                model.attention_cache[layer_idx] = module._cached_attn_weights
        return hook

    # Register hooks on each block
    for layer_idx, block in enumerate(model.blocks):
        block.register_forward_hook(make_attention_hook(layer_idx))


def modify_block_to_cache_attention(block: nn.Module):
    """Modify TransformerBlock.forward() to cache attention weights.

    This is a runtime monkey-patch. For production, modify the class directly.
    """
    original_forward = block.forward

    def forward_with_cache(x, layer_idx, layer_head_config=None, attn_mask=None):
        # Recompute attention weights for caching
        # This is inefficient but necessary for monitoring
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
    config,
    output_dir: Path,
    monitor_interval: int = 500,
    kill_test_frequency: int = 2,
    omega_sweep: Optional[list] = None,
):
    """Training loop with integrated developmental monitoring.

    Args:
        config: Training configuration (ParityConfig or similar)
        output_dir: Directory to save results
        monitor_interval: Steps between monitoring checkpoints
        kill_test_frequency: Run kill test every N monitoring checkpoints
        omega_sweep: Omega values for kill testing
    """
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
    print("Loading data...")
    train_data = load_parity_data(Path(config.data_path_train))
    test_data = load_parity_data(Path(config.data_path_test))
    print(f"Train: {len(train_data)} examples, Test: {len(test_data)} examples")

    # Initialize model
    model = ParityTransformer(config).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")

    # Add attention caching for monitoring
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
        # Parity model already reduces to [batch, 2] - no need for [:, -1, :]
        loss = F.cross_entropy(logits, targets)

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
                test_ids, test_targets = prepare_batch(test_batch, config, device)
                test_logits = model(test_ids)
                test_loss = F.cross_entropy(test_logits, test_targets)
                test_acc = (
                    (test_logits.argmax(dim=-1) == test_targets)
                    .float()
                    .mean()
                    .item()
                )

            print(
                f"Step {step:5d} | "
                f"Loss: {loss.item():.4f} | "
                f"Test Loss: {test_loss.item():.4f} | "
                f"Test Acc: {test_acc:.3f}"
            )

            metrics_log.append(
                {
                    "step": step,
                    "train_loss": loss.item(),
                    "test_loss": test_loss.item(),
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
                # Prepare monitoring batch (use subset of test data)
                monitor_batch_size = min(128, len(test_data))
                monitor_data = np.random.choice(
                    test_data, size=monitor_batch_size, replace=False
                )
                monitor_ids, _ = prepare_batch(monitor_data, config, device)

                # Get Layer 0 activations for MI estimation
                # Forward through embeddings
                bsz, seq_len = monitor_ids.shape
                pos = (
                    torch.arange(seq_len, device=device)
                    .unsqueeze(0)
                    .expand(bsz, seq_len)
                )
                layer0_input = model.token_emb(monitor_ids) + model.pos_emb(pos)
                layer0_output = model.blocks[0](
                    layer0_input, layer_idx=0, attn_mask=model.attn_mask
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
                            f"   Phase: {checkpoint.developmental_phase} -> snap_window"
                        )
                        monitor.snap_detected = True
                        monitor.snap_result = snap_result

                # Report kill test
                if checkpoint.kill_test_result:
                    kt = checkpoint.kill_test_result
                    print(f"\nB. Homeostatic Kill Test:")
                    print(
                        f"   Le Chatelier detected: {kt.le_chatelier_detected}"
                    )
                    print(f"   Compensation score: {kt.compensation_score:.4f}")
                    print(f"   Signatures:")
                    for sig in kt.compensation_signatures:
                        if sig.omega != 1.0:  # Skip baseline
                            print(
                                f"     ω={sig.omega:.1f}: "
                                f"{sig.compensation_count}/{len(sig.vdi_other_heads)} heads compensating, "
                                f"strength={sig.compensation_strength:.4f}"
                            )

                # Report MI saturation
                if checkpoint.mi_snapshot:
                    mi = checkpoint.mi_snapshot
                    is_saturated, diagnosis = check_saturation_boundary(mi)
                    print(f"\nC. Mutual Information:")
                    print(f"   {diagnosis}")
                    if is_saturated:
                        print(
                            f"   ⚠️  WARNING: Brittle attractor risk - "
                            f"system may not generalize robustly"
                        )

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

    print(f"\nCompensation by Phase:")
    for phase, score in summary["compensation_by_phase"].items():
        print(f"  {phase}: {score:.4f}")

    print(f"\nLe Chatelier Confirmed: {summary['le_chatelier_confirmed']}")

    if summary["saturation_warning"]:
        print(f"\n⚠️  Saturation Warning:")
        print(f"  Saturated at steps: {summary['saturated_steps']}")

    print(f"\nTotal Checkpoints: {summary['total_checkpoints']}")

    # Save results
    monitor.save_trajectory(output_dir / "developmental_trajectory.json")

    with open(output_dir / "training_metrics.jsonl", "w") as f:
        for m in metrics_log:
            f.write(json.dumps(m) + "\n")

    print(f"\nResults saved to {output_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Train with developmental monitoring"
    )
    parser.add_argument("--task", type=str, default="parity", help="Task name")
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
    if args.task == "parity":
        if ParityConfig is None:
            print("Error: Could not import ParityConfig")
            return
        config = ParityConfig(device=args.device, n_steps=args.steps)
    else:
        print(f"Unknown task: {args.task}")
        return

    # Output directory
    output_dir = Path(
        f"reports/developmental_monitoring/{args.task}_omega{args.omega}_seed{args.seed}"
    )

    print(f"\nDevelopmental Monitoring Experiment")
    print(f"Task: {args.task}")
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
