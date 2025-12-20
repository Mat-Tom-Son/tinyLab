#!/usr/bin/env python3
"""
VDI Target Sweep Experiment

Simplified training script that accepts target_vdi as a parameter.
Based on train_phase2_dual_timescale.py but with command-line control
over all homeostatic parameters.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Import from existing codebase
from scripts.train_stage1b_grokking import GrokkingConfig, GrokkingTransformer, load_modular_data, prepare_batch
from lab.src.losses.homeostasis_aware_loss import HomeostasisAwareLoss
from lab.src.wrappers.dual_timescale_wrapper import create_wrapper


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


def main():
    parser = argparse.ArgumentParser(
        description="VDI Target Sweep: Test if equilibrium tracks target"
    )
    parser.add_argument("--target_vdi", type=float, required=True,
                       help="Target VDI for set-point loss")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--steps", type=int, default=10000, help="Training steps")
    parser.add_argument("--lambda_compensation", type=float, default=0.5,
                       help="Compensation loss weight")
    parser.add_argument("--lambda_convergence", type=float, default=0.3,
                       help="Convergence loss weight")
    parser.add_argument("--lambda_setpoint", type=float, default=0.2,
                       help="Set-point loss weight")
    parser.add_argument("--fast_lr_scale", type=float, default=1.0,
                       help="Fast layer LR multiplier")
    parser.add_argument("--slow_lr_scale", type=float, default=0.1,
                       help="Slow layer LR multiplier")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--p", type=int, default=113, help="Modulus")
    parser.add_argument("--monitor_interval", type=int, default=500,
                       help="Monitoring checkpoint interval")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (auto/cpu/cuda/mps)")

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

    # Create custom condition config
    condition_config = {
        'name': f'VDI Sweep (target={args.target_vdi})',
        'lambda_compensation': args.lambda_compensation,
        'lambda_convergence': args.lambda_convergence,
        'lambda_setpoint': args.lambda_setpoint,
        'target_vdi': args.target_vdi,
        'fast_lr_scale': args.fast_lr_scale,
        'slow_lr_scale': args.slow_lr_scale,
        'description': f'Test if equilibrium tracks target_vdi={args.target_vdi}',
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"VDI Target Sweep Experiment")
    print(f"{'='*70}")
    print(f"Target VDI: {args.target_vdi}")
    print(f"Lambda compensation: {args.lambda_compensation}")
    print(f"Lambda convergence: {args.lambda_convergence}")
    print(f"Lambda setpoint: {args.lambda_setpoint}")
    print(f"Fast LR scale: {args.fast_lr_scale}")
    print(f"Slow LR scale: {args.slow_lr_scale}")
    print(f"Seed: {args.seed}")
    print(f"Steps: {args.steps}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")

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

    # Initialize homeostatic loss
    loss_fn = HomeostasisAwareLoss(vocab_size=config.modulus, config=condition_config)

    # Initialize dual-timescale wrapper
    wrapper = create_wrapper(
        model=model,
        condition_config=condition_config,
        base_lr=config.lr,
        weight_decay=config.weight_decay
    )

    # Note: We skip detailed developmental monitoring for VDI sweep experiments
    # to keep them lightweight. Only Phase 2 metrics (VDI, loss, accuracy) are logged.

    # Training tracking
    grokking_step = None
    crystallization_start = None
    crystallization_end = None
    phase2_log = []

    print("Starting training...")

    for step in range(args.steps):
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
            attention_layer0 = None

        # Compute loss (task + homeostatic)
        loss, loss_dict = loss_fn(
            logits[:, 5],  # Position 5 for modular arithmetic
            targets,
            attention_weights=attention_layer0,
            intermediates=None
        )

        # Backward + update
        grad_info = wrapper.training_step(loss, phase='both', grad_clip=config.grad_clip)

        # Evaluate every 100 steps
        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                # Evaluate on test set
                test_batch = test_data[: min(1000, len(test_data))]
                test_ids, test_targets = prepare_batch(test_batch, config, device)
                test_logits = model(test_ids)
                test_loss = torch.nn.functional.cross_entropy(
                    test_logits[:, 5], test_targets
                )
                test_acc = (test_logits[:, 5].argmax(dim=-1) == test_targets).float().mean()

                # Train accuracy (on sample batch)
                train_batch = train_data[: min(1000, len(train_data))]
                train_ids, train_targets = prepare_batch(train_batch, config, device)
                train_logits = model(train_ids)
                train_acc = (train_logits[:, 5].argmax(dim=-1) == train_targets).float().mean()

            # Track grokking
            if grokking_step is None and test_acc > 0.95:
                grokking_step = step
                print(f"\n🎯 GROKKING at step {step}! Test acc: {test_acc:.3f}\n")

            # Track crystallization
            vdi_std = loss_dict.get('vdi_std', None)
            vdi_mean = loss_dict.get('vdi_mean', None)

            if vdi_std is not None:
                if grokking_step is not None and crystallization_start is None:
                    if vdi_std < 0.001:
                        crystallization_start = step
                        print(f"\n🔷 CRYSTALLIZATION START at step {step}! VDI std: {vdi_std:.6f}\n")

                if crystallization_start is not None and crystallization_end is None:
                    if vdi_std < 0.0001:
                        crystallization_end = step
                        print(f"\n✨ CRYSTALLIZATION END at step {step}! VDI: {vdi_mean:.6f}\n")

            # Log metrics
            phase2_metrics = {
                'step': step,
                'train_loss': loss.item(),
                'test_loss': test_loss.item(),
                'train_acc': train_acc.item(),
                'test_acc': test_acc.item(),
                **loss_dict,
                'fast_grad_norm': grad_info['fast_grad_norm'],
                'slow_grad_norm': grad_info['slow_grad_norm'],
                'is_grokking': grokking_step is not None,
                'is_crystallized': crystallization_end is not None,
            }
            phase2_log.append(phase2_metrics)

            # Print progress
            if step % 1000 == 0:
                vdi_str = f"{vdi_mean:.4f}" if vdi_mean is not None else "N/A"
                vdi_std_str = f"{vdi_std:.6f}" if vdi_std is not None else "N/A"
                print(f"Step {step}: loss={loss.item():.4f}, test_acc={test_acc:.3f}, "
                      f"VDI={vdi_str}, VDI_std={vdi_std_str}")

        # Note: Detailed monitoring checkpoints skipped for VDI sweep

        # Early stopping
        if crystallization_end is not None and step > crystallization_end + 1000:
            print(f"\n✓ Crystallization stable. Stopping early at step {step}.")
            break

    # Save results
    print("\nSaving results...")

    # Phase 2 metrics
    with open(output_dir / "phase2_metrics.jsonl", 'w') as f:
        for entry in phase2_log:
            f.write(json.dumps(entry) + '\n')

    # Note: Developmental trajectory skipped for VDI sweep (monitoring disabled)

    # Summary
    summary = {
        'target_vdi': args.target_vdi,
        'condition_config': condition_config,
        'phase1_baseline': {
            'vdi_target': 0.611992,
            'crystallization_mean': 1500,
            'crystallization_std': 400,
        },
        'results': {
            'grokking_step': grokking_step,
            'crystallization_start': crystallization_start,
            'crystallization_end': crystallization_end,
            'crystallization_duration': (
                crystallization_end - crystallization_start
                if crystallization_start and crystallization_end
                else None
            ),
            'final_train_acc': phase2_log[-1]['train_acc'] if phase2_log else None,
            'final_test_acc': phase2_log[-1]['test_acc'] if phase2_log else None,
            'final_vdi_mean': phase2_log[-1].get('vdi_mean', None) if phase2_log else None,
            'final_vdi_std': phase2_log[-1].get('vdi_std', None) if phase2_log else None,
        }
    }

    with open(output_dir / "sweep_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ VDI sweep experiment complete!")
    print(f"   Target VDI: {args.target_vdi}")
    print(f"   Final VDI: {summary['results']['final_vdi_mean']:.4f}" if summary['results']['final_vdi_mean'] else "N/A")
    print(f"   Grokking: step {grokking_step}")
    print(f"   Crystallization: {crystallization_start} → {crystallization_end}")


if __name__ == "__main__":
    main()
