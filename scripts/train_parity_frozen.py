#!/usr/bin/env python3
"""
Train parity task with FROZEN compensation heads.

This is the KILL TEST for homeostatic compensation.

Experiment design:
1. Perturb head-0 with omega != 1.0
2. FREEZE other heads (heads 1, 2, 3) to prevent gradient updates
3. Measure if grokking still occurs

Predictions:
- If compensation is ACTIVE/NECESSARY: Freezing prevents grokking (or severely delays it)
- If compensation is PASSIVE: Grokking occurs normally (just via different mechanism)

This distinguishes:
- Active homeostasis (system fights back, needs compensation)
- Passive convergence (system finds same attractor via any route)
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from train_parity import *

def train_parity_frozen(args, cfg: ParityConfig):
    """
    Train parity with frozen heads.

    Args:
        args.freeze_heads: List of head indices to freeze in Layer-0 (e.g., [1,2,3])
    """
    device = torch.device(cfg.device if cfg.device != "auto" else
                         ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load data
    train_data = load_parity_data(Path(cfg.data_path_train))
    test_data = load_parity_data(Path(cfg.data_path_test))

    print(f"Loading data from {cfg.data_path_train} and {cfg.data_path_test}")
    print(f"Loaded {len(train_data)} train, {len(test_data)} test examples")
    print()

    # Build model
    model = ParityTransformer(cfg).to(device)

    # Setup omega perturbation
    layer_head_config = {(0, args.head): args.omega}

    # Setup optimizer with frozen heads
    # We'll create parameter groups: frozen vs trainable
    frozen_params = []
    trainable_params = []

    freeze_heads = getattr(args, 'freeze_heads', [])

    if freeze_heads:
        print(f"[FROZEN HEADS EXPERIMENT]")
        print(f"  Perturbing: Layer-0, Head-{args.head} with omega={args.omega}")
        print(f"  Freezing: Layer-0, Heads {freeze_heads}")
        print()

        # Identify which parameters belong to frozen heads
        # In our architecture, each head's parameters are in W_q, W_k, W_v, W_o
        # We need to freeze the specific slices corresponding to frozen heads

        # For simplicity, we'll use a hook-based approach:
        # Register a hook that zeros gradients for frozen heads
        def freeze_head_gradients(module, grad_input, grad_output):
            """Zero out gradients for frozen heads."""
            # grad_output[0] is the gradient w.r.t. the output
            # Shape: [batch, seq, heads, d_head] for context
            if grad_output[0] is not None:
                for head_idx in freeze_heads:
                    if 0 <= head_idx < cfg.n_heads:
                        # Zero gradients for this head
                        # This happens in the context tensor before W_o projection
                        pass  # We'll handle this differently
            return grad_input

        # Actually, easier approach: freeze by masking gradients after backward
        # We'll store the freeze mask and apply it in training loop

    # Create optimizer (all parameters trainable initially)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    # Output directory
    freeze_str = f"_frozen{''.join(map(str, freeze_heads))}" if freeze_heads else ""
    tag = f"_{args.tag}" if getattr(args, "tag", "") else ""
    run_name = f"parity_head{args.head}_omega{args.omega}_seed{args.seed}{freeze_str}{tag}"
    out_dir = Path(f"reports/parity/train/{run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[parity] Starting {run_name} on {device}")
    print(f"[parity] omega={args.omega}, head={args.head}, steps={cfg.n_steps}")
    print(f"[parity] layer_head_config={layer_head_config}")
    if freeze_heads:
        print(f"[parity] FREEZING heads {freeze_heads} in Layer-0")
    print()

    # Save config
    config_dict = asdict(cfg)
    config_dict.update({
        'omega': args.omega,
        'head': args.head,
        'seed': args.seed,
        'freeze_heads': freeze_heads,
        'tag': getattr(args, "tag", ""),
    })
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    metrics_file = out_dir / "metrics.jsonl"

    # Training state
    T_grok = None
    start_time = time.time()

    # Checkpoints
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    for step in range(1, cfg.n_steps + 1):
        model.train()

        # Sample batch
        batch = random.sample(train_data, cfg.batch_size)
        input_ids, targets = prepare_batch(batch, cfg, device)

        # Forward
        logits = model(input_ids, layer_head_config=layer_head_config)
        loss = F.cross_entropy(logits, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # FREEZE SPECIFIC HEADS by zeroing their gradients
        if freeze_heads:
            block_0 = model.blocks[0]

            # Freeze W_q, W_k, W_v for frozen heads
            # These are shape [d_model, d_model] where d_model = n_heads * d_head
            # Each head owns a slice of size d_head
            d_head = cfg.d_model // cfg.n_heads

            for head_idx in freeze_heads:
                if 0 <= head_idx < cfg.n_heads:
                    start_idx = head_idx * d_head
                    end_idx = (head_idx + 1) * d_head

                    # Zero gradients for this head's parameters
                    if block_0.W_q.weight.grad is not None:
                        block_0.W_q.weight.grad[:, start_idx:end_idx] = 0
                    if block_0.W_k.weight.grad is not None:
                        block_0.W_k.weight.grad[:, start_idx:end_idx] = 0
                    if block_0.W_v.weight.grad is not None:
                        block_0.W_v.weight.grad[:, start_idx:end_idx] = 0

                    # Also freeze the output projection for this head
                    if block_0.W_o.weight.grad is not None:
                        block_0.W_o.weight.grad[start_idx:end_idx, :] = 0

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        # Update
        optimizer.step()

        # Logging
        if step % cfg.log_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                # Train accuracy (on batch)
                train_preds = logits.argmax(dim=-1)
                train_acc = (train_preds == targets).float().mean().item()

                # Test accuracy (on test set)
                test_batch = test_data
                test_input, test_targets = prepare_batch(test_batch, cfg, device)
                test_logits = model(test_input, layer_head_config=layer_head_config)
                test_preds = test_logits.argmax(dim=-1)
                test_acc = (test_preds == test_targets).float().mean().item()

                # Check if grokked
                if T_grok is None and test_acc >= 0.90:
                    T_grok = step

                elapsed = time.time() - start_time

                print(f"[step {step:5d}] loss={loss.item():.4f}, "
                      f"train_acc={train_acc:.3f}, test_acc={test_acc:.3f}, "
                      f"T_grok={T_grok}, elapsed={elapsed/60:.1f}m")

                # Log metrics
                with open(metrics_file, "a") as f:
                    metrics = {
                        "step": step,
                        "loss": loss.item(),
                        "train_acc": train_acc,
                        "test_acc": test_acc,
                        "T_grok": T_grok,
                        "elapsed_s": elapsed,
                    }
                    f.write(json.dumps(metrics) + "\n")

        # Checkpoint
        if step % cfg.checkpoint_every == 0:
            torch.save(model.state_dict(), checkpoint_dir / f"step_{step}.pt")

    # Save final model
    torch.save(model.state_dict(), out_dir / "final_model.pt")
    print(f"\n[parity] Training complete. Outputs in {out_dir}")


def parse_args_frozen():
    parser = argparse.ArgumentParser(description="Parity task with frozen heads")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--omega", type=float, required=True)
    parser.add_argument("--head", type=int, required=True, help="Head to perturb")
    parser.add_argument("--freeze-heads", type=int, nargs="+", default=[],
                       help="Heads to freeze (e.g., --freeze-heads 1 2 3)")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data-dir", type=str, default="data_parity")
    parser.add_argument("--tag", type=str, default="", help="Optional run name suffix")
    return parser.parse_args()


def main():
    args = parse_args_frozen()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = ParityConfig(
        n_steps=args.steps,
        device=args.device,
        data_path_train=f"{args.data_dir}/parity_train.jsonl",
        data_path_test=f"{args.data_dir}/parity_test.jsonl"
    )

    train_parity_frozen(args, cfg)


if __name__ == "__main__":
    main()
