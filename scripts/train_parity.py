#!/usr/bin/env python3
"""
Train transformer on parity task with omega perturbations.

This tests whether suppressor scaling (omega) affects the developmental
trajectory of circuit formation for compositional reasoning.

Task: Given binary string, predict if number of 1s is ODD or EVEN

Why this works:
- Requires temporal state tracking (can't solve by lookup)
- Forces circuit formation (must maintain parity state across tokens)
- Has clear phase transition (when counting emerges)
- Small enough for CPU training

Usage:
    python scripts/train_parity.py --omega 1.0 --head 0 --seed 0 --steps 10000
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@dataclass
class ParityConfig:
    # Model architecture - small but not tiny
    vocab_size: int = 4  # 0='0', 1='1', 2=PAD, 3=CLS
    d_model: int = 64
    n_layers: int = 2   # Need at least 2 for state tracking
    n_heads: int = 4
    d_mlp: int = 128
    max_seq_len: int = 32  # Handle up to length-32 sequences

    # Training
    batch_size: int = 256
    n_steps: int = 10000
    lr: float = 1e-3
    weight_decay: float = 1.0  # High regularization to destabilize memorization
    grad_clip: float = 1.0

    # Logging
    log_every: int = 100
    checkpoint_every: int = 1000

    # Data
    data_path_train: str = "data_parity/parity_train.jsonl"
    data_path_test: str = "data_parity/parity_test.jsonl"

    device: str = "auto"


class TransformerBlock(nn.Module):
    """Transformer block with per-head scaling."""

    def __init__(self, d_model: int, n_heads: int, d_mlp: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int,
        layer_head_config: Dict[Tuple[int, int], float] | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        # Self-attention
        h = self.ln1(x)
        q = self.W_q(h).view(bsz, seq_len, self.n_heads, self.d_head)
        k = self.W_k(h).view(bsz, seq_len, self.n_heads, self.d_head)
        v = self.W_v(h).view(bsz, seq_len, self.n_heads, self.d_head)

        attn_scores = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(self.d_head)
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        attn_weights = F.softmax(attn_scores, dim=-1)

        context = torch.einsum("bhts,bshd->bthd", attn_weights, v)

        # Per-head scaling (omega perturbation)
        if layer_head_config:
            for (layer, head_idx), alpha in layer_head_config.items():
                if layer == layer_idx and 0 <= head_idx < self.n_heads:
                    context[:, :, head_idx, :] *= alpha

        context = context.reshape(bsz, seq_len, self.d_model)
        attn_out = self.W_o(context)
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.ln2(x))
        return x


class ParityTransformer(nn.Module):
    """Transformer for parity task."""

    def __init__(self, cfg: ParityConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_mlp)
            for _ in range(cfg.n_layers)
        ])

        self.ln_f = nn.LayerNorm(cfg.d_model)
        # Binary classification: ODD vs EVEN
        self.classifier = nn.Linear(cfg.d_model, 2, bias=True)

        # Causal mask
        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("attn_mask", (mask == 0).float() * -1e9, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        layer_head_config: Dict[Tuple[int, int], float] | None = None,
    ) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, seq_len)
        x = self.token_emb(input_ids) + self.pos_emb(pos)

        attn_mask = self.attn_mask[:, :, :seq_len, :seq_len]

        for layer_idx, block in enumerate(self.blocks):
            x = block(x, layer_idx, layer_head_config, attn_mask)

        x = self.ln_f(x)

        # Classify from last token (CLS position)
        logits = self.classifier(x[:, -1, :])  # [batch, 2]

        return logits


def load_parity_data(path: Path) -> List[Dict]:
    """Load JSONL dataset."""
    data = []
    with path.open() as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def prepare_batch(
    examples: List[Dict],
    cfg: ParityConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert examples to input_ids and targets.

    Format: [CLS] [seq] where seq is binary string
    Tokens: 0='0', 1='1', 2=PAD, 3=CLS
    """
    input_ids = []
    targets = []

    for ex in examples:
        seq = ex['sequence']
        target = 1 if ex['target'] == 'ODD' else 0  # ODD=1, EVEN=0

        # Convert to tokens: CLS + binary + padding
        tokens = [3]  # CLS token
        tokens.extend([int(c) for c in seq])  # Binary digits

        # Pad to max length
        if len(tokens) < cfg.max_seq_len:
            tokens.extend([2] * (cfg.max_seq_len - len(tokens)))  # PAD
        elif len(tokens) > cfg.max_seq_len:
            tokens = tokens[:cfg.max_seq_len]  # Truncate

        input_ids.append(tokens)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)

    return input_ids, targets


def compute_accuracy(
    model: ParityTransformer,
    data: List[Dict],
    cfg: ParityConfig,
    device: torch.device,
    layer_head_config: Dict[Tuple[int, int], float] | None = None,
) -> float:
    """Compute accuracy on dataset."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        batch_size = cfg.batch_size
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            input_ids, targets = prepare_batch(batch, cfg, device)

            logits = model(input_ids, layer_head_config)
            preds = logits.argmax(dim=-1)

            correct += (preds == targets).sum().item()
            total += len(batch)

    return correct / total if total > 0 else 0.0


def train_parity(args: argparse.Namespace, cfg: ParityConfig):
    """Main training loop."""
    # Device
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    # Seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    print(f"Loading data from {cfg.data_path_train} and {cfg.data_path_test}")
    train_data = load_parity_data(Path(cfg.data_path_train))
    test_data = load_parity_data(Path(cfg.data_path_test))
    print(f"Loaded {len(train_data)} train, {len(test_data)} test examples")

    # Model
    model = ParityTransformer(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    # Layer-head config
    layer_head_config: Dict[Tuple[int, int], float] = {}
    if abs(args.omega - 1.0) > 1e-6:
        layer_head_config[(0, int(args.head))] = float(args.omega)

    # Output directory
    tag = f"_{args.tag}" if getattr(args, "tag", "") else ""
    run_name = f"parity_head{args.head}_omega{args.omega:.1f}_seed{args.seed}{tag}"
    out_dir = Path("reports/parity/train") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # Save config
    full_cfg = {
        "config": asdict(cfg),
        "args": vars(args),
        "device": str(device),
        "layer_head_config": {f"{k[0]}_{k[1]}": v for k, v in layer_head_config.items()},
    }
    (out_dir / "config.json").write_text(json.dumps(full_cfg, indent=2))

    metrics_path = out_dir / "metrics.jsonl"
    metrics_f = metrics_path.open("w", encoding="utf-8")

    print(f"\n[parity] Starting {run_name} on {device}")
    print(f"[parity] omega={args.omega}, head={args.head}, steps={cfg.n_steps}")
    print(f"[parity] layer_head_config={layer_head_config}\n")

    # Phase tracking
    T_grok = None
    start_time = time.time()

    for step in range(1, cfg.n_steps + 1):
        model.train()

        # Sample batch
        batch_indices = random.sample(range(len(train_data)), min(cfg.batch_size, len(train_data)))
        batch = [train_data[i] for i in batch_indices]
        input_ids, targets = prepare_batch(batch, cfg, device)

        # Forward
        logits = model(input_ids, layer_head_config)
        loss = F.cross_entropy(logits, targets)

        # Backward
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        # Logging
        if step % cfg.log_every == 0 or step == 1:
            with torch.no_grad():
                train_preds = logits.argmax(dim=-1)
                train_acc = (train_preds == targets).float().mean().item()

            test_acc = compute_accuracy(
                model, test_data[:min(1000, len(test_data))], cfg, device, layer_head_config
            )

            # Track T_grok
            if T_grok is None and test_acc >= 0.9:
                T_grok = step

            elapsed = time.time() - start_time

            msg = {
                "step": step,
                "loss": float(loss.item()),
                "train_acc": train_acc,
                "test_acc": test_acc,
                "T_grok": T_grok,
                "elapsed_s": elapsed,
            }

            print(
                f"[step {step:5d}] loss={loss.item():.4f}, "
                f"train_acc={train_acc:.3f}, test_acc={test_acc:.3f}, "
                f"T_grok={T_grok}, elapsed={elapsed/60:.1f}m"
            )

            metrics_f.write(json.dumps(msg) + "\n")
            metrics_f.flush()

        # Checkpointing
        if step % cfg.checkpoint_every == 0 or step == cfg.n_steps:
            ckpt_path = ckpt_dir / f"step_{step:05d}.pt"
            checkpoint = {
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "test_acc": test_acc if 'test_acc' in locals() else None,
            }
            torch.save(checkpoint, ckpt_path)

    metrics_f.close()
    torch.save(model.state_dict(), out_dir / "final_model.pt")
    print(f"\n[parity] Training complete. Outputs in {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parity task training")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--omega", type=float, required=True, help="Head scaling factor")
    parser.add_argument("--head", type=int, required=True, help="Layer-0 head to scale")
    parser.add_argument("--steps", type=int, default=10000, help="Training steps")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--data-dir", type=str, default="data_parity", help="Data directory")
    parser.add_argument("--tag", type=str, default="", help="Optional run name suffix (e.g., medium)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = ParityConfig(
        n_steps=args.steps,
        device=args.device,
        data_path_train=f"{args.data_dir}/parity_train.jsonl",
        data_path_test=f"{args.data_dir}/parity_test.jsonl"
    )
    train_parity(args, cfg)


if __name__ == "__main__":
    main()
