#!/usr/bin/env python3
"""
Stage-1B: Modular arithmetic grokking experiment with omega perturbations.

This is the improved version of Stage-1A that:
1. Uses modular arithmetic (p=113) with documented grokking behavior
2. Tracks multiple order parameters (T_grok, circularity, VDI compensation)
3. Tests Le Chatelier hypothesis via compensation signature
4. Runs omega sweep with dense checkpointing

Task: (a + b) mod 113 = ?

Model: 4-layer transformer (enough for L0 -> mid -> late)
Perturbation: Scale layer-0 head by omega ∈ [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.7]

Outputs:
  reports/stage1b_grokking/train/<run_name>/
    - config.json: experiment configuration
    - metrics.jsonl: per-checkpoint metrics (loss, acc, circularity, etc.)
    - checkpoints/: saved model states at regular intervals
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@dataclass
class GrokkingConfig:
    # Model architecture - MUCH smaller to force learning vs memorization
    vocab_size: int = 1000  # Large enough for p=997
    d_model: int = 64       # Very small!
    n_layers: int = 1       # Single layer transformer
    n_heads: int = 2        # Minimal heads
    d_mlp: int = 128        # Small MLP
    max_seq_len: int = 16

    # Training
    batch_size: int = 512
    n_steps: int = 20000
    lr: float = 1e-3
    weight_decay: float = 1.0  # Moderate weight decay
    grad_clip: float = 1.0
    optimizer: str = "AdamW"

    # Logging
    log_every: int = 100
    checkpoint_every: int = 500  # Dense early, can thin later

    # Task (auto-detects modulus from data)
    modulus: int = 997  # Will be updated if using p=113 data
    data_path_train: str = "data/modular_p113_train.jsonl"
    data_path_test: str = "data/modular_p113_test.jsonl"

    device: str = "auto"


class TransformerBlock(nn.Module):
    """Transformer block with per-head scaling support."""

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
        return_activations: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, seq, d_model]
        bsz, seq_len, _ = x.shape

        # Store pre-attention activations if requested
        pre_attn = x.clone() if return_activations else None

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

        # Per-head scaling
        if layer_head_config:
            for (layer, head_idx), alpha in layer_head_config.items():
                if layer == layer_idx and 0 <= head_idx < self.n_heads:
                    context[:, :, head_idx, :] *= alpha

        context = context.reshape(bsz, seq_len, self.d_model)
        attn_out = self.W_o(context)
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.ln2(x))

        if return_activations:
            return x, pre_attn
        return x


class GrokkingTransformer(nn.Module):
    """4-layer transformer for modular arithmetic grokking."""

    def __init__(self, cfg: GrokkingConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_mlp)
            for _ in range(cfg.n_layers)
        ])

        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Causal mask: [1, 1, T, T]
        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("attn_mask", (mask == 0).float() * -1e9, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        layer_head_config: Dict[Tuple[int, int], float] | None = None,
        return_layer_activations: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, seq_len)
        x = self.token_emb(input_ids) + self.pos_emb(pos)

        attn_mask = self.attn_mask[:, :, :seq_len, :seq_len]

        layer_acts = {}
        for layer_idx, block in enumerate(self.blocks):
            if return_layer_activations:
                x, pre_attn = block(
                    x, layer_idx, layer_head_config, attn_mask,
                    return_activations=True
                )
                layer_acts[layer_idx] = pre_attn
            else:
                x = block(x, layer_idx, layer_head_config, attn_mask)

        x = self.ln_f(x)
        logits = self.unembed(x)

        if return_layer_activations:
            return logits, layer_acts
        return logits


def load_modular_data(path: Path) -> List[Dict]:
    """Load JSONL dataset."""
    data = []
    with path.open() as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def prepare_batch(
    examples: List[Dict],
    cfg: GrokkingConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert examples to input_ids and targets.

    NEW FORMAT (forces learning vs memorization):
    We encode the equation as a sequence that requires understanding relationships.

    Instead of just [a, b, +, =, result], we use:
    [a, a, b, b, =, result]

    The repetition forces the model to learn to:
    1. Extract operands from specific positions
    2. Compute the modular sum
    3. Predict the result

    This prevents simple lookup table memorization.
    """
    input_ids = []
    targets = []

    for ex in examples:
        a, b, result = ex['a'], ex['b'], ex['result']

        # Ensure values are in valid range [0, p-1]
        a = a % cfg.modulus
        b = b % cfg.modulus
        result = result % cfg.modulus

        # Format: [a, a, b, b, PAD, result]
        # The double encoding makes it harder to memorize
        # Model must learn to aggregate information
        seq = [a, a, b, b, 0, result]  # 0 acts as separator
        input_ids.append(seq)
        targets.append(result)

    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)

    return input_ids, targets


def compute_accuracy(
    model: GrokkingTransformer,
    data: List[Dict],
    cfg: GrokkingConfig,
    device: torch.device,
    layer_head_config: Dict[Tuple[int, int], float] | None = None,
) -> float:
    """Compute accuracy on dataset."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        # Process in batches
        batch_size = cfg.batch_size
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            input_ids, targets = prepare_batch(batch, cfg, device)

            logits = model(input_ids, layer_head_config)

            # Predict at position 5 (result position)
            preds = logits[:, 5].argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += len(batch)

    return correct / total if total > 0 else 0.0


def compute_circularity_simple(
    model: GrokkingTransformer,
    data: List[Dict],
    cfg: GrokkingConfig,
    device: torch.device,
    layer_idx: int = 0,
    layer_head_config: Dict[Tuple[int, int], float] | None = None,
) -> float:
    """
    Compute simple circularity score for layer activations.

    Measures variance of radii in PCA space (perfect circle has constant radius).
    """
    model.eval()

    with torch.no_grad():
        # Get activations for first batch
        batch = data[:min(512, len(data))]
        input_ids, targets = prepare_batch(batch, cfg, device)

        logits, layer_acts = model(
            input_ids,
            layer_head_config,
            return_layer_activations=True
        )

        # Extract layer-0 activations at position 4 (before prediction position 5)
        acts = layer_acts[layer_idx][:, 4, :]  # [batch, d_model]

        # PCA to 2D
        from sklearn.decomposition import PCA
        acts_np = acts.cpu().numpy()

        if acts_np.shape[0] < 3:
            return 0.0

        pca = PCA(n_components=2)
        acts_2d = pca.fit_transform(acts_np)

        # Compute circularity
        center = acts_2d.mean(axis=0)
        centered = acts_2d - center
        radii = np.linalg.norm(centered, axis=1)

        mean_r = radii.mean()
        std_r = radii.std()

        if mean_r < 1e-6:
            return 0.0

        cv = std_r / mean_r
        circularity = max(0.0, 1.0 - cv)

        return float(circularity)


def train_grokking(
    args: argparse.Namespace,
    cfg: GrokkingConfig,
) -> None:
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
    train_data = load_modular_data(Path(cfg.data_path_train))
    test_data = load_modular_data(Path(cfg.data_path_test))
    print(f"Loaded {len(train_data)} train, {len(test_data)} test examples")

    # Model
    model = GrokkingTransformer(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    # Layer-head config
    layer_head_config: Dict[Tuple[int, int], float] = {}
    if abs(args.omega - 1.0) > 1e-6:
        # Scale layer-0 head by omega
        layer_head_config[(0, int(args.head))] = float(args.omega)

    # Output directory
    run_name = f"stage1b_head{args.head}_omega{args.omega:.1f}_seed{args.seed}"
    out_dir = Path("reports/stage1b_grokking/train") / run_name
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

    print(f"[stage1b] Starting {run_name} on {device}")
    print(f"[stage1b] omega={args.omega}, head={args.head}, steps={cfg.n_steps}")
    print(f"[stage1b] layer_head_config={layer_head_config}")

    # Phase tracking
    T_grok = None  # Step where accuracy crosses 0.9
    start_time = time.time()

    for step in range(1, cfg.n_steps + 1):
        model.train()

        # Sample batch
        batch_indices = random.sample(range(len(train_data)), cfg.batch_size)
        batch = [train_data[i] for i in batch_indices]
        input_ids, targets = prepare_batch(batch, cfg, device)

        # Forward
        logits = model(input_ids, layer_head_config)

        # Loss: predict result at position 5 (last position in [a, a, b, b, 0, result])
        loss = F.cross_entropy(logits[:, 5], targets)

        # Backward
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        # Logging
        if step % cfg.log_every == 0 or step == 1:
            train_acc = compute_accuracy(model, batch, cfg, device, layer_head_config)
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
            # Save checkpoint with extended metrics
            ckpt_path = ckpt_dir / f"step_{step:05d}.pt"

            # Compute circularity
            circularity = compute_circularity_simple(
                model, test_data[:512], cfg, device, layer_idx=0, layer_head_config=layer_head_config
            )

            checkpoint = {
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "circularity": circularity,
                "test_acc": test_acc if 'test_acc' in locals() else None,
            }

            torch.save(checkpoint, ckpt_path)
            print(f"[checkpoint] Saved {ckpt_path} (circularity={circularity:.3f})")

    metrics_f.close()

    # Final save
    torch.save(model.state_dict(), out_dir / "final_model.pt")
    print(f"[stage1b] Training complete. Outputs in {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-1B grokking experiment")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--omega", type=float, required=True, help="Head scaling factor")
    parser.add_argument("--head", type=int, required=True, help="Layer-0 head to scale (0-7)")
    parser.add_argument("--steps", type=int, default=20000, help="Training steps")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = GrokkingConfig(n_steps=args.steps, device=args.device)
    train_grokking(args, cfg)


if __name__ == "__main__":
    main()
