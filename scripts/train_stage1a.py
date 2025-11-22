#!/usr/bin/env python3
"""
Stage-1A pilot: minimal 2-layer transformer training script with per-head scaling (omega).

This is a self-contained, small-model trainer intended to be launched via
`scripts/run_stage1a_prereg.sh` using a template such as:

  TRAIN_CMD_TEMPLATE='python scripts/train_stage1a.py --cond {cond} --seed {seed} --omega {omega} --head {head} --head-kind {head_kind}'

It trains a 2-layer decoder-only transformer on a simple induction-style Task A:
  sequences of the form ABABAB..., optimized with next-token prediction.

Per-head scaling:
  - Layer 0 head `head` is scaled by alpha=omega when `cond` is "suppressor" or "random".
  - Baseline condition ignores omega and uses alpha=1.0 for all heads.

Outputs:
  - reports/pilot_stage1a/train/<run_name>/config.json  : training config and hyperparams
  - reports/pilot_stage1a/train/<run_name>/metrics.jsonl: per-step loss/accuracy logs
  - reports/pilot_stage1a/train/<run_name>/model.pt     : final model state_dict
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
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TrainConfig:
    vocab_size: int = 64
    d_model: int = 256
    n_layers: int = 2
    n_heads: int = 8
    d_mlp: int = 1024
    max_seq_len: int = 16
    batch_size: int = 64
    n_steps: int = 5000
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    log_every: int = 50
    device: str = "auto"


class TransformerBlock(nn.Module):
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
        # x: [batch, seq, d_model]
        bsz, seq_len, _ = x.shape

        # Self-attention block
        h = self.ln1(x)
        q = self.W_q(h).view(bsz, seq_len, self.n_heads, self.d_head)
        k = self.W_k(h).view(bsz, seq_len, self.n_heads, self.d_head)
        v = self.W_v(h).view(bsz, seq_len, self.n_heads, self.d_head)

        # [batch, n_heads, seq, seq]
        attn_scores = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(self.d_head)
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask  # mask should contain 0 or -inf
        attn_weights = F.softmax(attn_scores, dim=-1)

        # [batch, seq, n_heads, d_head]
        context = torch.einsum("bhts,bshd->bthd", attn_weights, v)

        # Per-head scaling: scale z for selected heads at this layer
        if layer_head_config:
            for (layer, head_idx), alpha in layer_head_config.items():
                if layer == layer_idx and 0 <= head_idx < self.n_heads:
                    context[:, :, head_idx, :] *= alpha

        # Merge heads
        context = context.reshape(bsz, seq_len, self.d_model)
        attn_out = self.W_o(context)
        x = x + attn_out

        # MLP block
        x = x + self.mlp(self.ln2(x))
        return x


class Stage1ATransformer(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_mlp)
                for _ in range(cfg.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Causal mask: [1, 1, T, T]
        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("attn_mask", (mask == 0).float() * -1e9, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        layer_head_config: Dict[Tuple[int, int], float] | None = None,
    ) -> torch.Tensor:
        # input_ids: [batch, seq]
        bsz, seq_len = input_ids.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError("Sequence length exceeds max_seq_len.")

        device = input_ids.device
        pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, seq_len)

        x = self.token_emb(input_ids) + self.pos_emb(pos)
        attn_mask = self.attn_mask[:, :, :seq_len, :seq_len]

        for layer_idx, block in enumerate(self.blocks):
            x = block(x, layer_idx, layer_head_config, attn_mask)

        x = self.ln_f(x)
        logits = self.unembed(x)  # [batch, seq, vocab]
        return logits


def make_induction_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate a batch of simple ABAB... induction sequences.

    For each sequence:
        - Sample tokens a, b in {1..vocab_size-1}, a != b
        - Construct [a, b, a, b, ..., a, b] of length seq_len (even)
    """
    if seq_len % 2 != 0:
        raise ValueError("seq_len must be even for AB pattern.")

    a = torch.randint(1, vocab_size, (batch_size, 1), device=device)
    # sample b != a
    offset = torch.randint(1, vocab_size - 1, (batch_size, 1), device=device)
    b = (a + offset) % (vocab_size - 1) + 1

    pair = torch.cat([a, b], dim=1)  # [batch, 2]
    reps = seq_len // 2
    tokens = pair.repeat(1, reps)  # [batch, seq_len]
    return tokens


def train_stage1a(
    args: argparse.Namespace,
    cfg: TrainConfig,
) -> None:
    # Device selection
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    # Reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    model = Stage1ATransformer(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # Build layer->head->alpha config
    layer_head_config: Dict[Tuple[int, int], float] = {}
    if args.cond != "baseline" and abs(args.omega - 1.0) > 1e-6:
        # Stage 1A: only layer 0 is scaled in this toy model
        layer_head_config[(0, int(args.head))] = float(args.omega)

    # Run name and output dir
    run_name = f"stage1a_{args.cond}_head{args.head}_omega{args.omega}_seed{args.seed}"
    out_dir = Path("reports/pilot_stage1a/train") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    full_cfg = {
        "train_config": asdict(cfg),
        "args": vars(args),
        "device": str(device),
        "layer_head_config": {f"{k[0]}_{k[1]}": v for k, v in layer_head_config.items()},
    }
    (out_dir / "config.json").write_text(json.dumps(full_cfg, indent=2))

    metrics_path = out_dir / "metrics.jsonl"
    metrics_f = metrics_path.open("w", encoding="utf-8")

    print(
        f"[stage1a-train] Starting run {run_name} on {device} "
        f"(steps={cfg.n_steps}, batch_size={cfg.batch_size}, seq_len={cfg.max_seq_len})"
    )
    print(f"[stage1a-train] layer_head_config={layer_head_config}")

    start_time = time.time()
    for step in range(1, cfg.n_steps + 1):
        model.train()
        input_ids = make_induction_batch(
            batch_size=cfg.batch_size,
            seq_len=cfg.max_seq_len,
            vocab_size=cfg.vocab_size,
            device=device,
        )

        logits = model(input_ids, layer_head_config)
        # Next-token prediction loss: predict token t+1 from position t
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, cfg.vocab_size),
            input_ids[:, 1:].reshape(-1),
        )

        with torch.no_grad():
            preds = logits[:, :-1].argmax(dim=-1)
            acc = (preds == input_ids[:, 1:]).float().mean().item()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % cfg.log_every == 0 or step == 1:
            elapsed = time.time() - start_time
            msg = {
                "step": step,
                "loss": float(loss.item()),
                "accuracy": acc,
                "elapsed_s": elapsed,
            }
            print(
                f"[step {step:5d}] loss={msg['loss']:.4f}, acc={msg['accuracy']:.3f}, "
                f"elapsed={elapsed/60:.1f} min"
            )
            metrics_f.write(json.dumps(msg) + "\n")
            metrics_f.flush()

    metrics_f.close()

    # Save model weights
    torch.save(model.state_dict(), out_dir / "model.pt")
    print(f"[stage1a-train] Run complete. Outputs in {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-1A 2-layer pilot trainer.")
    parser.add_argument(
        "--cond",
        type=str,
        choices=["baseline", "suppressor", "random"],
        required=True,
        help="Condition label (baseline/suppressor/random).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for this run.",
    )
    parser.add_argument(
        "--omega",
        type=float,
        required=True,
        help="Scaling factor for the targeted head (ignored for baseline if ==1.0).",
    )
    parser.add_argument(
        "--head",
        type=int,
        required=True,
        help="Head index to scale (0-based) in layer 0.",
    )
    parser.add_argument(
        "--head-kind",
        type=str,
        choices=["baseline", "suppressor", "random"],
        required=True,
        help="Semantic label for the selected head (for logging only).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5000,
        help="Number of training steps.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Device to use ("auto", "cuda", "cpu", etc.).',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        batch_size=args.batch_size,
        n_steps=args.steps,
        lr=args.lr,
        device=args.device,
    )
    train_stage1a(args, cfg)


if __name__ == "__main__":
    main()

