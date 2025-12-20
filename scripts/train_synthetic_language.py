#!/usr/bin/env python3
"""
Synthetic Structured Language Experiment

Tests whether EDI homeostasis extends from discrete algorithmic tasks
to distributional language-like tasks.

Task: Structured narrative completion
Format: [the] SUBJ VERB [the] OBJ . [the] SUBJ VERB [to] LOC .

Hypothesis: Position-specific EDI equilibria (different positions 
stabilize to different values based on their K).
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    # Vocabulary
    n_subjects: int = 30      # tokens 0-29
    n_verbs: int = 20         # tokens 30-49
    n_objects: int = 30       # tokens 50-79
    n_locations: int = 20     # tokens 80-99
    # Special tokens: 100="the", 101="to", 102=".", 103="[PAD]"
    vocab_size: int = 104
    
    # Architecture
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_mlp: int = 128
    max_seq_len: int = 14     # 12 tokens + padding buffer
    
    # Training
    batch_size: int = 256
    n_steps: int = 20000
    lr: float = 1e-3
    weight_decay: float = 1.0
    
    # Logging
    log_every: int = 100
    checkpoint_every: int = 1000
    device: str = "auto"
    seed: int = 0
    output_dir: str = "outputs/synthetic_language"


# ============================================================================
# Data Generation
# ============================================================================

def generate_sentence(cfg: Config) -> List[int]:
    """
    Generate a structured sentence.
    Format: [the] SUBJ VERB [the] OBJ . [the] SUBJ VERB [to] LOC .
    
    Token positions (0-indexed):
    0: "the" (100)
    1: SUBJ (0-29)
    2: VERB (30-49)
    3: "the" (100)
    4: OBJ (50-79)
    5: "." (102)
    6: "the" (100)
    7: SUBJ (0-29)
    8: VERB (30-49)
    9: "to" (101)
    10: LOC (80-99)
    11: "." (102)
    """
    subj1 = random.randint(0, cfg.n_subjects - 1)
    verb1 = random.randint(0, cfg.n_verbs - 1) + cfg.n_subjects
    obj1 = random.randint(0, cfg.n_objects - 1) + cfg.n_subjects + cfg.n_verbs
    subj2 = random.randint(0, cfg.n_subjects - 1)
    verb2 = random.randint(0, cfg.n_verbs - 1) + cfg.n_subjects
    loc2 = random.randint(0, cfg.n_locations - 1) + cfg.n_subjects + cfg.n_verbs + cfg.n_objects
    
    tokens = [
        100,    # the
        subj1,  # SUBJ
        verb1,  # VERB
        100,    # the
        obj1,   # OBJ
        102,    # .
        100,    # the
        subj2,  # SUBJ
        verb2,  # VERB
        101,    # to
        loc2,   # LOC
        102,    # .
    ]
    return tokens


class SyntheticLanguageDataset(torch.utils.data.Dataset):
    """Dataset of structured sentences for next-token prediction."""
    
    def __init__(self, cfg: Config, size: int = 10000):
        self.cfg = cfg
        self.data = []
        for _ in range(size):
            tokens = generate_sentence(cfg)
            # Input: all tokens except last
            # Target: all tokens except first (shifted by 1)
            self.data.append((
                torch.tensor(tokens[:-1], dtype=torch.long),
                torch.tensor(tokens[1:], dtype=torch.long)
            ))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


# ============================================================================
# Model Architecture (with exposed attention for EDI tracking)
# ============================================================================

class ManualAttentionBlock(nn.Module):
    """Transformer block with manual attention to expose attention weights."""
    
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        
        self.W_q = nn.Linear(cfg.d_model, cfg.d_model)
        self.W_k = nn.Linear(cfg.d_model, cfg.d_model)
        self.W_v = nn.Linear(cfg.d_model, cfg.d_model)
        self.W_o = nn.Linear(cfg.d_model, cfg.d_model)
        
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_mlp),
            nn.ReLU(),
            nn.Linear(cfg.d_mlp, cfg.d_model)
        )
    
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Forward pass.
        
        Args:
            x: [B, T, D]
            return_attention: if True, also return attention weights
            
        Returns:
            output: [B, T, D]
            attention: [B, H, T, T] if return_attention else None
        """
        B, T, D = x.shape
        
        # Pre-norm
        normed = self.ln1(x)
        
        # QKV projections
        Q = self.W_q(normed).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(normed).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(normed).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Softmax
        attn = F.softmax(scores, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.W_o(out)
        
        # Residual
        x = x + out
        
        # MLP
        x = x + self.mlp(self.ln2(x))
        
        if return_attention:
            return x, attn
        return x


class SyntheticLanguageTransformer(nn.Module):
    """Transformer for synthetic language task with EDI tracking."""
    
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([ManualAttentionBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Forward pass.
        
        Args:
            x: [B, T] token IDs
            return_attention: if True, return attention weights from all layers
            
        Returns:
            logits: [B, T, V]
            attentions: List[[B, H, T, T]] if return_attention
        """
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        
        h = self.token_emb(x) + self.pos_emb(pos)
        
        attentions = []
        for block in self.blocks:
            if return_attention:
                h, attn = block(h, return_attention=True)
                attentions.append(attn)
            else:
                h = block(h, return_attention=False)
        
        h = self.ln_final(h)
        logits = self.head(h)
        
        if return_attention:
            return logits, attentions
        return logits


# ============================================================================
# EDI Computation
# ============================================================================

def compute_edi(attn: torch.Tensor) -> Dict[str, float]:
    """
    Compute Entropy Dispersion Index from attention weights.
    
    EDI = H(attention) / H_max = H(attention) / log(N)
    
    Args:
        attn: [B, H, T, T] attention weights
        
    Returns:
        Dict with per-head and mean EDI values
    """
    B, H, T, _ = attn.shape
    
    # Average over batch
    mean_attn = attn.mean(dim=0)  # [H, T, T]
    
    results = {}
    head_edis = []
    
    for h in range(H):
        # For each position, compute attention entropy
        # attn_h: [T, T] where each row is attention distribution for that position
        attn_h = mean_attn[h]  # [T, T]
        
        position_edis = []
        for pos in range(T):
            # Get attention distribution for this position
            # Only consider positions up to current (causal)
            valid_len = pos + 1
            p = attn_h[pos, :valid_len]
            
            # Compute entropy
            entropy = -torch.sum(p * torch.log(p + 1e-12))
            max_entropy = math.log(valid_len) if valid_len > 1 else 1.0
            edi = entropy.item() / max_entropy if max_entropy > 0 else 0.0
            position_edis.append(edi)
        
        mean_edi = np.mean(position_edis)
        head_edis.append(mean_edi)
        results[f"head_{h}"] = mean_edi
    
    results["mean"] = np.mean(head_edis)
    results["std"] = np.std(head_edis)
    
    return results


def compute_position_specific_edi(attn: torch.Tensor) -> Dict[str, float]:
    """
    Compute EDI per position (averaged over heads).
    
    This tests whether different positions stabilize to different EDI values.
    """
    B, H, T, _ = attn.shape
    mean_attn = attn.mean(dim=0)  # [H, T, T]
    
    results = {}
    
    for pos in range(T):
        valid_len = pos + 1
        if valid_len <= 1:
            results[f"pos_{pos}"] = 0.0
            continue
        
        pos_edis = []
        for h in range(H):
            p = mean_attn[h, pos, :valid_len]
            entropy = -torch.sum(p * torch.log(p + 1e-12)).item()
            max_entropy = math.log(valid_len)
            edi = entropy / max_entropy
            pos_edis.append(edi)
        
        results[f"pos_{pos}"] = np.mean(pos_edis)
    
    return results


# ============================================================================
# Training
# ============================================================================

def train(cfg: Config):
    """Main training loop."""
    
    # Setup
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    device = torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(cfg.output_dir) / f"seed_{cfg.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    
    # Create dataset and model
    print("Generating dataset...")
    train_dataset = SyntheticLanguageDataset(cfg, size=10000)
    test_dataset = SyntheticLanguageDataset(cfg, size=1000)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True
    )
    
    model = SyntheticLanguageTransformer(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    logs = []
    step = 0
    epoch = 0
    
    print("Starting training...")
    
    while step < cfg.n_steps:
        epoch += 1
        for inputs, targets in train_loader:
            if step >= cfg.n_steps:
                break
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward
            logits = model(inputs)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            step += 1
            
            # Logging
            if step % cfg.log_every == 0:
                model.eval()
                with torch.no_grad():
                    # Get attention weights
                    _, attentions = model(inputs, return_attention=True)
                    
                    # Compute train accuracy
                    preds = logits.argmax(dim=-1)
                    train_acc = (preds == targets).float().mean().item()
                    
                    # Compute EDI for each layer
                    layer_edis = {}
                    layer_pos_edis = {}
                    for layer_idx, attn in enumerate(attentions):
                        edi_stats = compute_edi(attn)
                        layer_edis[f"layer_{layer_idx}_mean_edi"] = edi_stats["mean"]
                        layer_edis[f"layer_{layer_idx}_std_edi"] = edi_stats["std"]
                        
                        pos_edis = compute_position_specific_edi(attn)
                        for k, v in pos_edis.items():
                            layer_pos_edis[f"layer_{layer_idx}_{k}"] = v
                    
                    # Test accuracy
                    test_inputs = test_dataset.data[0][0].unsqueeze(0).to(device)
                    test_targets = test_dataset.data[0][1].unsqueeze(0).to(device)
                    test_logits = model(test_inputs)
                    test_preds = test_logits.argmax(dim=-1)
                    # Simple test: sample a few
                    test_accs = []
                    for i in range(min(100, len(test_dataset))):
                        ti, tt = test_dataset.data[i]
                        ti, tt = ti.unsqueeze(0).to(device), tt.unsqueeze(0).to(device)
                        tl = model(ti)
                        tp = tl.argmax(dim=-1)
                        test_accs.append((tp == tt).float().mean().item())
                    test_acc = np.mean(test_accs)
                
                model.train()
                
                # Log entry
                log_entry = {
                    "step": step,
                    "loss": loss.item(),
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    **layer_edis,
                    **layer_pos_edis
                }
                logs.append(log_entry)
                
                print(f"Step {step:5d} | Loss: {loss.item():.4f} | "
                      f"Train: {train_acc:.3f} | Test: {test_acc:.3f} | "
                      f"L0 EDI: {layer_edis.get('layer_0_mean_edi', 0):.4f} | "
                      f"L1 EDI: {layer_edis.get('layer_1_mean_edi', 0):.4f}")
            
            # Checkpointing
            if step % cfg.checkpoint_every == 0:
                ckpt_path = output_dir / f"checkpoint_{step}.pt"
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, ckpt_path)
                
                # Save logs
                with open(output_dir / "logs.json", "w") as f:
                    json.dump(logs, f, indent=2)
    
    # Final save
    final_path = output_dir / "model_final.pt"
    torch.save(model.state_dict(), final_path)
    
    with open(output_dir / "logs.json", "w") as f:
        json.dump(logs, f, indent=2)
    
    print(f"\nTraining complete! Logs saved to {output_dir}")
    return logs


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_steps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--checkpoint_every", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="outputs/synthetic_language")
    args = parser.parse_args()
    
    cfg = Config(
        seed=args.seed,
        n_steps=args.n_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        output_dir=args.output_dir,
    )
    
    train(cfg)


if __name__ == "__main__":
    main()
