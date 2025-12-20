
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import json
import argparse
import random
import os
import numpy as np
from dataclasses import dataclass
import math

# Reusing ParityTransformer Architecture exactly to isolate Task variable
@dataclass
class Config:
    vocab_size: int = 30 # 0-9 keys, 10-19 values, 20-22 special
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_mlp: int = 128
    max_seq_len: int = 12 
    
    batch_size: int = 256
    n_steps: int = 20000
    lr: float = 1e-3
    weight_decay: float = 1.0
    device: str = "auto"
    log_every: int = 100

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_mlp),
            nn.ReLU(),
            nn.Linear(cfg.d_mlp, cfg.d_model)
        )
        
    def forward(self, x):
        attn_out, weights = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, weights

class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)
    
    def forward(self, x):
        b, t = x.shape
        pos = torch.arange(t, device=x.device).unsqueeze(0)
        h = self.emb(x) + self.pos_emb(pos)
        
        attns = []
        for block in self.blocks:
            h, w = block(h)
            attns.append(w)
            
        logits = self.head(h)
        return logits, attns

def compute_edi(attns):
    # attn shape: [B, H, T, T] or [B, T, T] depending on implementation
    # nn.MultiheadAttention returns [B, H, T, T] if we enable it? 
    # Actually nn.MultiheadAttention returns [B, T, T] (averaged) or [B, H, T, T] if strictly processed.
    # Wait, nn.MultiheadAttention returns (attn_output, attn_output_weights).
    # attn_output_weights is [B, T, T] (averaged over heads) by default unles average_attn_weights=False
    # We need per-head.
    pass 
    # Actually, let's stick to the manual implementation from train_parity.py to ensure we measure what we think we measure.
    # The pytorch MHA module averages weights by default which destroys our measurement.
    
# Copying valid manual block from train_parity.py for consistency
class ManualBlock(nn.Module):
    def __init__(self, cfg):
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
    
    def forward(self, x):
        b, t, d = x.shape
        h = self.ln1(x)
        q = self.W_q(h).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(h).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_v(h).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        att = F.softmax(att, dim=-1) # [B, H, T, T]
        
        out = att @ v # [B, H, T, D_h]
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        
        x = x + self.W_o(out)
        x = x + self.mlp(self.ln2(x))
        return x, att

class ManualTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([ManualBlock(cfg) for _ in range(cfg.n_layers)])
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)
        
    def forward(self, x):
        b, t = x.shape
        pos = torch.arange(t, device=x.device).unsqueeze(0)
        h = self.emb(x) + self.pos_emb(pos)
        
        attns = []
        for block in self.blocks:
            h, w = block(h)
            attns.append(w) # [B, H, T, T]
            
        logits = self.head(h)
        return logits, attns

# Data Generation
class AssociativeDataset(Dataset):
    def __init__(self, size=1000, seq_len=6):
        self.data = []
        for _ in range(size):
            # Keys: 0-9
            # Vals: 10-19
            pairs = [(k, k+10) for k in range(10)]
            random.shuffle(pairs)
            
            # Select subset
            chosen = pairs[:seq_len//2]
            # Flatten: k1, v1, k2, v2...
            seq = []
            for k, v in chosen:
                seq.extend([k, v])
            
            # Query one of the keys present
            q_idx = random.randint(0, (seq_len//2)-1)
            q_key = chosen[q_idx][0]
            target = chosen[q_idx][1]
            
            # Input: seq + query
            inp = seq + [q_key]
            self.data.append((torch.tensor(inp), torch.tensor(target)))
            
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def compute_edi_metric(attns):
    # attns: List of [B, H, T, T]
    # We want EDI of Layer 0 (or all).
    # EDI = 1 - (entropy / max_entropy) normalized?
    # Or just Entropy?
    # The paper defines EDI ~ 0.61.
    # Let's compute simple ER (Effective Rank) of the attention matrix averaged over batch
    
    # Take Layer 0, Head 0 for simplicity, or average all
    # Let's compute global mean ER
    total_er = 0
    count = 0
    
    for layer_att in attns: # [B, H, T, T]
        # Average over batch to get "Mean Attention Matrix"
        mean_att = layer_att.mean(dim=0) # [H, T, T]
        for h in range(mean_att.shape[0]):
            matrix = mean_att[h] # [T, T]
            # SVD
            try:
                _, S, _ = torch.svd(matrix)
                p = S / S.sum()
                entropy = -torch.sum(p * torch.log(p + 1e-12))
                er = torch.exp(entropy).item()
                total_er += er
                count += 1
            except: pass
            
    return total_er / max(1, count)

def train(args):
    cfg = Config()
    cfg.seed = args.seed
    cfg.weight_decay = args.wd
    
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): device = torch.device("mps")
    
    train_ds = AssociativeDataset(size=5000)
    test_ds = AssociativeDataset(size=100)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    
    model = ManualTransformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    run_name = f"assoc_seed{cfg.seed}"
    log_dir = f"reports/assoc/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    f_log = open(f"{log_dir}/metrics.jsonl", "w")
    
    step = 0
    while step < cfg.n_steps:
        for x, y in train_loader:
            step += 1
            x, y = x.to(device), y.to(device)
            
            logits, attns = model(x)
            # Predict only last token
            last_logits = logits[:, -1, :] # [B, V]
            loss = F.cross_entropy(last_logits, y)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            if step % cfg.log_every == 0:
                acc = (last_logits.argmax(-1) == y).float().mean().item()
                edi = compute_edi_metric(attns)
                
                log = {"step": step, "loss": loss.item(), "acc": acc, "edi": edi}
                f_log.write(json.dumps(log) + "\n")
                f_log.flush()
                print(f"[step {step}] loss={loss.item():.3f} acc={acc:.3f} edi={edi:.3f}")
                
            if step >= cfg.n_steps: break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--wd", type=float, default=1.0)
    args = parser.parse_args()
    train(args)
