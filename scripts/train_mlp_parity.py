
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
from typing import List, Dict, Tuple

# Config
@dataclass
class MLPConfig:
    vocab_size: int = 4
    n_embd: int = 64
    hidden_size: int = 256
    n_layers: int = 3 # Input -> Hidden1 -> Hidden2 -> Output
    max_seq_len: int = 16 # Parity length (padded)
    
    # Training
    batch_size: int = 256
    n_steps: int = 40000
    lr: float = 1e-3
    weight_decay: float = 0.5 # adjusted for MLP
    seed: int = 1
    
    device: str = "auto"
    log_every: int = 100

# Model
class ParityMLP(nn.Module):
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.n_embd)
        
        # Simple MLP: Flatten(Embd) -> Linear -> ReLU -> ...
        self.input_dim = cfg.n_embd * cfg.max_seq_len
        
        self.layers = nn.ModuleList()
        prev_dim = self.input_dim
        
        for _ in range(cfg.n_layers - 1):
            self.layers.append(nn.Linear(prev_dim, cfg.hidden_size))
            self.layers.append(nn.ReLU())
            prev_dim = cfg.hidden_size
            
        self.head = nn.Linear(prev_dim, 2) # Binary classification
        
    def forward(self, x):
        b, t = x.shape
        pos = torch.arange(t, device=x.device).unsqueeze(0)
        
        # Embed + Pos
        h = self.emb(x) + self.pos_emb(pos) # [B, T, D]
        
        # Flatten
        h = h.view(b, -1) # [B, T*D]
        
        # MLP
        for layer in self.layers:
            h = layer(h)
            
        logits = self.head(h)
        return logits

def compute_er(matrix):
    """Compute Effective Rank of a matrix."""
    # Singular values
    try:
        _, S, _ = torch.svd(matrix)
        # Normalize
        p = S / S.sum()
        # Entropy
        entropy = -torch.sum(p * torch.log(p + 1e-12))
        return torch.exp(entropy).item()
    except:
        return 0.0

# Data
class ParityDataset(Dataset):
    def __init__(self, path):
        self.data = []
        with open(path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
                
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        seq = [int(c) for c in item['sequence']]
        
        # Pad to max_len (16 to be safe)
        max_len = 16 
        if len(seq) < max_len:
            seq = seq + [2] * (max_len - len(seq)) # 2 is PAD token
        else:
            seq = seq[:max_len]
            
        tgt = 0 if item['target'] == "EVEN" else 1
        return torch.tensor(seq, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

def train(args):
    cfg = MLPConfig()
    cfg.seed = args.seed
    cfg.weight_decay = args.wd
    
    # Setup
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): device = torch.device("mps")
    
    # Data
    # Use existing parity data
    train_ds = ParityDataset("data_parity/parity_train.jsonl")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_ds = ParityDataset("data_parity/parity_test.jsonl")
    
    # Model
    model = ParityMLP(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    # Logging
    run_name = f"mlp_seed{cfg.seed}_wd{cfg.weight_decay}"
    log_dir = f"reports/parity/mlp/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    metric_file = open(f"{log_dir}/metrics.jsonl", "w")
    
    print(f"Starting MLP training: {run_name}")
    
    step = 0
    while step < cfg.n_steps:
        for x, y in train_loader:
            step += 1
            x, y = x.to(device), y.to(device)
            
            # Forward
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            
            # Backward
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            if step % cfg.log_every == 0:
                # Test Acc
                model.eval()
                with torch.no_grad():
                    # Quick test on subset
                    test_x, test_y = next(iter(DataLoader(test_ds, batch_size=256)))
                    test_x, test_y = test_x.to(device), test_y.to(device)
                    acc = (model(test_x).argmax(-1) == test_y).float().mean().item()
                    
                # Compute ER of weights
                # W1 (First Linear)
                w1 = model.layers[0].weight
                er1 = compute_er(w1)
                
                # W2 (Second Linear)
                w2 = model.layers[2].weight
                er2 = compute_er(w2)
                
                metrics = {
                    "step": step,
                    "loss": loss.item(),
                    "test_acc": acc,
                    "ER_W1": er1,
                    "ER_W2": er2
                }
                metric_file.write(json.dumps(metrics) + "\n")
                metric_file.flush()
                
                print(f"[step {step}] loss={loss.item():.4f}, acc={acc:.3f}, er1={er1:.2f}, er2={er2:.2f}")
                
                model.train()
                
            if step >= cfg.n_steps: break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--wd", type=float, default=0.5)
    args = parser.parse_args()
    train(args)
