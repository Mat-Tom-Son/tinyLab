
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

@dataclass
class RNNConfig:
    vocab_size: int = 4
    n_embd: int = 64
    hidden_size: int = 256
    n_layers: int = 2
    max_seq_len: int = 16 
    
    batch_size: int = 256
    n_steps: int = 20000
    lr: float = 1e-3
    weight_decay: float = 0.5
    device: str = "auto"
    log_every: int = 100

class ParityRNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # Using LSTM as the representative RNN
        self.rnn = nn.LSTM(
            input_size=cfg.n_embd,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.n_layers,
            batch_first=True
        )
        self.head = nn.Linear(cfg.hidden_size, 2)
        
    def forward(self, x):
        h = self.emb(x)
        out, (hn, cn) = self.rnn(h)
        # Use last hidden state
        # out: [B, T, H]
        last_out = out[:, -1, :]
        logits = self.head(last_out)
        return logits

def compute_er(matrix):
    try:
        _, S, _ = torch.svd(matrix)
        p = S / S.sum()
        entropy = -torch.sum(p * torch.log(p + 1e-12))
        return torch.exp(entropy).item()
    except:
        return 0.0

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
        max_len = 16
        if len(seq) < max_len:
            seq = seq + [2] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        tgt = 0 if item['target'] == "EVEN" else 1
        return torch.tensor(seq, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

def train(args):
    cfg = RNNConfig()
    cfg.seed = args.seed
    cfg.weight_decay = args.wd
    
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): device = torch.device("mps")
    
    train_ds = ParityDataset("data_parity/parity_train.jsonl")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    
    model = ParityRNN(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    run_name = f"rnn_seed{cfg.seed}"
    log_dir = f"reports/parity/rnn/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    f_log = open(f"{log_dir}/metrics.jsonl", "w")
    
    step = 0
    while step < cfg.n_steps:
        for x, y in train_loader:
            step += 1
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            if step % cfg.log_every == 0:
                acc = (logits.argmax(-1) == y).float().mean().item()
                # Compute ER of LSTM weights (Input-Hidden)
                # weight_ih_l0: [4*H, I]
                w_ih = model.rnn.weight_ih_l0
                # weight_hh_l0: [4*H, H]
                w_hh = model.rnn.weight_hh_l0
                
                er_in = compute_er(w_ih)
                er_rec = compute_er(w_hh)
                
                log = {"step": step, "loss": loss.item(), "acc": acc, "er_in": er_in, "er_rec": er_rec}
                f_log.write(json.dumps(log) + "\n")
                f_log.flush()
                print(f"[step {step}] loss={loss.item():.3f} acc={acc:.3f} er_in={er_in:.2f}")
                
            if step >= cfg.n_steps: break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--wd", type=float, default=0.5)
    args = parser.parse_args()
    train(args)
