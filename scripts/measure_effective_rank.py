
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Tuple
import glob

# --- Model Definition ---

@dataclass
class GrokkingConfig:
    vocab_size: int = 114
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    d_mlp: int = 512
    max_seq_len: int = 16
    batch_size: int = 512
    n_steps: int = 3000
    lr: float = 1e-3
    weight_decay: float = 5.0
    grad_clip: float = 1.0
    optimizer: str = "AdamW"
    log_every: int = 100
    checkpoint_every: int = 500
    modulus: int = 113
    data_path_train: str = "data/modular_p113_train.jsonl"
    data_path_test: str = "data/modular_p113_test.jsonl"
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

    def forward(self, x, layer_idx, layer_head_config=None, attn_mask=None):
        bsz, seq_len, _ = x.shape
        h = self.ln1(x)
        q = self.W_q(h).view(bsz, seq_len, self.n_heads, self.d_head)
        k = self.W_k(h).view(bsz, seq_len, self.n_heads, self.d_head)
        v = self.W_v(h).view(bsz, seq_len, self.n_heads, self.d_head)
        
        attn_scores = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(self.d_head)
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        context = torch.einsum("bhts,bshd->bthd", attn_weights, v)
        context = context.reshape(bsz, seq_len, self.d_model)
        attn_out = self.W_o(context)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, attn_weights

class GrokkingTransformer(nn.Module):
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
        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("attn_mask", (mask == 0).float() * -1e9, persistent=False)

    def forward(self, input_ids):
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, seq_len)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        attn_mask = self.attn_mask[:, :, :seq_len, :seq_len]
        
        all_attn_weights = []
        for i, block in enumerate(self.blocks):
            x, attn_weights = block(x, i, attn_mask=attn_mask)
            all_attn_weights.append(attn_weights)
            
        x = self.ln_f(x)
        logits = self.unembed(x)
        logits = logits[:, :seq_len, :] # Ensure output matches sequence length
        return logits, all_attn_weights

# --- Metrics ---

def effective_rank(matrix):
    try:
        # matrix: [..., N, N]
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        S_sum = S.sum(dim=-1, keepdim=True) + 1e-9
        p = S / S_sum
        entropy = -torch.sum(p * torch.log(p + 1e-9), dim=-1)
        er = torch.exp(entropy)
        return er
    except Exception as e:
        # print(f"SVD Error: {e}")
        return torch.tensor(0.0)

def load_config(ckpt_path):
    # Try to find config.json in parent directories
    p = Path(ckpt_path).parent
    # Check parent (e.g. checkpoints/) -> parent (run_dir/)
    cfg_path = p.parent / "config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                data = json.load(f)
            # config is inside "config" key
            c = data.get("config", {})
            return GrokkingConfig(**c)
        except Exception as e:
            print(f"Error loading config {cfg_path}: {e}")
    return GrokkingConfig() # fallback

def measure_checkpoint(ckpt_path, device):
    cfg = load_config(ckpt_path)
    model = GrokkingTransformer(cfg).to(device)
    
    try:
        data = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(data["model_state"])
    except Exception as e:
        print(f"Failed to load {ckpt_path}: {e}")
        return None

    model.eval()
    
    # Synthetic batch
    batch_size = 64
    inputs = []
    mod = cfg.modulus
    for _ in range(batch_size):
        a = torch.randint(0, mod, (1,)).item()
        b = torch.randint(0, mod, (1,)).item()
        res = (a + b) % mod
        inputs.append([a, a, b, b, 0, res]) 
    
    input_ids = torch.tensor(inputs, device=device)
    
    with torch.no_grad():
        logits, all_attn_weights = model(input_ids)

    metrics = {}
    
    # Global ER average
    all_er = []
    
    for layer_idx, att in enumerate(all_attn_weights):
        # att: [batch, heads, seq, seq]
        er = effective_rank(att) # [batch, heads]
        er_mean = er.mean(dim=0) # [heads]
        
        for h in range(len(er_mean)):
            val = er_mean[h].item()
            metrics[f"L{layer_idx}H{h}_ER"] = val
            all_er.append(val)
            
    metrics["Mean_ER"] = sum(all_er) / len(all_er) if all_er else 0.0
    metrics["circularity"] = data.get("circularity", 0.0)
    metrics["step"] = data.get("step", 0)
    metrics["test_acc"] = data.get("test_acc", 0.0)
    
    return metrics, cfg

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    roots = glob.glob("reports/stage1b_grokking/train/stage1b_head0_omega1.0_seed*/checkpoints")
    roots.sort()
    
    print(f"{'Seed':<5} | {'Step':<5} | {'TestAcc':<7} | {'Circ':<6} | {'Mean ER':<7} | {'L0H0 ER':<7} | {'L0H1 ER':<7}")
    print("-" * 70)
    
    for root in roots:
        seed = root.split("seed")[-1].split("/")[0]
        files = glob.glob(f"{root}/step_*.pt")
        files.sort()
        
        results = []
        for f in files:
            m, _ = measure_checkpoint(f, device)
            if m:
                results.append(m)
        
        results.sort(key=lambda x: x["step"])
        
        for r in results:
            print(f"{seed:<5} | {r['step']:<5} | {r.get('test_acc',0):<7.3f} | {r['circularity']:<6.3f} | {r['Mean_ER']:<7.3f} | {r.get('L0H0_ER', 0):<7.3f} | {r.get('L0H1_ER', 0):<7.3f}")
        print("-" * 70)

if __name__ == "__main__":
    main()
