
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
import glob

# --- Model Definition (Similar to Grokking, slightly different config) ---

@dataclass
class ParityConfig:
    vocab_size: int = 4
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_mlp: int = 128
    max_seq_len: int = 32
    batch_size: int = 256
    device: str = "cpu"

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.W_q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_mlp),
            nn.GELU(),
            nn.Linear(cfg.d_mlp, cfg.d_model),
        )

    def forward(self, x, attn_mask=None):
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

class ParityTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        # Checkpoint has 'classifier.weight' of size [2, 64]
        # This means output dim is 2, even if vocab_size is 4.
        self.classifier = nn.Linear(cfg.d_model, 2, bias=True)
        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("attn_mask", (mask == 0).float() * -1e9, persistent=False)

    def forward(self, input_ids):
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, seq_len)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        attn_mask = self.attn_mask[:, :, :seq_len, :seq_len]
        
        all_attn_weights = []
        # Support both old GrokkingTransformer style (LayerHeadConfig) and simple style
        for block in self.blocks:
             # Try simple call first
             try:
                 x, attn_weights = block(x, attn_mask=attn_mask)
             except TypeError:
                 # Might be the GrokkingBlock signature
                 x, attn_weights = block(x, 0, attn_mask=attn_mask)
                 
             all_attn_weights.append(attn_weights)
            
        x = self.ln_f(x)
        logits = self.classifier(x)
        return logits, all_attn_weights

# --- Metrics ---

def effective_rank(matrix):
    try:
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        S_sum = S.sum(dim=-1, keepdim=True) + 1e-9
        p = S / S_sum
        entropy = -torch.sum(p * torch.log(p + 1e-9), dim=-1)
        return torch.exp(entropy)
    except:
        return torch.tensor(0.0)

def compute_edi(attn_weights):
    # attn_weights: [B, H, T, T]
    # Entropy over last dim (T)
    # H = -sum p log p
    p = attn_weights + 1e-9
    entropy = -torch.sum(p * torch.log(p), dim=-1) # [B, H, T]
    
    # We care about the entropy specifically at the last token position (where prediction happens)
    # or averaged over all positions?
    # Usually grokking papers measure attention entropy at the prediction token.
    # But let's look at average over sequence for robustness, or just last token.
    # Let's take mean over Batch and Head and T
    
    # EDI = H / H_max
    # H_max for causal attention on token t (0-indexed) is log(t+1)
    
    # Create H_max tensor [1, 1, T]
    seq_len = attn_weights.shape[-1]
    # Range 1..seq_len
    # device = attn_weights.device
    idx = torch.arange(1, seq_len + 1, device=attn_weights.device).float()
    h_max = torch.log(idx).view(1, 1, -1)
    
    # Avoid div by zero for first token (log(1)=0) -> set to 1.0 (EDI=0 there anyway)
    h_max[0, 0, 0] = 1.0
    
    edi = entropy / h_max
    return edi

def measure_checkpoint(ckpt_path, device):
    # Load config from parent
    p = Path(ckpt_path).parent.parent / "config.json"
    if p.exists():
        with open(p) as f:
            c = json.load(f)["config"]
            cfg = ParityConfig(**{k:v for k,v in c.items() if k in ParityConfig.__annotations__})
    else:
        cfg = ParityConfig() # default

    model = ParityTransformer(cfg).to(device)
    try:
        data = torch.load(ckpt_path, map_location=device)
        # Check keys
        # keys = data.keys()
        # print(f"Keys: {keys}")
        if "model_state" in data:
            model.load_state_dict(data["model_state"])
        elif "model" in data: # sometimes saved as 'model'
             model.load_state_dict(data["model"])
        else:
            # Maybe just state dict?
            model.load_state_dict(data)
            
    except Exception as e:
        print(f"Load failed {ckpt_path}: {e}")
        return None

    model.eval()
    
    # Synthetic Batch: Random 0/1 sequences
    # vocab: 0, 1. maybe 2=start, 3=pad?
    # The data file usually has arrays of ints.
    # Let's assume standard parity: random 0/1 sequence.
    # Based on config max_seq_len=32.
    
    batch_size = 64
    # Create random binary sequences
    # We'll use length 32 (max)
    inputs = torch.randint(0, 2, (batch_size, cfg.max_seq_len)).to(device)
    
    with torch.no_grad():
        logits, all_attn_weights = model(inputs)
    
    # Measure Metrics
    metrics = {}
    
    # EDI (Layer 0, Head 0 - usually the interesting one, but let's do Mean EDI)
    total_edi = 0
    count = 0
    
    for l, att in enumerate(all_attn_weights):
        # att: [B, H, T, T]
        # Focus on "causal" attention, so entropy varies by position t.
        # But we normalize by log(t+1) usually? Or log(seq_len)?
        # Our paper uses log(seq_len) for fixed size.
        
        edi = compute_edi(att) # [B, H, T]
        
        # Take mean over B, H, T
        mean_edi = edi.mean().item()
        
        metrics[f"L{l}_EDI"] = mean_edi
        
        # Also ER
        er = effective_rank(att) # [B, H]
        metrics[f"L{l}_ER"] = er.mean().item()
        
    metrics['step'] = data.get('step', -1)
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, default="reports/parity/train/*/checkpoints", help="Glob pattern for checkpoint directories")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    # Debug: Print the glob pattern
    pattern = args.pattern
    print(f"Searching pattern: {pattern}")
    
    roots = glob.glob(pattern)
    roots.sort()
    
    print(f"Found {len(roots)} checkpoint directories.")
    for r in roots:
        print(f"  - {r}")
    
    print(f"{'Seed':<5} | {'Step':<5} | {'L0 EDI':<8} | {'L1 EDI':<8} | {'L0 ER':<8}")
    print("-" * 50)
    
    results = []
    
    for root in roots:
        # Extract seed if possible, else just use hash or something
        try:
            seed = root.split("_seed")[1].split("_")[0]
        except:
            seed = "?"
            
        files = glob.glob(f"{root}/step_*.pt")
        # Just check the last few checkpoints to simulate "convergence"
        files.sort()
        
        # print(f"Root: {root}, Files: {len(files)}")
        
        if not files: 
            print(f"No files found in {root}")
            continue
        
        # Check last file
        last_file = files[-1]
        print(f"Processing {last_file}...")
        m = measure_checkpoint(last_file, device)
        if m:
            print(f"{seed:<5} | {m['step']:<5} | {m['L0_EDI']:.4f}   | {m['L1_EDI']:.4f}   | {m['L0_ER']:.4f}")
            results.append(m['L0_EDI'])

    if results:
        print("-" * 50)
        print(f"Mean L0 EDI: {sum(results)/len(results):.4f}")

if __name__ == "__main__":
    main()
