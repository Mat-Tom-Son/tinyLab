
import argparse
import json
import math
import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

# Add local directory to path
sys.path.append(os.getcwd())
try:
    from scripts.train_stage1b_grokking import GrokkingTransformer, GrokkingConfig, prepare_batch, load_modular_data
except ImportError:
    sys.path.append(str(Path(os.getcwd()).parent))
    from scripts.train_stage1b_grokking import GrokkingTransformer, GrokkingConfig, prepare_batch, load_modular_data

def compute_edi_differentiable(attn_scores):
    """
    Compute EDI in a differentiable way using PyTorch.
    attn_scores: [Batch, Heads, Seg, Seq] (Logits or Scores before softmax? No, usually after softmax for EDI def)
    Wait, usually EDI is defined on probabilities.
    Input `attn_scores` here are pre-softmax scores from the model if we use the flag I added.
    Let's check the model: modified to return 'scores' (pre-softmax) or we can grab post-softmax.
    Actually, my modification returns `attn_scores` which are PRE-softmax (line 115 in original file).
    Line 118: `attn_weights = F.softmax(attn_scores, dim=-1)`
    
    The sensitivity we want is d(EDI)/d(Theta).
    EDI = Entropy(Softmax(Scores(Theta))) / H_max
    """
    # Softmax
    probs = F.softmax(attn_scores, dim=-1)
    
    # Entropy: -sum(p * log(p))
    # Add eps for stability
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    
    # Normalize by H_max
    n_tokens = probs.shape[-1]
    h_max = math.log(n_tokens)
    
    edi = entropy / h_max
    
    # Average over batch/heads/sequence if needed, or specific target
    return edi

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    device = torch.device(args.device)

    # Load checkpoint
    print(f"Loading checkpoint {args.ckpt}...")
    try:
        ckpt = torch.load(args.ckpt, map_location=device)
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {args.ckpt}")
        return

    # Config
    ckpt_path = Path(args.ckpt)
    config_path = ckpt_path.parent.parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg_dict = json.load(f)["config"]
        valid_keys = GrokkingConfig.__annotations__.keys()
        cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid_keys}
        cfg = GrokkingConfig(**cfg_dict)
    else:
        cfg = GrokkingConfig()

    model = GrokkingTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    # Set to eval mode but enable grad
    model.eval() 
    
    # Load Data
    test_data = load_modular_data(Path(cfg.data_path_test))
    # Use a fixed subset for consistent sensitivity measurement
    batch = test_data[:64]
    input_ids, targets = prepare_batch(batch, cfg, device)

    # We want gradients w.r.t parameters
    for p in model.parameters():
        p.requires_grad = True

    model.zero_grad()
    
    # Forward pass
    # We returned (logits, layer_scores)
    logits, layer_scores = model(input_ids, return_attention_scores=True)
    
    # Target: Layer 0, Head 0, Last Token (pos 5)
    # layer_scores[0]: [B, H, T, T]
    scores_l0 = layer_scores[0]
    
    # Focus on Head 0
    # Shape: [B, 6, 6] (assuming causal mask handled inside block, checking indices)
    # We want the attention distribution of the "Result" token (pos 5) attending to previous tokens
    target_scores = scores_l0[:, 0, 5, :6] # [B, 6] -> Attending to 0..5
    
    # Compute EDI
    edi_val = compute_edi_differentiable(target_scores)
    
    # We want sensitivity of the MEAN EDI over the batch
    # Scalar objective
    loss = edi_val.mean()
    
    print(f"Mean EDI: {loss.item():.4f}")
    
    # Backward
    loss.backward()
    
    # Collect Gradient Norms
    # We focus on W_q, W_k of Layer 0 as they directly shape attention
    # layer 0 is model.blocks[0]
    
    block0 = model.blocks[0]
    
    norms = {}
    total_norm = 0.0
    
    for name, param in block0.named_parameters():
        if param.grad is not None:
            # Frobenius norm
            grad_norm = param.grad.norm(2).item()
            norms[name] = grad_norm
            total_norm += grad_norm**2
            
    total_norm = math.sqrt(total_norm)
    
    print(f"\n[Sensitivity / Susceptibility]")
    print(f"Total Gradient Norm (Layer 0): {total_norm:.4f}")
    print(f"Breakdown:")
    for name in sorted(norms.keys()):
        print(f"  {name}: {norms[name]:.4f}")
        
    # Also Check Total Model Susceptibility
    full_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            full_norm += p.grad.norm(2).item()**2
    full_norm = math.sqrt(full_norm)
    print(f"Full Model Sensitivity: {full_norm:.4f}")

if __name__ == "__main__":
    main()
