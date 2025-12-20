
import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import sys
import os
import json
import math

# Add local directory to path
sys.path.append(os.getcwd())
try:
    from scripts.train_stage1b_grokking import GrokkingTransformer, GrokkingConfig, prepare_batch, load_modular_data
except ImportError:
    sys.path.append(str(Path(os.getcwd()).parent))
    from scripts.train_stage1b_grokking import GrokkingTransformer, GrokkingConfig, prepare_batch, load_modular_data

def compute_edi_differentiable(attn_scores):
    # Same as calc_sensitivity.py
    import torch.nn.functional as F
    probs = F.softmax(attn_scores, dim=-1)
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    n_tokens = probs.shape[-1]
    h_max = math.log(n_tokens)
    return entropy / h_max

def get_sensitivity(model, input_ids, layer_idx=0):
    for p in model.parameters():
        p.requires_grad = True
    model.zero_grad()
    
    logits, layer_scores = model(input_ids, return_attention_scores=True)
    scores_l0 = layer_scores[layer_idx]
    # Head 0, Pos 5, Context 0..5
    target_scores = scores_l0[:, 0, 5, :6]
    
    edi_val = compute_edi_differentiable(target_scores)
    loss = edi_val.mean()
    loss.backward()
    
    total_norm = 0.0
    # Layer 0 params
    for param in model.blocks[layer_idx].parameters():
        if param.grad is not None:
             total_norm += param.grad.norm(2).item()**2
    return math.sqrt(total_norm)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="Run directory containing checkpoints")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    
    # Load Config
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg_dict = json.load(f)["config"]
        valid_keys = GrokkingConfig.__annotations__.keys()
        cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid_keys}
        cfg = GrokkingConfig(**cfg_dict)
    else:
        cfg = GrokkingConfig()
        
    # Load Data
    test_data = load_modular_data(Path(cfg.data_path_test))
    batch = test_data[:64]
    input_ids, targets = prepare_batch(batch, cfg, device)
    
    data_points = []
    
    # Scan checkpoints
    ckpts = sorted(list(ckpt_dir.glob("step_*.pt")))
    print(f"Found {len(ckpts)} checkpoints")
    
    model = GrokkingTransformer(cfg).to(device)
    
    for ckpt_path in ckpts:
        step = int(ckpt_path.stem.split("_")[1])
        # We sample points to be fast: 500, 1000...
        # Or just do all if small number
        
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            
            sens = get_sensitivity(model, input_ids)
            print(f"Step {step}: Sensitivity {sens:.4f}")
            data_points.append({"step": step, "sensitivity": sens})
        except Exception as e:
            print(f"Error loading {ckpt_path}: {e}")
            
    # Plot
    df = pd.DataFrame(data_points)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="step", y="sensitivity", marker="o")
    plt.axvline(x=3700, color='r', linestyle='--', label='Grokking Transition (~3700)')
    plt.title("Susceptibility Trajectory: ||∇ EDI|| over Training")
    plt.xlabel("Training Step")
    plt.ylabel("Gradient Norm (Layer 0)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = Path("paper/unified_homeostasis/figures/sensitivity_trajectory.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
