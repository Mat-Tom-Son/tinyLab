#!/usr/bin/env python3
"""Tiny OV-steer demo on GPT-2 (facts + neg).

Computes head 0:2 residual direction and injects α·v at blocks.0.hook_resid_post
for α in {-0.5, 0.0, +0.5}. Reports mean LD and ECE across prompts on facts and
negation probes, and renders a two-panel figure (LD/ECE vs α).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from cycler import cycler
import numpy as np
import torch
from transformer_lens import HookedTransformer

ROOT = Path(__file__).resolve().parents[2]
OUT_FIG = ROOT / "paper" / "figures" / "ov_steer_demo.pdf"
OUT_JSON = ROOT / "paper" / "supplement" / "ov_steer_demo.json"
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'savefig.dpi': 200,
    'lines.linewidth': 1.8,
})
mpl.rcParams['axes.unicode_minus'] = True
mpl.rcParams['axes.prop_cycle'] = cycler(color=[
    '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9', '#E69F00', '#000000'
])

FACTS = ROOT / "lab" / "data" / "corpora" / "facts_single_token_v1.jsonl"
NEG = ROOT / "lab" / "data" / "corpora" / "negation_single_token_v1.jsonl"


def load_jsonl(path: Path) -> List[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def head02_vector(model: HookedTransformer, texts: List[str]) -> torch.Tensor:
    tokens = model.to_tokens(texts)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    z = cache['z', 0][:, :, 2, :]  # [B, T, d_head]
    last = z[:, -1, :]  # [B, d_head]
    resid = last @ model.W_O[0, 2]  # [B, d_model]
    vec = resid.mean(dim=0)
    return vec / (vec.norm() + 1e-9)


def ece_from_logits(target_ids: np.ndarray, foil_ids: np.ndarray, logits: torch.Tensor, n_bins: int = 10) -> float:
    # logits: [B, V] at last position
    targ = logits[torch.arange(logits.size(0)), torch.from_numpy(target_ids)]
    foil = logits[torch.arange(logits.size(0)), torch.from_numpy(foil_ids)]
    diff = targ - foil
    probs = torch.sigmoid(diff).cpu().numpy()
    labels = (targ > foil).cpu().numpy().astype(int)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(probs, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        mask = idx == i
        if mask.any():
            acc = labels[mask].mean()
            conf = probs[mask].mean()
            ece += abs(acc - conf) * mask.mean()
    return float(ece)


def run_eval(model: HookedTransformer, rows: List[dict], alpha: float, vec: torch.Tensor) -> Tuple[float, float]:
    clean = [r['clean'] for r in rows]
    target_ids = np.array([model.to_single_token(r['target']) for r in rows])
    foil_ids = np.array([model.to_single_token(r['foil']) for r in rows])
    toks = model.to_tokens(clean)

    def steer_hook(act, hook):
        return act + alpha * vec

    with torch.no_grad():
        if alpha == 0.0:
            logits = model(toks)[:, -1, :]
        else:
            logits = model.run_with_hooks(toks, fwd_hooks=[('blocks.0.hook_resid_post', steer_hook)])[:, -1, :]
    targ = logits[torch.arange(logits.size(0)), torch.from_numpy(target_ids)]
    foil = logits[torch.arange(logits.size(0)), torch.from_numpy(foil_ids)]
    ld = float((targ - foil).mean())
    ece = ece_from_logits(target_ids, foil_ids, logits)
    return ld, ece


def main() -> None:
    model = HookedTransformer.from_pretrained('gpt2-medium', device='cpu', dtype=torch.float32)
    facts_rows = load_jsonl(FACTS)
    # Build vector from a subset of facts prompts for stability
    vec = head02_vector(model, [r['clean'] for r in facts_rows[:160]])

    results = {}
    for tag, path in [('facts', FACTS), ('neg', NEG)]:
        rows = load_jsonl(path)
        metrics = []
        for alpha in [-0.5, 0.0, 0.5]:
            ld, ece = run_eval(model, rows, alpha, vec)
            metrics.append({'alpha': alpha, 'ld': ld, 'ece': ece, 'n': len(rows)})
        results[tag] = metrics

    OUT_JSON.write_text(json.dumps(results, indent=2))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.2), sharex=True)
    for i, metric in enumerate(['ld', 'ece']):
        ax = axes[i]
        for tag in ['facts', 'neg']:
            data = results[tag]
            xs = [d['alpha'] for d in data]
            ys = [d[metric] for d in data]
            ax.plot(xs, ys, marker='o', label=tag.upper())
        ax.set_xlabel('α (residual steer)')
        ax.set_ylabel(metric.upper())
        ax.grid(True, alpha=0.3)
        ax.set_xticks([-0.5, -0.25, 0.0, 0.25, 0.5])
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        ax.set_xlim(-0.55, 0.55)
        if i == 0:
            ax.legend(frameon=False)
    fig.suptitle('OV-steer demo (GPT-2 Medium, head 0:2)')
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])
    fig.savefig(OUT_FIG)
    print(f'Wrote {OUT_FIG} and {OUT_JSON}')


if __name__ == '__main__':
    main()
