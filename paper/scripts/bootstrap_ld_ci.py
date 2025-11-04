#!/usr/bin/env python3
"""Bootstrap 95% CIs over prompts for ΔLD (GPT-2 only).

Computes per-prompt logit-difference (target - foil) on clean prompts, with and
without simultaneous ablation of layer-0 heads {0:2, 0:4, 0:7}. Bootstraps the
mean ΔLD over prompts and emits a bar chart with error bars plus a JSON summary.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np
import torch
from transformer_lens import HookedTransformer

ROOT = Path(__file__).resolve().parents[2]
OUT_FIG = ROOT / "paper" / "figures" / "bootstrap_ld_ci.pdf"
OUT_JSON = ROOT / "paper" / "supplement" / "bootstrap_ld_ci.json"
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'savefig.dpi': 200,
    'lines.linewidth': 1.8,
})
mpl.rcParams['axes.prop_cycle'] = cycler(color=[
    '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9', '#E69F00', '#000000'
])

DATASETS = [
    ("facts", ROOT / "lab" / "data" / "corpora" / "facts_single_token_v1.jsonl"),
    ("neg", ROOT / "lab" / "data" / "corpora" / "negation_single_token_v1.jsonl"),
    ("cf", ROOT / "lab" / "data" / "corpora" / "counterfactual_single_token_v1.jsonl"),
    ("logic", ROOT / "lab" / "data" / "corpora" / "logical_single_token_v1.jsonl"),
]

SUPPRESSOR_HEADS = {(0, 2), (0, 4), (0, 7)}


def load_jsonl(path: Path) -> List[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def hook_builder(layer: int):
    indices = [h for (l, h) in SUPPRESSOR_HEADS if l == layer]
    def hook(z, hook):
        z = z.clone()
        z[:, :, indices, :] = 0
        return z
    return hook


HOOKS = [(f"blocks.{layer}.attn.hook_z", hook_builder(layer)) for layer, _ in SUPPRESSOR_HEADS]


def per_prompt_delta_ld(model: HookedTransformer, rows: List[dict]) -> np.ndarray:
    """Return ΔLD per prompt: (ablated LD - baseline LD)."""
    clean_texts = [r['clean'] for r in rows]
    target_ids = [model.to_single_token(r['target']) for r in rows]
    foil_ids = [model.to_single_token(r['foil']) for r in rows]

    toks = model.to_tokens(clean_texts)
    with torch.no_grad():
        base_logits = model(toks)[:, -1, :]
        abl_logits = model.run_with_hooks(toks, fwd_hooks=HOOKS)[:, -1, :]

    base = base_logits[torch.arange(base_logits.size(0)), torch.tensor(target_ids)] - \
           base_logits[torch.arange(base_logits.size(0)), torch.tensor(foil_ids)]
    abl = abl_logits[torch.arange(abl_logits.size(0)), torch.tensor(target_ids)] - \
          abl_logits[torch.arange(abl_logits.size(0)), torch.tensor(foil_ids)]
    delta = (abl - base).cpu().numpy()
    return delta


def bootstrap_ci(x: np.ndarray, n_boot: int = 5000, seed: int = 0) -> Tuple[float, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = len(x)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means.append(np.mean(x[idx]))
    means = np.array(means)
    mean = float(np.mean(x))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return mean, (float(lo), float(hi))


def main() -> None:
    model = HookedTransformer.from_pretrained('gpt2-medium', device='cpu', dtype=torch.float32)
    summary = {}
    means, los, his, labels = [], [], [], []
    for tag, path in DATASETS:
        rows = load_jsonl(path)
        delta = per_prompt_delta_ld(model, rows)
        mean, (lo, hi) = bootstrap_ci(delta, n_boot=5000, seed=13)
        summary[tag] = {
            'n_prompts': len(rows),
            'mean_delta_ld': mean,
            'ci95': [lo, hi],
        }
        labels.append(tag)
        means.append(mean)
        los.append(mean - lo)
        his.append(hi - mean)

    # Save JSON
    OUT_JSON.write_text(json.dumps(summary, indent=2))

    # Plot bars with error bars
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.bar(x, means, yerr=[los, his], capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in labels])
    ax.set_ylabel('ΔLD (ablated − baseline)', labelpad=8)
    ax.set_title('Bootstrap 95% CI over prompts (GPT-2 Medium)')
    fig.tight_layout()
    fig.savefig(OUT_FIG)
    print(f'Wrote {OUT_FIG} and {OUT_JSON}')


if __name__ == '__main__':
    main()
