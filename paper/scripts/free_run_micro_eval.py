#!/usr/bin/env python3
"""Free-run micro-evaluation on GPT-2: hedge rate + factuality with/without L0 ablation.

Generates short completions for a small sample of prompts from each probe family,
under greedy and nucleus sampling, with and without suppressor head ablation.
Outputs a simple bar chart and a JSON summary.
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
OUTPUT_DIR = ROOT / "paper" / "figures"
SUPP_DIR = ROOT / "paper" / "supplement"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUPP_DIR.mkdir(parents=True, exist_ok=True)

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

LEXICON_PATH = ROOT / "data" / "lexicons" / "hedge_booster.json"


def load_jsonl(path: Path) -> List[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_token(s: str) -> str:
    import re
    s = s.strip().lower()
    return re.sub(r"[\W_]+", "", s)


def build_hooks():
    def hook_builder(layer: int):
        indices = [h for (l, h) in SUPPRESSOR_HEADS if l == layer]
        if not indices:
            return None

        def hook(z, hook):
            z[:, :, indices, :] = 0
            return z

        return hook

    hooks = [(f"blocks.{layer}.attn.hook_z", hook_builder(layer)) for layer, _ in SUPPRESSOR_HEADS]
    return hooks


def generate(model: HookedTransformer, prompt: str, max_new_tokens: int, nucleus_p: float | None, ablate: bool) -> str:
    toks = model.to_tokens(prompt, prepend_bos=True)
    out = toks
    hooks = build_hooks() if ablate else None
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if hooks:
                logits = model.run_with_hooks(out, fwd_hooks=hooks)[:, -1, :]
            else:
                logits = model(out)[:, -1, :]
            if nucleus_p is None:
                next_id = torch.argmax(logits, dim=-1).unsqueeze(-1)
            else:
                probs = torch.softmax(logits, dim=-1)
                # nucleus sampling
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum > nucleus_p
                mask[..., 0] = False
                sorted_probs[mask] = 0
                sorted_probs = sorted_probs / sorted_probs.sum()
                idx = torch.multinomial(sorted_probs, num_samples=1)
                next_id = sorted_indices.gather(-1, idx)
            out = torch.cat([out, next_id], dim=-1)
    gen = model.to_string(out[0])
    return gen[len(prompt):]


def main() -> None:
    model = HookedTransformer.from_pretrained('gpt2-medium', device='cpu', dtype=torch.float32)
    lex = json.loads(LEXICON_PATH.read_text())
    hedges = {normalize_token(t) for t in lex['hedges']}

    rng = np.random.default_rng(13)

    results = []
    for tag, path in DATASETS:
        rows = load_jsonl(path)
        # sample 25 prompts deterministically
        idx = rng.choice(len(rows), size=min(25, len(rows)), replace=False)
        sample = [rows[i] for i in idx]
        for ablate in [False, True]:
            for mode in [(None, 'greedy'), (0.9, 'nucleus')]:
                p, mode_name = mode
                hedge_hits, factual_hits = [], []
                for r in sample:
                    prompt = r['clean']
                    target = r['target']
                    cont = generate(model, prompt, max_new_tokens=6, nucleus_p=p, ablate=ablate)
                    # hedge rate
                    tokens = cont.split()
                    norm_tokens = [normalize_token(t) for t in tokens]
                    hedge = any(t in hedges for t in norm_tokens)
                    hedge_hits.append(int(hedge))
                    # factual correctness (first token exact match on normalized form)
                    factual = normalize_token(cont).startswith(normalize_token(target))
                    factual_hits.append(int(factual))
                results.append({
                    'tag': tag,
                    'mode': mode_name,
                    'ablate': ablate,
                    'hedge_rate': float(np.mean(hedge_hits)),
                    'factual_rate': float(np.mean(factual_hits)),
                    'n': len(sample),
                })

    # Save JSON
    (SUPP_DIR / 'free_run_micro_eval.json').write_text(json.dumps(results, indent=2))

    # Bar chart per metric
    for metric in ['hedge_rate', 'factual_rate']:
        fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.8), sharey=True)
        for i, (tag, _) in enumerate(DATASETS):
            ax = axes[i]
            rows = [r for r in results if r['tag'] == tag and r['mode'] == 'nucleus']
            rows = sorted(rows, key=lambda r: r['ablate'])
            vals = [r[metric] for r in rows]
            labels = ['baseline', 'ablated']
            ax.bar(labels, vals)
            ax.set_title(tag)
            ax.set_ylabel(metric.replace('_', ' '))
        fig.tight_layout()
        out = OUTPUT_DIR / f'free_run_{metric}.pdf'
        fig.savefig(out)
        print(f'Wrote {out}')


if __name__ == '__main__':
    main()
