import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformer_lens import HookedTransformer

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / 'paper' / 'figures'
SUPPLEMENT = ROOT / 'paper' / 'supplement'
DATASETS = [
    ('facts', ROOT / 'lab' / 'data' / 'corpora' / 'facts_single_token_v1.jsonl'),
    ('neg', ROOT / 'lab' / 'data' / 'corpora' / 'negation_single_token_v1.jsonl'),
    ('cf', ROOT / 'lab' / 'data' / 'corpora' / 'counterfactual_single_token_v1.jsonl'),
    ('logic', ROOT / 'lab' / 'data' / 'corpora' / 'logical_single_token_v1.jsonl')
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUPPLEMENT.mkdir(parents=True, exist_ok=True)

model = HookedTransformer.from_pretrained('gpt2-medium', device='cpu', dtype=torch.float32)
SUPPRESSOR_HEADS = {(0, 2), (0, 4), (0, 7)}


def zero_hook_builder(layer: int):
    indices = [head for (l, head) in SUPPRESSOR_HEADS if l == layer]
    if not indices:
        return None

    def hook(z, hook):
        z[:, :, indices, :] = 0
        return z

    return hook

HOOKS = [(f'blocks.{layer}.attn.hook_z', zero_hook_builder(layer)) for layer, _ in SUPPRESSOR_HEADS]


def load_dataset(path: Path) -> List[dict]:
    items = []
    with path.open() as f:
        for line in f:
            items.append(json.loads(line))
    return items


def to_token_id(token: str) -> int:
    return model.to_single_token(token)


records = []
for tag, path in DATASETS:
    dataset = load_dataset(path)
    for rec in dataset:
        clean = rec['clean']
        target_id = to_token_id(rec['target'])
        foil_id = to_token_id(rec['foil'])
        tokens = model.to_tokens(clean, prepend_bos=True)
        with torch.no_grad():
            logits = model(tokens)[:, -1, :]
            logits_zero = model.run_with_hooks(tokens, fwd_hooks=HOOKS)[:, -1, :]
        base_diff = logits[0, target_id] - logits[0, foil_id]
        ablated_diff = logits_zero[0, target_id] - logits_zero[0, foil_id]
        base_prob = torch.sigmoid(base_diff).item()
        ablated_prob = torch.sigmoid(ablated_diff).item()
        base_correct = logits[0, target_id] > logits[0, foil_id]
        ablated_correct = logits_zero[0, target_id] > logits_zero[0, foil_id]
        records.append({'condition': 'baseline', 'prob': base_prob, 'correct': int(base_correct)})
        records.append({'condition': 'ablated', 'prob': ablated_prob, 'correct': int(ablated_correct)})

bins = np.linspace(0, 1, 11)
centers = 0.5 * (bins[1:] + bins[:-1])

fig, ax = plt.subplots(figsize=(4, 3))
for condition in ['baseline', 'ablated']:
    subset = [r for r in records if r['condition'] == condition]
    probs = np.array([r['prob'] for r in subset])
    labels = np.array([r['correct'] for r in subset])
    bin_indices = np.digitize(probs, bins) - 1
    bin_acc = []
    bin_conf = []
    for i in range(len(centers)):
        mask = bin_indices == i
        if mask.any():
            bin_acc.append(labels[mask].mean())
            bin_conf.append(probs[mask].mean())
        else:
            bin_acc.append(np.nan)
            bin_conf.append(np.nan)
    ax.plot(bin_conf, bin_acc, marker='o', label=condition)

ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
ax.set_xlabel('Predicted probability (target)')
ax.set_ylabel('Empirical accuracy')
ax.set_title('Calibration on probe suite (10 bins)')
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'calibration_curve.pdf')

metrics = {}
for condition in ['baseline', 'ablated']:
    subset = [r for r in records if r['condition'] == condition]
    probs = np.array([r['prob'] for r in subset])
    labels = np.array([r['correct'] for r in subset])
    bin_indices = np.digitize(probs, bins) - 1
    ece = 0.0
    for i in range(len(centers)):
        mask = bin_indices == i
        if mask.any():
            acc = labels[mask].mean()
            conf = probs[mask].mean()
            ece += np.abs(acc - conf) * mask.sum() / len(probs)
    brier = float(np.mean((probs - labels) ** 2))
    clipped = np.clip(probs, 1e-6, 1 - 1e-6)
    nll = float(np.mean(-labels * np.log(clipped) - (1 - labels) * np.log(1 - clipped)))
    metrics[condition] = {
        'ece': float(ece),
        'brier': brier,
        'nll': nll
    }

(SUPPLEMENT / 'calibration_metrics.json').write_text(json.dumps(metrics, indent=2))
print('Wrote calibration curve and metrics')
