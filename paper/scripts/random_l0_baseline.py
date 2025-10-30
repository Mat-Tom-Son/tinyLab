import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "paper" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

h1_runs = {
    'facts': 'h1_cross_condition_physics_balanced_facts_ec98a447a635',
    'neg': 'h1_cross_condition_physics_balanced_neg_2f50f1d455db',
    'cf': 'h1_cross_condition_physics_balanced_cf_ccea18594296',
    'logic': 'h1_cross_condition_physics_balanced_logic_a8830c85dd3d'
}

pairs_runs = {
    'facts': 'h5_layer0_pairs_balanced_facts_082fd4192397',
    'neg': 'h5_layer0_pairs_balanced_neg_8a8c3499827a',
    'cf': 'h5_layer0_pairs_balanced_cf_d5f81639b85d',
    'logic': 'h5_layer0_pairs_balanced_logic_d40e627e439d'
}

triplet_runs = {
    'facts': 'h5_layer0_triplet_balanced_facts_e5134e079282',
    'neg': 'h5_layer0_triplet_balanced_neg_898b1e89b906',
    'cf': 'h5_layer0_triplet_balanced_cf_d25f7f77aba5',
    'logic': 'h5_layer0_triplet_balanced_logic_7380df62633b'
}


def summary_value(path: Path) -> float:
    data = json.loads(path.read_text())
    return data['logit_diff']['mean']


def load_baselines() -> Dict[str, float]:
    baselines = {}
    for tag, run in h1_runs.items():
        summary_path = ROOT / "lab" / "runs" / run / "metrics" / "summary.json"
        baselines[tag] = summary_value(summary_path)
    return baselines


def collect_single_deltas(baselines: Dict[str, float]) -> pd.DataFrame:
    frames = []
    for tag, run in h1_runs.items():
        parquet = ROOT / "lab" / "runs" / run / "metrics" / "head_impact.parquet"
        df = pd.read_parquet(parquet)
        subset = df[(df.metric == 'logit_diff') & (df.layer == 0)].copy()
        subset['delta'] = subset['value'] - baselines[tag]
        subset['tag'] = tag
        subset['head'] = subset['head'].astype(str)
        frames.append(subset[['tag', 'head', 'delta']])
    return pd.concat(frames, ignore_index=True)


def collect_pair_means(baselines: Dict[str, float]) -> Dict[str, float]:
    pair_deltas: Dict[str, List[float]] = {}
    for tag, run in pairs_runs.items():
        parquet = ROOT / "lab" / "runs" / run / "metrics" / "head_impact.parquet"
        df = pd.read_parquet(parquet)
        subset = df[(df.metric == 'logit_diff') & (df.layer == 0)].copy()
        subset['delta'] = subset['value'] - baselines[tag]
        for _, row in subset.iterrows():
            pair_deltas.setdefault(row['head'], []).append(row['delta'])
    return {pair: float(np.mean(vals)) for pair, vals in pair_deltas.items()}


def collect_triplet_delta(baselines: Dict[str, float]) -> float:
    deltas = []
    for tag, run in triplet_runs.items():
        summary_path = ROOT / "lab" / "runs" / run / "metrics" / "summary.json"
        deltas.append(summary_value(summary_path) - baselines[tag])
    return float(np.mean(deltas))


def main() -> None:
    baselines = load_baselines()

    singles = collect_single_deltas(baselines)
    suppressor_heads = {'2', '4', '7'}
    baseline_single = singles[~singles['head'].isin(suppressor_heads)]['delta'].to_numpy()
    suppressor_means = singles[singles['head'].isin(suppressor_heads)].groupby('head')['delta'].mean().to_dict()

    rng = np.random.default_rng(0)
    sim_pairs = []
    for _ in range(1000):
        sample = rng.choice(baseline_single, size=2, replace=False)
        sim_pairs.append(sample.sum())
    sim_pairs = np.array(sim_pairs)

    pair_means = collect_pair_means(baselines)
    triplet_mean = collect_triplet_delta(baselines)

    single_percentiles = {head: float((baseline_single < val).mean()) for head, val in suppressor_means.items()}
    pair_percentiles = {pair: float((sim_pairs < val).mean()) for pair, val in pair_means.items()}
    triplet_percentile = float((sim_pairs < triplet_mean).mean())

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    sorted_single = np.sort(baseline_single)
    ecdf = np.linspace(0, 1, len(sorted_single), endpoint=False)
    axes[0].plot(sorted_single, ecdf, label='Random L0 head')
    q95, q99 = np.quantile(sorted_single, [0.95, 0.99])
    axes[0].axvline(q95, color='gray', linestyle=':', label='95th pct.')
    axes[0].axvline(q99, color='gray', linestyle='-.', label='99th pct.')
    for head, val in suppressor_means.items():
        axes[0].axvline(val, linestyle='--', label=f'Head {head}: {val:+.3f}')
    axes[0].set_xlabel('ΔLD (single head ablation; ↑ better factual preference)')
    axes[0].set_ylabel('ECDF')
    axes[0].set_title('Single head ablations')
    axes[0].legend(fontsize=7, loc='lower right')

    sorted_pairs = np.sort(sim_pairs)
    ecdf_pairs = np.linspace(0, 1, len(sorted_pairs), endpoint=False)
    axes[1].plot(sorted_pairs, ecdf_pairs, label='Random L0 pair (resampled)')
    pq95, pq99 = np.quantile(sorted_pairs, [0.95, 0.99])
    axes[1].axvline(pq95, color='gray', linestyle=':', label='95th pct.')
    axes[1].axvline(pq99, color='gray', linestyle='-.', label='99th pct.')
    for pair, val in pair_means.items():
        axes[1].axvline(val, linestyle='--', label=f'Pair {pair}: {val:+.3f}')
    axes[1].axvline(triplet_mean, color='black', linestyle='-', label=f'Triplet 2-4-7: {triplet_mean:+.3f}')
    axes[1].set_xlabel('ΔLD (pair sum; ↑ better factual preference)')
    axes[1].set_ylabel('ECDF')
    axes[1].set_title('Pair ablations (simulated)')
    axes[1].legend(fontsize=7, loc='lower right')

    fig.tight_layout()
    fig_path = OUTPUT_DIR / 'random_l0_baseline.pdf'
    fig.savefig(fig_path)

    metrics = {
        'single_percentiles': single_percentiles,
        'pair_percentiles': pair_percentiles,
        'triplet_percentile': triplet_percentile,
        'triplet_mean_delta': triplet_mean,
        'pair_means': pair_means,
        'single_means': suppressor_means,
        'quantiles': {
            'single_95': float(q95),
            'single_99': float(q99),
            'pair_95': float(pq95),
            'pair_99': float(pq99)
        }
    }
    metrics_path = ROOT / 'paper' / 'supplement' / 'random_l0_baseline.json'
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f'Wrote {fig_path}')
    print(f'Wrote {metrics_path}')


if __name__ == "__main__":
    main()
