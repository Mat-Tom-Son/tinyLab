"""Scan reports/ and lab/runs/ to produce a RESULTS_MANIFEST.json.

Captures per-model, per-probe pointers to:
- summary table CSV
- trio percentiles JSON
- head ranking CSVs (full + top/bottom slices)
- OV token reports (if present)
"""

from __future__ import annotations

import json
from pathlib import Path


def collect(prefix: str) -> dict:
    r = Path('reports')
    out = {
        'summary_table': str(r / f'{prefix}_summary_table.csv') if (r / f'{prefix}_summary_table.csv').exists() else None,
        'trio_percentiles': str(r / f'{prefix}_trio_percentiles.json') if (r / f'{prefix}_trio_percentiles.json').exists() else None,
        'head_rankings': {},
        'ov_reports': [],
    }
    for probe in ['facts','neg','cf','logic']:
        full = r / f'{prefix}_{probe}_head_ranking.csv'
        top = r / f'{prefix}_{probe}_top12.csv'
        bot = r / f'{prefix}_{probe}_bottom6.csv'
        if full.exists():
            out['head_rankings'][probe] = {
                'full': str(full),
                'top': str(top) if top.exists() else None,
                'bottom': str(bot) if bot.exists() else None,
            }
    # OV: list any that include the prefix in filename
    for p in r.glob('ov_report_*json'):
        if prefix in p.name:
            out['ov_reports'].append(str(p))
    return out


def main() -> None:
    manifest = {
        'mistral': collect('mistral'),
        'gpt2_medium': collect('gpt2_medium'),
        'gpt2_large': collect('gpt2_large'),
        'gpt2m': collect('gpt2m'),
        'gpt2l': collect('gpt2l'),
        'gpt2': collect('gpt2'),
    }
    out = Path('reports/RESULTS_MANIFEST.json')
    out.write_text(json.dumps(manifest, indent=2))
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
