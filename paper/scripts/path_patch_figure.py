import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

ROOT = Path(__file__).resolve().parents[2]
mpl.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'savefig.dpi': 200,
})
mpl.rcParams['axes.prop_cycle'] = cycler(color=[
    '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9', '#E69F00', '#000000'
])
PARTIAL = ROOT / 'reports' / 'facts_partial_summary.json'
REVERSE = ROOT / 'reports' / 'h6_reverse_patch_summary.json'

partial_data = json.loads(PARTIAL.read_text())
head2 = next(entry for entry in partial_data if entry['name'] == 'facts_partial_patch_head2')

baseline = head2['baseline_logit_diff']
zero = head2['zero_logit_diff']
patched = head2['patched_logit_diff']
full_delta = zero - baseline
remaining = patched - baseline
mediated = full_delta - remaining
fraction = mediated / full_delta if full_delta else 0.0

reverse_delta = json.loads(REVERSE.read_text())['facts']['reverse_delta']

fig, ax = plt.subplots(figsize=(6.2, 3.0))
ax.axis('off')

nodes = {
    'suppressor': (0.08, 0.5),
    'target': (0.48, 0.5),
    'logit': (0.88, 0.5)
}

box = dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='black', linewidth=2.0)
ax.text(*nodes['suppressor'], 'Layer 0\nsuppressor', ha='center', va='center', bbox=box)
ax.text(*nodes['target'], 'Layer 11\nresidual', ha='center', va='center', bbox=box)
ax.text(*nodes['logit'], 'Unembedding → logits', ha='center', va='center', bbox=box)

arrow = dict(arrowstyle='->', color='black', linewidth=2.0)
# Natural forward path (solid)
ax.annotate('', xy=nodes['target'], xytext=nodes['suppressor'], arrowprops=arrow)
ax.annotate('', xy=nodes['logit'], xytext=nodes['target'], arrowprops=arrow)
# Forward patch cue (reinstated path) as dashed overlay from L0→L11
ax.annotate('', xy=(nodes['target'][0], nodes['target'][1] + 0.08),
            xytext=(nodes['suppressor'][0], nodes['suppressor'][1] + 0.08),
            arrowprops=dict(arrowstyle='->', color='#0072B2', linewidth=2.0, linestyle='--'))

def fmt_signed(x: float) -> str:
    # Use ASCII hyphen-minus for negative to ensure rendering in all figure fonts
    return (f'+{x:.2f}' if x >= 0 else f'-{abs(x):.2f}')

med_pct = int(round(fraction * 100))
ax.text(nodes['suppressor'][0], 0.73, f'Full ablation: ΔLD = {fmt_signed(full_delta)}',
        ha='center', va='center', fontsize=10)
ax.text(0.28, 0.64, 'Forward patch (reinstated path)',
        ha='center', va='center', color='#0072B2', fontsize=10)
ax.text(0.48, 0.73, f'ΔLD = {fmt_signed(remaining)}',
        ha='center', va='center', color='#0072B2', fontsize=10)
ax.text(0.48, 0.30, f'Mediated fraction = {med_pct}%', ha='center', va='center')
ax.text(0.75, 0.62, f'Reverse patch at L11: ΔLD = {fmt_signed(reverse_delta)}',
        ha='center', va='center', fontsize=10)

fig.tight_layout()
fig_path = ROOT / 'paper' / 'figures' / 'path_patch_dag.pdf'
fig.savefig(fig_path)
print(f'Wrote {fig_path}')
