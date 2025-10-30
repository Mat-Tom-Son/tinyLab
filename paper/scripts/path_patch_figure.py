import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
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

fig, ax = plt.subplots(figsize=(4.2, 2.4))
ax.axis('off')

nodes = {
    'suppressor': (0.05, 0.5),
    'target': (0.45, 0.5),
    'logit': (0.85, 0.5)
}

ax.text(*nodes['suppressor'], 'Layer-0\nsuppressor', ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
ax.text(*nodes['target'], 'Layer 11\nresidual', ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
ax.text(*nodes['logit'], 'Logits', ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

ax.annotate('', xy=nodes['target'], xytext=nodes['suppressor'], arrowprops=dict(arrowstyle='->', color='black'))
ax.annotate('', xy=nodes['logit'], xytext=nodes['target'], arrowprops=dict(arrowstyle='->', color='black'))

ax.text(0.25, 0.64, f'Full ablation ΔLD = +{full_delta:.2f}', ha='center', fontsize=8)
ax.text(0.65, 0.64, f'Path reinstated ΔLD = +{remaining:.2f}', ha='center', fontsize=8)
ax.text(0.5, 0.32, f'Mediated fraction = {fraction:.2%}', ha='center', fontsize=8)
ax.text(0.65, 0.44, f'Reverse patch ΔLD = {reverse_delta:.2f}', ha='center', fontsize=8)

fig.tight_layout()
fig_path = ROOT / 'paper' / 'figures' / 'path_patch_dag.pdf'
fig.savefig(fig_path)
print(f'Wrote {fig_path}')
