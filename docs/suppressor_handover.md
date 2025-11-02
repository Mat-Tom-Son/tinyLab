# Tiny Ablation Lab – Suppressor Validation Handover

> For a concise reproduction checklist (commands and expected outputs), see [REPLICATION.md](REPLICATION.md).

This document captures the full context of the suppressor-analysis work completed so far and provides a quick-start guide for extending the harness to new researchers/models. It consolidates everything we ran on GPT‑2, what happened with LLaMA, the ongoing Mistral validation, and the reusable tooling we built along the way.

---

## 1. High-Level Story

1. **GPT‑2 (medium & large)**  
   - Layer‑0 head 2 injects a “hedge/meta-commentary” direction (OV top tokens like `totally`, `solid`, `...`) while suppressing factual/logical continuations (`Recomm`, `trave`, …).  
   - Heads 4 and 7 amplify head 2 but contribute much smaller logit deltas.  
   - Partial interventions (zero all three, patch one back) show head 2 alone recovers ~60–70 % of the damage.  
   - Pair/Triplet runs confirm destructive cooperation, reverse patch windows stay asymmetric.

2. **LLaMA attempt**  
   - TransformerLens currently only loads LLaMA checkpoints that require Meta gating. `meta-llama/Llama-2-7b-hf` returned 403, and `openlm-research/open_llama_7b` isn’t in the official allowlist; so the run was blocked.  
   - Path forward: request access for an allowed LLaMA repo (`meta-llama/Meta-Llama-3-8B`, etc.) or extend the loader to handle unlisted models manually.

3. **Mistral-7B (in progress)**  
   - Tokenizer-specific corpora built via `scripts/build_tokenizer_variants.py` produce `*_mistral.jsonl` datasets & splits.  
   - H1/H5 runs completed per condition (`seed=0`, 24-train split). Layer‑0 head 21 emerged as the primary suppressor (logit diff ~0.09–0.35 on neg/logic), mirroring the GPT‑2 pattern.  
   - Reverse-patch (H6) per condition is queued next; the orchestrator timeouts mean we’ll run each config through `lab/src/harness` directly.

---

## 2. Reproducible Workflow

### 2.1 Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
If you need gated models (e.g., Meta LLaMA), log in once: `python -c "from huggingface_hub import login; login()"`.

### 2.2 Core Harness Commands
The orchestrator is here: `lab/src/orchestrators/conditions.py`. For single configs use `lab/src/harness.py`. Each config describes:
- `shared`: seeds, model family/name/dtype, dataset fields, metrics.
- `battery`: which ablation or patching module to run.
- `conditions`: dataset overrides per tag (one or more corpora).

Example GPT‑2 commands (compiled in `docs/suppressor_analysis_runbook.md`):
```bash
# GPT-2 Medium cross-condition heads
python -m lab.src.orchestrators.conditions lab/configs/run_h1_cross_condition_balanced.json

# Pair & triplet ablation
python -m lab.src.orchestrators.conditions lab/configs/run_h5_layer0_pairs_balanced.json
python -m lab.src.orchestrators.conditions lab/configs/run_h5_layer0_triplet_balanced.json

# Reverse patch windows (layer resid)
python -m lab.src.orchestrators.conditions lab/configs/run_h6_layer_targets_window_balanced.json
```

The current Mistral configs ship with multi-seed defaults (H1: five seeds, H5: three seeds) and run a small CPU `verify_slice` before each sweep to catch MPS drift. Tweak the `seeds` array or `verify_slice` block in the JSON if you need different coverage.

### 2.3 Tokenizer-aware corpora
Use `scripts/build_tokenizer_variants.py` for any new architecture:
```bash
python scripts/build_tokenizer_variants.py \
  --tokenizer mistralai/Mistral-7B-v0.1 \
  --suffix mistral \
  --datasets facts_single_token_v1 negation_single_token_v1 counterfactual_single_token_v1 logical_single_token_v1
```
This filters each corpus to single-token targets/foils under the given tokenizer and rewrites the split files. Update configs to consume the new IDs (e.g., `facts_single_token_v1_mistral`).

### 2.4 Partial patch & OV tooling
- `lab/analysis/h5_partial_patch.py`: zero {head2,4,7}, patch one back; outputs summary JSON (baseline/zero/patch deltas).  
- `lab/analysis/ov_report.py`: full per-head OV projections (top/bottom tokens, vector norms).  
- `lab/analysis/cluster_ov_tokens.py`: cluster tokens across multiple reports.

Example OV command:
```bash
python -m lab.analysis.ov_report \
  --config lab/configs/run_h1_cross_condition_balanced.json \
  --tag facts --heads 0:2 0:4 0:7 --samples 160 --top-k 150 \
  --output reports/ov_report_facts.json

python lab/analysis/cluster_ov_tokens.py \
  reports/ov_report_facts.json reports/ov_report_facts_large.json \
  --limit 100 --top-n 50 --output reports/ov_token_clusters_facts.json
```

---

## 3. Repository Layout (key paths)

```
lab/
  src/
    harness.py                # main driver for single configs
    orchestrators/
      conditions.py           # cross-condition runner
    ablations/                # heads_zero, heads_pair_zero, activation_patch, …
    analysis/                 # OV reports, clustering, partial patch etc.
  configs/                    # JSON configs for all runs (GPT-2, Mistral, etc.)
  data/
    corpora/*.jsonl           # original + tokenizer-specific corpora
    splits/*.split.json       # matching split metadata
docs/
  suppressor_analysis_runbook.md  # step-by-step gpt2 workflow log
  suppressor_handover.md          # (this file)
reports/
  ov_report_*.json            # per-model OV dumps
  ov_token_clusters_*.json    # aggregated token clusters
scripts/
  build_tokenizer_variants.py # batch filter for new tokenizers
  curate_single_token_dataset.py   # single tokenizer run
  … (legacy dataset scripts, pair config generator, etc.)
```

---

## 4. Status by Model Family

| Model        | H1 (heads) | H5 (pairs/triplet) | H6 (reverse patch) | OV Reports | Notes |
|--------------|------------|--------------------|--------------------|------------|-------|
| GPT‑2 medium | ✅ complete | ✅ complete        | ✅ clean           | ✅ (facts/neg/cf/logic + clustering) | Layer‑0 head 2 suppressor confirmed. |
| GPT‑2 large  | ✅ complete | focal H5/H6 done | ✅ clean windows   | ✅ `ov_report_facts_large.json` etc. | Same suppressor pattern; pair/triplet results in `lab/runs/h5_layer0_pairs_balanced_large_*`. |
| LLaMA        | ⛔ blocked  | –                  | –                  | –          | Requires Meta gating (403). |
| Mistral 7B   | ✅ per-condition runs (seed=0) | ✅ pair/triplet (seed=0) | 🚧 run each condition via harness | 🚧 (next step) | Layer‑0 head 21 emerging as suppressor. |

Next actions (short term):
1. Finish Mistral H6 by launching each condition individually and collect the layer-window results.
2. Generate OV reports/clusters for Mistral suppressor heads (compare vectors to GPT‑2).
3. Once Mistral is stable, port the workflow to GPT-NeoX/Pythia or another architecture in the allowlist for broader generality.

---

## 5. Quick Start for New Researchers
1. **Install** dependencies, activate `.venv`, log in to Hugging Face.  
2. **Pick a config** (e.g., `run_h1_cross_condition_balanced.json`) and run it:  
   `python -m lab.src.orchestrators.conditions lab/configs/<config>.json`.  
3. **Inspect outputs** under `lab/runs/<run_id>/metrics/*.json`/`*.parquet`; cross-condition matrices are in `.../artifacts/cross_condition/`.  
4. Use the **analysis scripts** (OV reports, partial patch, clustering) to interpret results.  
5. To adapt to a new tokenizer/model, run `scripts/build_tokenizer_variants.py`, adjust configs, and repeat.

The workflow is intentionally modular: every new architecture just needs ① matching corpora/splits, ② a set of configs pointing to the right model name, and ③ the same command sequence. All intermediate and final artefacts stay under `lab/runs/` and `reports/`, so results are traceable and reproducible.

---

## 6. Working With Outputs & Analysis Artefacts

### 6.1 Where results are saved

Every run gets its own directory under `lab/runs/<run_name>_<hash>/` with:

- `metrics/summary.json`          – aggregated seed metrics (`logit_diff`, `acc_flip_rate`, `p_drop`, …) plus per-seed values.
- `metrics/head_impact.parquet`   – long-form table of head-level metrics (for H1/H5 conditions).
- `metrics/layer_impact.parquet`  – layer-level tables (activation patch H6).
- `metrics/per_example.parquet`   – per input example/seed breakdown (argmax flips, predicted ids, etc.).
- `artifacts/impact_heatmap.*`    – visual heatmaps (one per seed, averaged).
- `artifacts/cross_condition/`    – (for orchestrator runs) aggregate matrices & summaries for easy comparison.
- `partial_summary.json`          – produced if a run aborts early; useful for debugging.
- `profile.json`                  – resource usage traces if profiling enabled.
- `provenance.json`               – runtime metadata (python/torch versions, device availability, backend flags).

The cross-condition rebuild helper (`lab/analysis/rebuild_cross_condition.py`) stitches together single-condition runs if the orchestrator ran into timeouts. Use it anytime a parent run is missing `artifacts/cross_condition/*.parquet`.

### 6.2 Inspecting head/layer impacts

Load the head or layer parquet into pandas for ad-hoc analysis:
```python
import pandas as pd
df = pd.read_parquet('lab/runs/<run_id>/metrics/head_impact.parquet')
layer0 = df[(df.metric == 'logit_diff') & (df.layer == 0)]
print(layer0.nsmallest(5, 'value'))
```

For cross-condition matrices (from orchestrators):
```python
matrix = pd.read_parquet('lab/runs/<parent_run>/artifacts/cross_condition/head_matrix.parquet')
print(matrix[(matrix.metric == 'logit_diff') & (matrix.layer == 0)].nsmallest(5, 'value'))
```

### 6.3 OV report analysis

- Raw OV dumps live in `reports/ov_report_*.json`. The structure includes vector norms, top/bottom token lists, and per-head metadata.
- Token clustering output (`reports/ov_token_clusters_*.json`) counts how frequently each token appears across the top/bottom lists for different models – great for spotting shared “hedge” directions.
- Use the provided scripts to generate or compare reports:
  ```bash
  # New OV report for Mistral facts suppressor
  python -m lab.analysis.ov_report \
    --config lab/configs/run_h1_cross_condition_balanced_mistral.json \
    --tag facts --heads 0:21 --samples 160 --top-k 150 \
    --output reports/ov_report_facts_mistral.json

  # Cluster against GPT-2 baseline
  python lab/analysis/cluster_ov_tokens.py \
    reports/ov_report_facts.json reports/ov_report_facts_mistral.json \
    --limit 100 --top-n 50 \
    --output reports/ov_token_clusters_facts_gpt2_vs_mistral.json
  ```

### 6.4 Partial patch summaries

Summary JSONs from `h5_partial_patch.py` contain `baseline`, `zero`, and `patched` sections. Example snippet:
```json
{
  "baseline": {"logit_diff": 1.483},
  "zero": {"logit_diff": 2.017},
  "patched": {"logit_diff": 1.648}
}
```
This lets you quantify how much of the suppressor effect a single head explains.

### 6.5 Visualisations

The repository ships pre-configured heatmap exporters (`lab/src/viz/heatmap.py`). Each run writes `artifacts/impact_heatmap.png` (and `.html` if available). These are useful for quick presentations.

---

## 7. Contacts / Next Steps
- **Current owner**: the suppressor harness is documented in `docs/suppressor_analysis_runbook.md` and this handover file.  
- **Open tasks**:  
  - Complete Mistral H6 + OV clustering.  
  - Automate OV report comparisons (scripts currently cover facts/neg/cf/logic separately).  
  - Once a LLaMA checkpoint is accessible, rerun the full suite for parity with GPT‑2/Mistral.

Happy probing!  
