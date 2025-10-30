# Layer-0 Suppressor Analysis Runbook

This note captures the full workflow we ran today to characterise the layer‑0 suppressor heads in GPT‑2 models, the key findings, and how to reuse the tooling on additional models or corpora.

---

## 1. Highlights & Findings

- **Suppressor circuit confirmed (gpt2‑medium & gpt2‑large)**  
  Layer‑0 head 2 consistently injects a “hedge/meta-commentary” direction, boosting tokens such as `totally`, `solid`, punctuation runs, etc., while suppressing factual stems (`Recomm`, `trave`, `circumst`, …). Partial interventions show that restoring head 2 alone recovers ~60–70 % of the damage; heads 4/7 add only minor residual effects.

- **Cooperative behaviour**  
  Pair ablations and triplet removal (`run_h5_layer0_pairs_balanced*.json`, `run_h5_layer0_triplet_balanced.json`) demonstrate that any pair containing head 2 remains highly destructive, confirming it is the core suppressor.

- **Directional asymmetry**  
  Reverse patches (`h2_cross_condition_physics_probe*.json`) remain asymmetric across ±2 layer windows; corrupt→clean swaps fail to recover clean logits, reinforcing that the circuit pushes activations away from factual answers.

- **Reusable OV tooling**  
  `lab/analysis/ov_report.py` + `cluster_ov_tokens.py` generalise to any model/config and give a full semantic fingerprint of each head’s contribution.

---

## 2. Command Log (gpt2-medium → gpt2-large)

> Activate the project environment first:
>
> ```bash
> source .venv/bin/activate
> ```

### 2.1 Balanced Heads & Suppressor Confirmation

```bash
# gpt2-medium
python -m lab.src.orchestrators.conditions lab/configs/run_h1_cross_condition_balanced.json
# gpt2-large (same pipeline, swap model name)
python -m lab.src.orchestrators.conditions lab/configs/run_h1_cross_condition_balanced_large.json
```

Artifacts:
- `lab/runs/h1_cross_condition_physics_balanced_*/artifacts/cross_condition/head_matrix.parquet`
- `lab/runs/h1_cross_condition_balanced_gpt2_large_*/artifacts/cross_condition/head_matrix.parquet`

### 2.2 Pair/Triplet Ablations

```bash
# gpt2-medium
python -m lab.src.orchestrators.conditions lab/configs/run_h5_layer0_pairs_balanced.json
python -m lab.src.orchestrators.conditions lab/configs/run_h5_layer0_triplet_balanced.json

# gpt2-large replicas
python -m lab.src.orchestrators.conditions lab/configs/run_h5_layer0_pairs_balanced_large.json
```

Outputs live under `lab/runs/h5_layer0_pairs_balanced*`.

### 2.3 Partial Patch (Zero trio, patch single back)

```bash
python -m lab.analysis.h5_partial_patch \
  --config lab/configs/run_h1_cross_condition_balanced.json \
  --tag facts --layer 0 --zero-heads 2 4 7 --patch-head 2 \
  --samples 80 --output reports/facts_partial_patch_head2.json
```

(Repeat for other heads/tags; summaries are generated with `lab/analysis/summarise_partial_patch.py`.)

### 2.4 Reverse Patching Windows

```bash
python -m lab.src.orchestrators.conditions lab/configs/run_h6_layer_targets_window_balanced.json
python -m lab.src.orchestrators.conditions lab/configs/run_h6_layer_targets_window_balanced_large.json
```

Artifacts:
- `lab/runs/h6_layer_targets_window_balanced_*/artifacts/cross_condition/layer_matrix.parquet`.

### 2.5 OV Projection & Clustering

```bash
# Generate detailed token projections
python -m lab.analysis.ov_report \
  --config lab/configs/run_h1_cross_condition_balanced.json \
  --tag facts --heads 0:2 0:4 0:7 --samples 160 --top-k 150 \
  --output reports/ov_report_facts.json

# Repeat for other corpora & models (see *_large configs)

# Cluster the vocab contributions
python lab/analysis/cluster_ov_tokens.py \
  reports/ov_report_facts.json reports/ov_report_facts_large.json \
  --limit 100 --top-n 50 \
  --output reports/ov_token_clusters_facts.json
```

Comparable commands were run for `neg`, `cf`, and `logic`, producing:
- `reports/ov_report_{neg,cf,logic}.json`
- `reports/ov_report_{neg,cf,logic}_large.json`
- `reports/ov_token_clusters_{neg,cf,logic}.json`

---

## 3. Plug-and-Play Instructions for New Models / Architectures

1. **Copy a baseline config**  
   Duplicate the relevant config (`run_h1_cross_condition_balanced.json`, `run_h5_layer0_pairs_balanced.json`, etc.) and change the `model.name` to the target architecture. Update `run_name` to avoid collisions.

2. **Run the orchestrator**  
   Execute the corresponding `lab/src/orchestrators/conditions` command; the harness automatically logs runs and builds cross-condition matrices.

3. **Generate OV reports**  
   Use `lab/analysis/ov_report.py` with the new config/tag to emit top/bottom tokens. Combine reports via `cluster_ov_tokens.py` to get semantic clusters.

4. **Partial patches & triplet runs**  
   `lab/analysis/h5_partial_patch.py` accepts any config; no model-specific changes required.

5. **Reverse patch windows**  
   Adjust `battery_h6_layer_window_*.json` to target the appropriate layer window for the new model, then run the orchestrator config.

6. **Automation tip**  
   All configs can be scripted: create a new directory (e.g., `configs/models/gptj/`) with copies of the JSON files and tweak model/device settings. The orchestrator picks up per-condition battery overrides automatically.

### 3.1 LLaMA Quick-Start

We already dropped LLaMA-ready configs in `lab/configs/` (look for files ending with `_llama.json`). To replicate the full suppressor suite on a LLaMA checkpoint:

1. **Accept / download the model**  
   Make sure `huggingface-cli login` is configured and the `llama-7b` weights are available to TransformerLens. Adjust `model.name` in the configs if you want a larger variant.

2. **Run the balanced heads scan**  
   ```bash
   python -m lab.src.orchestrators.conditions lab/configs/run_h1_cross_condition_balanced_llama.json
   ```

3. **Pairs & triplet**  
   ```bash
   python -m lab.src.orchestrators.conditions lab/configs/run_h5_layer0_pairs_balanced_llama.json
   python -m lab.src.orchestrators.conditions lab/configs/run_h5_layer0_triplet_balanced_llama.json
   ```

4. **Reverse-patch windows**  
   ```bash
   python -m lab.src.orchestrators.conditions lab/configs/run_h6_layer_targets_window_balanced_llama.json
   ```

5. **OV projections & clustering**  
   Swap in the LLaMA config when calling `ov_report.py`, e.g.:
   ```bash
   python -m lab.analysis.ov_report \
     --config lab/configs/run_h1_cross_condition_balanced_llama.json \
     --tag facts --heads 0:2 0:4 0:7 --samples 160 --top-k 150 \
     --output reports/ov_report_facts_llama.json
   python lab/analysis/cluster_ov_tokens.py \
     reports/ov_report_facts_llama.json reports/ov_report_facts.json \
     --limit 100 --top-n 50 \
     --output reports/ov_token_clusters_facts_llama_vs_gpt2.json
   ```

6. **Partial interventions**  
   Reuse `h5_partial_patch.py` by pointing to the LLaMA config and desired heads; e.g., `--patch-head 2` isolates the suppressor.

Tip: If you experiment with LLaMA‑13B/65B, duplicate the configs again and update `model.name`. All supporting scripts work unchanged.

---

## 4. Next Diagnostic Ideas (Work in Progress)

- Training-dynamics probe: capturing early checkpoints or simulating curriculum steps to see when head 2 emerges.
- OOD fact evaluation: compare accuracy with and without head 2 to distinguish “regulariser” vs “artifact”.
- Rank/compression measurements: inspect representation covariance with head 2 zeroed to test the compression hypothesis.

Feel free to extend this document with new experiments; the structure here should scale to additional models, datasets, or head groups.
