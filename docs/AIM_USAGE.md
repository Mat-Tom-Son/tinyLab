# Aim Usage Guide for tinyLab

This document explains how to use Aim with tinyLab once the environment is set up and data has been imported.

See `AIM_INTEGRATION_PLAN.md` for the full design; this file is a practical, short guide.

## 1. Setup (once per environment)

From the repository root:

```bash
source .venv/bin/activate
bash scripts/setup_aim.sh
python scripts/import_to_aim.py
```

This:
- Installs Aim into the active virtualenv.
- Ensures `.aim/` is gitignored.
- Initializes the Aim repo under `.aim/`.
- Imports all standardized results from `reports/` into Aim (head rankings, summaries, entropy scans, Pythia drift, etc.).

## 2. Launching the UI

```bash
source .venv/bin/activate
aim up
```

Then open `http://localhost:43800` in your browser.

Aim will list experiments such as:
- `h1_heads_zero`, `h1_metrics`, `h5_metrics`, `h6_metrics` – live runs from the harness.
- `imported_head_ranking`, `imported_head_slice`, `imported_summary` – standardized tables from `reports/`.
- `imported_entropy_scan`, `imported_pca_rank`, `imported_drift_trajectories` – geometric and Pythia dynamics.
- `imported_binder_sweep`, `imported_cross_model_summary`, `imported_metrics` – binder, cross-model, and JSON metrics.

## 3. Suppressor Atlas (H1/H5/H6)

**Goal:** Visualize which heads/layers act as suppressors across models and probe families.

1. In the left sidebar, select experiment `imported_head_ranking`.
2. Filter runs:
   - Example (GPT‑2 Medium, facts): `run.model == "gpt2-medium" and run.condition == "facts"`.
3. In the Metrics view:
   - Choose the metric that represents suppressor strength (e.g. `logit_diff` or a similar column imported from the CSV).
   - X‑axis: `step` (will be 0 for imported tables).
   - Group by:
     - `context.layer` and `context.head` to see per-head curves.
4. Save a dashboard with panels for:
   - GPT‑2 Medium vs GPT‑2 Large (one chart per model).
   - Mistral‑7B.
   - Pair/triplet experiments from `imported_head_slice` (H5) and `imported_summary` (H6).

For live runs via `lab/src/harness.py`:
- Experiments: `h1_heads_zero`, `h5_metrics`, `h6_metrics`.
- Metrics: `logit_diff_mean`, `acc_flip_rate_mean`, `p_drop_mean`.
- Group by run tags (e.g. `H1`, `H5`, `H6`, model name, dataset id).

## 4. Geometry Panel (Entropy, Curvature, OV)

**Goal:** Connect suppressor behaviour to geometry (entropy, curvature) and OV fingerprints.

1. Entropy scans:
   - Experiment: `imported_entropy_scan`.
   - Filter runs by model and condition (e.g. `run.model == "gpt2-medium" and run.condition == "facts"`).
   - Metrics: look for `activation_entropy_*` metrics imported from `layer_entropy_scan_*` CSVs.
   - Group by: `context.layer` to get entropy vs. layer curves.
2. Curvature metrics:
   - If you ran `lab.analysis.activation_entropy` and imported the JSONs, look for metrics containing `curv_early`, `curv_mean`, `curv_late`.
   - Plot versus `context.layer` or compare baseline vs ablated variants via tags.
3. OV reports:
   - OV report JSONs are imported under `imported_metrics`.
   - Filter runs where `run.artifact == "metrics"` and the `source_file` matches `ov_report_*`.
   - Explore per‑head metrics and use `context.layer`/`context.head` for grouping.

Combine these into a single dashboard showing:
- Entropy vs. layer.
- OV metrics vs. layer/head.
- Curvature metrics where available.

## 5. Stage‑1A Development (VDI, Circularity, Drift)

**Goal:** Visualize developmental dynamics from Stage‑1A tooling.

1. VDI (Variance Dampening Index):
   - After running `scripts/identify_suppressors.py` and importing its CSV, look for runs with metrics named `vdi_effect`, `vdi_full`, `vdi_minus`.
   - Filter by layer (e.g. `run.layer == 0`) and model.
   - Group by `context.head` to see per-head VDI profiles.
2. Circularity (Task B weekday representations):
   - After running `scripts/measure_circularity.py` and importing, find the corresponding run in `imported_metrics`.
   - Metrics: `circularity_score`, `angle_correlation`, `radial_consistency`.
   - Plot `circularity_score` against checkpoint or configuration tags.
3. Pythia drift:
   - Experiment: `imported_drift_trajectories` (from `pythia_layer0_drift_trajectories.csv`).
   - Metrics: `mean_drift`, `mean_entropy`.
   - Group by `context.layer_label` and plot vs `step` to see trajectories at `resid0`, `resid_mid`, `resid_final`.

These views help track when suppressor‑like VDI profiles stabilize and how they relate to emergent geometry (circularity and drift).

## 6. Regenerating and Re‑importing

Whenever you regenerate results under `reports/` (e.g. after running new H1/H5/H6 experiments or Stage‑1A probes):

```bash
make postprocess        # or targeted analysis scripts
python scripts/import_to_aim.py
aim up                  # reloads UI with updated runs
```

Aim can always be rebuilt from the DVC‑managed `reports/` directory; `.aim/` itself is never committed.
