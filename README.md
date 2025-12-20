# Tiny Ablation Lab

A reproducible, local‑first workspace for mechanistic interpretability on Apple Silicon. This repository accompanies the October 2025 study **“Layer‑0 Suppressors Ground Hallucination Inevitability”**, and ships end‑to‑end code to replicate the findings on GPT‑2 Medium and Mistral‑7B. It now includes the geometric validation of suppressors via output entropy and trajectory curvature.

Paper PDF: `paper/main.pdf` | Archive: https://zenodo.org/records/17524770

Key idea: circuits that implement the factuality vs hedging tradeoff crystallize at the first bottleneck (layer 0). We validate this prediction with dual observables (power: ΔLD; information: calibration), random head baselines, cross‑architecture checks, path mediation, and now geometric signatures.

New in this repo: a preregistered Stage‑1A pilot on **early‑layer synchronization control of induction‑head emergence** ("developmental interpretability"). The preregistration PDF lives at `docs/prereg_stage1a/prereg_stage1a.pdf` and is backed by small‑model utilities for variance‑dampening (VDI), Task‑B weekday modular addition data, circularity measurements, and per‑head α‑scaling hooks.

**NEW: Developmental Monitoring Framework** for phase-transition control. Track the crystallization of Layer-0 "Gatekeepers" during training via: (A) VDI "snap" detection, (B) homeostatic kill testing (Le Chatelier compensation), and (C) MI saturation boundaries. See [docs/PHASE_TRANSITION_CONTROL.md](docs/PHASE_TRANSITION_CONTROL.md) for implementation details and [docs/DEVELOPMENTAL_MONITORING.md](docs/DEVELOPMENTAL_MONITORING.md) for usage guide.

## Highlights

- Strong geometric signature under suppressor ablation across all probe families (facts, counterfactual, negation, logic):
  - Output entropy reduction: ΔH = −2.4 to −3.8 nats (lower is sharper), all in the extreme lower tail of random layer‑0 head controls (p < 0.02).
  - Early trajectory straightening in layer‑0 residuals: Δ curvature (early) ≈ −14 to −16.
- Activation space expands under suppressors (negative Δ activation entropy under ablation across estimators), consistent with a rotation‑plus‑expansion mechanism coupled with output flattening.
- Location is forced by geometry: the operation appears at layer 0; implementation varies by model.
- Fully reproducible harness and figure scripts; all key reports are committed.


## Table of Contents

1. [Quick Start](#quick-start)
2. [Repository Layout](#repository-layout)
3. [Everyday Workflow](#everyday-workflow)
4. [Reproducing the Study](#reproducing-the-study)
5. [Available Artefacts](#available-artefacts)
6. [Documentation](#documentation)
7. [Contributing](#contributing)
8. [Citation](#citation)

## Quick Start

### Apple Silicon (MPS)

```bash
git clone https://github.com/Mat-Tom-Son/tinyLab.git
cd tinyLab
bash scripts/setup_env.sh
source .venv/bin/activate
python smoke_test.py  # optional sanity check
```

### NVIDIA GPUs (CUDA)

```bash
git clone https://github.com/Mat-Tom-Son/tinyLab.git
cd tinyLab
bash scripts/setup_env_cuda.sh
source .venv/bin/activate
python smoke_test_cuda.py  # optional sanity check
```

### Debian 12 / GCE (CUDA, one-liner)

SSH into your NVIDIA T4 VM (Debian 12 with CUDA 12.x) and run:

```bash
bash -c "$(curl -fsSL https://raw.githubusercontent.com/Mat-Tom-Son/tinyLab/main/install_tinylab_linux.sh)"
```

This clones `tinyLab`, installs system deps, sets up the venv, installs CUDA wheels, pulls DVC data (if configured), runs the CUDA smoke test, and executes the Stage‑1A pilot dry‑run.

- `setup_env.sh` (MPS) or `setup_env_cuda.sh` (CUDA) installs all pinned dependencies (see `pyproject.toml`) and validates PyTorch.
- Smoke tests load GPT‑2‑small and check the harness wiring; skip if you know your environment is ready.
- For CUDA-specific setup and optimization guide, see [docs/CUDA_SETUP.md](docs/CUDA_SETUP.md).
- For an annotated walkthrough of the harness, see [QUICKSTART.md](QUICKSTART.md).

### Stage‑1A Pilot Utilities

To sanity‑check the new Stage‑1A utilities (Task‑B data, circularity metrics, VDI probe), run:

```bash
bash scripts/run_pilot_dry_run.sh
```

This runs a small, end‑to‑end dry‑run on `gpt2-small` to validate the geometry and structural probes without training the 2‑layer pilot model.

## Everyday Workflow

Once the repo is set up, a typical “pull → run → log → push” cycle looks like:

1. **Sync code and data**

   ```bash
   git pull
   dvc pull
   ```

2. **Activate the environment**

   ```bash
   source .venv/bin/activate
   ```

3. **Run experiments or analysis**

   Use the harness / scripts you need (e.g. orchestrators under `lab/src/orchestrators`, analysis modules under `lab/analysis`, or helper shell scripts under `scripts/`).

4. **Update tracked results (when you care about them)**

   For a full standardized refresh:

   ```bash
   make postprocess          # regenerate summaries, rankings, manifest, etc.
   dvc add reports/          # update DVC pointer for reports
   git add reports.dvc
   dvc push                  # push updated artefacts to the DVC remote
   ```

   For a small new subdirectory of results:

   ```bash
   dvc add reports/<new_subdir>/
   git add reports/<new_subdir>.dvc
   dvc push
   ```

5. **Run pre‑commit and basic checks**

   ```bash
   .venv/bin/pre-commit run --all-files
   # or, for the full checklist:
   ./scripts/pre_submit_check.sh
   ```

6. **Commit and push**

   ```bash
   git status        # optional: inspect changes
   git add .         # or select files
   git commit -m "Brief message about this run"
   git push
   ```

With this loop, CI should pass consistently, DVC stays in sync with Git, and `reports/RESULTS_MANIFEST.json` stays up to date.

## Data Management with DVC

This project uses [DVC (Data Version Control)](https://dvc.org) to manage datasets, results, and artifacts. DVC keeps large data files out of Git while maintaining full version control and reproducibility.

### First-time Setup

After cloning the repository, pull all tracked data:

```bash
# Install DVC
pip install dvc

# Pull datasets and results
dvc pull
```

This downloads:
- Raw datasets (`lab/data/corpora/`)
- Data splits (`lab/data/splits/`)
- Results and metrics (`reports/`)
- Paper supplements (`paper/supplement/`)

### GCS remote (for GCE)

To point DVC at a GCS bucket on GCE:

```bash
gcloud auth application-default login
GCS_BUCKET=<your-bucket> GCS_PREFIX=tinylab bash scripts/configure_dvc_gcs.sh
dvc pull  # or dvc push after runs
```

### Why DVC?

- **Version control for data** - Track dataset and result versions alongside code
- **Efficient storage** - Large files stored separately from Git
- **Reproducibility** - Exact data versions tied to code commits
- **Scalability** - Seamlessly migrate to S3/GCS/Azure when needed

### Documentation

- [DVC_SETUP.md](DVC_SETUP.md) - Complete setup and usage guide
- [DVC_MIGRATION_DESIGN.md](DVC_MIGRATION_DESIGN.md) - Architecture and design decisions
- [DVC_TROUBLESHOOTING.md](DVC_TROUBLESHOOTING.md) - Common issues and solutions

## Monitoring

- GPU usage logging for cost-awareness: `bash scripts/log_gpu_usage.sh` (set `INTERVAL` and `OUT_FILE` as needed). CSV logs land under `logs/` by default.
- After a run, back up results + commit quickly:
  - Configure GCS remote once (see above).
  - `bash scripts/post_run_backup.sh reports/pilot_stage1a "pilot: baseline run"` (DVC add + dvc push + git commit; set `GIT_PUSH=1` to push git, `DVC_PUSH=0` to skip data push).
- Stage‑1A helper to reproduce the prereg battery (baseline circularity + VDI + dry run):  
  `bash scripts/run_stage1a_suite.sh` (env toggles: `SEEDS`, `RUN_CIRCULARITY`, `RUN_VDI`, `RUN_DRY_RUN`, `DEVICE`, `TASK_B_SIZE`, `REPORTS_DIR`).
- Stage‑1A prereg orchestrator (suite + auto head selection + templated training launcher):  
  `bash scripts/run_stage1a_prereg.sh` with envs such as `TRAIN_CMD_TEMPLATE='python train_stage1a.py --cond {cond} --seed {seed} --omega {omega} --head {head} --head-kind {head_kind}'`.  
  Other envs: `SEEDS`, `TRAIN_SEEDS`, `SUPPRESSOR_OMEGAS`, `RANDOM_OMEGAS`, `RUN_BASELINE/RUN_SUPPRESSOR/RUN_RANDOM`, `SUITE_FIRST`, `DEVICE`, `REPORTS_DIR`.

## Aim Experiment Tracking

This repo logs runs to MLflow by default and can optionally mirror key metrics into [Aim](https://aimstack.io) for an interactive web UI.

To enable Aim locally:

```bash
source .venv/bin/activate
bash scripts/setup_aim.sh        # install Aim + init .aim/
python scripts/import_to_aim.py  # import reports/ into Aim
bash scripts/launch_aim_ui.sh    # launches UI at http://localhost:43800
```

- Historical results under `reports/` (head rankings, summaries, entropy scans, Pythia drift, Stage‑1A probes, …) are imported into Aim with descriptive experiments such as `imported_head_ranking`, `imported_entropy_scan`, `imported_drift_trajectories`.
- New runs launched via `lab/src/harness.py` log per‑seed and aggregated metrics into Aim experiments named after `run_name` (e.g., `h1_heads_zero`, `h5_layer0_pairs_balanced_*`) when Aim is installed.
- For a short, task‑oriented overview of how to explore H1/H5/H6 suppressors, geometric signatures, and Stage‑1A VDI/circularity in Aim, see [docs/AIM_USAGE.md](docs/AIM_USAGE.md).

## Repository Layout

| Path | Purpose |
| --- | --- |
| `lab/` | Experiment harness, batteries, configs, and recorded runs |
| `lab/runs/` | Hash-stamped outputs referenced in the paper (configs, metrics, artefacts) |
| `reports/` | Aggregated analysis dumps (OV projections, partial patch summaries) |
| `paper/` | LaTeX source, figure scripts, and the compiled PDF |
| `docs/` | Project notes, replication guides, and prereg materials (`prereg_stage1a/prereg_stage1a.pdf`) |

## Reproducing the Study

A full, step‑by‑step guide covering dataset preparation, GPT‑2 and Mistral runs, analysis scripts, and paper compilation is in `docs/REPLICATION.md`. The high‑level flow is:

1. Corpora — use the provided single‑token probe datasets under `lab/data/corpora/`. If you need fresh tokenizer variants, run `scripts/build_tokenizer_variants.py`.
2. GPT‑2 experiments — orchestrated configs under `lab/configs/run_h1_cross_condition_balanced.json`, `run_h5_layer0_triplet_balanced.json`, and `run_h6_layer_targets_window_balanced.json` reproduce the H1/H5/H6 batteries.
3. Mistral experiments — `lab/configs/run_h1_cross_condition_balanced_mistral*.json` cover per‑condition sweeps; pair and triplet follow‑ups live alongside (`run_h5_*mistral*.json`).
4. Analysis scripts — regenerate figures and tables via the Python utilities in `paper/scripts/`.
5. Paper build — `cd paper && make` uses `latexmk` to compile `paper/main.pdf` from source.

All runs log reproducibility metadata (`config.json`, hashes, seeds, git commit) into their respective `lab/runs/<run_id>/` directories.

## Containers

An nvidia-docker image is available for CUDA 12.6 on Debian 12:

```bash
docker build --platform=linux/amd64 -t tinylab:cuda .
docker run --gpus all --shm-size=16g -it tinylab:cuda bash scripts/run_pilot_dry_run.sh
```

Mount a host reports directory if you want outputs persisted: `-v $PWD/reports:/app/reports`.

For a quick orientation of directories, see `docs/STRUCTURE.md`.

### Multi‑token Evaluation

Set `"metric_span": "full_target"` in any config to enable span-aware metrics that score the entire target vs foil continuation under teacher forcing. New metrics are added alongside existing first‑token metrics and flow into the standard tables:

- `seq_logprob_diff` — mean over examples of log p(target seq) − log p(foil seq) under the ablated model.
- `seq_p_drop` — mean drop in log p(target seq) from clean to ablated.
- `seq_kl_mean` — mean KL(p_clean || p_ablated) across continuation positions (target path).

This works with current single‑token corpora and generalizes automatically to multi‑token corpora when provided.

### Regenerate Standardized Reports

After running H1/H5/H6, regenerate all standardized exports (summaries, rankings, OV token tables, H5/H6 consolidated CSVs, manifest) with:

```
make postprocess
```

Key outputs land in `reports/` and are indexed by `reports/RESULTS_MANIFEST.json`.

## Reproduce the Geometric Signature

Run the activation‑entropy and curvature analysis for a given task (examples use GPT‑2 Medium, heads 0:2, 0:4, 0:7; 64 samples and 50 random controls):

```bash
source .venv/bin/activate
python -m lab.analysis.activation_entropy \
  --config lab/configs/run_h1_cross_condition_balanced.json \
  --tag facts \
  --device mps \
  --samples 64 \
  --heads 2 4 7 \
  --random-samples 50 \
  --entropy-methods subspace,diagonal,per_token \
  --output reports/activation_entropy_gpt2medium_facts_robust.json

# Repeat with --tag cf, neg, logic
```

Generate the figure shown above:

```bash
python paper/scripts/geometric_signature.py
# writes paper/figures/geometric_signature.{pdf,png}
```

Observed deltas for GPT‑2 Medium:
- Facts: ΔH_out = −2.44 (p < 0.02), Δ curvature (early) ≈ −14.6
- Counterfactual: ΔH_out = −3.49 (p < 0.02), Δ curvature (early) ≈ −15.2
- Negation: ΔH_out = −3.81 (p < 0.02), Δ curvature (early) ≈ −15.2
- Logic: ΔH_out = −3.08 (p < 0.02), Δ curvature (early) ≈ −16.1

---

## Explore Other Layers (Binder + Bottlenecks + Sharpeners)

Fast, exploratory scripts to probe beyond layer‑0:

- Binder‑head sweep (L6–L14) on a tiny synthetic binding dataset:

  ```bash
  python -m lab.analysis.binder_sweep \
    --model-name gpt2-medium \
    --device mps \
    --layer-range 6:14 \
    --output reports/binder_sweep_gpt2medium.json
  ```

  Outputs CSV/JSON with per‑head Δ metrics (ΔLD, Δacc, Δp_drop, ΔKL) to spot high‑impact “binder” heads.

- PCA rank curve (intrinsic dimension vs layer) on clean prompts of a condition:

  ```bash
  python -m lab.analysis.layer_pca_rank \
    --config lab/configs/run_h1_cross_condition_balanced.json \
    --tag facts \
    --model-name gpt2-medium \
    --device mps \
    --samples 256 \
    --var-frac 0.90 \
    --output reports/layer_pca_rank_gpt2medium_facts.json
  ```

  Writes a CSV and a simple figure plotting layer index vs PCs @ 90% variance.

- Late‑layer “sharpeners”: baseline entropy profile + scan heads in the last K layers:

  ```bash
  python -m lab.analysis.layer_entropy_and_sharpener_scan \
    --config lab/configs/run_h1_cross_condition_balanced.json \
    --tag facts \
    --model-name gpt2-medium \
    --device mps \
    --samples 128 \
    --last-k 3 \
    --output reports/layer_entropy_scan_gpt2medium_facts.json
  ```

  Heads with positive d_entropy_final (ablated − baseline) are candidates that force commitment late.

## Release and Submission

This repository is tagged with a release for the paper (e.g., `v1.0-suppressor-paper`). To access the exact version used for submission:

```
git checkout v1.0-suppressor-paper
pip install -e .
python3 scripts/verify_manifest.py  # Verify integrity
```

### For Reviewers

Download a bundle of key artifacts:

```
make bundle_review
# Creates: build/results_bundle_YYYYMMDD.tar.gz (+ .sha256)
```

Verify and extract:

```
sha256sum -c build/results_bundle_*.tar.gz.sha256 || shasum -a 256 -c build/results_bundle_*.tar.gz.sha256
tar -xzf build/results_bundle_*.tar.gz
```

The bundle includes the manifest, head rankings, OV reports, and docs for independent verification.

## Available Artefacts

- GPT‑2 head sweeps — `lab/runs/h1_cross_condition_physics_balanced_*`
- Mistral head sweeps — `lab/runs/h1_cross_condition_balanced_mistral_*`
- Pair and triplet ablations — `lab/runs/h5_layer0_*`
- Reverse patching — `lab/runs/h6_layer_targets_window_balanced_*`
- OV projections and partial patches — `reports/ov_report_*.json`, `reports/facts_partial_summary.json`
- Geometric signature reports — `reports/activation_entropy_gpt2medium_*_robust.json`
- Geometric signature reports — JSONs in `reports/` (see above). Figure scripts in `paper/scripts/` can regenerate plots locally.
- Stage‑1A pilot utilities — Task‑B weekday modular data (`lab/data/task_b_weekdays.jsonl`), circularity summaries (`reports/task_b_circularity_*.json`), and small‑model VDI runs (`reports/pilot_stage1a/vdi_layer0_*.csv`) generated via the new scripts in `scripts/`.

Feel free to inspect these directly or rerun analyses using the scripts referenced in `Results.md`.

## Development and Quality Checks

- Install dev dependencies: `pip install -e .[dev]`
- Enable git hooks: `pre-commit install`
- Run format/lint locally: `pre-commit run --all-files`
- Ensure reports manifest is consistent: `python scripts/verify_manifest.py`
- GitHub Actions runs the same checks plus `smoke_test.py` on every PR.

- `docs/suppressor_handover.md` — narrative overview, status, and next steps.
- `docs/REPLICATION.md` — definitive reproduction checklist with expected outputs.
- `Results.md` — mapping from each figure and table to the generating script and its inputs.

## Contributing

Bug reports, replication notes, and PRs are welcome. The harness targets macOS on Apple Silicon with PyTorch MPS, but now has full CUDA/NVIDIA support. CPU support is unoptimized but functional. Before opening a PR, please:

1. Run `python smoke_test.py` (MPS) or `python smoke_test_cuda.py` (CUDA).
2. Regenerate any touched figures via the scripts in `paper/scripts/`.
3. Ensure `cd paper && make` completes without errors.
4. For CUDA changes, verify on at least one NVIDIA GPU and document VRAM requirements.

## Citation

```bibtex
@misc{tinyablation2025suppressors,
  title        = {Layer-0 Suppressors Ground Hallucination Inevitability},
  author       = {Mat Thompson},
  year         = {2025},
  howpublished = {Tiny Ablation Lab, GitHub repository},
  note         = {\url{https://github.com/Mat-Tom-Son/tinyLab}}
}
```

For changelog details predating the suppressor work, see [CHANGELOG_v1.1.md](CHANGELOG_v1.1.md).
