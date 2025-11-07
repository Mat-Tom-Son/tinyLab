# Suppressor Study Reproduction Guide

This note walks through regenerating every result in the "Layer-0 Suppressors Ground Hallucination Inevitability" paper.

## 0. Prerequisites

- macOS on Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- ~60 GB free disk (models, runs, reports)
- Hugging Face CLI (only required if your account needs to accept model terms)

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
python smoke_test.py  # optional sanity check
```

The harness assumes PyTorch MPS; set `"device": "cpu"` in configs if you need CPU fallback (slow).

Multi-token evaluation: set `"metric_span": "full_target"` to add span-aware metrics (`seq_logprob_diff`, `seq_p_drop`, `seq_kl_mean`) while preserving first-token metrics for comparability. These flow into the standard parquet tables.

## 1. Data & Tokeniser Variants

The repository ships the exact corpora used:

- `lab/data/corpora/*_single_token_v1_balanced.jsonl`
- `lab/data/corpora/*_single_token_v1_mistral.jsonl`
- Matching splits in `lab/data/splits/`

To regenerate the Mistral-specific corpora:

```bash
python scripts/build_tokenizer_variants.py \
  --tokenizer mistralai/Mistral-7B-v0.1 \
  --suffix mistral \
  --datasets \
    facts_single_token_v1 \
    negation_single_token_v1 \
    counterfactual_single_token_v1 \
    logical_single_token_v1
```

## 2. GPT-2 Experiments

### Cross-condition head ablations (H1)

```bash
python -m lab.src.orchestrators.conditions \
  lab/configs/run_h1_cross_condition_physics_balanced.json
```

This emits multi-seed runs under `lab/runs/h1_cross_condition_physics_balanced_*` with per-condition `head_impact.parquet` tables.

### Pair/triplet cooperation (H5)

```bash
python -m lab.src.orchestrators.conditions \
  lab/configs/run_h5_layer0_triplet_balanced.json
```

### Reverse patching (H6)

```bash
python -m lab.src.orchestrators.conditions \
  lab/configs/run_h6_layer_targets_window_balanced.json
```

Outputs land in `lab/runs/h6_layer_targets_window_balanced_*`.

## 3. Mistral-7B Experiments

The orchestrated config targets all four probe families with seeds `[0,1,2,3,4]` and stores a 32-example CPU verify slice before each sweep:

```bash
python -m lab.src.orchestrators.conditions \
  lab/configs/run_h1_cross_condition_balanced_mistral.json
```

If orchestration timeouts occur, run the per-condition configs (all set to the same five seeds and verify slice):

```bash
python -m lab.src.harness lab/configs/run_h1_cross_condition_balanced_mistral_facts.json
python -m lab.src.harness lab/configs/run_h1_cross_condition_balanced_mistral_neg.json
python -m lab.src.harness lab/configs/run_h1_cross_condition_balanced_mistral_cf.json
python -m lab.src.harness lab/configs/run_h1_cross_condition_balanced_mistral_logic.json
```

### Pair/triplet analyses

- Pair sweep (counterfactual focus):
  ```bash
  python -m lab.src.harness lab/configs/run_h5_layer0_pairs_balanced_mistral_cf_minimal.json
  ```
- Triplet variants:
  ```bash
  python -m lab.src.harness lab/configs/run_h5_layer0_triplet_balanced_mistral_*.json
  ```
  (facts/neg/cf/logic corrected configs match the paper; each defaults to seeds `[0,1,2]` plus a 24-example CPU verify slice.)

Each run logs to `lab/runs/` with matching names.


After all probes complete, regenerate standardized reports with:

```
make postprocess
```

This refreshes `reports/` (head rankings, OV tokens, H5/H6 tables) and rebuilds
`reports/RESULTS_MANIFEST.json` for reviewers.

## 4. Analysis Scripts → Figures / Tables

All figure/table scripts live in `paper/scripts/`.

| Output | Command | Inputs |
| --- | --- | --- |
| Random L0 baseline (Fig. 1) | `python paper/scripts/random_l0_baseline.py` | H1/H5 GPT-2 runs |
| Path patch DAG (Fig. 2) | `python paper/scripts/path_patch_figure.py` | `reports/facts_partial_summary.json`, `reports/h6_reverse_patch_summary.json` |
| Calibration diagram (Fig. 3) | `python paper/scripts/calibration_curve.py` | Probe corpora, GPT-2 weights |
| OV token tables (Appendix) | `python paper/scripts/token_tables.py` | `reports/ov_report_*.json` |

`Results.md` repeats this mapping for quick reference.

## 5. Regenerate the Paper

```bash
cd paper
make      # runs latexmk + bibtex
open main.pdf
```

All intermediate artefacts (figures, tables, bibliography) are rebuilt from source.

## 6. Inspect Existing Runs

- **Per-run metrics**: `lab/runs/<run_id>/metrics/summary.json`
- **Head/layer tables**: `lab/runs/<run_id>/metrics/head_impact.parquet`, `layer_impact.parquet`
- **Artefacts**: `artifacts/impact_heatmap.*`, `artifacts/cross_condition/`
- **Hash provenance**: `config.json`, `config_hash.txt`, `data_hash.txt`, `git_commit.txt`

## 7. Hardware Notes

- GPT-2 Medium runs finish in ~20–30 minutes per config on an M3 Max (3 seeds).
- Mistral-7B runs take 60–90 minutes per config; the harness checkpoints progress if interrupted.
- For reproducibility sanity checks, the `verify_slice` option in configs runs a CPU slice (disabled by default).

## 8. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| Hugging Face 403 on Mistral | Model requires accepting license | `huggingface-cli login` then accept terms on HF |
| `NaN` KL terms | Deterministic completions (logits saturate) | Expected; handled in scripts |
| `RuntimeError: MPS` | Older macOS/PyTorch | Ensure macOS ≥ 14 and PyTorch ≥ 2.2, rerun `scripts/setup_env.sh` |

## 9. Cross-Platform Validation

We repeated the GPT-2 Medium H1 suppressor experiments on an NVIDIA CUDA backend (torch 2.9.0, CUDA 12.8) using the exact corpora and seeds {0, 1, 2}. Logit-difference means remained in the expected 1.23–1.64 band, and the layer‑0 suppressor trio (heads 0:2, 0:4, 0:7) led every probe exactly as on Apple M‑series hardware. VRAM profiling during the higher-capacity config peaked at ≈2.3 GB, confirming that batch_size = 4 with span-aware metrics comfortably fits on 8 GB GPUs. Mistral‑7B validation on CUDA is deferred: loading the 7B weights requires >16 GB VRAM/RAM, so the Apple M‑series results in the main text remain authoritative for that model. All CUDA artefacts (metrics, head rankings, VRAM logs) are bundled under `paper/supplement/cuda_validation/`.

Questions? Open an issue on GitHub or ping via repo discussions.
