# Results Overview

## Table 1 (Layer-0 suppressor impact)
- Source metrics: `paper/supplement/supplement.md` (config/data hashes, per-task means).
- Confidence intervals computed from `lab/runs/h1_cross_condition_*` summaries (GPT-2 Medium, seeds `{0,1,2}`).
- Regenerate manuscript table via `cd paper && make`.

## Figures
1. **Random L0 baselines** (`paper/figures/random_l0_baseline.pdf`)
   - Script: `python paper/scripts/random_l0_baseline.py`
   - Inputs: H1/H5 runs under `lab/runs/` (GPT-2 Medium).
2. **Path patch mediation** (`paper/figures/path_patch_dag.pdf`)
   - Script: `python paper/scripts/path_patch_figure.py`
   - Inputs: `reports/facts_partial_summary.json`, `reports/h6_reverse_patch_summary.json`.
3. **Calibration curves** (`paper/figures/calibration_curve.pdf`)
   - Script: `source .venv/bin/activate && python paper/scripts/calibration_curve.py`
   - Inputs: probe corpora in `lab/data/corpora/`, GPT-2 Medium weights via TransformerLens.

All commands assume the repository root as the working directory.
