# Contributing to Tiny Ablation Lab

Thanks for your interest in contributing! This project is an open, reproducible
workspace for mechanistic interpretability experiments. We welcome issues,
replication notes, and PRs that improve clarity, reliability, or extendability.

## Quick Start (Dev)

- Python 3.10+
- macOS + Apple Silicon recommended (MPS). CPU works; CUDA is untested here.

```
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
python smoke_test.py
```

## Reproducing Results

- Follow docs/REPLICATION.md to run H1/H5/H6.
- Then regenerate standard exports for review:
  - `make postprocess`
  - Open `notebooks/results_summary.ipynb` for a quick viewer.
  - All artifacts are indexed by `reports/RESULTS_MANIFEST.json`.

## Coding Style

- Install hooks: `pre-commit install`
- Run `pre-commit run --all-files` before pushing.

- Black (line length 88) and Ruff rules are configured in `pyproject.toml`.
- Prefer small, focused changes; keep new code close to existing patterns.
- Add/extend scripts under `lab/analysis/` for post‑processing rather than inline notebooks.

## Tests / Validation

- Keep `smoke_test.py` green.
- If you add new post‑processing, ensure it can run from a clean clone with only
  `lab/runs/` present and produces outputs under `reports/`.

## Datasets & Models

- Datasets live under `lab/data/`; tokenizer‑aware corpora can be built with
  `scripts/build_tokenizer_variants.py`.
- TransformerLens is used for model loading; ensure new configs use
  explicit `model.family/name/dtype`.

## Submitting PRs

- Describe the goal, scope, and how to validate.
- Include commands run and outputs touched (preferably in `reports/`).
- Avoid reformatting unrelated files; keep diffs reviewable.

Thanks again for helping make Tiny Ablation Lab easy to read, reproduce, and extend!
