# Reproducibility Guide

This folder contains the LaTeX manuscript, supplementary tables, and build harness for the layer-0 suppressor study.

## Dependencies
- TeX Live 2025 (or any distribution with `pdflatex`, `latexmk`, `siunitx`, `booktabs`, `hyperref`)
- Python 3.10+
- Poetry/virtualenv not required for compilation (all data rendered in-table).

## Build steps
```bash
cd paper
make  # runs latexmk -pdf main.tex
```
The compiled PDF is written to `paper/main.pdf`. Intermediate artefacts live alongside the source and can be cleaned with `make clean` (preserves the PDF) or `make distclean` (removes the PDF).

## Supplementary materials
- `supplement/supplement.md` – configuration/data hashes, per-head rankings, OV token lists, statistical notes.
- `internal_review.md` – internal peer-review feedback and resolution notes.

## Data + run references
All experiments referenced in the paper live under `lab/runs/` with immutable `config_hash.txt` and `data_hash.txt` files. The supplementary table provides the mapping from run label to hashes for fast verification.

## Blog post
A 1,000-word explainer for practitioners is located at `docs/blog_post.md` and can be cross-linked from the release announcement.
