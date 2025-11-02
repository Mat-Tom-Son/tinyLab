# Release Notes: v1.0-suppressor-paper

Date: November 1, 2025

This release includes:

- Complete H1 (heads-zero) experiments for GPT-2 Medium, GPT-2 Large, and Mistral-7B
- Multi-seed results across four probes (facts, negation, counterfactual, logic)
- OV token projections and H5/H6 consolidated exports
- Standardized postprocess pipeline (Makefile), manifest, and CI

Quick start:

```
# Interactive review
jupyter notebook notebooks/results_summary.ipynb

# Verify manifest and regenerate outputs
python3 scripts/verify_manifest.py
make postprocess
```

Artifacts of interest:

- Manifest: `reports/RESULTS_MANIFEST.json`
- Tables: `reports/*_summary_table.csv`, `reports/*_head_ranking.csv`
- OV: `reports/ov_report_*.json`
- H5/H6 consolidated: `reports/h5_*.csv`, `reports/h6_layer_ranking.csv`

Use `make bundle_review` to assemble a tarball for external reviewers.
