# RESULTS_MANIFEST.json

The file `reports/RESULTS_MANIFEST.json` indexes all standardized outputs
produced by Tiny Ablation Lab post‑processing scripts. It is organized by model
key (e.g., `mistral`, `gpt2_medium`, `gpt2_large`, `gpt2m`, `gpt2l`). Each entry
contains:

- `summary_table` – CSV of per‑probe logit‑diff means and seed counts.
- `trio_percentiles` – JSON with suppressor trio percentiles per probe.
- `head_rankings` – map of probe→CSV paths (full/top/bottom slices).
- `ov_reports` – list of OV token report JSONs for that model.

Regenerate it after runs with:

```
make manifest
```

Or regenerate everything:

```
make postprocess
```

You can quickly preview the results using
`notebooks/results_summary.ipynb`, which reads this manifest and renders the
key tables.
