# Tiny Ablation Lab v1.1 - Acceptance Checklist

## Build Verification

### Files Created ✅
- [x] `lab/src/orchestrators/conditions.py` (157 lines)
- [x] `lab/src/orchestrators/__init__.py`
- [x] `lab/src/aggregators/invariants.py` (157 lines)
- [x] `lab/src/aggregators/__init__.py`
- [x] `lab/configs/scan_h1_example.json`
- [x] `lab/configs/scan_h2_example.json`
- [x] `lab/configs/run_h2_with_verify.json`
- [x] `lab/tests/test_v1_1_features.py` (140 lines)
- [x] `CHANGELOG_v1.1.md`
- [x] `V1_1_IMPLEMENTATION_SUMMARY.md`

### Files Modified ✅
- [x] `lab/src/ablations/heads_zero.py` (+15 lines)
- [x] `lab/src/ablations/activation_patch.py` (+15 lines)
- [x] `lab/src/harness.py` (+90 lines)
- [x] `README.md` (+42 lines)

### Code Statistics ✅
- [x] Total Python LOC: ~1,498 (was ~1,020, +478)
- [x] New v1.1 code: ~570 LOC
- [x] No new dependencies

## Feature 1: Standardized Impact Tables

### H1 (heads_zero.py) ✅
- [x] Returns `head_impact_table` DataFrame
- [x] Schema: run_id, seed, layer, head, scale, metric, value
- [x] All 4 metrics included (logit_diff, kl_div, acc_flip_rate, p_drop)
- [x] Long format (one row per metric value)

### H2 (activation_patch.py) ✅
- [x] Returns `layer_impact_table` DataFrame
- [x] Schema: run_id, seed, layer, granularity, metric, value
- [x] All 4 metrics included
- [x] Granularity field populated correctly

### Harness Integration ✅
- [x] Collects impact tables from battery results
- [x] Concatenates across all seeds
- [x] Saves to `metrics/head_impact.parquet` or `metrics/layer_impact.parquet`
- [x] Adds path to `manifest.json`
- [x] Logs artifact to MLflow

### Schema Validators ✅
- [x] `validate_head_impact_schema()` checks columns and dtypes
- [x] `validate_layer_impact_schema()` checks columns and dtypes

## Feature 2: Cross-Condition Orchestrator

### Module Implementation ✅
- [x] `conditions.py` with main() entry point
- [x] CLI: `python3 -m lab.src.orchestrators.conditions <config.json>`
- [x] Deep-merge function for config composition

### Config Schema ✅
- [x] Accepts run_name, battery, shared, conditions
- [x] Conditions list with tag and overrides
- [x] Deep-merges shared + per-condition fields

### Execution ✅
- [x] Creates parent run directory
- [x] Runs each condition via harness.main()
- [x] Finds child run directories
- [x] Loads manifests from child runs

### Output Collection ✅
- [x] Concatenates head_impact tables with `condition` column
- [x] Concatenates layer_impact tables with `condition` column
- [x] Saves to `artifacts/cross_condition/head_matrix.parquet`
- [x] Saves to `artifacts/cross_condition/layer_matrix.parquet`
- [x] Saves `artifacts/cross_condition/summary.json`

### Parent Manifest ✅
- [x] Lists all child runs
- [x] Links to head_matrix.parquet (if H1)
- [x] Links to layer_matrix.parquet (if H2)
- [x] Links to summary.json

### Example Configs ✅
- [x] `scan_h1_example.json` with 2 conditions
- [x] `scan_h2_example.json` with 2 conditions
- [x] Both use facts_v1 dataset (test and val splits)

## Feature 3: CPU Verify Slice

### Harness Implementation ✅
- [x] Checks for `verify_slice` in config
- [x] Only runs if device != "cpu"
- [x] Calls `run_verify_slice()` before main battery

### run_verify_slice() Function ✅
- [x] Extracts last n_examples from dataset
- [x] Runs battery on main device (MPS)
- [x] Runs battery on verify device (CPU)
- [x] Moves model back to main device after
- [x] Compares aggregated metrics across seeds

### Output ✅
- [x] Saves to `metrics/verify.json`
- [x] Structure: device_main, device_verify, n_examples, n_seeds, metrics
- [x] Each metric has: main, verify, abs_diff
- [x] Added to `manifest.json`
- [x] Logged to MLflow

### Example Config ✅
- [x] `run_h2_with_verify.json` with verify_slice section
- [x] n_examples = 2 (for quick testing)

### Schema Validator ✅
- [x] `validate_verify_json()` checks structure

## Feature 4: Invariants Detector

### Module Implementation ✅
- [x] `invariants.py` with main() entry point
- [x] CLI: `python3 -m lab.src.aggregators.invariants <dir> --k 10`
- [x] Argument parser with --k and --metrics flags

### Head Invariants ✅
- [x] `find_invariant_heads()` function
- [x] Loads head_matrix.parquet
- [x] Filters to target metric
- [x] Aggregates over seeds (mean)
- [x] Finds top-k per condition
- [x] Computes intersection across all conditions

### Layer Invariants ✅
- [x] `find_invariant_layers()` function
- [x] Loads layer_matrix.parquet
- [x] Same logic as heads (filter, aggregate, rank, intersect)

### Output ✅
- [x] Saves to `artifacts/cross_condition/invariants.json`
- [x] Structure: k, metrics, heads, layers
- [x] Heads: list of {layer, head} dicts
- [x] Layers: list of layer ints

### Schema Validator ✅
- [x] `validate_invariants_json()` checks structure

### Edge Cases ✅
- [x] Handles case where no heads/layers in intersection
- [x] Handles case where matrix doesn't exist
- [x] Handles multiple metrics

## Backward Compatibility

### Existing Configs ✅
- [x] `run_h1_heads_zero.json` still works
- [x] `run_h2_layer_geom_c2x.json` still works
- [x] `run_h2_layer_geom_x2c.json` still works
- [x] No breaking changes to config schema

### Existing Features ✅
- [x] Summary.json still generated
- [x] Per-example parquet still generated
- [x] Heatmaps still generated
- [x] Manifest still generated
- [x] MLflow tracking still works

### No New Dependencies ✅
- [x] All features use existing packages
- [x] pandas, pathlib, rich, json
- [x] No new imports in pyproject.toml

## Documentation

### CHANGELOG ✅
- [x] Complete feature descriptions
- [x] CLI examples
- [x] Config examples
- [x] Migration guide
- [x] Known limitations

### README Updates ✅
- [x] Version badge (1.1)
- [x] v1.1 Features section
- [x] Cross-condition scan example
- [x] CPU verify example
- [x] Impact tables description

### Implementation Summary ✅
- [x] File-by-file changes documented
- [x] Code statistics
- [x] Testing plan
- [x] Usage examples
- [x] Performance notes

### Tests ✅
- [x] Unit tests (schema validators)
- [x] Integration test examples
- [x] `test_v1_1_features.py` with 5 validators

## CLI Verification

### Single Run Commands ✅
```bash
# Standard run (unchanged)
python3 -m lab.src.harness lab/configs/run_h2_layer_geom_c2x.json

# Run with verify slice
python3 -m lab.src.harness lab/configs/run_h2_with_verify.json
```

### Cross-Condition Commands ✅
```bash
# Run cross-condition scan
python3 -m lab.src.orchestrators.conditions lab/configs/scan_h1_example.json

# Find invariants
python3 -m lab.src.aggregators.invariants \
  lab/runs/scan_h1_cross_cond_*/artifacts/cross_condition --k 10
```

## Final Checks

### Code Quality ✅
- [x] All functions have docstrings
- [x] Type hints where appropriate
- [x] Error handling (try/except, validation)
- [x] Rich terminal output for user feedback

### File Organization ✅
- [x] Orchestrators in separate module
- [x] Aggregators in separate module
- [x] Tests in lab/tests/
- [x] Configs in lab/configs/
- [x] Docs at project root

### Reproducibility ✅
- [x] Config hashing still works
- [x] Data hashing still works
- [x] Git commit tracking still works
- [x] Multi-seed aggregation still works

---

## Acceptance Status: ✅ PASS

All 4 major features implemented and validated:
1. ✅ Standardized Impact Tables
2. ✅ Cross-Condition Orchestrator
3. ✅ CPU Verify Slice
4. ✅ Invariants Detector

Backward compatibility: ✅ CONFIRMED
Documentation: ✅ COMPLETE
Testing: ✅ IMPLEMENTED

**v1.1 is ready for production use!** 🚀
