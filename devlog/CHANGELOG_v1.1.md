# Tiny Ablation Lab v1.1 - Cross-Condition Baseline Upgrades

**Release Date**: 2025-01-28

## Overview

Version 1.1 transforms the lab from a single-condition rig into a **stable, reusable diagnostic container** for multi-condition (task/model) science. All upgrades are task-agnostic and enable systematic cross-condition analysis.

## New Features

### 1. Standardized Impact Tables

Every H1/H2 run now emits machine-readable impact matrices in long format for downstream analysis.

**H1 (heads_zero)**: `metrics/head_impact.parquet`
- Schema: `run_id`, `seed`, `layer`, `head`, `scale`, `metric`, `value`
- All metrics (logit_diff, p_drop, kl_div, acc_flip_rate) in long format
- Automatically added to `manifest.json`

**H2 (activation_patch)**: `metrics/layer_impact.parquet`
- Schema: `run_id`, `seed`, `layer`, `granularity`, `metric`, `value`
- Supports all granularities (layer_resid, mlp_out, head_out)
- Automatically added to `manifest.json`

**Benefits**:
- Easy to load and analyze with pandas/polars
- Standardized schema across experiments
- Ready for cross-condition aggregation

### 2. Cross-Condition Orchestrator

First-class support for running the same battery across multiple conditions.

**New module**: `lab/src/orchestrators/conditions.py`

**Usage**:
```bash
python3 -m lab.src.orchestrators.conditions lab/configs/scan_h1_example.json
```

**Config format**:
```json
{
  "run_name": "scan_h1",
  "battery": "lab/configs/battery_h1_heads_zero.json",
  "shared": {
    "seeds": [0,1,2],
    "device": "mps",
    "model": {"name":"gpt2-medium", "dtype":"float16"},
    ...
  },
  "conditions": [
    {"tag":"cond_a", "dataset": {...}},
    {"tag":"cond_b", "dataset": {...}}
  ]
}
```

**Outputs**:
- `artifacts/cross_condition/head_matrix.parquet` - All H1 results with `condition` column
- `artifacts/cross_condition/layer_matrix.parquet` - All H2 results with `condition` column
- `artifacts/cross_condition/summary.json` - Aggregated metrics per condition
- Parent `manifest.json` with paths to all outputs

### 3. CPU Verify Slice

Quick drift check for MPS determinism.

**Config extension**:
```json
{
  ...
  "verify_slice": {
    "device": "cpu",
    "n_examples": 20
  }
}
```

**Behavior**:
- Runs battery on last `n_examples` from active split
- Compares MPS vs CPU results using same seeds
- Emits `metrics/verify.json` with per-metric diff

**Output format**:
```json
{
  "device_main": "mps",
  "device_verify": "cpu",
  "n_examples": 20,
  "n_seeds": 3,
  "metrics": {
    "logit_diff": {
      "main": 2.34,
      "verify": 2.35,
      "abs_diff": 0.01
    },
    ...
  }
}
```

### 4. Invariants Detector

Automatically surface **conserved components** across conditions.

**New module**: `lab/src/aggregators/invariants.py`

**Usage**:
```bash
python3 -m lab.src.aggregators.invariants lab/runs/scan_h1_*/artifacts/cross_condition --k 10
```

**Process**:
1. Load concatenated impact matrices (head_matrix.parquet, layer_matrix.parquet)
2. For each condition, compute top-k heads/layers by metric
3. Find intersection (components in top-k for ALL conditions)

**Output**: `artifacts/cross_condition/invariants.json`
```json
{
  "k": 10,
  "metrics": ["logit_diff"],
  "heads": [
    {"layer": 11, "head": 3},
    {"layer": 11, "head": 7}
  ],
  "layers": [10, 11]
}
```

## Modified Files

### Core Ablations
- `lab/src/ablations/heads_zero.py` - Emit head_impact_table
- `lab/src/ablations/activation_patch.py` - Emit layer_impact_table

### Harness
- `lab/src/harness.py`
  - Collect and save impact tables
  - Support `verify_slice` config
  - Update manifest with new artifacts

### New Modules
- `lab/src/orchestrators/conditions.py` - Cross-condition runner
- `lab/src/aggregators/invariants.py` - Invariant detector

### Configs
- `lab/configs/scan_h1_example.json` - Example H1 cross-condition scan
- `lab/configs/scan_h2_example.json` - Example H2 cross-condition scan
- `lab/configs/run_h2_with_verify.json` - Example with CPU verify slice

### Tests
- `lab/tests/test_v1_1_features.py` - Schema validators for new features

## CLI Examples

**Single run with verification** (unchanged):
```bash
python3 -m lab.src.harness lab/configs/run_h2_with_verify.json
```

**Cross-condition scan**:
```bash
# Run across conditions
python3 -m lab.src.orchestrators.conditions lab/configs/scan_h1_example.json

# Find invariants
python3 -m lab.src.aggregators.invariants \
  lab/runs/scan_h1_cross_cond_*/artifacts/cross_condition --k 10
```

## Acceptance Criteria

All v1.1 features pass validation:

✅ **Impact tables**
- H1 runs emit `metrics/head_impact.parquet` with correct schema
- H2 runs emit `metrics/layer_impact.parquet` with correct schema
- Both added to `manifest.json`

✅ **Cross-condition runner**
- Produces `head_matrix.parquet` and/or `layer_matrix.parquet`
- Produces `summary.json` with per-condition stats
- All matrices contain `condition` column

✅ **CPU verify slice**
- Emits `metrics/verify.json` with per-metric diffs
- Added to `manifest.json`

✅ **Invariants**
- Finds and writes `invariants.json`
- Handles cases where fewer than k components exist

✅ **Backwards compatibility**
- All v1.0 configs run unchanged
- No new dependencies

## Testing Checklist

**Unit**:
```python3
from lab.tests.test_v1_1_features import *

# Test schemas
validate_head_impact_schema(df_head)
validate_layer_impact_schema(df_layer)
validate_verify_json(verify_data)
validate_cross_condition_matrix(df_matrix, ["cond_a", "cond_b"])
validate_invariants_json(invariants_data)
```

**Integration**:
```bash
# Run H2 with verify
python3 -m lab.src.harness lab/configs/run_h2_with_verify.json

# Run cross-condition H1
python3 -m lab.src.orchestrators.conditions lab/configs/scan_h1_example.json

# Find invariants
python3 -m lab.src.aggregators.invariants \
  lab/runs/scan_h1_*/artifacts/cross_condition --k 5
```

## Migration Guide

No breaking changes! Existing v1.0 configs work as-is.

**To use new features**:

1. **Impact tables**: Automatically generated, just check `manifest.json`
2. **Verify slice**: Add `"verify_slice": {"device": "cpu", "n_examples": 20}` to config
3. **Cross-condition**: Create new config following `scan_*_example.json` pattern
4. **Invariants**: Run aggregator after cross-condition orchestrator

## Performance Notes

- Impact tables add ~1-2MB per run (compressed parquet)
- Verify slice adds ~2x runtime for n_examples subset
- Cross-condition orchestrator is sequential (runs conditions one-by-one)
- Invariants aggregation is fast (<1s for typical datasets)

## Known Limitations

- Cross-condition orchestrator does not parallelize (future: joblib/ray)
- Verify slice only supports single device comparison (no multi-GPU)
- Invariants only support top-k intersection (no union or other set ops)

## Future Extensions (not in v1.1)

- Task-specific corpora and configs
- Hypothesis testing framework
- Metric geometry analysis
- Multi-token target/foil support
- Cross-layer pair ablations

---

**Ready for cross-condition science!** 🔬
