# Tiny Ablation Lab - Quick Start Guide

Get running in 10 minutes.

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- ~5GB disk space (for models + venv)

## Setup (5 minutes)

### 1. Install Dependencies

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

This will:
- Create virtual environment
- Install PyTorch, TransformerLens, MLflow, etc.
- Validate MPS availability

Expected output:
```
--- Setup Validation ---
MPS available: True
macOS: 14.x.x, Python: 3.10.x
PyTorch: 2.x.x
------------------------
```

### 2. Verify Setup

```bash
python3 smoke_test.py
```

Should see:
```
=== Tiny Ablation Lab Smoke Test ===

1. Checking MPS availability...
   ✓ MPS available
2. Loading GPT-2 small...
   ✓ Model loaded (12 layers, 12 heads)
...
=== ✓ All checks passed! ===
```

### 3. Check Sample Data

```bash
cat lab/data/corpora/facts_v1.jsonl
cat lab/data/splits/facts_v1.split.json
```

You should see 10 factual examples and a split file with train/val/test indices.

## Run First Experiment (3 minutes)

### Option A: Quick Test (H2, Small Model, 1 Seed)

Edit `lab/configs/run_h2_layer_geom_c2x.json`:
```json
{
  "seeds": [0],  // Change from [0,1,2] to [0] for speed
  ...
}
```

Run:
```bash
python3 -m lab.src.harness lab/configs/run_h2_layer_geom_c2x.json
```

**Runtime**: ~2-3 minutes on M1/M2

### Option B: Full Multi-Seed Run

```bash
python3 -m lab.src.harness lab/configs/run_h2_layer_geom_c2x.json
```

**Runtime**: ~6-8 minutes (3 seeds)

## View Results

### 1. Terminal Output

You'll see:
```
Loading model: gpt2-small to mps with torch.float32
Loaded 2 examples from split 'test'
[blue]Starting battery 'activation_patch' for N=3 seeds...[/blue]
[cyan]Running seed 0...[/cyan]
...
[green]Done[/green]. Run dir: lab/runs/h2_layer_geom_c2x_<hash>
```

### 2. Check Run Directory

```bash
ls lab/runs/h2_layer_geom_c2x_*/
```

You'll find:
- `config.json` - Exact config used
- `config_hash.txt` - Reproducibility hash
- `data_hash.txt` - Dataset fingerprint
- `git_commit.txt` - Code version
- `metrics/summary.json` - Aggregated results
- `metrics/per_example.parquet` - Per-example predictions
- `artifacts/impact_heatmap.html` - Interactive viz

### 3. View Summary

```bash
cat lab/runs/h2_layer_geom_c2x_*/metrics/summary.json | python3 -m json.tool
```

Example output:
```json
{
  "logit_diff": {
    "mean": 2.34,
    "ci95": [2.10, 2.58],
    "values": [2.20, 2.45, 2.38]
  },
  "kl_div": { ... },
  ...
}
```

### 4. Open Heatmap

```bash
open lab/runs/h2_layer_geom_c2x_*/artifacts/impact_heatmap.html
```

You'll see a Plotly interactive heatmap showing per-layer impact.

### 5. MLflow UI (Optional)

```bash
mlflow ui --backend-store-uri file://$(pwd)/mlruns
```

Open http://localhost:5000 to browse all experiments.

## Run More Experiments

### H1: Find Critical Heads

```bash
python3 -m lab.src.harness lab/configs/run_h1_heads_zero.json
```

**Runtime**: ~15-20 minutes (12 layers × 12 heads × 3 seeds = 432 runs)

**Output**: Heatmap showing which heads are most critical.

### H6: Test Asymmetry

```bash
# Clean -> Corrupt
python3 -m lab.src.harness lab/configs/run_h2_layer_geom_c2x.json

# Corrupt -> Clean
python3 -m lab.src.harness lab/configs/run_h2_layer_geom_x2c.json
```

Compare the two heatmaps to see if patching is symmetric.

### H5: Find Backup Circuits

First, generate pairs from H1 results:
```bash
python3 scripts/make_pairs_from_h1.py lab/runs/h1_heads_zero_* --top-k 5
```

Creates `lab/configs/battery_h5_pairs.json`.

Then create a main config pointing to it and run.

## Troubleshooting

### "MPS not available"

Check macOS version:
```bash
sw_vers
```

MPS requires macOS 12.3+. If not available, experiments will run on CPU (slower but works).

### "Data hash mismatch"

Regenerate split:
```bash
python3 scripts/facts_make_split.py facts_v1 --seed 13
```

### "Model download fails"

Check internet connection. TransformerLens downloads from HuggingFace.

If behind firewall, set:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Out of memory

Reduce batch size in config:
```json
{
  "batch_size": 4,  // Default is 8
  ...
}
```

Or use smaller model:
```json
{
  "model": {
    "name": "gpt2-small",  // Instead of gpt2-medium
    "dtype": "float16"     // Instead of float32
  }
}
```

## Next Steps

1. **Add Your Data**: Replace `facts_v1.jsonl` with your own dataset
2. **Tune Configs**: Adjust seeds, batch_size, layers, heads
3. **Extend Ablations**: Add new battery types in `lab/src/ablations/`
4. **Implement SAE**: Fill in `sae_train.py` and `sae_toggle.py`
5. **Add Viz**: Create feature panels, UMAP plots in `lab/src/viz/`

## Useful Commands

```bash
# List all runs
ls -lt lab/runs/

# Find most recent run
ls -t lab/runs/ | head -1

# View summary of latest run
cat lab/runs/$(ls -t lab/runs/ | head -1)/metrics/summary.json | python3 -m json.tool

# Clean old runs (careful!)
rm -rf lab/runs/*

# Clean MLflow data
rm -rf mlruns/*
```

## Getting Help

- Check [README.md](README.md) for full documentation
- Check [BUILD_SUMMARY.md](BUILD_SUMMARY.md) for architecture details
- Check `lab/tests/test_determinism.py` for MPS validation
- Check `smoke_test.py` for component validation

## Performance Tips

1. **Use float16**: Cuts memory in half, minimal accuracy loss
2. **Batch size**: 4-8 for sweeps, 16+ for single runs
3. **Subset layers**: Set `"layers": [6,7,8,9,10,11]` instead of `"all"`
4. **Subset heads**: Set `"heads": [0,1,2]` for quick tests
5. **Single seed first**: Use `"seeds": [0]` to validate before full run

## Expected Runtimes (M1 Pro, MPS, float32)

| Experiment | Model      | Seeds | Runtime  |
|------------|------------|-------|----------|
| H2 (c→x)   | gpt2-small | 1     | ~2 min   |
| H2 (c→x)   | gpt2-small | 3     | ~6 min   |
| H1 (all)   | gpt2-small | 3     | ~20 min  |
| H2 (c→x)   | gpt2-medium| 3     | ~15 min  |

Runtimes scale linearly with:
- Number of seeds
- Dataset size
- Model size
- Sweep size (layers × heads)

---

**Ready to explore!** 🚀
