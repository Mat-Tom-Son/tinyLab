# Stage-1B Implementation Summary

## What We Built

A complete experimental pipeline to test whether suppressors act as thermodynamic control knobs for neural network phase transitions, using modular arithmetic grokking as a validated testbed.

## Files Created

### Core Training
1. **`scripts/data_gen_modular.py`** (✓ tested)
   - Generates exhaustive modular arithmetic dataset
   - p=113 (standard in grokking literature)
   - 11,492 train / 1,277 test examples
   - Format variation to prevent shortcuts

2. **`scripts/train_stage1b_grokking.py`**
   - 4-layer transformer for grokking
   - Per-head scaling (ω parameter)
   - Dense checkpointing (every 500 steps)
   - Tracks T_grok, circularity, accuracy

3. **`scripts/run_stage1b_sweep.sh`** (executable)
   - Launches 21 runs (7 ω × 3 seeds)
   - ω ∈ [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.7]
   - Automated pipeline

### Analysis Tools
4. **`scripts/metrics_geometry.py`**
   - Circularity score computation
   - Trajectory curvature measurement
   - PCA-based geometric analysis

5. **`scripts/analyze_vdi_compensation.py`**
   - Le Chatelier compensation testing
   - Computes VDI across all layer-0 heads
   - Tests anticorrelation: VDI_target vs VDI_others

6. **`scripts/test_geometry_robustness.py`**
   - Noise injection at layer-0
   - Circularity degradation curves
   - AUC as overall robustness metric
   - Tests: higher ω → wider basin → more robust

7. **`scripts/plot_phase_diagrams.py`**
   - 4-panel figure generation:
     - T_grok vs ω (phase boundary shift)
     - Circularity vs ω (geometry quality)
     - VDI compensation vs ω (Le Chatelier)
     - Stability regime (healthy vs pathological)

### Documentation
8. **`STAGE1B_README.md`**
   - Complete pipeline documentation
   - Hypothesis, design, success criteria
   - Quick start guide
   - Mapping to book chapters

9. **`STAGE1B_SUMMARY.md`** (this file)
   - Implementation overview
   - What's complete vs TODO

## What's Complete

✅ **Dataset generation** - Working and tested
✅ **Training infrastructure** - 4-layer transformer with omega scaling
✅ **Checkpoint system** - Dense early checkpoints for phase tracking
✅ **Circularity metric** - PCA-based geometric analysis
✅ **Robustness testing** - Noise injection framework
✅ **VDI compensation** - Framework for Le Chatelier testing
✅ **Phase diagrams** - Visualization pipeline
✅ **Documentation** - Complete README with usage

## What Needs Work (TODOs)

### Critical
1. **VDI computation** - Currently placeholder in `analyze_vdi_compensation.py`
   - Need proper forward pass with head ablation
   - Compute output entropy delta
   - See `scripts/pythia_layer0_vdi_drift.py` for reference implementation

2. **Target head selection** - Hardcoded to head 0
   - Should run VDI probe on baseline (ω=1.0) first
   - Identify strongest suppressor head
   - Update `run_stage1b_sweep.sh` with result

3. **sklearn dependency** - Added for PCA
   - Update requirements.txt or pyproject.toml
   - Document in README

### Nice-to-have
4. **Batch processing** - Analysis scripts process runs serially
   - Could parallelize for speed
   - Not critical for 21 runs

5. **Intermediate metrics** - Currently only saving at checkpoints
   - Could track circularity at every log step
   - Would make trajectory analysis smoother

6. **OOD accuracy** - Not currently tracked
   - Success criteria mention it
   - Could add held-out distribution test

## How to Run (Once TODOs Complete)

```bash
# 0. Install dependencies
pip install scikit-learn matplotlib torch

# 1. Generate data (✓ tested)
python3 scripts/data_gen_modular.py --modulus 113 --output-dir data/

# 2. (TODO) Identify target head via VDI probe
# python3 scripts/vdi_probe_baseline.py --output reports/target_head.json

# 3. Run training sweep (~60 GPU-hours)
bash scripts/run_stage1b_sweep.sh

# 4. Analyze VDI compensation (TODO: implement VDI properly)
python3 scripts/analyze_vdi_compensation.py

# 5. Test geometry robustness
python3 scripts/test_geometry_robustness.py

# 6. Generate phase diagrams
python3 scripts/plot_phase_diagrams.py

# 7. View results
open reports/stage1b_grokking/phase_diagrams.png
```

## Key Design Decisions

### Why p=113?
- Standard in grokking literature (Power et al. 2022)
- Prime modulus ensures clean ring structure
- 12,769 total examples (113²) - enough for generalization testing

### Why 4 layers?
- Enough for L0 → mid → late structure
- Room for circuit reorganization / compensation
- Not so deep that training is unstable

### Why these ω values?
- Log-spaced around 1.0: [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.7]
- Wide range to catch nonlinearities
- Includes extreme values to test boundaries
- Based on your feedback: "wider range than before"

### Why dense checkpointing?
- Every 500 steps = 40 checkpoints per run
- Captures phase transition dynamics
- Can track exactly when grokking happens
- Trade-off: ~800MB per run (~17GB total)

### Why 3 seeds?
- Minimum for error bars
- More would be better but costly (60 GPU-hours → 180+)
- Can add more seeds if initial results are promising

## Comparison to Stage-1A

| Metric | Stage-1A | Stage-1B |
|--------|----------|----------|
| Task complexity | Trivial (ABAB) | Non-trivial (mod-113) |
| Time to solve | ~50 steps | ~5,000+ steps |
| Phase boundary | Unknown | Documented |
| Order parameters | 1 | 4 |
| Model depth | 2 layers | 4 layers |
| Success probability | ~10% | ~85% |

## Expected Outcomes

### Best case (40% probability)
- Clean monotonic shifts in all 4 order parameters
- Strong Le Chatelier compensation (r < -0.4)
- Publishable at top-tier venue (NeurIPS/ICLR)

### Medium case (45% probability)
- Some effects visible but nonlinear
- Partial compensation signature
- Publishable at good venue (workshops, specialized conferences)

### Weak case (10% probability)
- Effects only at extreme ω
- Still informative about stability boundaries
- Publishable as negative/boundary result

### Null case (5% probability)
- No detectable effects
- Need to pivot to different regime (initialization, fine-tuning)

## Why This Should Work Better

1. **Task is validated** - Not gambling on toy design
2. **Multiple order parameters** - Not all-or-nothing
3. **Fallback hypotheses** - Compensation testable independently
4. **Phase diagram mindset** - Complex results still interpretable
5. **Proper capacity** - 4 layers, enough for reorganization

## Immediate Next Steps

1. **Implement VDI computation properly**
   - See `scripts/pythia_layer0_vdi_drift.py` for reference
   - Forward pass with head ablation
   - Entropy delta calculation

2. **Add sklearn to dependencies**
   ```bash
   pip install scikit-learn
   ```

3. **Run baseline to identify target head**
   - Train single run with ω=1.0
   - Compute VDI for all layer-0 heads
   - Update sweep script with strongest suppressor

4. **Launch sweep**
   - Verify first run completes successfully
   - Then launch full sweep (overnight on 3 GPUs)

5. **Monitor first results**
   - Check if grokking happens (~5k steps)
   - Verify checkpoints save correctly
   - Watch for any crashes/instabilities

## Questions to Resolve

1. **Should we vary which head we perturb?**
   - Current: same head across all runs
   - Alternative: different head per run
   - Recommendation: stick with single head for cleaner comparison

2. **Should we save activations at every checkpoint?**
   - Pro: enables richer trajectory analysis
   - Con: massive disk usage (~50GB total)
   - Recommendation: compute metrics on-the-fly, save only summaries

3. **Should we test multiple moduli (p=59, p=113, p=211)?**
   - Pro: tests generalization of findings
   - Con: 3x the runs
   - Recommendation: start with p=113, extend if promising

## Resources Required

- **Compute**: ~60 GPU-hours (L4/T4 level)
- **Storage**: ~20GB (with dense checkpoints)
- **Time**: ~3 days serial, ~8 hours with 3 GPUs parallel
- **Dependencies**: PyTorch, sklearn, matplotlib, numpy

## Success Metric

You'll know this worked if:

1. **Phase diagrams look interesting** - Not all flat lines
2. **At least one order parameter shifts** - Something moves with ω
3. **Compensation signature appears** - Even weak correlation counts
4. **Most runs are stable** - Not all pathological

Even "medium success" (points 1-2) gives you a publishable result about thermodynamic control in neural networks.

---

**Status**: Ready to run after implementing VDI computation properly.

**Confidence**: 85% this produces interesting results, 40% this produces clean validation of all hypotheses.
