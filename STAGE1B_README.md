# Stage-1B: Modular Arithmetic Grokking Experiment

## Overview

This is an improved experimental design to test the thermodynamic hypothesis that suppressors act as Le Chatelier stabilizers during neural network training.

**Key improvement over Stage-1A:** Uses modular arithmetic (p=113) with documented grokking behavior, instead of trivial ABAB sequences.

## Hypothesis

Suppressors at the bottleneck (layer-0 heads) control the phase transition:

1. **Timing**: Scaling suppressor strength (ω) shifts T_grok (when grokking occurs)
2. **Geometry**: Higher ω → delayed crystallization → more robust circular geometry
3. **Compensation**: When we perturb one suppressor, other heads compensate (Le Chatelier signature)

## Experimental Design

### Task
- **Problem**: (a + b) mod 113 = ?
- **Why p=113**: Prime modulus, standard in grokking literature (Power et al. 2022)
- **Phase transition**: Models show dramatic sudden transition from ~50% to >95% accuracy
- **Geometry**: Activations form circular structure in representation space

### Model
- **Architecture**: 4-layer transformer (enough for L0 → mid → late structure)
- **Heads**: 8 per layer
- **Hidden dim**: 256
- **Total params**: ~2M (small but not toy)

### Perturbation
- **Control parameter**: ω (scaling factor for layer-0 head)
- **Range**: [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.7]
- **Seeds**: 3 per ω value
- **Total runs**: 21

### Order Parameters
1. **T_grok**: First step where test accuracy ≥ 0.9
2. **Circularity**: How well layer-0 activations form a circle
3. **VDI compensation**: Total suppression across other heads
4. **Robustness**: Circularity degradation under noise injection

## Pipeline

### 1. Data Generation

```bash
python scripts/data_gen_modular.py \
    --modulus 113 \
    --train-fraction 0.9 \
    --output-dir data/
```

**Output**: `data/modular_p113_{train,test}.jsonl`

### 2. Training Sweep

```bash
bash scripts/run_stage1b_sweep.sh
```

This runs 21 training jobs (7 ω × 3 seeds).

**Expected runtime**:
- Per run: ~2-3 hours on GPU (20k steps)
- Serial: ~60 hours
- Parallel (3 GPUs): ~20 hours

**Output structure**:
```
reports/stage1b_grokking/train/
├── stage1b_head0_omega0.3_seed0/
│   ├── config.json
│   ├── metrics.jsonl
│   ├── final_model.pt
│   └── checkpoints/
│       ├── step_00500.pt
│       ├── step_01000.pt
│       └── ...
├── stage1b_head0_omega0.5_seed0/
└── ...
```

### 3. VDI Compensation Analysis

```bash
python scripts/analyze_vdi_compensation.py \
    --base-dir reports/stage1b_grokking/train \
    --target-head 0
```

**Tests**: Does VDI across other heads anticorrelate with ω?

**Expected**: Negative correlation (Le Chatelier compensation)

**Output**: `reports/stage1b_grokking/vdi_compensation.json`

### 4. Geometry Robustness Testing

```bash
python scripts/test_geometry_robustness.py \
    --base-dir reports/stage1b_grokking/train \
    --noise-levels 0.0 0.1 0.2 0.5 1.0
```

**Tests**: How does circularity degrade under noise?

**Expected**: Higher ω → higher AUC (more robust)

**Output**:
- `reports/stage1b_grokking/geometry_robustness.json`
- `reports/stage1b_grokking/robustness_curves.png`

### 5. Phase Diagrams

```bash
python scripts/plot_phase_diagrams.py \
    --base-dir reports/stage1b_grokking/train
```

**Generates 4-panel figure**:
1. T_grok vs ω (phase boundary shift)
2. Final circularity vs ω (geometry quality)
3. VDI compensation vs ω (Le Chatelier signature)
4. Stability regime (healthy vs pathological runs)

**Output**: `reports/stage1b_grokking/phase_diagrams.png`

## Success Criteria

### Strong Success (40% probability)
- ✓ T_grok shifts monotonically with ω
- ✓ Circularity robustness correlates with ω (r > 0.4)
- ✓ VDI compensation anticorrelates with ω (r < -0.3)
- ✓ All ω values in [0.5, 1.5] are stable (acc > 0.7)

**Interpretation**: Clean validation of thermodynamic control hypothesis

### Medium Success (45% probability)
- ✓ T_grok shifts (but nonlinearly)
- ✓ Some geometry quality variation with ω
- ✓ Partial compensation signature visible
- ✓ Most ω values stable

**Interpretation**: Thermodynamic control exists but phase structure is complex

### Weak Success (10% probability)
- ✓ Effects visible only at extremes (ω < 0.5 or ω > 1.5)
- ✓ Pathologies emerge at boundaries

**Interpretation**: Suppressor strength has stability boundaries (still publishable)

### Null Result (5% probability)
- ✗ All ω converge to same T_grok (variance < 500 steps)
- ✗ No compensation signature (correlation ≈ 0)
- ✗ Geometry quality independent of ω

**Interpretation**: This regime doesn't engage developmental control (need to pivot)

## Key Differences from Stage-1A

| Aspect | Stage-1A (Failed) | Stage-1B (Current) |
|--------|------------------|-------------------|
| **Task** | ABAB sequences (invented) | Modular arithmetic (validated) |
| **Phase boundary** | Unknown if exists | Documented in literature |
| **Circuit complexity** | Trivial (solved in ~50 steps) | Non-trivial (grokking at ~5k+ steps) |
| **Model depth** | 2 layers | 4 layers |
| **Order parameters** | 1 (accuracy only) | 4 (timing, geometry, VDI, robustness) |
| **Hypothesis tests** | 1 (does timing shift?) | 4 (timing, quality, compensation, stability) |
| **Success modes** | Binary (works or doesn't) | Graded (strong/medium/weak/null) |

## Why This Should Work

1. **Validated task**: Grokking on mod-113 is well-documented (not a gamble)
2. **Multiple observables**: Not all-or-nothing (4 independent tests)
3. **Fallback hypotheses**: Compensation testable even if timing is messy
4. **Phase diagram mindset**: Complex/nonlinear results still interpretable
5. **Capacity for compensation**: 4 layers × 8 heads = enough room for circuit reorganization

## Mapping to Book Chapters

### Chapter 3: Gatekeepers of Doubt
> "Suppressors inject uncertainty at the bottleneck. When we scale them (ω), we shift the balance between exploration and commitment."

**Evidence**: T_grok shifts, geometry quality varies with ω

### Chapter 5: Structured Noise
> "In the sub-saturated regime—early training—suppressors control the 'temperature'. Higher ω = warmer system = longer plasticity window."

**Evidence**: Robustness testing shows wider basins at higher ω

### Chapter 6: Crystallization
> "The phase transition (grokking) happens when the system cools enough. We show ω tunes when this happens, without breaking the final structure."

**Evidence**: Phase diagrams show boundary shift

### Chapter 7: Critical Window
> "By scaling suppressors during training, we shift the critical window—the model 'grows up' faster or slower. This is the thermostat of development."

**Evidence**: T_grok vs ω plot

### Epilogue: Legibility
> "Le Chatelier compensation: when we perturb one suppressor, others respond to maintain balance. The system is regulating itself according to physical principles, not learned behavior."

**Evidence**: VDI compensation signature (correlation plot)

## Quick Start

```bash
# 1. Generate data
python scripts/data_gen_modular.py --modulus 113 --output-dir data/

# 2. Run sweep (this takes a while!)
bash scripts/run_stage1b_sweep.sh

# 3. Analyze results
python scripts/analyze_vdi_compensation.py
python scripts/test_geometry_robustness.py
python scripts/plot_phase_diagrams.py

# 4. View phase diagrams
open reports/stage1b_grokking/phase_diagrams.png
```

## Next Steps After Running

1. **If strong success**: Write up for NeurIPS/ICLR, emphasize Le Chatelier prediction
2. **If medium success**: Dig into nonlinear phase structure, publish as "complex developmental control"
3. **If weak success**: Study boundary pathologies, frame as "stability regime identification"
4. **If null**: Pivot to earlier interventions (initialization) or later (fine-tuning)

## Notes

- VDI computation is currently a placeholder - needs proper implementation
- Consider pre-running a baseline (ω=1.0) to identify strongest suppressor head via VDI probe
- Checkpoint schedule is dense early (every 500 steps) - can thin for disk space
- sklearn dependency added for PCA (needed for circularity computation)

## Contact

For questions about this experiment design, see the conversation history where we designed it collaboratively.
