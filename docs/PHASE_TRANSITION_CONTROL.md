# Phase-Transition Control: Implementation Summary

## Executive Summary

This document summarizes the implementation of the **Developmental Monitoring Framework** for tracking and controlling the crystallization of Layer-0 "Gatekeeper" circuits during transformer pre-training.

The framework operationalizes three research leads from the Internal Memo on **Architecture of Intentionality & Phase-Transition Control**:

1. **VDI "Snap" Metric** - Detects crystallization timing
2. **Homeostatic Kill Testing** - Measures compensation dynamics
3. **Saturation Boundary Guardrails** - Prevents brittle attractors

## What Was Built

### Core Components

#### 1. Developmental Monitor ([lab/src/components/developmental_monitor.py](../lab/src/components/developmental_monitor.py))

**A. VDI Snap Detection**

```python
@dataclass
class VDISnapshot:
    """Tracks variance dampening index over time"""
    step: int
    vdi_per_head: List[float]
    mean_vdi: float
    vdi_velocity: float  # d(VDI)/dt
    vdi_acceleration: float  # d²(VDI)/dt²

def detect_vdi_snap(vdi_history) -> VDISnapDetection:
    """Detects phase transition via negative acceleration"""
    # Looks for:
    # 1. Sharp inflection (d²VDI/dt² < threshold)
    # 2. Transition from high variance to structure
    # 3. Post-snap stabilization
```

**B. Le Chatelier Kill Testing**

```python
@dataclass
class CompensationSignature:
    """Measures homeostatic response to ω perturbation"""
    omega: float
    compensation_count: int  # Heads showing inverse correlation
    compensation_strength: float
    compensating_heads: List[Tuple[int, int]]

def run_kill_test(model, data_batch, target_head, omega_values):
    """Applies ω-sweep and measures compensation"""
    # For ω < 1: Expect other heads to become more suppressive (↑VDI)
    # For ω > 1: Expect other heads to become less suppressive (↓VDI)
```

**C. MI Saturation Monitoring**

```python
@dataclass
class MutualInfoSnapshot:
    """Tracks channel utilization"""
    mi_estimate: float  # bits
    channel_capacity: float
    saturation_ratio: float  # MI / capacity
    saturation_phase: str  # "healthy", "approaching", "saturated"

def check_saturation_boundary(mi_snapshot) -> (is_saturated, diagnosis):
    """Warns if MI > 90% of capacity (brittle attractor risk)"""
```

**D. Integrated Monitor**

```python
class DevelopmentalMonitor:
    """Orchestrates all three monitoring systems"""

    def record_checkpoint(self, step, data_batch, layer0_activations):
        """Records VDI, runs kill tests, estimates MI"""

    def analyze_trajectory(self):
        """Identifies phases and confirms Le Chatelier"""

    def save_trajectory(self, output_path):
        """Exports JSON with full developmental history"""
```

### Integration Scripts

#### 2. Training with Monitoring ([scripts/train_with_developmental_monitoring.py](../scripts/train_with_developmental_monitoring.py))

Complete example showing how to:
- Add attention caching to model
- Initialize monitor
- Record checkpoints during training
- Detect snap in real-time
- Save developmental trajectory

#### 3. Visualization ([scripts/visualize_developmental_trajectory.py](../scripts/visualize_developmental_trajectory.py))

Generates diagnostic plots:
- VDI trajectory with snap markers
- Compensation strength across phases
- MI saturation curve
- Integrated phase diagram

#### 4. Demo Script ([scripts/run_developmental_demo.sh](../scripts/run_developmental_demo.sh))

One-command demonstration:
```bash
bash scripts/run_developmental_demo.sh
```

Runs omega sweep (0.5, 1.0, 1.5) and generates visualizations.

### Documentation

#### 5. User Guide ([docs/DEVELOPMENTAL_MONITORING.md](DEVELOPMENTAL_MONITORING.md))

Comprehensive documentation covering:
- Conceptual framework
- Installation and usage
- Integration patterns
- Output format and interpretation
- Troubleshooting

## Experimental Predictions

Based on the Internal Memo, we expect:

### Timeline

| Phase | Steps | VDI Behavior | Compensation | MI |
|-------|-------|--------------|--------------|-----|
| **Pre-Crystallization** | 0-2000 | Noisy, high variance | Minimal (<0.005) | <60% |
| **The Window** | 2000-3000 | Sharp drop (snap) | Emerging (0.005-0.015) | 60-80% |
| **Stability** | 3000+ | Stable, structured | Active (>0.015) | Plateaus 70-85% |

### Omega Perturbation Effects

| Condition | Snap Timing | Compensation Strength | Interpretation |
|-----------|-------------|----------------------|----------------|
| **ω=0.5** (weaken) | Delayed | High | Other heads must compensate more |
| **ω=1.0** (baseline) | Normal | Moderate | Natural equilibrium |
| **ω=1.5** (strengthen) | Accelerated | Low | Target head dominates, less need for compensation |

### Saturation Boundaries

- **Healthy (<80%)**: Room for learning, snap can still occur
- **Approaching (80-90%)**: Optimal intervention window
- **Saturated (>90%)**: Brittle attractor risk - may memorize without generalizing

## Key Design Decisions

### 1. VDI as Proxy for Suppressor Strength

**Rationale**: VDI = H / H_max directly measures attention distribution flatness. A suppressor flattens attention (high VDI), an amplifier focuses it (low VDI).

**Advantage**: Computationally cheap, interpretable, doesn't require ground truth labels.

**Limitation**: Only measures attention patterns, not full circuit behavior. Could be supplemented with OV projection analysis.

### 2. Second Derivative for Snap Detection

**Rationale**: Crystallization is a phase transition, characterized by rapid change in the rate of change (acceleration).

**Advantage**: Robust to noise, captures genuine structural shifts vs gradual drift.

**Limitation**: Requires sufficient sampling density (checkpoints) to estimate derivatives accurately.

### 3. Le Chatelier via Inverse Correlation

**Rationale**: If perturbing head A causes head B to respond in the opposite direction, that's evidence of homeostatic regulation.

**Advantage**: Mechanistically grounded, doesn't require predefined "compensation" patterns.

**Limitation**: Assumes compensation operates within the same layer. May miss cross-layer compensation.

### 4. Gaussian MI Estimation

**Rationale**: Fast, closed-form estimate based on covariance structure.

**Advantage**: No need for discrete binning or expensive kernel methods.

**Limitation**: Assumes approximately Gaussian activations. Can fail for highly non-linear representations.

## Validation Strategy

### Phase 1: Synthetic Task (Parity/Grokking)

**Goal**: Establish baseline behavior in controlled setting.

**Experiments**:
- Run omega sweep on parity task (quick, CPU-friendly)
- Verify snap detection correlates with task performance
- Confirm Le Chatelier signature

**Success Criteria**:
- Snap detected in 80%+ of runs
- Compensation increases after snap (window → stability)
- MI saturation < 90% in successful runs

### Phase 2: Stage-1B Integration

**Goal**: Apply to modular arithmetic grokking (documented phase transition).

**Experiments**:
- Monitor p=113 modular addition training
- Compare snap timing to known grokking transition
- Test if ω-perturbation delays/accelerates grokking

**Success Criteria**:
- Snap precedes grokking (evidence of Layer-0 as prerequisite)
- ω=0.5 delays both snap and grokking
- ω=1.5 accelerates both

### Phase 3: Pre-training on Language

**Goal**: Demonstrate framework scales to realistic settings.

**Experiments**:
- Monitor GPT-2 Small pre-training from scratch
- Track snap across multiple heads
- Identify which heads crystallize first

**Success Criteria**:
- Layer 0 heads snap before downstream induction heads
- Compensation signatures persist in stability phase
- MI provides early warning for overfitting

## Limitations and Future Work

### Current Limitations

1. **Attention-Only Monitoring**: Doesn't track MLP contributions or OV circuit structure
2. **Single-Layer Focus**: Monitors Layer 0 but not multi-layer coordination
3. **Computational Overhead**: Kill testing requires multiple forward passes
4. **MI Estimation**: Gaussian assumption may not hold for all layers

### Planned Extensions

1. **Multi-Head Tracking**: Monitor multiple heads simultaneously, detect coordination patterns
2. **Gradient Flow Analysis**: Add eigenspectrum monitoring of Hessian (loss landscape curvature)
3. **Automated Intervention**: Trigger learning rate adjustments or early stopping based on snap/saturation
4. **Dashboard**: Real-time visualization during long training runs
5. **Causal Validation**: Patch experiments to verify snap → capability causality

## Usage Examples

### Basic Usage

```bash
# Train with monitoring
python scripts/train_with_developmental_monitoring.py \
    --task parity --omega 1.0 --seed 0 --steps 10000 \
    --monitor-interval 500

# Visualize
python scripts/visualize_developmental_trajectory.py \
    reports/developmental_monitoring/parity_omega1.0_seed0/developmental_trajectory.json
```

### Omega Sweep

```bash
for omega in 0.5 1.0 1.5; do
    python scripts/train_with_developmental_monitoring.py \
        --omega $omega --seed 0 --steps 10000
done
```

### Integration with Existing Code

```python
from lab.src.components.developmental_monitor import DevelopmentalMonitor

# Add to your training loop
monitor = DevelopmentalMonitor(model, target_head=(0, 0))

for step in training_loop:
    # ... train ...

    if step % 500 == 0:
        checkpoint = monitor.record_checkpoint(step, data_batch, layer0_acts)

        if checkpoint.kill_test_result:
            print(f"Compensation: {checkpoint.kill_test_result.compensation_score:.4f}")

# After training
monitor.save_trajectory(output_dir / "developmental_trajectory.json")
```

## File Structure

```
tinyLab/
├── lab/src/components/
│   └── developmental_monitor.py       # Core monitoring framework
├── scripts/
│   ├── train_with_developmental_monitoring.py  # Training integration
│   ├── visualize_developmental_trajectory.py   # Visualization
│   ├── integrate_monitoring_stage1b.py         # Stage-1B patch
│   └── run_developmental_demo.sh               # One-command demo
└── docs/
    ├── DEVELOPMENTAL_MONITORING.md     # User guide
    └── PHASE_TRANSITION_CONTROL.md     # This document
```

## Key Metrics Glossary

| Metric | Range | Interpretation |
|--------|-------|----------------|
| **VDI** | 0-1 | 1.0 = suppressor (flat), 0.0 = amplifier (peaked) |
| **VDI Velocity** | ℝ | Negative = becoming more suppressive |
| **VDI Acceleration** | ℝ | Large negative = snap point |
| **Compensation Score** | 0+ | >0.01 = active homeostasis |
| **Compensation Count** | 0-N | How many heads show inverse correlation |
| **MI Estimate** | 0-∞ bits | Information flow through layer |
| **Saturation Ratio** | 0-1 | >0.9 = brittle attractor risk |

## Theoretical Grounding

This implementation operationalizes concepts from:

1. **Le Chatelier's Principle**: System responds to perturbations by shifting equilibrium to counteract the change
2. **Information Theory**: Channel capacity limits how much information a layer can transmit
3. **Phase Transitions**: Sharp qualitative changes in system behavior at critical points
4. **Homeostatic Regulation**: Distributed compensation mechanisms maintain system invariants

The key insight: **Structural alignment can be engineered by controlling the timing and geometry of crystallization at architectural bottlenecks.**

## Next Steps

1. **Run Demo**: `bash scripts/run_developmental_demo.sh`
2. **Read Guide**: [DEVELOPMENTAL_MONITORING.md](DEVELOPMENTAL_MONITORING.md)
3. **Integrate**: Add monitoring to Stage-1B experiments
4. **Validate**: Compare snap timing to grokking transition
5. **Publish**: Document findings for transparency

## Contact

For questions or collaboration:
- GitHub Issues: https://github.com/Mat-Tom-Son/tinyLab/issues
- Email: mat@tinylab.ai

## Citation

```bibtex
@misc{tinylab2025phasecontrol,
  title        = {Phase-Transition Control for Safe AGI Development},
  author       = {Mat Thompson},
  year         = {2025},
  howpublished = {Tiny Ablation Lab},
  note         = {Developmental monitoring framework for transformer crystallization}
}
```

---

**Status**: Implementation complete, ready for validation experiments.

**Last Updated**: 2025-12-18
