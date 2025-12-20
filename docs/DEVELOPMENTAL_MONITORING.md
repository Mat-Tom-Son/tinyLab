# Developmental Monitoring Framework

## Overview

This framework implements the three-part monitoring system for detecting and controlling the "crystallization point" during transformer pre-training, as described in the Internal Memo on **Architecture of Intentionality & Phase-Transition Control**.

The system tracks:

1. **VDI "Snap" Metric** - Detects when Layer 0 transitions from noise amplifier to structured bottleneck
2. **Homeostatic Kill Testing** - Measures Le Chatelier compensation response to ω-perturbations
3. **Saturation Boundary Guardrails** - Monitors mutual information vs channel capacity

## Key Concepts

### The Dual-Loop Architecture

The framework is designed to monitor the emergence of two distinct optimization timescales:

- **Fast Loop (Adaptation)**: Shallow circuits (Layers 0-2) capable of rapid in-context learning
- **Slow Loop (Regulation)**: Deep structures (Layers 3+) that maintain invariant properties (Q) like corrigibility

The "crystallization point" occurs when Layer 0 transitions from a noise amplifier to a structured bottleneck that implements the **Gatekeeper of Doubt** - the circuit that trades off truth against caution.

### VDI (Variance Dampening Index)

VDI measures whether a head acts as a **suppressor** (flattens distribution) or **amplifier** (sharpens distribution):

```
VDI = H / H_max
```

where H is the entropy of attention weights.

- **VDI ≈ 1.0**: Suppressor (flat/uniform attention)
- **VDI ≈ 0.0**: Amplifier (peaked/focused attention)

The **VDI "snap"** is characterized by:
1. Sharp inflection in mean VDI (negative acceleration)
2. Transition from high variance (noise) to low variance (structure)
3. Stabilization of VDI trajectory after the snap

### Le Chatelier Compensation

When we perturb Layer-0 head strength via omega (ω) scaling:

- **ω < 1.0** (weakening): Other heads should become more suppressive (↑VDI)
- **ω > 1.0** (strengthening): Other heads should become less suppressive (↓VDI)

This inverse correlation indicates **homeostatic compensation** - the system actively restores equilibrium.

### Saturation Boundary

Mutual information (MI) between layer input and output indicates channel utilization:

```
Saturation Ratio = MI / Channel Capacity
```

- **< 80%**: Healthy - room for learning
- **80-90%**: Approaching saturation - crystallization window
- **> 90%**: Saturated - risk of brittle attractor

## Installation

The framework is integrated into the existing tinyLab codebase:

```bash
cd tinyLab
source .venv/bin/activate
```

No additional dependencies required beyond the base environment.

## Usage

### Basic Example

Train a model with developmental monitoring:

```bash
python scripts/train_with_developmental_monitoring.py \
    --task parity \
    --omega 1.0 \
    --head 0 \
    --seed 0 \
    --steps 10000 \
    --monitor-interval 500 \
    --kill-test-frequency 2
```

This will:
- Train a parity transformer
- Record VDI snapshots every 500 steps
- Run kill tests every 2nd checkpoint (every 1000 steps)
- Save results to `reports/developmental_monitoring/parity_omega1.0_seed0/`

### Visualize Results

Generate diagnostic plots:

```bash
python scripts/visualize_developmental_trajectory.py \
    reports/developmental_monitoring/parity_omega1.0_seed0/developmental_trajectory.json
```

This creates `developmental_trajectory.png` with:
- VDI trajectory and snap detection
- Compensation strength across phases
- MI saturation curve
- Integrated phase diagram

## Integration with Custom Training Loops

### Step 1: Add Attention Caching

Modify your `TransformerBlock.forward()` to cache attention weights:

```python
def forward(self, x, layer_idx, layer_head_config=None, attn_mask=None):
    # ... compute attention ...
    attn_scores = torch.einsum("bthd,bshd->bhts", q, k) / sqrt(d_head)
    attn_weights = F.softmax(attn_scores + attn_mask, dim=-1)

    # Cache for monitoring
    self._cached_attn_weights = attn_weights

    # ... rest of forward pass ...
```

Then add a hook to collect these weights:

```python
model.attention_cache = []

def make_attention_hook(layer_idx):
    def hook(module, input, output):
        if hasattr(module, '_cached_attn_weights'):
            model.attention_cache[layer_idx] = module._cached_attn_weights
    return hook

for layer_idx, block in enumerate(model.blocks):
    block.register_forward_hook(make_attention_hook(layer_idx))
```

### Step 2: Initialize Monitor

```python
from lab.src.components.developmental_monitor import DevelopmentalMonitor

monitor = DevelopmentalMonitor(
    model=model,
    target_head=(0, 0),  # Layer 0, Head 0
    omega_sweep=[0.5, 0.7, 1.0, 1.3, 1.5],
    kill_test_frequency=2  # Every 2nd checkpoint
)
```

### Step 3: Record Checkpoints

In your training loop:

```python
if step % monitor_interval == 0:
    # Get Layer 0 activations
    layer0_activations = model.blocks[0](embeddings, layer_idx=0)

    # Record checkpoint
    checkpoint = monitor.record_checkpoint(
        step=step,
        data_batch=input_ids,
        layer0_activations=layer0_activations
    )

    # Check for snap
    if not monitor.snap_detected and len(monitor.vdi_history) >= 5:
        snap_result = detect_vdi_snap(monitor.vdi_history)
        if snap_result.snap_detected:
            print(f"🔔 VDI SNAP at step {snap_result.snap_step}!")
            monitor.snap_detected = True
```

### Step 4: Analyze Trajectory

After training:

```python
summary = monitor.analyze_trajectory()
monitor.save_trajectory(output_dir / "developmental_trajectory.json")

print(f"Snap detected: {summary['snap_detected']}")
print(f"Le Chatelier confirmed: {summary['le_chatelier_confirmed']}")
```

## Output Format

### Developmental Trajectory JSON

```json
{
  "target_head": [0, 0],
  "omega_sweep": [0.5, 1.0, 1.5],
  "snap_result": {
    "detected": true,
    "step": 2500,
    "confidence": 0.87
  },
  "vdi_history": [
    {
      "step": 500,
      "mean_vdi": 0.72,
      "vdi_std": 0.15,
      "vdi_velocity": -0.0002,
      "vdi_acceleration": -0.0015
    }
  ],
  "checkpoints": [
    {
      "step": 500,
      "developmental_phase": "pre_snap",
      "vdi_mean": 0.72,
      "kill_test": {
        "performed": true,
        "le_chatelier_detected": false,
        "compensation_score": 0.003
      },
      "mi": {
        "estimate": 4.2,
        "saturation_ratio": 0.65,
        "phase": "healthy"
      }
    }
  ],
  "summary": {
    "snap_detected": true,
    "snap_step": 2500,
    "snap_confidence": 0.87,
    "compensation_by_phase": {
      "pre_crystallization": 0.002,
      "window": 0.012,
      "stability": 0.018
    },
    "le_chatelier_confirmed": true,
    "saturation_warning": false,
    "total_checkpoints": 20
  }
}
```

## Interpreting Results

### VDI Snap Detection

**Positive detection** (confidence > 0.7):
- Sharp negative acceleration in VDI trajectory
- Post-snap stabilization
- Reduced variance after snap

**Interpretation**:
- Layer 0 has crystallized into a structured bottleneck
- The "Gatekeeper of Doubt" circuit is now active
- Downstream circuits will inherit this equilibrium

**No detection**:
- VDI remains noisy throughout training
- May indicate insufficient model capacity
- Or task doesn't require Layer-0 bottleneck

### Le Chatelier Compensation

**Confirmed** (compensation score > 0.01 in window/stability phases):
- Other heads compensate when target head is perturbed
- Homeostatic regulation is active
- System maintains equilibrium via distributed compensation

**Not confirmed**:
- Compensation may be weak or operate through different mechanisms
- May indicate the system hasn't entered the stability phase yet
- Or compensation occurs at different layers

### MI Saturation

**Healthy** (< 80%):
- Room for continued learning
- Safe to continue training

**Approaching** (80-90%):
- Crystallization window - optimal time for interventions
- Monitor closely for snap

**Saturated** (> 90%):
- ⚠️ Risk of brittle attractor
- System may solve loss without robust generalization
- Consider early stopping or architecture changes

## Expected Timeline

Based on preliminary experiments:

| Phase | Steps | VDI | Compensation | MI Saturation |
|-------|-------|-----|--------------|---------------|
| **Phase 1: Pre-Crystallization** | 0-2000 | High variance, noisy | Minimal (<0.005) | < 60% |
| **Phase 2: The Window** | 2000-3000 | Sharp drop, inflection | Emerging (0.005-0.015) | 60-80% |
| **Phase 3: Stability** | 3000+ | Stable, low variance | Active (>0.015) | Plateaus ~70-85% |

The **VDI snap** typically occurs during Phase 2, marking the transition from noise amplification to structured suppression.

## Advanced Usage

### Custom Omega Sweep

Test more granular perturbations:

```python
monitor = DevelopmentalMonitor(
    model=model,
    target_head=(0, 0),
    omega_sweep=[0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7],
    kill_test_frequency=1  # Every checkpoint
)
```

### Multi-Head Monitoring

Monitor multiple heads simultaneously:

```python
monitors = {
    (0, 0): DevelopmentalMonitor(model, target_head=(0, 0)),
    (0, 1): DevelopmentalMonitor(model, target_head=(0, 1)),
    (1, 0): DevelopmentalMonitor(model, target_head=(1, 0)),
}

for step in training_loop:
    if step % monitor_interval == 0:
        for head, monitor in monitors.items():
            checkpoint = monitor.record_checkpoint(step, data_batch, activations)
```

### Early Intervention

Modulate training based on snap detection:

```python
if checkpoint.vdi_snapshot.vdi_acceleration < -0.002:
    print("Approaching snap - consider reducing learning rate")
    optimizer.param_groups[0]['lr'] *= 0.5

if checkpoint.mi_snapshot and checkpoint.mi_snapshot.saturation_ratio > 0.9:
    print("Saturation detected - early stopping")
    break
```

## Experimental Protocol

### Preregistered Omega Sweep

To replicate the memo's research leads:

```bash
# Baseline (ω=1.0)
python scripts/train_with_developmental_monitoring.py \
    --omega 1.0 --seed 0 --monitor-interval 250

# Weakening Layer-0 Head (ω=0.5)
python scripts/train_with_developmental_monitoring.py \
    --omega 0.5 --seed 0 --monitor-interval 250

# Strengthening Layer-0 Head (ω=1.5)
python scripts/train_with_developmental_monitoring.py \
    --omega 1.5 --seed 0 --monitor-interval 250
```

Expected observations:
- **ω=0.5**: Delayed snap, increased compensation from other heads
- **ω=1.0**: Normal developmental trajectory (baseline)
- **ω=1.5**: Accelerated snap, reduced need for compensation

### Batch Analysis

Process multiple runs:

```bash
for omega in 0.5 1.0 1.5; do
    for seed in 0 1 2; do
        python scripts/train_with_developmental_monitoring.py \
            --omega $omega --seed $seed --steps 10000 \
            --monitor-interval 500
    done
done

# Aggregate results
python scripts/aggregate_developmental_trajectories.py \
    reports/developmental_monitoring/ \
    --output reports/developmental_summary.json
```

## Troubleshooting

### "Model does not cache attention weights"

**Problem**: Monitor can't access attention patterns.

**Solution**: Add attention caching to your `TransformerBlock`:

```python
self._cached_attn_weights = attn_weights
```

And ensure the model has `attention_cache` list initialized.

### "Kill test failed"

**Problem**: Omega perturbation not supported by model.

**Solution**: Ensure your model's forward pass accepts `layer_head_config`:

```python
def forward(self, x, layer_head_config=None):
    for layer_idx, block in enumerate(self.blocks):
        x = block(x, layer_idx, layer_head_config)
```

### VDI snap not detected

**Possible causes**:
- Not enough checkpoints (need at least 5)
- Training stopped before crystallization
- Task doesn't require Layer-0 bottleneck
- Model too small or too large

**Solution**: Train longer, increase checkpoint frequency, or try different task.

### MI estimation errors

**Problem**: Covariance matrix singular or ill-conditioned.

**Solution**: The Gaussian MI estimator can fail for degenerate distributions. This is informative - it suggests the layer hasn't learned meaningful representations yet. The monitor will skip MI measurement and continue.

## Citation

If you use this framework in your research:

```bibtex
@misc{tinylab2025developmental,
  title        = {Developmental Monitoring Framework for Phase-Transition Control},
  author       = {Mat Thompson},
  year         = {2025},
  howpublished = {Tiny Ablation Lab},
  note         = {\url{https://github.com/Mat-Tom-Son/tinyLab}}
}
```

## References

- Internal Memo: Architecture of Intentionality & Phase-Transition Control
- Power et al. (2022): Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets
- Le Chatelier's Principle in Neural Networks: Compensatory Dynamics in Attention Mechanisms
- Shannon (1948): A Mathematical Theory of Communication (Channel Capacity)

## Support

For questions or issues:
- Open an issue: https://github.com/Mat-Tom-Son/tinyLab/issues
- Email: mat@tinylab.ai

## Next Steps

See [docs/roadmap_next_steps.md](roadmap_next_steps.md) for planned extensions:

- Automated intervention strategies
- Multi-layer crystallization tracking
- Integration with Stage-1B grokking experiments
- Real-time monitoring dashboard
