# Homeostatic Phase Transitions in Neural Network Learning

This unified paper integrates findings from three independent research threads:

## Paper Components

### Experiment 1: Anatomy of the Equilibrium
**Source:** `../main.pdf` (Layer-0 Suppressors paper)
- Identifies layer-0 suppressor heads in GPT-2 and Mistral-7B
- Shows equilibrium precision: VDI = 0.611992 (exact to 6 decimals, 5 seeds)
- Effect size: ΔLD = +0.40 to +0.85 across tasks
- Validates dual-observable framework (power + information)

### Experiment 2: The Boundaries of Perturbation
**Source:** Prior saturation analysis + scale studies
- Maps information-saturation boundaries (MI ≈ 2-3 bits)
- Shows suppressor role transitions at d_eff ≈ 30-50
- Demonstrates capacity-contingent attractor dynamics

### Experiment 3: The Kill Test and Active Compensation
**Source:** `../homeostatic_compensation/main.pdf`
- Validates Le Chatelier's principle through kill tests
- Shows 4.9× slowdown when compensation is blocked
- Demonstrates brittle solutions when homeostasis is disabled
- Evidence from 4 seeds on parity grokking task

### Experiment 4: The Geometry of Designability
**Source:** Phase 1-2 crystallization experiments (`../../FINDINGS.md`)
- Achieves 93% acceleration of crystallization (1500 → 100 steps)
- Discovers forced attractor at VDI ≈ 0.44-0.46
- Shows 29× worse tracking for high VDI targets
- Proves equilibria are constrained by information geometry

## Key Findings

1. **Homeostatic equilibrium is real:** VDI = 0.611992 ± 0.000000 (5 seeds)
2. **Equilibria are constrained:** Forced attractor at 0.44-0.46, unreachable region > 0.50
3. **Compensation is active:** 4.9× slowdown when blocked, brittle solutions emerge
4. **Engineering is possible:** 93% acceleration within geometric constraints
5. **Le Chatelier validated:** System actively resists perturbations through distributed compensation

## Compilation

```bash
cd paper/unified_homeostasis
make
```

This will produce `main.pdf` (26 pages, ~250KB).

## Structure

```
unified_homeostasis/
├── main.tex                          # Main document
├── sections/
│   ├── introduction.tex              # Motivation and overview
│   ├── experiment1_anatomy.tex       # Layer-0 suppressors
│   ├── experiment2_boundaries.tex    # Information saturation limits
│   ├── experiment3_mechanism.tex     # Kill tests and Le Chatelier
│   ├── experiment4_geometry.tex      # Crystallization and forced attractor
│   ├── synthesis.tex                 # Unified picture
│   ├── discussion.tex                # Implications and limitations
│   ├── related_work.tex              # Literature review
│   ├── future_work.tex               # Open questions
│   └── conclusion.tex                # Summary and impact
├── Makefile
└── README.md (this file)
```

## Citation

If you build on this work, please cite:

```
@article{thompson2025homeostatic,
  title={Homeostatic Phase Transitions in Neural Network Learning: How Information Geometry Constrains Learning Dynamics},
  author={Thompson, Mat},
  year={2025},
  note={Independent Research, Raleigh, NC}
}
```

## Related Files

- `../main.pdf` - Paper 1: Layer-0 Suppressors
- `../homeostatic_compensation/main.pdf` - Paper 2: Kill Test Evidence
- `../../FINDINGS.md` - Complete experimental results from all 35 runs
- `../../reports/phase2/` - Raw data from crystallization experiments
- `../../reports/developmental_monitoring/` - Phase 1 equilibrium data
