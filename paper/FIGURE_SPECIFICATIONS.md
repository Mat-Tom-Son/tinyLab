# TinyLab Methods Paper - Figure Specifications

**Purpose:** This document provides detailed specifications for all figures to be created for the paper. Each specification includes: purpose, data sources, visual design, annotations, and LaTeX integration details.

---

## Figure 1: Path-Patch DAG (Lead Figure) ⭐ PRIORITY

### Purpose
Visualize the causal mediation structure discovered by H6 battery, showing that 67% of head 0:2's effect on factuality travels through the layer-11 residual stream.

### Data Source
- Table 3 from case study (paper/sections_methods/case_study.tex, lines ~240-260)
- Path patching results:
  - Direct effect (no mediation): 12%
  - Via layer-1 residual: 18%
  - Via layer-2 residual: 20%
  - **Via layer-11 residual: 67%** ← Primary path
  - Via layer-23 residual: 8%

### Visual Design

**Type:** Directed acyclic graph (DAG) with nodes and weighted edges

**Nodes:**
```
[Layer 0: Head 0:2]
    ↓
[Residual Stream Insertion Points]
├─ Layer 1
├─ Layer 2
├─ Layer 11 ← THICK EDGE, HIGHLIGHTED
└─ Layer 23
    ↓
[Output: Factuality Change]
```

**Edge Properties:**
- Edge thickness proportional to mediation fraction
- Layer-11 path: **thick solid line** (67%)
- Other paths: thin dashed lines (8-20%)
- Color scheme: Layer-11 path in **accent color** (e.g., teal/blue), others in gray
- Edge labels show percentages

**Annotations:**
- Box around "Layer 11" node: "Primary Mediation Path (67%)"
- Note: "H6 Battery: Reverse Path Patching"
- Caption reference to Table 3

### Dimensions
- Width: 3.5 inches (half-page column width for double-column format)
- Height: 4 inches
- Resolution: 300 DPI minimum for publication

### LaTeX Integration
```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.48\textwidth]{figures/path_patch_dag.pdf}
\caption{Causal mediation structure for head~0:2's factuality effect (H6~battery). Path patching reveals 67\% of the effect travels through the layer-11 residual stream, with alternative paths showing only 8--20\% mediation. Solid thick line indicates primary pathway; dashed lines show minor paths. This quantified mediation structure suggests an attractor-like information channel.}
\label{fig:path_dag}
\end{figure}
```

### Implementation Notes
- Use `networkx` for graph layout (hierarchical/layered)
- Use `matplotlib` or `tikz` for rendering
- Export as PDF (vector format for crisp edges)
- Alternative: Manual TikZ diagram in LaTeX for full control

### References in Text
- Section 6.3 (H6 Mechanism): "Figure~\ref{fig:path_dag} visualizes this mediation structure..."
- Discussion 7.4 (Quantifying Mechanisms): "...67\% of head~0:2's effect is mediated through the layer-11 residual stream (Figure~\ref{fig:path_dag})"

---

## Figure 2: Random Baseline ECDF

### Purpose
Show that the hedging coalition ranks at the 99th-100th percentile of 1,000 random layer-0 head ablations, providing statistical context for "this head matters" claims.

### Data Source
- Random baseline generation (Appendix C, paper/sections_methods/appendix.tex, lines ~177-218)
- Expected data:
  - Random baseline ΔLD: mean = 0.05, std = 0.03, 99th percentile = 0.169
  - Head 0:2 observed ΔLD: **0.406**
  - Head 0:4 observed ΔLD: **0.520**
  - Head 0:7 observed ΔLD: **0.329**

### Visual Design

**Type:** Empirical Cumulative Distribution Function (ECDF)

**X-axis:** ΔLD (logit difference improvement)
- Range: [0.0, 0.6]
- Label: "Δ Logit Difference (Factual Probes)"

**Y-axis:** Cumulative Probability
- Range: [0.0, 1.0]
- Label: "Cumulative Probability"

**Plot Elements:**
1. **Gray curve:** ECDF of 1,000 random L0 ablations
2. **Vertical lines** (dashed, colored):
   - Red: Head 0:2 at ΔLD = 0.406
   - Blue: Head 0:4 at ΔLD = 0.520
   - Green: Head 0:7 at ΔLD = 0.329
3. **Annotations:**
   - "99th %ile" marker on ECDF curve
   - Labels for each head: "0:2 (100th %ile)", "0:4 (100th %ile)", "0:7 (99th %ile)"

**Shading:**
- Light gray region between mean ± 1 std
- Darker gray for 90th-99th percentile region

### Dimensions
- Width: 3.5 inches
- Height: 2.5 inches
- Resolution: 300 DPI

### LaTeX Integration
```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.48\textwidth]{figures/random_baseline_ecdf.pdf}
\caption{Statistical validation via random baseline comparison. Gray curve shows empirical CDF of $\Delta$LD from 1,000~random layer-0 head ablations (mean~$= 0.05$, 99th~percentile~$= 0.169$). Coalition heads~\{0:2, 0:4, 0:7\} (vertical lines) rank at or above the 99th~percentile, confirming they are statistical outliers, not typical layer-0 heads.}
\label{fig:baseline_ecdf}
\end{figure}
```

### Implementation Notes
- Use `numpy.percentile()` for percentile calculations
- Use `matplotlib.pyplot.step()` for ECDF rendering
- Add `axvline()` for head markers
- Include grid for readability

### References in Text
- Section 6.1 (H1 Discovery): "Coalition heads rank at the 99th-100th percentile (Figure~\ref{fig:baseline_ecdf})"
- Design Section 4.3 (Random Baselines): "Figure~\ref{fig:baseline_ecdf} illustrates..."

---

## Figure 3: Calibration Reliability Diagram

### Purpose
Demonstrate dual-observable measurement: coalition ablation improves both accuracy (ΔLD) AND calibration (ECE reduction from 0.122 → 0.091).

### Data Source
- Table 5 from case study (calibration results)
- Expected data (10 bins, [0.0-0.1), [0.1-0.2), ..., [0.9-1.0]):
  - **Baseline:** bins with confidence vs. accuracy gap (ECE = 0.122)
  - **Ablated:** bins with reduced gap (ECE = 0.091)

### Visual Design

**Type:** Reliability diagram (calibration curve)

**X-axis:** Predicted Confidence (mean probability in bin)
- Range: [0.0, 1.0]
- Label: "Mean Predicted Confidence"

**Y-axis:** Empirical Accuracy (fraction correct in bin)
- Range: [0.0, 1.0]
- Label: "Empirical Accuracy"

**Plot Elements:**
1. **Diagonal dashed line:** Perfect calibration (y = x)
2. **Red squares + line:** Baseline (before ablation)
   - Label: "Baseline (ECE = 0.122)"
3. **Blue circles + line:** Ablated (after removing coalition)
   - Label: "Coalition Ablated (ECE = 0.091)"
4. **Markers sized by bin count** (larger = more examples)

**Annotations:**
- Arrow showing "Improved Calibration" from baseline to ablated
- Shaded region between curves to emphasize improvement

### Dimensions
- Width: 3.5 inches
- Height: 3.5 inches (square aspect ratio for calibration plots)
- Resolution: 300 DPI

### LaTeX Integration
```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.48\textwidth]{figures/calibration_reliability.pdf}
\caption{Calibration improvement from coalition ablation (H1~battery, dual observables). Baseline model (red squares) shows ECE~$= 0.122$; ablating heads~\{0:2, 0:4, 0:7\} (blue circles) reduces ECE to 0.091 ($-25\%$). Deviation from diagonal (perfect calibration) decreases, confirming coalition trades calibration for raw accuracy. Marker size indicates bin count.}
\label{fig:calibration}
\end{figure}
```

### Implementation Notes
- Use `sklearn.calibration.calibration_curve()` or manual binning
- Plot with `plt.scatter()` for markers, `plt.plot()` for lines
- Add `plt.fill_between()` for shaded improvement region
- Include legend with ECE values

### References in Text
- Section 6.5 (Calibration): "Figure~\ref{fig:calibration} shows the reliability diagram..."
- Discussion 7.1 (Dual Observables): "...improves both ΔLD and ECE (Figure~\ref{fig:calibration})"

---

## Figure 4: Cross-Architecture Head Rankings Heatmap

### Purpose
Visualize cross-architecture validation: GPT-2 Small → Medium shows conservation (same heads, ρ = 0.94), GPT-2 → Mistral shows adaptation (different heads, but layer-0 motif preserved).

### Data Source
- Section 7 (Cross-Architecture Validation)
- Expected data:
  - **GPT-2 Small:** Head rankings for layer-0 heads on Facts task
  - **GPT-2 Medium:** Head rankings (should match Small for {0:2, 0:4, 0:7})
  - **Mistral-7B:** Head rankings (different heads: {0:22, 0:23}, but still layer-0)

### Visual Design

**Type:** Heatmap with annotations

**Rows:** Models (GPT-2 Small, GPT-2 Medium, Mistral-7B)

**Columns:** Layer-0 head indices (0:0, 0:1, 0:2, ..., 0:11 for GPT-2; 0:0-0:31 for Mistral)

**Cell Values:** ΔLD on Facts task
- Color scale: White (0.0) → Dark Blue (max ΔLD, e.g., 0.6)
- Annotate cells with exact values for coalition heads

**Annotations:**
1. **Red boxes** around GPT-2 coalition heads {0:2, 0:4, 0:7}
2. **Green boxes** around Mistral coalition heads {0:22, 0:23}
3. **Text annotations:**
   - "Conserved" arrow between GPT-2 Small and Medium
   - "Adapted Motif" arrow to Mistral
4. **Spearman ρ values:**
   - GPT-2 Small ↔ Medium: ρ = 0.94
   - GPT-2 ↔ Mistral: ρ = 0.37 (low, as expected)

### Dimensions
- Width: 7 inches (full page width for double-column)
- Height: 3 inches
- Resolution: 300 DPI

### LaTeX Integration
```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/cross_arch_heatmap.pdf}
\caption{Cross-architecture validation of layer-0 hedging motif. Heatmap shows $\Delta$LD for all layer-0 heads on factual probes across three models. Red boxes: GPT-2 coalition heads~\{0:2, 0:4, 0:7\} conserved from Small to Medium (Spearman~$\rho = 0.94$). Green boxes: Mistral's adapted coalition~\{0:22, 0:23\} (different heads, same layer-0 motif). Color intensity indicates impact magnitude.}
\label{fig:cross_arch}
\end{figure*}
```

### Implementation Notes
- Use `seaborn.heatmap()` for rendering
- Use `matplotlib.patches.Rectangle()` for red/green boxes
- Export as PDF (wide figure spanning both columns)
- Alternative: Create separate heatmaps per model, arrange as subfigures

### References in Text
- Section 7.1 (GPT-2 Conservation): "Figure~\ref{fig:cross_arch} shows identical rankings..."
- Section 7.2 (Mistral Adaptation): "...different heads but same layer-0 motif (Figure~\ref{fig:cross_arch})"

---

## Figure 5: TinyLab Architecture Diagram (Optional)

### Purpose
High-level system overview showing how batteries, harness, config management, and determinism infrastructure interact.

### Visual Design

**Type:** Block diagram with data flow arrows

**Components (boxes):**
1. **Config File** (JSON) → feeds into →
2. **Harness** (central orchestrator) → controls →
3. **Batteries** (H1, H5, H6, H2, H7) → execute ablations on →
4. **Model** (TransformerLens) → produces →
5. **Dual Observables** (Power + Info metrics) → stored in →
6. **Results** (Parquet, JSON, manifests)

**Side annotations:**
- "Determinism Layer" (seeds, flags, parity checks)
- "Random Baseline Generator" (feeding into statistical comparison)
- "Cross-Architecture Pipeline" (multiple model inputs)

**Color Scheme:**
- Configs: Yellow
- Harness: Blue
- Batteries: Green
- Model: Gray
- Outputs: Purple

### Dimensions
- Width: 7 inches (full page width)
- Height: 4 inches
- Resolution: 300 DPI

### LaTeX Integration
```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/tinylab_architecture.pdf}
\caption{TinyLab system architecture. Config files specify model, battery, and dataset. The harness orchestrates ablation execution with determinism enforcement (seeds, flags, parity checks). Batteries (H1--H7) implement standardized protocols. Dual observables (power + information metrics) are computed and stored with full reproducibility metadata (config hashes, git state, hardware).}
\label{fig:architecture}
\end{figure*}
```

### Implementation Notes
- Use `draw.io`, `Lucidchart`, or TikZ
- Keep high-level (avoid implementation details)
- Export as PDF vector graphic
- **Priority: LOW** (nice-to-have, not critical)

### References in Text
- Section 5.1 (Infrastructure Overview): "Figure~\ref{fig:architecture} shows the system architecture..."
- Optional: Could be moved to appendix if page limits tight

---

## Figure 6: OV Token Projection (Top-K Tokens) - OPTIONAL

### Purpose
Visualize semantic coherence of head 0:2's OV projection: hedge words upweighted, factual stems downweighted.

### Data Source
- Table 4 from case study (OV tokens)
- Top-10 upweighted: perhaps, maybe, possibly, likely, seemingly, etc.
- Top-10 downweighted: Paris, France, Berlin, Germany, etc.

### Visual Design

**Type:** Horizontal bar chart (diverging)

**Y-axis:** Token (text)
- Top section: Hedge words
- Bottom section: Factual tokens

**X-axis:** OV Logit Contribution
- Range: [-4, +6]
- Positive (right): Upweighted tokens
- Negative (left): Downweighted tokens

**Colors:**
- Green bars (right): Hedge words
- Red bars (left): Factual stems
- Annotate with token text + logit value

### Dimensions
- Width: 3.5 inches
- Height: 5 inches
- Resolution: 300 DPI

### LaTeX Integration
```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.48\textwidth]{figures/ov_tokens.pdf}
\caption{OV projection semantics for head~0:2. Top-10 upweighted tokens (green, positive logits) are hedge words (``perhaps'', ``maybe'', ``possibly''). Top-10 downweighted tokens (red, negative logits) are factual answer stems (``Paris'', ``France'', ``Berlin''). This semantic coherence confirms the circuit's functional interpretation.}
\label{fig:ov_tokens}
\end{figure}
```

### Implementation Notes
- Use `plt.barh()` for horizontal bars
- Sort by logit magnitude
- Export as PDF
- **Priority: MEDIUM** (helpful but text table may suffice)

### References in Text
- Section 6.4 (OV Semantics): "Figure~\ref{fig:ov_tokens} shows the top-10..."

---

## Summary of Priorities

### Must-Have (Critical Path)
1. **Figure 1: Path-Patch DAG** ⭐ - Visualizes core H6 result
2. **Figure 2: Random Baseline ECDF** - Statistical validation
3. **Figure 3: Calibration Reliability Diagram** - Dual-observable proof

### Should-Have (Strong Support)
4. **Figure 4: Cross-Architecture Heatmap** - Replication evidence

### Nice-to-Have (Optional)
5. **Figure 5: TinyLab Architecture** - System overview (could be appendix)
6. **Figure 6: OV Token Projection** - Semantic coherence (table may suffice)

---

## Implementation Plan

### Phase 1: Generate Data Files (if not already exists)
- Extract path patching results into `data/path_patch_results.json`
- Generate random baseline samples into `data/random_baseline_samples.csv`
- Compute calibration bins into `data/calibration_bins.json`
- Export cross-architecture rankings into `data/cross_arch_rankings.csv`

### Phase 2: Create Python Scripts
- `scripts/figures/fig1_path_dag.py`
- `scripts/figures/fig2_baseline_ecdf.py`
- `scripts/figures/fig3_calibration.py`
- `scripts/figures/fig4_cross_arch_heatmap.py`

### Phase 3: Generate Figures
```bash
python scripts/figures/fig1_path_dag.py
python scripts/figures/fig2_baseline_ecdf.py
python scripts/figures/fig3_calibration.py
python scripts/figures/fig4_cross_arch_heatmap.py
```

### Phase 4: LaTeX Integration
- Create `paper/figures/` directory
- Copy generated PDFs into `paper/figures/`
- Ensure all `\includegraphics{}` paths are correct
- Compile with `pdflatex` to verify rendering

---

## Next Steps

1. **Confirm figure data availability** - Check if actual experimental results exist or if we need placeholder/schematic versions
2. **Choose implementation approach** - Python matplotlib vs. TikZ vs. manual design tools
3. **Create Figure 1 (Path DAG) first** - Highest priority, most impactful
4. **Iterate on aesthetics** - Ensure publication quality (fonts, line weights, colors)

---

**Status:** Figure specifications complete. Ready for implementation.

**Estimated time:**
- Figure 1 (DAG): 2-3 hours (most complex)
- Figure 2 (ECDF): 1 hour
- Figure 3 (Calibration): 1 hour
- Figure 4 (Heatmap): 1.5 hours
- **Total: ~6-7 hours for core 4 figures**
