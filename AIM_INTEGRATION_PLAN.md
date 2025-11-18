# Aim Experiment Tracking Integration for tinyLab

**Status:** Design Complete - Ready for Implementation
**Date:** 2025-11-18
**Purpose:** Add comprehensive experiment tracking with web UI for mechanistic interpretability research

---

## Overview

This document outlines the integration of [Aim](https://aimstack.io/) experiment tracking into tinyLab. Aim will provide an interactive web UI to browse experiments, compare runs, and visualize mechanistic interpretability metrics in real-time.

### Why Aim?

- **Self-hosted** - No cloud dependencies, works offline
- **Python-native** - Easy integration with existing codebase
- **Rich visualizations** - Interactive plots, comparisons, filtering
- **Flexible** - Supports custom metrics, images, text, distributions
- **Fast** - Efficient storage and querying
- **Open source** - MIT license, no vendor lock-in

### What Gets Tracked

```
Run Metadata         Core Metrics              MI Metrics                    Artifacts
├── model_name       ├── logit_diff           ├── ov_fidelity_by_layer     ├── attention_heatmaps
├── condition        ├── accuracy             ├── qk_pattern_strength      ├── ov_projections
├── probe_type       ├── p_drop               ├── activation_entropy       ├── calibration_curves
├── layers           ├── kl_divergence        ├── geometric_curvature      ├── confusion_matrices
├── heads            ├── calibration_ece      ├── pca_rank_by_layer        ├── token_clouds
├── seed             ├── mediation_fraction   ├── path_patching_effects    └── trajectory_plots
├── git_commit       └── bootstrap_ci         └── emergence_curves
├── timestamp
└── device
```

---

## Architecture

### Directory Structure

```
tinyLab/
├── .aim/                       # Aim storage (gitignored)
│   ├── meta/                   # Metadata index
│   ├── runs/                   # Run data (metrics, logs)
│   └── seqs/                   # Sequence storage
│
├── lab/
│   ├── tracking/               # NEW: Aim integration code
│   │   ├── __init__.py
│   │   ├── tracker.py          # Main tracking class
│   │   ├── metrics.py          # Metric definitions
│   │   ├── visualizations.py  # Custom plots
│   │   └── migrate.py          # Import existing results
│   │
│   ├── harness.py              # MODIFIED: Add tracking hooks
│   └── configs/                # EXISTING: Experiment configs
│
├── scripts/
│   ├── import_to_aim.py        # Import historical results
│   └── launch_aim_ui.sh        # Start Aim web UI
│
└── docs/
    └── AIM_USAGE.md            # User guide for Aim UI
```

### Data Flow

```
Experiment Run (harness.py)
    ↓
TinyLabTracker.log_metrics()
    ↓
Aim Run Storage (.aim/)
    ↓
Aim UI (http://localhost:43800)
    ↓
Interactive Visualizations
```

---

## Implementation Plan

### Phase 1: Core Integration (30 min)

**Goal:** Basic tracking of runs with metadata and core metrics

#### 1. Install Aim

```bash
pip install aim
```

#### 2. Create Tracking Module

**`lab/tracking/__init__.py`:**
```python
"""Experiment tracking with Aim."""
from .tracker import TinyLabTracker

__all__ = ['TinyLabTracker']
```

**`lab/tracking/tracker.py`:**
```python
"""Main tracking class for tinyLab experiments."""
from aim import Run
from pathlib import Path
from typing import Dict, Any, Optional
import json

class TinyLabTracker:
    """
    Wrapper around Aim for mechanistic interpretability experiments.

    Example usage:
        tracker = TinyLabTracker(
            experiment_name="h1_suppressor_sweep",
            config=config_dict,
            tags=["gpt2-medium", "facts", "layer0"]
        )

        # Log metrics
        tracker.log_metric("logit_diff", 2.45, step=0)
        tracker.log_metric("accuracy", 0.89, step=0)

        # Log custom MI metrics
        tracker.log_ov_fidelity(ov_scores_by_layer, step=0)

        # Log artifacts
        tracker.log_attention_pattern(attn_matrix, head=(0, 2))

        # Finish
        tracker.finish()
    """

    def __init__(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        tags: Optional[list] = None,
        repo_path: Optional[str] = None
    ):
        """
        Initialize tracker for an experiment run.

        Args:
            experiment_name: Name of experiment (e.g., "h1_cross_condition")
            config: Full experiment configuration dict
            tags: List of tags for filtering (e.g., ["gpt2-medium", "facts"])
            repo_path: Path to .aim directory (default: project root)
        """
        self.experiment_name = experiment_name
        self.config = config

        # Initialize Aim run
        self.run = Run(
            repo=repo_path,
            experiment=experiment_name,
            tags=tags or []
        )

        # Log all config as hyperparameters
        self.run['hparams'] = config

        # Log key metadata
        self.run['model_name'] = config.get('model_name', 'unknown')
        self.run['condition'] = config.get('tag', 'unknown')
        self.run['probe_type'] = config.get('probe', 'unknown')
        self.run['device'] = config.get('device', 'unknown')
        self.run['seed'] = config.get('seed', None)

        # Git info
        import git
        try:
            repo = git.Repo(search_parent_directories=True)
            self.run['git_commit'] = repo.head.commit.hexsha[:8]
            self.run['git_branch'] = repo.active_branch.name
        except:
            pass

    def log_metric(self, name: str, value: float, step: int = 0, context: Optional[Dict] = None):
        """
        Log a scalar metric.

        Args:
            name: Metric name (e.g., "logit_diff")
            value: Metric value
            step: Step/iteration (0 for final metrics)
            context: Additional context (e.g., {"head": "0:2"})
        """
        self.run.track(value, name=name, step=step, context=context or {})

    def log_metrics_dict(self, metrics: Dict[str, float], step: int = 0, prefix: str = ""):
        """
        Log multiple metrics at once.

        Args:
            metrics: Dict of {metric_name: value}
            step: Step/iteration
            prefix: Prefix to add to all metric names
        """
        for name, value in metrics.items():
            full_name = f"{prefix}/{name}" if prefix else name
            self.log_metric(full_name, value, step=step)

    def log_head_metrics(self, head: tuple, metrics: Dict[str, float], step: int = 0):
        """
        Log metrics for a specific attention head.

        Args:
            head: Tuple of (layer, head_idx)
            metrics: Dict of {metric_name: value}
            step: Step/iteration
        """
        context = {"layer": head[0], "head": head[1]}
        for name, value in metrics.items():
            self.run.track(value, name=name, step=step, context=context)

    def log_layer_metrics(self, layer: int, metrics: Dict[str, float], step: int = 0):
        """
        Log metrics for a specific layer.

        Args:
            layer: Layer index
            metrics: Dict of {metric_name: value}
            step: Step/iteration
        """
        context = {"layer": layer}
        for name, value in metrics.items():
            self.run.track(value, name=name, step=step, context=context)

    def log_ov_fidelity(self, fidelity_by_layer: Dict[int, float], step: int = 0):
        """
        Log OV circuit fidelity across layers.

        Args:
            fidelity_by_layer: {layer_idx: fidelity_score}
            step: Step/iteration
        """
        for layer, fidelity in fidelity_by_layer.items():
            self.run.track(
                fidelity,
                name="ov_fidelity",
                step=step,
                context={"layer": layer}
            )

    def log_activation_entropy(
        self,
        layer: int,
        entropy: float,
        entropy_type: str = "subspace",
        step: int = 0
    ):
        """
        Log activation entropy for a layer.

        Args:
            layer: Layer index
            entropy: Entropy value
            entropy_type: Type of entropy ("subspace", "diagonal", "per_token")
            step: Step/iteration
        """
        self.run.track(
            entropy,
            name=f"activation_entropy_{entropy_type}",
            step=step,
            context={"layer": layer}
        )

    def log_geometric_metrics(
        self,
        curvature: float,
        output_entropy: float,
        step: int = 0,
        phase: str = "final"
    ):
        """
        Log geometric signature metrics.

        Args:
            curvature: Trajectory curvature
            output_entropy: Output distribution entropy
            step: Step/iteration
            phase: Phase of trajectory ("early", "mid", "final")
        """
        self.run.track(curvature, name="curvature", step=step, context={"phase": phase})
        self.run.track(output_entropy, name="output_entropy", step=step, context={"phase": phase})

    def log_image(self, name: str, image, step: int = 0, context: Optional[Dict] = None):
        """
        Log an image (attention pattern, plot, etc.).

        Args:
            name: Image name
            image: PIL Image, numpy array, or matplotlib figure
            step: Step/iteration
            context: Additional context
        """
        from aim import Image
        self.run.track(Image(image), name=name, step=step, context=context or {})

    def log_attention_pattern(self, pattern, layer: int, head: int, step: int = 0):
        """
        Log attention pattern heatmap.

        Args:
            pattern: Attention matrix (numpy array or matplotlib figure)
            layer: Layer index
            head: Head index
            step: Step/iteration
        """
        import matplotlib.pyplot as plt
        from aim import Image

        # If pattern is numpy array, create heatmap
        if hasattr(pattern, 'shape'):
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(pattern, cmap='viridis', aspect='auto')
            ax.set_title(f'Attention Pattern L{layer}H{head}')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            plt.colorbar(im, ax=ax)
            self.run.track(
                Image(fig),
                name="attention_pattern",
                step=step,
                context={"layer": layer, "head": head}
            )
            plt.close(fig)
        else:
            # Assume it's already a figure
            self.run.track(
                Image(pattern),
                name="attention_pattern",
                step=step,
                context={"layer": layer, "head": head}
            )

    def log_distribution(self, name: str, values, step: int = 0, context: Optional[Dict] = None):
        """
        Log a distribution of values.

        Args:
            name: Distribution name
            values: Array of values
            step: Step/iteration
            context: Additional context
        """
        from aim import Distribution
        self.run.track(
            Distribution(values),
            name=name,
            step=step,
            context=context or {}
        )

    def log_text(self, name: str, text: str, step: int = 0):
        """
        Log text (e.g., model output, errors).

        Args:
            name: Text identifier
            text: Text content
            step: Step/iteration
        """
        from aim import Text
        self.run.track(Text(text), name=name, step=step)

    def log_artifact(self, name: str, artifact: Any):
        """
        Log arbitrary Python object as artifact.

        Args:
            name: Artifact name
            artifact: Any JSON-serializable object
        """
        self.run[name] = artifact

    def finish(self, final_metrics: Optional[Dict[str, float]] = None):
        """
        Finalize the run.

        Args:
            final_metrics: Optional final metrics to log
        """
        if final_metrics:
            self.log_metrics_dict(final_metrics, prefix="final")

        self.run.close()
```

#### 3. Integrate with Harness

**Modify `lab/harness.py`:**

```python
# At top of file
from lab.tracking import TinyLabTracker

# In the main experiment function:
def run_experiment(config_path: str):
    # Load config
    config = load_config(config_path)

    # Initialize tracker
    tracker = TinyLabTracker(
        experiment_name=config.get('experiment', 'unnamed'),
        config=config,
        tags=[
            config['model_name'],
            config.get('tag', 'unknown'),
            f"layer{config.get('target_layer', 0)}"
        ]
    )

    try:
        # Run experiment
        results = run_ablation_sweep(config)

        # Log results
        for head, metrics in results.items():
            tracker.log_head_metrics(
                head=head,
                metrics={
                    'logit_diff': metrics['ld'],
                    'accuracy': metrics['acc'],
                    'p_drop': metrics['p_drop'],
                    'kl_divergence': metrics['kl']
                }
            )

        # Log aggregate metrics
        tracker.log_metrics_dict({
            'mean_logit_diff': np.mean([m['ld'] for m in results.values()]),
            'max_logit_diff': np.max([m['ld'] for m in results.values()]),
            'top_head_ld': sorted(results.items(), key=lambda x: x[1]['ld'])[-1][1]['ld']
        })

    finally:
        tracker.finish()
```

#### 4. Start Aim UI

```bash
# Launch web UI
aim up

# Opens at http://localhost:43800
```

---

### Phase 2: Historical Data Import (1 hour)

**Goal:** Import existing results from `reports/` into Aim

**`scripts/import_to_aim.py`:**
```python
#!/usr/bin/env python3
"""
Import historical tinyLab results into Aim.

Usage:
    python scripts/import_to_aim.py
    python scripts/import_to_aim.py --reports-dir reports/
"""
import argparse
import json
from pathlib import Path
from aim import Run
import re

def parse_filename(filename: str):
    """Extract metadata from filename."""
    # Examples:
    # gpt2m_facts_ranking.csv
    # mistral_cf_l0_ranking.csv
    # h1_head_rank_stats.json

    parts = filename.stem.split('_')
    metadata = {}

    # Extract model
    if 'gpt2' in filename.stem:
        if 'gpt2m' in filename.stem:
            metadata['model'] = 'gpt2-medium'
        elif 'gpt2l' in filename.stem:
            metadata['model'] = 'gpt2-large'
        else:
            metadata['model'] = 'gpt2'
    elif 'mistral' in filename.stem:
        metadata['model'] = 'mistral-7b'
    elif 'pythia' in filename.stem:
        metadata['model'] = 'pythia'

    # Extract condition
    conditions = ['facts', 'cf', 'logic', 'neg', 'counterfactual', 'negation', 'logical']
    for cond in conditions:
        if cond in filename.stem:
            metadata['condition'] = cond
            break

    # Extract hypothesis
    h_match = re.search(r'h(\d+)', filename.stem)
    if h_match:
        metadata['hypothesis'] = f"H{h_match.group(1)}"

    return metadata

def import_head_rankings(csv_path: Path, repo_path: str = None):
    """Import head ranking CSV."""
    import pandas as pd

    metadata = parse_filename(csv_path)

    run = Run(
        repo=repo_path,
        experiment=f"imported_{metadata.get('hypothesis', 'ranking')}",
        tags=['imported', 'historical'] + list(metadata.values())
    )

    # Log metadata
    run['source_file'] = str(csv_path)
    run['imported'] = True
    for k, v in metadata.items():
        run[k] = v

    # Load and log data
    df = pd.read_csv(csv_path)

    for idx, row in df.iterrows():
        layer = row.get('layer', row.get('Layer', 0))
        head = row.get('head', row.get('Head', idx))

        context = {"layer": int(layer), "head": int(head)}

        # Log available metrics
        for col in df.columns:
            if col.lower() in ['layer', 'head', 'rank']:
                continue
            try:
                value = float(row[col])
                run.track(value, name=col.lower(), step=0, context=context)
            except:
                pass

    run.close()
    print(f"✓ Imported {csv_path.name}")

def import_json_metrics(json_path: Path, repo_path: str = None):
    """Import JSON metric file."""
    metadata = parse_filename(json_path)

    run = Run(
        repo=repo_path,
        experiment=f"imported_{metadata.get('hypothesis', 'metrics')}",
        tags=['imported', 'historical'] + list(metadata.values())
    )

    # Log metadata
    run['source_file'] = str(json_path)
    run['imported'] = True
    for k, v in metadata.items():
        run[k] = v

    # Load data
    with open(json_path) as f:
        data = json.load(f)

    # Log all metrics
    def log_nested(obj, prefix=""):
        """Recursively log nested dict."""
        if isinstance(obj, dict):
            for key, val in obj.items():
                new_prefix = f"{prefix}/{key}" if prefix else key
                log_nested(val, new_prefix)
        elif isinstance(obj, (int, float)):
            run.track(float(obj), name=prefix, step=0)
        elif isinstance(obj, list) and all(isinstance(x, (int, float)) for x in obj):
            # Log as distribution
            from aim import Distribution
            run.track(Distribution(obj), name=prefix, step=0)

    log_nested(data)

    run.close()
    print(f"✓ Imported {json_path.name}")

def main():
    parser = argparse.ArgumentParser(description="Import tinyLab results to Aim")
    parser.add_argument('--reports-dir', default='reports/', help='Path to reports directory')
    parser.add_argument('--repo', default=None, help='Path to .aim directory')
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)

    # Import CSVs
    print("Importing CSV files...")
    for csv_file in reports_dir.glob('**/*.csv'):
        try:
            import_head_rankings(csv_file, repo_path=args.repo)
        except Exception as e:
            print(f"✗ Failed to import {csv_file.name}: {e}")

    # Import JSONs
    print("\nImporting JSON files...")
    for json_file in reports_dir.glob('**/*.json'):
        # Skip manifest files
        if 'manifest' in json_file.name.lower():
            continue
        try:
            import_json_metrics(json_file, repo_path=args.repo)
        except Exception as e:
            print(f"✗ Failed to import {json_file.name}: {e}")

    print("\n✓ Import complete! Launch UI with: aim up")

if __name__ == '__main__':
    main()
```

**Run import:**
```bash
python scripts/import_to_aim.py
```

---

### Phase 3: Custom Visualizations (2 hours)

**Goal:** Add tinyLab-specific visualizations to Aim UI

**`lab/tracking/visualizations.py`:**
```python
"""Custom visualizations for Aim UI."""
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any

class MIVisualizations:
    """Mechanistic interpretability visualizations."""

    @staticmethod
    def plot_layer_metrics(metrics_by_layer: Dict[int, Dict[str, float]], metric_name: str):
        """
        Plot metric evolution across layers.

        Args:
            metrics_by_layer: {layer_idx: {metric_name: value}}
            metric_name: Which metric to plot

        Returns:
            matplotlib.Figure
        """
        layers = sorted(metrics_by_layer.keys())
        values = [metrics_by_layer[l][metric_name] for l in layers]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(layers, values, marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{metric_name.replace("_", " ").title()} Across Layers', fontsize=14)
        ax.grid(True, alpha=0.3)

        return fig

    @staticmethod
    def plot_head_heatmap(head_metrics: Dict[tuple, float], n_layers: int, n_heads: int):
        """
        Plot heatmap of head-level metrics.

        Args:
            head_metrics: {(layer, head): metric_value}
            n_layers: Number of layers
            n_heads: Number of heads per layer

        Returns:
            matplotlib.Figure
        """
        # Create matrix
        matrix = np.zeros((n_layers, n_heads))
        for (layer, head), value in head_metrics.items():
            matrix[layer, head] = value

        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')

        ax.set_xlabel('Head', fontsize=12)
        ax.set_ylabel('Layer', fontsize=12)
        ax.set_title('Head Ablation Effects (ΔLD)', fontsize=14)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Logit Difference', fontsize=12)

        # Add grid
        ax.set_xticks(np.arange(n_heads))
        ax.set_yticks(np.arange(n_layers))
        ax.grid(which='major', color='white', linewidth=0.5)

        return fig

    @staticmethod
    def plot_emergence_curve(
        checkpoint_metrics: Dict[int, float],
        checkpoint_steps: List[int],
        metric_name: str = "logit_diff"
    ):
        """
        Plot metric emergence across training checkpoints (for Pythia).

        Args:
            checkpoint_metrics: {checkpoint_step: metric_value}
            checkpoint_steps: List of checkpoint steps
            metric_name: Metric to plot

        Returns:
            matplotlib.Figure
        """
        steps = sorted(checkpoint_steps)
        values = [checkpoint_metrics.get(s, 0) for s in steps]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps, values, marker='o', linewidth=2, markersize=8, color='#2E86AB')
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{metric_name.replace("_", " ").title()} Emergence', fontsize=14)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

        # Add shaded region for crystallization
        if len(values) > 2:
            # Find inflection point (simple heuristic)
            diffs = np.diff(values)
            inflection = np.argmax(diffs) + 1
            ax.axvspan(steps[0], steps[inflection], alpha=0.1, color='red', label='Pre-crystallization')
            ax.axvspan(steps[inflection], steps[-1], alpha=0.1, color='green', label='Post-crystallization')
            ax.legend()

        return fig

    @staticmethod
    def plot_ov_token_projection(
        token_embeddings: np.ndarray,
        token_labels: List[str],
        title: str = "OV Circuit Token Projection"
    ):
        """
        Plot 2D projection of OV-projected tokens.

        Args:
            token_embeddings: (n_tokens, embedding_dim) array
            token_labels: List of token strings
            title: Plot title

        Returns:
            matplotlib.Figure
        """
        from sklearn.decomposition import PCA

        # Project to 2D
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(token_embeddings)

        fig, ax = plt.subplots(figsize=(12, 8))

        # Scatter plot
        scatter = ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=range(len(token_labels)),
            cmap='viridis',
            s=100,
            alpha=0.6
        )

        # Add labels
        for i, label in enumerate(token_labels):
            ax.annotate(
                label,
                (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                fontsize=9,
                alpha=0.8
            )

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)

        return fig
```

---

### Phase 4: DVC Integration (30 min)

**Goal:** Ensure Aim storage works with DVC

**Update `.gitignore`:**
```gitignore
# Aim tracking (local only, regenerate from DVC data)
/.aim/
```

**Optional:** Track Aim exports with DVC:

```python
# scripts/export_aim_reports.py
"""Export Aim runs to static JSON for DVC tracking."""
from aim import Repo
import json

repo = Repo('.')

# Export all runs metadata
runs_data = []
for run in repo.iter_runs():
    runs_data.append({
        'hash': run.hash,
        'name': run.name,
        'experiment': run.experiment,
        'creation_time': run.creation_time.isoformat(),
        'params': run.get('hparams', {}),
        'metrics': {
            track.name: track.values.last_value()
            for track in run.metrics()
        }
    })

# Save to reports/
with open('reports/aim_runs_export.json', 'w') as f:
    json.dump(runs_data, f, indent=2)

# Track with DVC
# dvc add reports/aim_runs_export.json
```

---

## Usage Guide

### Running Experiments with Tracking

**Before (no tracking):**
```bash
python -m lab.battery --config lab/configs/run_h1_cross_condition_balanced.json
```

**After (with Aim tracking):**
```bash
# Tracking is automatic! Just run as before
python -m lab.battery --config lab/configs/run_h1_cross_condition_balanced.json

# View in UI
aim up
```

### Browsing Experiments

**Launch UI:**
```bash
aim up
# Opens http://localhost:43800
```

**UI Features:**

1. **Runs Table** - View all runs with metadata, hyperparameters, metrics
2. **Metrics Explorer** - Compare metrics across runs with interactive plots
3. **Images** - Browse attention patterns, OV projections, calibration curves
4. **Text Logs** - View model outputs, errors, notes
5. **Params** - Filter and group by hyperparameters
6. **Custom Dashboards** - Create saved views for specific analyses

### Filtering and Grouping

**In UI:**
- Filter by model: `run.model_name == "gpt2-medium"`
- Filter by condition: `run.condition == "facts"`
- Group by hypothesis: Group by `run.hypothesis`
- Compare top heads: Filter by `logit_diff > 2.0` with context `{layer: 0}`

**Programmatically:**
```python
from aim import Repo

repo = Repo('.')

# Find all GPT-2 Medium facts runs
runs = repo.query_runs(
    "run.model_name == 'gpt2-medium' and run.condition == 'facts'"
).iter()

for run in runs:
    print(f"Run {run.hash}: LD = {run.metrics()['logit_diff'].last_value()}")
```

### Creating Custom Dashboards

**Example: Suppressor Analysis Dashboard**

```python
# In Aim UI → Metrics → Create new dashboard
# Add charts:
# 1. Logit Diff by Layer (line plot, group by layer)
# 2. Head Heatmap (table, context: {layer, head})
# 3. OV Fidelity Over Time (line plot, group by checkpoint)
# 4. Attention Patterns (image grid, filter: layer == 0)

# Save as "Suppressor Analysis"
```

---

## Advanced Features

### 1. Compare Runs Side-by-Side

```python
from aim import Repo

repo = Repo('.')

# Get two runs
run1 = repo.get_run('abc123')  # GPT-2 Medium facts
run2 = repo.get_run('def456')  # Mistral facts

# Compare metrics
for metric_name in ['logit_diff', 'accuracy', 'calibration_ece']:
    val1 = run1.metrics()[metric_name].last_value()
    val2 = run2.metrics()[metric_name].last_value()
    print(f"{metric_name}: GPT-2={val1:.3f}, Mistral={val2:.3f}")
```

### 2. Export for Paper

```python
# Export specific metrics for LaTeX table
from aim import Repo
import pandas as pd

repo = Repo('.')

# Query runs
runs = repo.query_runs("run.condition == 'facts'").iter()

# Build dataframe
data = []
for run in runs:
    data.append({
        'Model': run['model_name'],
        'ΔLD': run.metrics()['logit_diff'].last_value(),
        'Accuracy': run.metrics()['accuracy'].last_value(),
        'ECE': run.metrics()['calibration_ece'].last_value(),
    })

df = pd.DataFrame(data)
print(df.to_latex(index=False, float_format='%.3f'))
```

### 3. Automated Analysis Pipelines

```python
# scripts/analyze_latest_run.py
"""Analyze most recent run and generate report."""
from aim import Repo

repo = Repo('.')

# Get latest run
run = sorted(repo.iter_runs(), key=lambda r: r.creation_time, reverse=True)[0]

print(f"Latest Run: {run.hash}")
print(f"Experiment: {run.experiment}")
print(f"Model: {run['model_name']}")
print(f"\nTop Metrics:")
for name in ['logit_diff', 'accuracy', 'p_drop']:
    print(f"  {name}: {run.metrics()[name].last_value():.3f}")

# Find top suppressor heads
head_metrics = {}
for track in run.metrics():
    if track.name == 'logit_diff' and track.context.get('layer') == 0:
        head = track.context['head']
        head_metrics[head] = track.values.last_value()

top_heads = sorted(head_metrics.items(), key=lambda x: x[1], reverse=True)[:5]
print(f"\nTop 5 Suppressor Heads (L0):")
for head, ld in top_heads:
    print(f"  Head {head}: ΔLD = {ld:.3f}")
```

---

## Migration Checklist

- [ ] Install Aim: `pip install aim`
- [ ] Create `lab/tracking/` module
- [ ] Add `TinyLabTracker` class
- [ ] Integrate tracking into `lab/harness.py`
- [ ] Test with single experiment run
- [ ] Import historical results: `python scripts/import_to_aim.py`
- [ ] Launch UI: `aim up`
- [ ] Verify metrics, images, distributions appear correctly
- [ ] Create custom dashboards for key analyses
- [ ] Add `.aim/` to `.gitignore`
- [ ] Update documentation (DVC_SETUP.md, README.md)
- [ ] Train team on Aim UI usage

---

## FAQ

**Q: How does Aim compare to MLflow?**
A: Aim has a more modern UI, better metric comparison, and is specifically designed for ML/DL experiments. MLflow is more general-purpose with deployment features we don't need.

**Q: Will this slow down experiments?**
A: Minimal overhead (<1% for typical runs). Logging is asynchronous.

**Q: Can I disable tracking?**
A: Yes, just don't initialize `TinyLabTracker`. Or use env var: `TINYLAB_DISABLE_TRACKING=1`.

**Q: How much storage does Aim use?**
A: ~1-5MB per run for metrics/metadata. Images/distributions increase this. Use `aim storage --clean` to remove old runs.

**Q: Can I query Aim from notebooks?**
A: Yes! See examples above. Full Python API available.

**Q: How do I backup Aim data?**
A: The `.aim/` directory contains everything. Can export to JSON for DVC tracking or copy entire directory.

---

## Next Steps

1. **Implement Phase 1** - Basic tracking (30 min)
2. **Test with one experiment** - Verify tracking works (15 min)
3. **Import historical data** - Run import script (30 min)
4. **Explore UI** - Familiarize with Aim interface (30 min)
5. **Add custom visualizations** - Implement MI-specific plots (2 hours)
6. **Create dashboards** - Build saved views for analyses (1 hour)
7. **Document for team** - Write usage guide (1 hour)

**Total time:** ~5-6 hours for complete integration

---

## Resources

- **Aim Docs:** https://aimstack.readthedocs.io/
- **Aim GitHub:** https://github.com/aimhubio/aim
- **Aim Discord:** https://community.aimstack.io/
- **Examples:** https://github.com/aimhubio/aim/tree/main/examples

---

**Document Version:** 1.0
**Author:** Claude
**Status:** Ready for Implementation
