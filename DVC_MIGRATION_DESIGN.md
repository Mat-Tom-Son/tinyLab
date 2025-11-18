# DVC Migration Design for tinyLab

## Executive Summary

This document outlines the design for migrating tinyLab to use DVC (Data Version Control) for all datasets, checkpoints, logs, and artifacts. The design prioritizes:
1. **Minimal code changes** - preserve existing workflows
2. **Clear organization** - logical grouping by purpose
3. **Future-proof** - ready for S3/GCS/Azure backends
4. **Reversibility** - all changes staged behind git branches

## Current State Inventory

### Data Currently in Git (to be moved to DVC)

| Category | Location | Files | Size | Purpose |
|----------|----------|-------|------|---------|
| Raw Data | `lab/data/corpora/` | 18 | ~370K | JSONL datasets (facts, counterfactual, logical, negation) |
| Data Splits | `lab/data/splits/` | 18 | ~29K | Train/val/test indices |
| Lexicons | `data/lexicons/` | 1 | 949B | Hedge/booster word lists |
| Results (CSV) | `reports/` | 161 | ~4.5MB | Head rankings, layer sweeps, summaries |
| Results (JSON) | `reports/` | 137 | ~2.8MB | Metrics, analyses, manifests |
| Paper Supplements | `paper/supplement/` | 20+ | ~150K | Bootstrap CI, calibration, validation data |

**Total data to track with DVC: ~7.4 MB across 355+ files**

### Data Already Gitignored (stays ignored)

- `lab/runs/*` - Empty (only .gitkeep)
- `mlruns/*` - Empty (only .gitkeep)
- `*.png`, `*.html`, `*.pdf` - Generated plots
- `*.log` - Generated logs
- `*.ipynb` - Jupyter notebooks

### Code/Config (stays in Git)

- `lab/configs/*.json` - 100 experiment configs (~124K)
- All Python scripts (analysis, training, figures)
- LaTeX source files
- Documentation and makefiles

---

## Proposed Directory Structure

### Option A: Minimal Restructure (RECOMMENDED)

Keep existing paths but organize DVC tracking by purpose. Minimal code changes required.

```
tinyLab/
├── .dvc/                      # DVC configuration
├── .dvcstore/                 # Local DVC cache (gitignored)
│
├── data/                      # Raw data - DVC tracked
│   ├── lexicons/              # [DVC] Lexicon files
│   │   └── hedge_booster.json
│   └── README.md              # Documents data sources
│
├── lab/
│   ├── data/                  # Lab datasets - DVC tracked
│   │   ├── corpora/           # [DVC] Raw experimental corpora
│   │   │   ├── facts_*.jsonl
│   │   │   ├── counterfactual_*.jsonl
│   │   │   ├── logical_*.jsonl
│   │   │   └── negation_*.jsonl
│   │   └── splits/            # [DVC] Processed train/test splits
│   │       └── *.split.json
│   │
│   ├── configs/               # [GIT] Experiment configurations
│   ├── analysis/              # [GIT] Analysis scripts
│   ├── runs/                  # [IGNORED] Generated training runs
│   └── tests/                 # [GIT] Test files
│
├── reports/                   # Results - DVC tracked
│   ├── *.csv                  # [DVC] All ranking CSVs
│   ├── *.json                 # [DVC] All metric JSONs
│   ├── layer_sweep_*/         # [DVC] Layer sweep subdirs
│   ├── appendices/            # [DVC] Additional analyses
│   ├── RESULTS_MANIFEST.json  # [DVC] Master results index
│   └── README.md              # Documents results structure
│
├── paper/
│   ├── sections/              # [GIT] LaTeX source
│   ├── scripts/               # [GIT] Figure generation scripts
│   ├── supplement/            # Paper supplement data - DVC tracked
│   │   ├── *.json             # [DVC] Supplement metrics
│   │   └── cuda_validation/   # [DVC] CUDA validation results
│   └── generated/             # [IGNORED] Auto-generated content
│
├── mlruns/                    # [IGNORED] MLflow tracking
├── figs/                      # [GIT] Figure descriptions (md)
│                              # [IGNORED] Rendered plots (png/pdf)
├── docs/                      # [GIT] Documentation
└── devlog/                    # [GIT] Development logs
```

**DVC Tracking Scheme:**
- `data/lexicons/*.json` → Track individual files
- `lab/data/corpora/` → Track entire directory
- `lab/data/splits/` → Track entire directory
- `reports/` → Track entire directory (includes all CSV/JSON)
- `paper/supplement/` → Track entire directory

---

### Option B: Full Reorganization (More disruptive)

Complete restructure following canonical data science layout. Requires updating all import paths.

```
tinyLab/
├── .dvc/
├── .dvcstore/                 # Local DVC cache
│
├── data/                      # ALL data - DVC tracked
│   ├── raw/                   # Raw immutable data
│   │   ├── corpora/           # [DVC] Moved from lab/data/corpora/
│   │   │   ├── facts/
│   │   │   ├── counterfactual/
│   │   │   ├── logical/
│   │   │   └── negation/
│   │   └── lexicons/          # [DVC] Moved from data/lexicons/
│   │
│   └── processed/             # Derived/transformed data
│       └── splits/            # [DVC] Moved from lab/data/splits/
│
├── results/                   # Renamed from reports/ - DVC tracked
│   ├── metrics/               # [DVC] JSON metric files
│   ├── rankings/              # [DVC] CSV ranking files
│   ├── analyses/              # [DVC] Specialized analyses
│   │   ├── layer_sweep/
│   │   ├── vdi_drift/
│   │   ├── entropy/
│   │   └── ov_reports/
│   └── MANIFEST.json          # Master index
│
├── models/                    # For future model artifacts
│   └── checkpoints/           # [DVC] Training checkpoints (currently empty)
│
├── lab/                       # Experimental code
│   ├── configs/               # [GIT] Experiment configs
│   ├── analysis/              # [GIT] Analysis scripts
│   └── tests/                 # [GIT] Test files
│
├── paper/
│   ├── supplement/            # [DVC] Paper supplement data
│   └── sections/              # [GIT] LaTeX source
│
├── logs/                      # Execution logs
│   ├── mlruns/                # [IGNORED] MLflow runs
│   └── training/              # [IGNORED] Training logs
│
└── notebooks/                 # [IGNORED] Jupyter notebooks
```

**Migration Impact:**
- Requires updating ~30 analysis scripts
- Need to update Makefile paths
- Configuration files need path updates
- More maintenance but cleaner long-term

---

## Recommendation: Option A (Minimal Restructure)

**Rationale:**
1. **Low risk** - existing code continues to work
2. **Fast migration** - can be completed in hours, not days
3. **Reversible** - easy to rollback if needed
4. **Sufficient** - achieves all DVC goals without unnecessary complexity

The current structure is already reasonably well-organized:
- `lab/data/` clearly separates experimental data
- `reports/` is an established convention
- `paper/supplement/` is logically placed

We can achieve clean DVC tracking without restructuring.

---

## DVC Configuration

### DVC Remote Structure

```bash
# Local remote inside repository (git-ignored)
.dvcstore/
  ├── files/
  │   └── md5/              # Content-addressable storage
  │       ├── ab/
  │       │   └── cdef123...
  │       └── ...
  └── tmp/
```

**Configuration:**
```bash
# .dvc/config.local
[core]
    remote = localstore

[remote "localstore"]
    url = .dvcstore
```

**Future S3 Migration:**
```bash
# Just add remote and push
dvc remote add s3store s3://tinylab-data/
dvc remote default s3store
dvc push
```

### .gitignore Updates

Add to `.gitignore`:
```gitignore
# DVC
/reports/*.csv
/reports/*.json
/reports/layer_sweep_*
/reports/appendices
/lab/data/corpora
/lab/data/splits
/data/lexicons
/paper/supplement/*.json
/paper/supplement/*.csv
/paper/supplement/cuda_validation
.dvcstore/
```

Keep tracking:
- `*.dvc` files (DVC pointers)
- `.dvc/config` (DVC configuration)
- `.dvc/.gitignore`

---

## DVC Tracking Strategy

### Granularity Decision Matrix

| Directory | Strategy | Rationale |
|-----------|----------|-----------|
| `lab/data/corpora/` | Single `.dvc` for entire dir | Files change together, versioned as unit |
| `lab/data/splits/` | Single `.dvc` for entire dir | Derived from corpora, versioned together |
| `data/lexicons/` | Individual `.dvc` per file | Small, independent files |
| `reports/` | Single `.dvc` for entire dir | Results regenerated together, large file count |
| `paper/supplement/` | Single `.dvc` for entire dir | Small, versioned with paper |

### Directory-Level Tracking

```bash
# Track entire directories
dvc add lab/data/corpora
dvc add lab/data/splits
dvc add reports
dvc add paper/supplement

# Track individual files
dvc add data/lexicons/hedge_booster.json
```

**Generated artifacts:**
```
lab/data/corpora.dvc          # Pointer file (goes in git)
lab/data/splits.dvc           # Pointer file (goes in git)
reports.dvc                   # Pointer file (goes in git)
paper/supplement.dvc          # Pointer file (goes in git)
data/lexicons/hedge_booster.json.dvc  # Pointer file (goes in git)
```

---

## Migration Workflow

### Phase 1: Preparation (No changes to working tree)

1. Create branch: `git checkout -b dvc-migration`
2. Install DVC: `pip install dvc`
3. Initialize DVC: `dvc init`
4. Configure local remote:
   ```bash
   dvc remote add localstore .dvcstore --local
   dvc remote default localstore
   ```
5. Update `.gitignore` with DVC patterns

### Phase 2: Add DVC Tracking

**Track data directories:**
```bash
# Add DVC tracking (data moved to .dvcstore, .dvc pointers created)
dvc add lab/data/corpora
dvc add lab/data/splits
dvc add data/lexicons/hedge_booster.json
dvc add reports
dvc add paper/supplement

# Check what was created
ls -la lab/data/*.dvc
ls -la *.dvc
ls -la paper/*.dvc
```

**Commit DVC pointers:**
```bash
git add lab/data/corpora.dvc lab/data/splits.dvc
git add data/lexicons/hedge_booster.json.dvc
git add reports.dvc paper/supplement.dvc
git add .gitignore .dvc/config .dvc/.gitignore
git commit -m "Add DVC tracking for datasets, results, and supplements"
```

### Phase 3: Verification

**Test data retrieval:**
```bash
# Remove data (simulate fresh clone)
rm -rf lab/data/corpora lab/data/splits reports paper/supplement
rm -f data/lexicons/hedge_booster.json

# Restore from DVC
dvc pull

# Verify all files restored
ls lab/data/corpora/*.jsonl
ls lab/data/splits/*.json
ls reports/*.csv
ls paper/supplement/*.json
```

**Test reproducibility:**
```bash
# Run smoke test
python smoke_test.py

# Run single analysis
python lab/analysis/export_head_rankings.py

# Verify outputs match
```

### Phase 4: Documentation and Push

```bash
# Create comprehensive docs
# (see Documentation section below)

# Push to remote
git push -u origin dvc-migration

# Create pull request for review
```

---

## Data Flows and Dependencies

### Data Generation Pipeline

```
Raw Data (DVC)
  ↓
lab/data/corpora/*.jsonl
  ↓
[scripts/facts_make_split.py]
  ↓
lab/data/splits/*.json (DVC)
  ↓
[lab/analysis/*.py scripts]
  ↓
reports/*.csv + *.json (DVC)
  ↓
[paper/scripts/*.py]
  ↓
paper/supplement/*.json (DVC)
  ↓
[pdflatex]
  ↓
paper/main.pdf (IGNORED)
```

### Reproducibility Requirements

To regenerate all results from scratch:

```bash
# 1. Clone repository
git clone <repo> && cd tinyLab

# 2. Restore data
dvc pull

# 3. Install dependencies
pip install -e .

# 4. Run analyses
make postprocess

# 5. Generate paper
cd paper && make
```

**Critical insight:** Only raw data and splits need DVC tracking. Results can be regenerated via `make postprocess`, but we track them anyway for:
- **Speed** - Avoid re-running expensive analyses
- **Reproducibility** - Preserve exact results for papers
- **Collaboration** - Share results without re-computation

---

## Documentation Requirements

### 1. DVC_SETUP.md (New file)

```markdown
# DVC Setup Guide for tinyLab

## Installation

# Prerequisites
- Python 3.11+
- Git

# Install DVC
pip install dvc

## First-time Setup (after cloning)

# Pull all data
dvc pull

# Verify
ls lab/data/corpora/*.jsonl
ls reports/*.csv

## Adding New Data

# Track new dataset
dvc add data/new_dataset.csv
git add data/new_dataset.csv.dvc
git commit -m "Add new dataset"

## Updating Tracked Data

# Modify data, then update tracking
dvc add reports/
git add reports.dvc
git commit -m "Update results after experiment X"

## Troubleshooting

See docs/DVC_TROUBLESHOOTING.md
```

### 2. Update README.md

Add DVC section:
```markdown
## Data Management with DVC

This project uses DVC to manage datasets and results. After cloning:

\`\`\`bash
pip install dvc
dvc pull
\`\`\`

See [DVC_SETUP.md](DVC_SETUP.md) for detailed instructions.
```

### 3. Update REPLICATION.md

Add DVC step:
```markdown
## Replication Steps

1. Clone repository
2. **Pull data with DVC**: `dvc pull`
3. Install dependencies: `pip install -e .`
4. Run experiments: `make postprocess`
```

---

## Migration Risks and Mitigations

### Risk 1: Large file count in single .dvc file

**Issue:** `reports/` has 298 files. If any single file changes, entire directory re-uploads.

**Mitigation:**
- Acceptable for ~7MB total size
- Can split later if needed: `reports/csv.dvc` + `reports/json.dvc`
- Monitor with `dvc status`

### Risk 2: Git repository growth

**Issue:** Multiple versions of `.dvc` files increase git repo size.

**Mitigation:**
- `.dvc` files are tiny (~100 bytes each)
- Only 5 `.dvc` files total
- Git handles small text files efficiently

### Risk 3: Accidental data loss

**Issue:** `dvc add` moves data to `.dvcstore`, could lose if .dvcstore deleted.

**Mitigation:**
- Create backup before migration: `tar czf tinylab-backup.tar.gz reports/ lab/data/`
- Test `dvc pull` restoration before deleting original data
- Keep branch protection on main/master

### Risk 4: Path breakage

**Issue:** Scripts might hardcode paths that DVC changes.

**Mitigation:**
- Option A (recommended) doesn't change any paths
- DVC creates symlinks/copies, paths remain valid
- Test suite runs before/after migration

### Risk 5: Merge conflicts with .dvc files

**Issue:** Two branches updating same data creates conflicts in `.dvc` files.

**Mitigation:**
- `.dvc` files are structured JSON, easy to merge
- Use `dvc diff` to understand changes
- Document conflict resolution in DVC_SETUP.md

---

## Testing Checklist

Before considering migration complete:

- [ ] `dvc status` shows all files tracked
- [ ] `dvc push` succeeds to localstore
- [ ] `dvc pull` restores all files correctly
- [ ] `python smoke_test.py` passes
- [ ] `make postprocess` completes without errors
- [ ] `cd paper && make` generates PDF
- [ ] All analysis scripts run successfully
- [ ] Git repository size reasonable (<50MB)
- [ ] `.dvcstore` size matches expected (~7-8MB)
- [ ] Fresh clone + `dvc pull` works on different machine
- [ ] Documentation clear and complete

---

## Future Enhancements

### Phase 2: Cloud Storage (S3/GCS)

```bash
# Add S3 remote
dvc remote add s3store s3://tinylab-data/dvc-cache
dvc remote default s3store

# Push to S3
dvc push

# Configure access
dvc remote modify s3store access_key_id XXX
dvc remote modify s3store secret_access_key YYY
```

### Phase 3: Data Versioning

```bash
# Tag dataset versions
git tag -a data-v1.0 -m "Initial dataset release"
git tag -a data-v1.1 -m "Added balanced variants"

# Checkout specific version
git checkout data-v1.0
dvc checkout
```

### Phase 4: Pipelines (Optional)

Define data pipelines in `dvc.yaml`:
```yaml
stages:
  split_data:
    cmd: python scripts/facts_make_split.py
    deps:
      - lab/data/corpora/
    outs:
      - lab/data/splits/

  analyze:
    cmd: python lab/analysis/head_rank_stats.py
    deps:
      - lab/data/splits/
    outs:
      - reports/h1_head_rank_stats.json
```

Run with: `dvc repro`

---

## Appendix: File Size Analysis

### Files by Size Category

| Size Range | Count | Category | DVC Strategy |
|------------|-------|----------|--------------|
| < 1KB | 45 | Config JSON, small JSONs | Track individually or as dir |
| 1-10KB | 89 | Data splits, small metrics | Track as directory |
| 10-50KB | 156 | Corpora, CSVs, metric JSONs | Track as directory |
| 50-100KB | 48 | Large CSVs, result manifests | Track as directory |
| 100KB-1MB | 15 | Large result files | Track as directory |
| > 1MB | 2 | Comprehensive reports | Track as directory |

**Total:** 355 files, ~7.4 MB

### Growth Projections

**Conservative (1 year):**
- New experiments: 10 runs/month × 12 months = 120 runs
- New results: ~200KB per run = 24MB
- New checkpoints: 0 (using pretrained models)
- **Total:** ~31MB

**Aggressive (1 year):**
- New experiments: 50 runs/month × 12 months = 600 runs
- New results: ~200KB per run = 120MB
- Model fine-tuning: 5 checkpoints × 500MB = 2.5GB
- **Total:** ~2.6GB

**Conclusion:** Even aggressive growth is manageable with S3/GCS backends.

---

## Appendix: DVC Commands Reference

### Essential Commands

```bash
# Initialize
dvc init

# Track data
dvc add <path>

# Save changes
git add <path>.dvc .gitignore
git commit -m "Track <path> with DVC"

# Push/pull data
dvc push                    # Upload to remote
dvc pull                    # Download from remote

# Status
dvc status                  # Check for changes
dvc diff                    # Compare versions

# Restore data
dvc checkout                # Restore to committed version
dvc fetch                   # Download without checking out
```

### Advanced Commands

```bash
# Remote management
dvc remote add <name> <url>
dvc remote modify <name> <option> <value>
dvc remote list

# Data management
dvc gc -w                   # Clean up unused cache
dvc cache dir               # Show cache location

# Versioning
dvc get <repo> <path>       # Download specific file
dvc import <repo> <path>    # Import and track from another repo
```

---

## Questions for Team Review

Before proceeding with implementation, please confirm:

1. **Structure:** Option A (minimal) or Option B (full reorganization)?
2. **Granularity:** Single `reports.dvc` or split by type (`reports/csv.dvc`, `reports/json.dvc`)?
3. **Configs:** Should `lab/configs/*.json` move to DVC? (Currently recommended: stay in git)
4. **Generated outputs:** Confirm `*.png`, `*.pdf`, `*.html` should remain gitignored?
5. **Timeline:** Migration in single PR or phased approach?

---

**Document Version:** 1.0
**Date:** 2025-11-18
**Author:** Claude
**Status:** Draft for Review
