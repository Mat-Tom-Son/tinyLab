# DVC Setup Guide for tinyLab

This guide covers setting up and using DVC (Data Version Control) in the tinyLab project.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [First-Time Setup](#first-time-setup)
4. [Daily Workflows](#daily-workflows)
5. [Adding New Data](#adding-new-data)
6. [Updating Tracked Data](#updating-tracked-data)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Usage](#advanced-usage)

---

## Quick Start

For new contributors cloning the repository:

```bash
# 1. Clone the repository
git clone <repository-url>
cd tinyLab

# 2. Install DVC
pip install dvc

# 3. Pull all data
dvc pull

# 4. Verify data
ls lab/data/corpora/*.jsonl
ls reports/*.csv

# 5. Run tests
python smoke_test.py
```

That's it! You now have all datasets and results.

---

## Installation

### Prerequisites

- Python 3.11 or higher
- Git
- pip or conda

### Install DVC

**Using pip:**
```bash
pip install dvc
```

**Using conda:**
```bash
conda install -c conda-forge dvc
```

**Verify installation:**
```bash
dvc version
# Should output: 3.x.x or higher
```

### Optional: Install cloud storage support

If you'll be working with S3, GCS, or Azure later:

```bash
# For S3
pip install 'dvc[s3]'

# For Google Cloud Storage
pip install 'dvc[gs]'

# For Azure
pip install 'dvc[azure]'

# For all backends
pip install 'dvc[all]'
```

---

## First-Time Setup

### After Cloning the Repository

1. **Navigate to the repository:**
   ```bash
   cd tinyLab
   ```

2. **Install DVC** (if not already installed):
   ```bash
   pip install dvc
   ```

3. **Pull all tracked data:**
   ```bash
   dvc pull
   ```

   This downloads:
   - Raw datasets (`lab/data/corpora/`)
   - Data splits (`lab/data/splits/`)
   - Lexicons (`data/lexicons/`)
   - Results and metrics (`reports/`)
   - Paper supplements (`paper/supplement/`)

4. **Verify data integrity:**
   ```bash
   # Check DVC status
   dvc status

   # Should output: "Data and pipelines are up to date."

   # Verify files exist
   ls -lh lab/data/corpora/
   ls -lh reports/
   ```

5. **Run smoke test:**
   ```bash
   python smoke_test.py
   ```

   If this passes, your setup is complete!

---

## Daily Workflows

### Checking Data Status

```bash
# Check if your data is up to date
dvc status

# See what changed
dvc diff

# Compare with a specific commit
dvc diff HEAD~1
```

### Pulling Latest Data

When collaborators update datasets or results:

```bash
# Update code
git pull

# Update data
dvc pull
```

**Pro tip:** Create a git alias:
```bash
git config alias.dvc-pull '!git pull && dvc pull'

# Now you can use:
git dvc-pull
```

### Updating to a Specific Version

```bash
# Checkout a specific commit
git checkout <commit-hash>

# Update data to match that commit
dvc checkout

# Return to latest
git checkout main
dvc checkout
```

---

## Adding New Data

### Adding a New Dataset

1. **Place your data in the appropriate directory:**
   ```bash
   # Example: new corpus file
   cp new_data.jsonl lab/data/corpora/
   ```

2. **Update DVC tracking:**
   ```bash
   # If the directory is already tracked, just update it
   dvc add lab/data/corpora

   # If it's a new standalone file
   dvc add data/new_dataset.csv
   ```

3. **Commit the changes:**
   ```bash
   # Add the updated .dvc pointer file
   git add lab/data/corpora.dvc

   # Commit
   git commit -m "Add new corpus data: new_data.jsonl"
   ```

4. **Push data to remote:**
   ```bash
   dvc push
   ```

### Adding a New Results Directory

```bash
# Add new results
dvc add reports/new_experiment/

# Commit
git add reports/new_experiment.dvc .gitignore
git commit -m "Add results from new experiment"

# Push
dvc push
```

---

## Updating Tracked Data

### Updating Existing Data

When you regenerate results or modify datasets:

1. **Make your changes** (run analysis, update data, etc.)

2. **Update DVC tracking:**
   ```bash
   # Update directory tracking
   dvc add reports/

   # DVC will detect changes and update the .dvc file
   ```

3. **Commit the updated pointer:**
   ```bash
   git add reports.dvc
   git commit -m "Update results after fixing analysis bug"
   ```

4. **Push to remote:**
   ```bash
   dvc push
   ```

### Example: Regenerating All Results

```bash
# Regenerate results
make postprocess

# Update DVC tracking
dvc add reports/
dvc add paper/supplement/

# Commit
git add reports.dvc paper/supplement.dvc
git commit -m "Regenerate results with updated analysis scripts"

# Push
dvc push
```

---

## Troubleshooting

### Issue: `dvc pull` fails with "file not found"

**Cause:** Data not pushed to remote, or remote not configured.

**Solution:**
```bash
# Check remote configuration
dvc remote list

# Should show:
# localstore    .dvcstore

# If empty, reconfigure
dvc remote add localstore .dvcstore --local
dvc remote default localstore
```

### Issue: Large data in git repository

**Cause:** Accidentally committed data files instead of .dvc pointers.

**Solution:**
```bash
# Remove data from git, keep locally
git rm --cached reports/*.csv
git rm --cached reports/*.json

# Track with DVC
dvc add reports/

# Commit
git add reports.dvc .gitignore
git commit -m "Fix: Move reports to DVC tracking"
```

### Issue: "Data is not in cache"

**Cause:** `.dvcstore` was deleted or data never pushed.

**Solution:**
```bash
# If you have the data files locally
dvc add <directory>
dvc push

# If not, pull from remote
dvc pull

# If all else fails, regenerate
make postprocess
dvc add reports/
git add reports.dvc
git commit -m "Regenerate missing results"
```

### Issue: Slow `dvc pull`

**Cause:** Many files, slow I/O, or network issues.

**Solutions:**
```bash
# Pull only specific directory
dvc pull reports.dvc

# Use parallel jobs
dvc pull -j 4

# Check cache location (move to faster disk if needed)
dvc cache dir
```

### Issue: Merge conflict in .dvc file

**Example conflict in `reports.dvc`:**
```diff
<<<<<<< HEAD
- md5: abc123
=======
- md5: def456
>>>>>>> feature-branch
```

**Solution:**
```bash
# Keep one version (choose based on which data you want)
# Edit reports.dvc to resolve conflict

# Restore data matching the chosen MD5
dvc checkout reports.dvc

# Complete the merge
git add reports.dvc
git commit
```

For more details, see [DVC_TROUBLESHOOTING.md](DVC_TROUBLESHOOTING.md).

---

## Advanced Usage

### Working with Multiple Remotes

```bash
# Add S3 remote
dvc remote add s3store s3://tinylab-data/dvc-cache
dvc remote modify s3store region us-west-2

# Add GCS remote
dvc remote add gcsstore gs://tinylab-data/dvc-cache

# Push to specific remote
dvc push -r s3store

# Set default
dvc remote default s3store
```

### Configuring AWS Credentials

```bash
# Option 1: Environment variables (recommended)
export AWS_ACCESS_KEY_ID=<your-key>
export AWS_SECRET_ACCESS_KEY=<your-secret>

# Option 2: DVC config (not recommended for credentials)
dvc remote modify s3store access_key_id <your-key>
dvc remote modify s3store secret_access_key <your-secret>

# Option 3: Use AWS CLI credentials
# DVC will automatically use ~/.aws/credentials
```

### Data Registry (Sharing Across Projects)

```bash
# Import data from another DVC project
dvc import https://github.com/org/ml-data data/raw/dataset.csv

# This tracks the external source and can be updated
dvc update dataset.csv.dvc
```

### DVC Pipelines (Future)

For reproducible workflows, define stages in `dvc.yaml`:

```yaml
stages:
  preprocess:
    cmd: python scripts/facts_make_split.py
    deps:
      - lab/data/corpora/
      - scripts/facts_make_split.py
    outs:
      - lab/data/splits/

  analyze:
    cmd: python lab/analysis/export_head_rankings.py
    deps:
      - lab/data/splits/
      - lab/analysis/export_head_rankings.py
    outs:
      - reports/
```

Then run with:
```bash
dvc repro
```

### Garbage Collection

Clean up old unused data:

```bash
# Dry run - see what would be removed
dvc gc -w -vv

# Actually remove
dvc gc -w

# Keep only data for last 3 commits
dvc gc -w --all-commits --rev HEAD~3
```

### Metrics and Plots

Track metrics for experiment comparison:

```bash
# Track metrics file
dvc metrics show reports/metrics.json

# Compare metrics across branches
dvc metrics diff main feature-branch

# Show plots
dvc plots show reports/training_curve.csv
```

---

## Integration with Git Workflows

### Feature Branch Workflow

```bash
# Create feature branch
git checkout -b feature/new-experiment

# Run experiment, generate new results
python run_experiment.py

# Track results
dvc add reports/
git add reports.dvc
git commit -m "Add results for new experiment"

# Push code and data
git push origin feature/new-experiment
dvc push

# Create PR
gh pr create
```

### Reviewing PRs with Data Changes

As a reviewer:

```bash
# Checkout PR branch
gh pr checkout 123

# Pull data changes
dvc pull

# Review data
ls reports/
head reports/new_results.csv

# Compare with main
dvc diff main
```

---

## Best Practices

### DO:
- ✅ Always run `dvc pull` after `git pull`
- ✅ Use `dvc status` before committing to check for data changes
- ✅ Add descriptive commit messages when updating data
- ✅ Push data (`dvc push`) after pushing code (`git push`)
- ✅ Use `.dvcignore` for files that shouldn't be tracked

### DON'T:
- ❌ Commit large data files directly to git
- ❌ Delete `.dvcstore` without backing up
- ❌ Modify `.dvc` files manually (use `dvc add` instead)
- ❌ Push to git without pushing to DVC (others can't get data)
- ❌ Use `git add .` blindly (might add large files)

---

## Quick Reference

### Common Commands

| Command | Purpose |
|---------|---------|
| `dvc pull` | Download data from remote |
| `dvc push` | Upload data to remote |
| `dvc status` | Check if data is up to date |
| `dvc diff` | Show data changes |
| `dvc add <path>` | Track file/directory with DVC |
| `dvc checkout` | Restore data to match git HEAD |
| `dvc remote list` | Show configured remotes |
| `dvc cache dir` | Show cache location |

### Directory Structure

```
tinyLab/
├── .dvc/                   # DVC configuration
│   ├── config              # Remote settings (tracked in git)
│   ├── config.local        # Local settings (not tracked)
│   └── .gitignore          # What to ignore in .dvc/
│
├── .dvcstore/              # Local data cache (not tracked in git)
│
├── lab/data/
│   ├── corpora/            # [DVC tracked] Raw datasets
│   ├── corpora.dvc         # [Git tracked] Pointer to data
│   └── splits.dvc          # [Git tracked] Pointer to splits
│
└── reports/                # [DVC tracked] Results
    └── reports.dvc         # [Git tracked] Pointer to results
```

---

## Getting Help

- **DVC Documentation:** https://dvc.org/doc
- **DVC Discord:** https://dvc.org/chat
- **Project Issues:** See GitHub issues
- **Internal Docs:** See [DVC_TROUBLESHOOTING.md](DVC_TROUBLESHOOTING.md)

---

## Next Steps

- Read [DVC_MIGRATION_DESIGN.md](DVC_MIGRATION_DESIGN.md) for architecture details
- See [REPLICATION.md](docs/REPLICATION.md) for full reproduction instructions
- Join the discussion in project Slack/Discord

---

**Last Updated:** 2025-11-18
**DVC Version:** 3.x
**Maintained By:** tinyLab Team
