# DVC Implementation Guide for tinyLab

**Status:** Ready for Implementation
**Date:** 2025-11-18
**Branch:** `claude/migrate-dvc-tracking-013aAqNvWvh6CwHntNnxvhjo`

## Overview

This guide provides the complete implementation plan for migrating tinyLab to DVC (Data Version Control). All design work, documentation, and automation scripts are complete and ready for execution.

## What Has Been Prepared

### 1. Data Inventory ✅
- Comprehensive catalog of all data files (355+ files, ~7.4 MB)
- Classification of what needs DVC tracking vs what stays in Git
- Size analysis and growth projections

### 2. Architecture Design ✅
- Two proposed structures (Option A: Minimal, Option B: Full)
- **Recommendation:** Option A (Minimal Restructure) for low risk and fast migration
- Future-proof design ready for S3/GCS/Azure backends

### 3. Documentation ✅
- **DVC_MIGRATION_DESIGN.md** - Complete architecture and design decisions
- **DVC_SETUP.md** - User guide for setup and daily workflows
- **DVC_TROUBLESHOOTING.md** - Comprehensive troubleshooting reference
- **README.md** - Updated with DVC quick start

### 4. Automation Scripts ✅
- **scripts/migrate_to_dvc.sh** - Fully automated migration script
- Dry-run support for safe testing
- Backup creation capability
- Comprehensive error checking

## Implementation Steps

### Prerequisites

Before starting, ensure:
- [ ] You have a clean working directory (`git status` shows clean)
- [ ] You're on the correct branch (`claude/migrate-dvc-tracking-013aAqNvWvh6CwHntNnxvhjo`)
- [ ] You've reviewed the design document (DVC_MIGRATION_DESIGN.md)
- [ ] You have a backup (optional but recommended)

### Step 1: Install DVC

```bash
# Using pip
pip install dvc

# Verify installation
dvc version
# Should output: 3.x.x or higher
```

**Troubleshooting:** If installation fails, see [DVC_TROUBLESHOOTING.md](DVC_TROUBLESHOOTING.md#installation-issues)

### Step 2: Run Migration Script (Dry Run)

Test the migration without making changes:

```bash
# Dry run - see what would happen
./scripts/migrate_to_dvc.sh --dry-run

# Dry run with backup preparation
./scripts/migrate_to_dvc.sh --dry-run --backup
```

Review the output carefully. The script will show:
- Which directories will be tracked
- What .dvc files will be created
- What changes will be made to .gitignore
- Git staging operations

### Step 3: Create Backup (Recommended)

```bash
# Create backup of all data
./scripts/migrate_to_dvc.sh --backup --dry-run

# Or manually:
mkdir -p backups
tar czf backups/tinylab_pre_dvc_$(date +%Y%m%d_%H%M%S).tar.gz \
  lab/data/corpora \
  lab/data/splits \
  data/lexicons \
  reports \
  paper/supplement
```

### Step 4: Execute Migration

Run the actual migration:

```bash
# Execute migration
./scripts/migrate_to_dvc.sh

# Or with backup
./scripts/migrate_to_dvc.sh --backup
```

**What happens:**
1. DVC is initialized (`.dvc/` directory created)
2. Local remote configured (`.dvcstore/`)
3. `.gitignore` updated with DVC patterns
4. Data directories tracked with DVC
5. `.dvc` pointer files created
6. Changes staged in Git

### Step 5: Verify Migration

Check that everything worked:

```bash
# Check DVC status
dvc status
# Should output: "Data and pipelines are up to date."

# List .dvc files created
find . -name "*.dvc"
# Should show:
# lab/data/corpora.dvc
# lab/data/splits.dvc
# data/lexicons/hedge_booster.json.dvc
# reports.dvc
# paper/supplement.dvc

# Check .dvcstore size
du -sh .dvcstore
# Should be ~7-8 MB

# Verify git status
git status
# Should show staged .dvc files and .gitignore
```

### Step 6: Test Data Retrieval

Simulate a fresh clone:

```bash
# In a temporary directory (don't do this in main repo!)
cd /tmp
git clone /home/user/tinyLab tinylab-test
cd tinylab-test

# Install DVC
pip install dvc

# Pull data
dvc pull

# Verify files
ls -lh lab/data/corpora/
ls -lh reports/

# Run smoke test
python smoke_test.py

# Clean up
cd ..
rm -rf tinylab-test
```

### Step 7: Commit Changes

If everything looks good:

```bash
cd /home/user/tinyLab

# Review what will be committed
git status
git diff --cached .gitignore
cat lab/data/corpora.dvc
cat reports.dvc

# Commit DVC migration
git commit -m "Add DVC tracking for datasets, results, and artifacts

- Initialize DVC with local remote (.dvcstore)
- Track lab/data/corpora (18 JSONL files, ~370K)
- Track lab/data/splits (18 JSON files, ~29K)
- Track data/lexicons/hedge_booster.json
- Track reports/ (298 CSV/JSON files, ~7.4MB)
- Track paper/supplement/ (20+ files, ~150K)
- Update .gitignore with DVC patterns

All data moved to .dvcstore, .dvc pointers tracked in git.
Total tracked: ~7.4 MB across 355+ files.

See DVC_MIGRATION_DESIGN.md for architecture details.
See DVC_SETUP.md for usage instructions."
```

### Step 8: Push to Remote

Push both code and data:

```bash
# Push code changes to GitHub
git push -u origin claude/migrate-dvc-tracking-013aAqNvWvh6CwHntNnxvhjo

# Data is already in .dvcstore (local remote)
# When ready for S3/GCS, add remote and push:
# dvc remote add s3store s3://tinylab-data/dvc-cache
# dvc push -r s3store
```

### Step 9: Test Cross-Machine Reproducibility

On a different machine (or fresh clone):

```bash
# Clone repository
git clone <repo-url> tinylab-fresh
cd tinylab-fresh

# Checkout DVC branch
git checkout claude/migrate-dvc-tracking-013aAqNvWvh6CwHntNnxvhjo

# Install DVC
pip install dvc

# Pull data
dvc pull

# Verify
ls lab/data/corpora/
ls reports/

# Run tests
python smoke_test.py
make postprocess
cd paper && make
```

## Manual Migration (If Script Fails)

If the automated script fails, follow these manual steps:

### 1. Initialize DVC
```bash
dvc init
```

### 2. Configure Local Remote
```bash
dvc remote add localstore .dvcstore --local
dvc remote default localstore
```

### 3. Update .gitignore

Add to `.gitignore`:
```gitignore
# DVC
/.dvcstore/
/reports/*.csv
/reports/*.json
/reports/layer_sweep_*
/reports/appendices
/reports/pythia_layer*_vdi_drift*
/lab/data/corpora
/lab/data/splits
/data/lexicons/*.json
/paper/supplement/*.json
/paper/supplement/*.csv
/paper/supplement/cuda_validation
```

### 4. Track Data with DVC
```bash
dvc add lab/data/corpora
dvc add lab/data/splits
dvc add data/lexicons/hedge_booster.json
dvc add reports
dvc add paper/supplement
```

### 5. Stage Git Changes
```bash
git add .dvc/.gitignore .dvc/config
git add .gitignore
git add lab/data/corpora.dvc
git add lab/data/splits.dvc
git add data/lexicons/hedge_booster.json.dvc
git add reports.dvc
git add paper/supplement.dvc
```

### 6. Commit
```bash
git commit -m "Add DVC tracking for datasets and results"
```

## Post-Migration Tasks

### Update Documentation

1. **Update REPLICATION.md**
   - Add DVC installation step
   - Add `dvc pull` before running experiments

2. **Update QUICKSTART.md**
   - Mention DVC setup after environment setup

3. **Update CI/CD** (if applicable)
   - Add DVC installation to CI workflows
   - Add `dvc pull` before running tests

### Team Onboarding

Share with team:
1. Link to [DVC_SETUP.md](DVC_SETUP.md)
2. Quick start: `pip install dvc && dvc pull`
3. When to use DVC: "Always `dvc pull` after `git pull`"

### Monitor and Maintain

1. **Check .dvcstore size regularly:**
   ```bash
   du -sh .dvcstore
   ```

2. **Garbage collect old versions:**
   ```bash
   dvc gc -w  # Remove unused cached data
   ```

3. **Monitor Git repo size:**
   ```bash
   du -sh .git
   # Should stay small (only .dvc pointer files)
   ```

## Migration to Cloud Storage (Future)

When ready to migrate to S3/GCS/Azure:

### Option 1: AWS S3

```bash
# Add S3 remote
dvc remote add s3store s3://tinylab-data/dvc-cache
dvc remote modify s3store region us-west-2

# Configure credentials (use environment variables)
export AWS_ACCESS_KEY_ID=xxx
export AWS_SECRET_ACCESS_KEY=yyy

# Push data to S3
dvc push -r s3store

# Set as default remote
dvc remote default s3store

# Update .dvc/config in git
git add .dvc/config
git commit -m "Set S3 as default DVC remote"
```

### Option 2: Google Cloud Storage

```bash
# Add GCS remote
dvc remote add gcsstore gs://tinylab-data/dvc-cache

# Authenticate
gcloud auth application-default login

# Push data
dvc push -r gcsstore

# Set as default
dvc remote default gcsstore
```

### Option 3: Azure Blob Storage

```bash
# Add Azure remote
dvc remote add azurestore azure://tinylab-data/dvc-cache
dvc remote modify azurestore account_name <account>

# Set credentials
export AZURE_STORAGE_ACCOUNT=<account>
export AZURE_STORAGE_KEY=<key>

# Push data
dvc push -r azurestore
```

## Rollback Procedure

If you need to undo the migration:

### Option 1: Git Reset (Before Push)

```bash
# Reset to before DVC commit
git reset HEAD~1

# Remove DVC initialization
rm -rf .dvc .dvcstore

# Restore .gitignore
git checkout HEAD .gitignore

# Data files should still be present
ls lab/data/corpora/
```

### Option 2: Restore from Backup

```bash
# Extract backup
tar xzf backups/tinylab_pre_dvc_YYYYMMDD_HHMMSS.tar.gz

# Remove DVC
rm -rf .dvc .dvcstore
rm **/*.dvc

# Reset .gitignore
git checkout origin/main .gitignore
```

### Option 3: Revert Commit (After Push)

```bash
# Revert the DVC migration commit
git revert <commit-hash>

# Remove DVC files
rm -rf .dvc .dvcstore
```

## Success Criteria

Migration is successful when:

- ✅ `dvc status` shows "Data and pipelines are up to date"
- ✅ All `.dvc` files created and tracked in Git
- ✅ `.dvcstore/` directory created and gitignored
- ✅ Data files gitignored (CSV, JSON in reports/, etc.)
- ✅ `dvc pull` works in fresh clone
- ✅ `python smoke_test.py` passes
- ✅ `make postprocess` completes successfully
- ✅ `cd paper && make` generates PDF
- ✅ Git repository size reasonable (<50MB)
- ✅ `.dvcstore` size matches expected (~7-8MB)

## Troubleshooting

For issues during migration, see [DVC_TROUBLESHOOTING.md](DVC_TROUBLESHOOTING.md).

Common issues:
- **DVC installation fails** → See [Installation Issues](DVC_TROUBLESHOOTING.md#installation-issues)
- **`dvc pull` fails** → See [Data Retrieval Problems](DVC_TROUBLESHOOTING.md#data-retrieval-problems)
- **Git repo too large** → See [Git Integration Issues](DVC_TROUBLESHOOTING.md#git-integration-issues)

## Support

- **Documentation:** See all `DVC_*.md` files in repository root
- **DVC Docs:** https://dvc.org/doc
- **Issues:** File issues on GitHub with `[DVC]` prefix
- **Questions:** Check [DVC_TROUBLESHOOTING.md](DVC_TROUBLESHOOTING.md) first

## Files Created

This migration preparation includes:

| File | Purpose |
|------|---------|
| `DVC_MIGRATION_DESIGN.md` | Architecture and design decisions |
| `DVC_SETUP.md` | User guide for setup and workflows |
| `DVC_TROUBLESHOOTING.md` | Troubleshooting reference |
| `DVC_IMPLEMENTATION_GUIDE.md` | This file - step-by-step implementation |
| `scripts/migrate_to_dvc.sh` | Automated migration script |
| `README.md` | Updated with DVC quick start |

## Timeline Estimate

- **Preparation (Review):** 30 minutes
- **Migration Execution:** 10 minutes
- **Verification:** 15 minutes
- **Testing:** 20 minutes
- **Documentation Updates:** 15 minutes
- **Total:** ~1.5 hours

## Next Steps

1. **Review** this guide and [DVC_MIGRATION_DESIGN.md](DVC_MIGRATION_DESIGN.md)
2. **Install** DVC: `pip install dvc`
3. **Test** migration: `./scripts/migrate_to_dvc.sh --dry-run`
4. **Execute** migration: `./scripts/migrate_to_dvc.sh --backup`
5. **Verify** and commit changes
6. **Push** to remote: `git push`
7. **Test** on fresh clone
8. **Celebrate** 🎉 - Your data is now version controlled!

---

**Questions or Issues?**

1. Check [DVC_TROUBLESHOOTING.md](DVC_TROUBLESHOOTING.md)
2. Review [DVC_SETUP.md](DVC_SETUP.md)
3. See DVC documentation: https://dvc.org/doc
4. File a GitHub issue with `[DVC]` prefix

**Ready to proceed?** Follow the steps above to implement DVC tracking.

---

**Document Version:** 1.0
**Last Updated:** 2025-11-18
**Author:** Claude
**Status:** Ready for Implementation
