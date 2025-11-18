# DVC Troubleshooting Guide

Comprehensive troubleshooting for common DVC issues in tinyLab.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Data Retrieval Problems](#data-retrieval-problems)
3. [Cache Issues](#cache-issues)
4. [Remote Storage Issues](#remote-storage-issues)
5. [Performance Issues](#performance-issues)
6. [Merge Conflicts](#merge-conflicts)
7. [Git Integration Issues](#git-integration-issues)
8. [Emergency Procedures](#emergency-procedures)

---

## Installation Issues

### Issue: DVC command not found

**Symptoms:**
```bash
$ dvc --version
bash: dvc: command not found
```

**Solutions:**

1. **Install DVC:**
   ```bash
   pip install dvc
   ```

2. **Check if installed in different environment:**
   ```bash
   which python
   python -m dvc --version
   ```

3. **Use full path:**
   ```bash
   python -m dvc pull
   ```

4. **Reinstall:**
   ```bash
   pip uninstall dvc
   pip install --no-cache-dir dvc
   ```

### Issue: Import errors after installation

**Symptoms:**
```
ImportError: cannot import name 'x' from 'dvc'
```

**Solutions:**

1. **Check Python version:**
   ```bash
   python --version  # Should be 3.11+
   ```

2. **Reinstall with dependencies:**
   ```bash
   pip install --upgrade 'dvc[all]'
   ```

3. **Clear Python cache:**
   ```bash
   find . -type d -name __pycache__ -exec rm -rf {} +
   pip install --force-reinstall dvc
   ```

---

## Data Retrieval Problems

### Issue: `dvc pull` fails with "file not found in cache"

**Symptoms:**
```
ERROR: failed to pull data from the cloud - ... file not found
```

**Diagnosis:**
```bash
# Check remote configuration
dvc remote list

# Check what DVC is looking for
dvc status -v
```

**Solutions:**

1. **Data never pushed - regenerate locally:**
   ```bash
   # Regenerate all results
   make postprocess

   # Re-add to DVC
   dvc add reports/
   dvc push
   ```

2. **Remote misconfigured:**
   ```bash
   # Fix local remote
   dvc remote add localstore .dvcstore --local
   dvc remote default localstore

   # Check if .dvcstore exists
   ls -la .dvcstore/
   ```

3. **Cache corrupted - fetch from another source:**
   ```bash
   # If you have another clone with data
   rsync -av other-clone/.dvcstore/ ./.dvcstore/

   # Or pull from backup remote
   dvc pull -r backup-remote
   ```

### Issue: `dvc pull` downloads nothing

**Symptoms:**
```bash
$ dvc pull
# No output, no data downloaded
```

**Diagnosis:**
```bash
dvc status
# Should show: "Data and pipelines are up to date"
# OR show missing files
```

**Solutions:**

1. **Data already in cache:**
   ```bash
   # Restore from cache
   dvc checkout
   ```

2. **No .dvc files in repository:**
   ```bash
   # Check for .dvc files
   find . -name "*.dvc"

   # If missing, you may need to:
   git checkout main  # or correct branch
   ```

3. **Wrong branch:**
   ```bash
   git branch  # Check current branch
   git checkout main
   dvc pull
   ```

### Issue: Partial data downloaded

**Symptoms:**
Some directories restored, others missing.

**Solutions:**

1. **Pull specific targets:**
   ```bash
   # Pull each .dvc file explicitly
   dvc pull lab/data/corpora.dvc
   dvc pull lab/data/splits.dvc
   dvc pull reports.dvc
   dvc pull paper/supplement.dvc
   ```

2. **Force pull:**
   ```bash
   dvc pull --force
   ```

3. **Check for errors:**
   ```bash
   dvc pull -v  # Verbose mode
   ```

---

## Cache Issues

### Issue: Cache directory missing

**Symptoms:**
```
ERROR: failed to pull data - cache directory not found
```

**Solutions:**

1. **Restore cache from backup:**
   ```bash
   # If you have a backup
   tar xzf tinylab_cache_backup.tar.gz -C .

   # Verify
   ls -la .dvcstore/
   ```

2. **Re-pull from remote:**
   ```bash
   dvc pull --force
   ```

3. **Regenerate data:**
   ```bash
   make postprocess
   dvc add reports/
   ```

### Issue: Cache taking too much space

**Symptoms:**
`.dvcstore` directory is very large.

**Diagnosis:**
```bash
# Check cache size
du -sh .dvcstore

# See what's in cache
dvc cache dir
ls -lh .dvcstore/files/md5/
```

**Solutions:**

1. **Clean up old versions:**
   ```bash
   # Keep only data for current commit
   dvc gc -w

   # Keep only last 5 commits
   dvc gc --all-commits --rev HEAD~5

   # Dry run first
   dvc gc -w -vv --dry
   ```

2. **Move cache to larger disk:**
   ```bash
   # Move cache
   mv .dvcstore /mnt/large-disk/tinylab-dvc-cache

   # Create symlink
   ln -s /mnt/large-disk/tinylab-dvc-cache .dvcstore
   ```

3. **Use external cache:**
   ```bash
   # Configure shared cache location
   dvc cache dir /shared/dvc-cache
   dvc config cache.shared group
   ```

### Issue: Cache corrupted

**Symptoms:**
```
ERROR: checksum mismatch
ERROR: corrupted cache file
```

**Solutions:**

1. **Remove corrupted file:**
   ```bash
   # DVC will tell you which file
   # Example: .dvcstore/files/md5/ab/cdef123456
   rm .dvcstore/files/md5/ab/cdef123456

   # Re-pull
   dvc pull --force
   ```

2. **Clear entire cache and re-pull:**
   ```bash
   # DANGER: Only if you have remote backup
   rm -rf .dvcstore/*
   dvc pull
   ```

3. **Verify integrity:**
   ```bash
   dvc status -v
   dvc checkout --force
   ```

---

## Remote Storage Issues

### Issue: Cannot push to remote

**Symptoms:**
```
ERROR: failed to push data to the cloud
```

**Diagnosis:**
```bash
# Check remote config
dvc remote list
dvc remote list --local

# Test remote access
dvc remote --help
```

**Solutions:**

1. **Local remote - check path:**
   ```bash
   # Verify .dvcstore exists
   mkdir -p .dvcstore

   # Reconfigure
   dvc remote add localstore .dvcstore --local --force
   dvc remote default localstore
   ```

2. **S3 remote - check credentials:**
   ```bash
   # Check AWS credentials
   aws s3 ls s3://your-bucket/

   # Set credentials
   export AWS_ACCESS_KEY_ID=xxx
   export AWS_SECRET_ACCESS_KEY=yyy

   # Or configure remote
   dvc remote modify s3store access_key_id xxx
   dvc remote modify s3store secret_access_key yyy
   ```

3. **Permission denied:**
   ```bash
   # For local remote
   chmod -R u+w .dvcstore

   # For S3/GCS - check IAM permissions
   ```

### Issue: Slow push/pull

**Symptoms:**
`dvc push` or `dvc pull` takes a very long time.

**Solutions:**

1. **Use parallel transfers:**
   ```bash
   dvc pull -j 8  # Use 8 parallel jobs
   dvc push -j 8
   ```

2. **Check network:**
   ```bash
   # For S3
   aws s3 ls s3://bucket/ --profile default

   # Check region
   dvc remote modify s3store region us-west-2
   ```

3. **Use faster remote:**
   ```bash
   # Add regional endpoint for S3
   dvc remote modify s3store endpoint_url https://s3.us-west-2.amazonaws.com
   ```

4. **Compress large files:**
   ```bash
   # For text files, enable compression
   gzip reports/*.csv
   dvc add reports/
   ```

---

## Performance Issues

### Issue: `dvc status` is slow

**Symptoms:**
`dvc status` takes minutes to complete.

**Solutions:**

1. **Check for large directories:**
   ```bash
   # See what DVC is tracking
   find . -name "*.dvc"

   # If tracking too many files, consider splitting
   ```

2. **Use directory tracking instead of individual files:**
   ```bash
   # Instead of:
   dvc add reports/*.csv  # Tracks each file separately

   # Use:
   dvc add reports/  # Tracks entire directory
   ```

3. **Disable status for CI:**
   ```bash
   # In CI/CD, skip status checks
   dvc pull --no-status
   ```

### Issue: Git operations are slow

**Symptoms:**
`git status`, `git add` are slow after DVC migration.

**Diagnosis:**
```bash
# Check git status
time git status

# Check if large files in git
git ls-files | xargs du -sh | sort -h | tail -20
```

**Solutions:**

1. **Remove large files from git:**
   ```bash
   # Find large files
   git ls-files | xargs ls -lh | sort -k5 -h | tail -20

   # Remove if shouldn't be in git
   git rm --cached path/to/large/file
   dvc add path/to/large/file
   git add path/to/large/file.dvc
   ```

2. **Use git-lfs for PDFs/images:**
   ```bash
   # If you have large binary files that aren't tracked by DVC
   git lfs track "*.pdf"
   git lfs track "*.png"
   ```

3. **Check .gitignore:**
   ```bash
   # Ensure DVC data is ignored
   cat .gitignore | grep dvc
   ```

---

## Merge Conflicts

### Issue: Conflict in .dvc file

**Symptoms:**
```diff
<<<<<<< HEAD
  md5: abc123456
=======
  md5: def789012
>>>>>>> feature-branch
```

**Understanding:**
- Both branches modified the same data
- Git doesn't know which version to keep

**Solutions:**

**Option 1: Keep HEAD version**
```bash
# Edit .dvc file, remove conflict markers, keep HEAD md5
vim reports.dvc  # or your editor

# Restore data matching HEAD
dvc checkout reports.dvc

# Complete merge
git add reports.dvc
git commit
```

**Option 2: Keep incoming version**
```bash
# Edit .dvc file, keep incoming md5
vim reports.dvc

# Restore data matching incoming
dvc checkout reports.dvc

# Complete merge
git add reports.dvc
git commit
```

**Option 3: Regenerate data (recommended)**
```bash
# Merge code first
git checkout HEAD reports.dvc  # or branch version
git add reports.dvc
git commit

# Regenerate fresh data
make postprocess

# Re-track
dvc add reports/
git add reports.dvc
git commit --amend -m "Merge with regenerated data"
```

### Issue: Both added .dvc file

**Symptoms:**
```
CONFLICT (both added): reports.dvc
```

**Solution:**
```bash
# Choose one version
git checkout --theirs reports.dvc
# OR
git checkout --ours reports.dvc

# Verify data
dvc checkout reports.dvc

# If data doesn't match, regenerate
make postprocess
dvc add reports/
git add reports.dvc
git commit
```

---

## Git Integration Issues

### Issue: .dvc files not being tracked

**Symptoms:**
`git status` doesn't show `.dvc` files after `dvc add`.

**Solutions:**

1. **Explicitly add:**
   ```bash
   git add *.dvc
   git add **/*.dvc
   ```

2. **Check .gitignore:**
   ```bash
   # Make sure .dvc files aren't ignored
   grep "\.dvc$" .gitignore

   # Should NOT have:
   # *.dvc  # This would ignore pointer files
   ```

3. **Check DVC config:**
   ```bash
   cat .dvc/.gitignore
   # Should NOT ignore *.dvc files
   ```

### Issue: Data files committed to git

**Symptoms:**
Large files tracked by both git and DVC.

**Diagnosis:**
```bash
# Check file size in git
git ls-files | xargs du -sh | sort -h | tail -20

# Check if tracked by both
git ls-files reports/*.csv
ls -la reports/*.csv.dvc
```

**Solution:**
```bash
# Remove from git, keep locally
git rm --cached reports/*.csv
git rm --cached reports/*.json

# Ensure tracked by DVC
dvc status

# Commit
git add .gitignore
git commit -m "Remove data files from git (tracked by DVC)"

# Push
git push
```

### Issue: Large git repository after DVC migration

**Diagnosis:**
```bash
# Check repo size
du -sh .git

# Check large objects
git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  awk '/^blob/ {print substr($0,6)}' | \
  sort -k2 -n | tail -20
```

**Solution:**
Use git-filter-repo to remove historical large files:
```bash
# Install git-filter-repo
pip install git-filter-repo

# Remove historical data files
git filter-repo --path reports/ --invert-paths
git filter-repo --path lab/data/corpora --invert-paths

# Force push (DANGER: coordinate with team)
git push --force
```

---

## Emergency Procedures

### Emergency: Lost all data

**Situation:** `.dvcstore` deleted, no backup.

**Recovery:**

1. **Check remote:**
   ```bash
   dvc pull -v
   ```

2. **Check other clones:**
   ```bash
   # Copy from teammate's machine
   rsync -av teammate-machine:/path/to/tinyLab/.dvcstore/ ./.dvcstore/
   ```

3. **Regenerate from scratch:**
   ```bash
   # Regenerate all results
   make postprocess

   # Re-track
   dvc add reports/
   dvc push

   # Commit
   git add reports.dvc
   git commit -m "Regenerate lost results"
   ```

### Emergency: Corrupted DVC configuration

**Situation:** DVC commands fail with config errors.

**Recovery:**

1. **Backup current config:**
   ```bash
   cp -r .dvc .dvc.backup
   ```

2. **Reinitialize:**
   ```bash
   rm -rf .dvc
   dvc init

   # Reconfigure remote
   dvc remote add localstore .dvcstore --local
   dvc remote default localstore
   ```

3. **Verify:**
   ```bash
   dvc status
   dvc pull
   ```

### Emergency: Cannot push, about to lose data

**Situation:** Need to save work but `dvc push` fails.

**Workaround:**

1. **Backup cache manually:**
   ```bash
   tar czf dvc-cache-backup-$(date +%Y%m%d).tar.gz .dvcstore/
   ```

2. **Push to alternative location:**
   ```bash
   # Temporary remote
   dvc remote add backup /mnt/external-drive/tinylab-dvc --local
   dvc push -r backup
   ```

3. **Upload to cloud manually:**
   ```bash
   # AWS S3
   aws s3 sync .dvcstore/ s3://my-backup-bucket/tinylab-dvc-emergency/

   # Later recover with:
   aws s3 sync s3://my-backup-bucket/tinylab-dvc-emergency/ .dvcstore/
   ```

---

## Diagnostic Commands

Run these to gather information before asking for help:

```bash
# DVC version and config
dvc version
dvc config --list
dvc config --list --local

# Remote configuration
dvc remote list
dvc remote list --local

# Status and cache
dvc status -v
dvc cache dir
du -sh .dvcstore

# Git integration
git status
find . -name "*.dvc"

# System info
df -h .
python --version
pip list | grep dvc
```

---

## Getting More Help

1. **Check DVC documentation:** https://dvc.org/doc/user-guide/troubleshooting
2. **Search DVC forum:** https://discuss.dvc.org
3. **File an issue:** Project GitHub issues
4. **Ask in Slack/Discord:** See team channels

When asking for help, include:
- Output of diagnostic commands above
- Full error message
- What you were trying to do
- What you expected to happen

---

## Prevention Tips

Avoid issues by following these practices:

1. **Always run `dvc status` before committing**
2. **Test `dvc pull` on fresh clone regularly**
3. **Keep backups of `.dvcstore` for critical projects**
4. **Document custom remote configurations**
5. **Use `--dry` flag for destructive operations**
6. **Monitor `.git` and `.dvcstore` sizes**
7. **Coordinate with team before force-pushing**

---

**Last Updated:** 2025-11-18
**Maintained By:** tinyLab Team
