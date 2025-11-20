#!/usr/bin/env bash
set -euo pipefail

# One-stop backup helper: DVC add -> DVC push -> git add/commit (optional push).
# Usage examples:
#   bash scripts/post_run_backup.sh reports/pilot_stage1a "pilot: baseline run"
#   DVC_PUSH=0 GIT_PUSH=1 bash scripts/post_run_backup.sh reports/new_run "commit only"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TARGETS=("$@")
if [[ "${#TARGETS[@]}" -eq 0 ]]; then
  TARGETS=("reports")
fi

COMMIT_MSG="${COMMIT_MSG:-Add results $(date +'%Y-%m-%d %H:%M')}"
DVC_PUSH="${DVC_PUSH:-1}"
GIT_PUSH="${GIT_PUSH:-0}"

if ! command -v dvc >/dev/null 2>&1; then
  echo "dvc not found; activate your venv first."
  exit 1
fi

changed=0
for path in "${TARGETS[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Skipping missing path: ${path}"
    continue
  fi
  echo "DVC tracking: ${path}"
  dvc add "${path}"
  dvc_file="${path%/}.dvc"
  if [[ -f "${dvc_file}" ]]; then
    git add "${dvc_file}"
  fi
  if [[ -f ".gitignore" ]]; then
    git add .gitignore
  fi
  changed=1
done

if [[ "${changed}" -eq 0 ]]; then
  echo "Nothing to back up."
  exit 0
fi

echo "Git status:"
git status --short

if [[ -n "$(git status --porcelain)" ]]; then
  git commit -m "${COMMIT_MSG}" || true
else
  echo "No git changes to commit."
fi

if [[ "${DVC_PUSH}" == "1" ]]; then
  echo "Pushing DVC data to remote..."
  dvc push
else
  echo "Skipping dvc push (DVC_PUSH=0)."
fi

if [[ "${GIT_PUSH}" == "1" ]]; then
  echo "Pushing git commit..."
  git push
else
  echo "Skipping git push (GIT_PUSH=0)."
fi

echo "Backup helper complete."
