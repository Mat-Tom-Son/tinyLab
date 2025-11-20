#!/usr/bin/env bash
set -euo pipefail

# Configure DVC to use a GCS bucket as the default remote.
# Usage:
#   GCS_BUCKET=my-bucket \
#   GCS_PREFIX=tinylab \
#   GCS_CREDENTIALS_PATH=~/.config/gcloud/application_default_credentials.json \
#   REMOTE_NAME=gcsremote \
#     bash scripts/configure_dvc_gcs.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

REMOTE_NAME="${REMOTE_NAME:-gcsremote}"
GCS_BUCKET="${GCS_BUCKET:-}"
GCS_PREFIX="${GCS_PREFIX:-tinylab}"
GCS_CREDENTIALS_PATH="${GCS_CREDENTIALS_PATH:-$HOME/.config/gcloud/application_default_credentials.json}"

if [[ -z "${GCS_BUCKET}" ]]; then
  echo "Set GCS_BUCKET to your bucket name (without gs://)."
  exit 1
fi

if ! command -v dvc >/dev/null 2>&1; then
  echo "dvc not found; activate your venv and install dvc first."
  exit 1
fi

REMOTE_URL="gs://${GCS_BUCKET%/}/${GCS_PREFIX}"
echo "Configuring DVC remote '${REMOTE_NAME}' -> ${REMOTE_URL}"

dvc remote add -f "${REMOTE_NAME}" "${REMOTE_URL}"

if [[ -n "${GCS_CREDENTIALS_PATH}" ]]; then
  dvc remote modify "${REMOTE_NAME}" credentialpath "${GCS_CREDENTIALS_PATH}"
fi

dvc remote default "${REMOTE_NAME}"

echo "DVC remotes:"
dvc remote list

echo "Done. To test: dvc pull (or dvc push) after logging in with 'gcloud auth application-default login'."
