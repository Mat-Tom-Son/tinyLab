#!/usr/bin/env bash
set -euo pipefail

echo "🔍 Pre-Submission Checklist"
echo "======"

echo "1) Lint & format (pre-commit)"
pre-commit run --all-files || true

echo "\n2) Smoke test"
python3 smoke_test.py

echo "\n3) Rebuild manifest"
python3 -m lab.analysis.build_manifest

echo "\n4) Verify manifest"
python3 scripts/verify_manifest.py --check-only || true

echo "\n5) Regenerate standardized reports"
make postprocess

echo "\n6) Verify standardized reports again"
python3 scripts/verify_manifest.py --check-only || true

echo "\n✅ Pre-submission checks complete."
