#!/usr/bin/env bash
#
# setup_aim.sh - Quick setup for Aim experiment tracking
#
# Usage:
#   ./scripts/setup_aim.sh

set -euo pipefail

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  tinyLab Aim Tracking Setup           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo

# 1. Install Aim (and dependencies)
echo -e "${BLUE}[1/4]${NC} Installing Aim..."
pip install --upgrade pip wheel setuptools
pip install Cython==3.0.10
pip install --pre aimrocks==0.5.3.dev8
pip install --no-build-isolation aim==3.21.0
echo -e "${GREEN}✓${NC} Aim installed\n"

# 2. Ensure .aim is gitignored
echo -e "${BLUE}[2/4]${NC} Updating .gitignore..."
if ! grep -q "^/.aim/" .gitignore 2>/dev/null; then
    cat >> .gitignore << 'EOF'

# Aim experiment tracking (regenerate from data, don't commit)
/.aim/
EOF
    echo -e "${GREEN}✓${NC} Updated .gitignore\n"
else
    echo -e "${YELLOW}→${NC} .aim/ already in .gitignore\n"
fi

# 3. Initialize Aim repo
echo -e "${BLUE}[3/4]${NC} Initializing Aim repository..."
python << 'EOF'
from aim import Repo
repo = Repo.from_path('.', init=True)
print(f"Initialized Aim repo at: {repo.path}")
EOF
echo -e "${GREEN}✓${NC} Aim repository initialized\n"

# Done!
echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Setup Complete! 🎉                    ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo
echo "Next steps:"
echo "  1. Import historical results: python scripts/import_to_aim.py"
echo "  2. Launch UI: aim up"
echo "  3. Browse experiments at http://localhost:43800"
echo
echo "See AIM_INTEGRATION_PLAN.md for detailed integration guide."
echo
