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

# 1. Install Aim
echo -e "${BLUE}[1/4]${NC} Installing Aim..."
pip install aim
echo -e "${GREEN}✓${NC} Aim installed\n"

# 2. Create tracking module
echo -e "${BLUE}[2/4]${NC} Creating tracking module..."
mkdir -p lab/tracking

# Create __init__.py
cat > lab/tracking/__init__.py << 'EOF'
"""Experiment tracking with Aim."""
from .tracker import TinyLabTracker

__all__ = ['TinyLabTracker']
EOF

echo -e "${GREEN}✓${NC} Created lab/tracking/\n"

# 3. Add .aim to .gitignore
echo -e "${BLUE}[3/4]${NC} Updating .gitignore..."
if ! grep -q "^/.aim/" .gitignore 2>/dev/null; then
    cat >> .gitignore << 'EOF'

# Aim experiment tracking (regenerate from data, don't commit)
/.aim/
EOF
    echo -e "${GREEN}✓${NC} Updated .gitignore\n"
else
    echo -e "${YELLOW}→${NC} .aim/ already in .gitignore\n"
fi

# 4. Initialize Aim repo
echo -e "${BLUE}[4/4]${NC} Initializing Aim repository..."
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
echo "  1. Copy tracker.py from AIM_INTEGRATION_PLAN.md to lab/tracking/"
echo "  2. Import historical results: python scripts/import_to_aim.py"
echo "  3. Launch UI: aim up"
echo "  4. Browse experiments at http://localhost:43800"
echo
echo "See AIM_INTEGRATION_PLAN.md for detailed integration guide."
echo
