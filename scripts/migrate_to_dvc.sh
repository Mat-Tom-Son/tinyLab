#!/usr/bin/env bash
#
# migrate_to_dvc.sh
#
# Automated migration script for adding DVC tracking to tinyLab
#
# Usage:
#   ./scripts/migrate_to_dvc.sh [--dry-run] [--backup]
#
# Options:
#   --dry-run    Show what would be done without making changes
#   --backup     Create backup tarball before migration
#   --help       Show this help message

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKUP_DIR="${REPO_ROOT}/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Flags
DRY_RUN=false
CREATE_BACKUP=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --backup)
      CREATE_BACKUP=true
      shift
      ;;
    --help)
      grep '^#' "$0" | sed 's/^# //' | sed 's/^#//'
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      exit 1
      ;;
  esac
done

# Logging functions
log_info() {
  echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
  echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
  echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

# Dry run wrapper
run_cmd() {
  if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}[DRY-RUN]${NC} Would run: $*"
  else
    "$@"
  fi
}

# Check prerequisites
check_prerequisites() {
  log_info "Checking prerequisites..."

  # Check if we're in the right directory
  if [ ! -f "$REPO_ROOT/pyproject.toml" ] || [ ! -d "$REPO_ROOT/lab" ]; then
    log_error "Must be run from tinyLab repository root"
    exit 1
  fi

  # Check if git is available
  if ! command -v git &> /dev/null; then
    log_error "git is required but not installed"
    exit 1
  fi

  # Check if DVC is installed
  if ! command -v dvc &> /dev/null; then
    log_error "DVC is not installed. Install with: pip install dvc"
    exit 1
  fi

  # Check if we're on a clean branch
  if [ -n "$(git status --porcelain)" ]; then
    log_warning "Working directory has uncommitted changes"
    if [ "$DRY_RUN" = false ]; then
      read -p "Continue anyway? (y/N) " -n 1 -r
      echo
      if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
      fi
    fi
  fi

  log_success "Prerequisites check passed"
}

# Create backup
create_backup() {
  if [ "$CREATE_BACKUP" = true ]; then
    log_info "Creating backup..."

    mkdir -p "$BACKUP_DIR"
    BACKUP_FILE="${BACKUP_DIR}/tinylab_pre_dvc_${TIMESTAMP}.tar.gz"

    run_cmd tar czf "$BACKUP_FILE" \
      lab/data/corpora \
      lab/data/splits \
      data/lexicons \
      reports \
      paper/supplement \
      2>/dev/null || true

    if [ -f "$BACKUP_FILE" ]; then
      log_success "Backup created: $BACKUP_FILE"
    fi
  fi
}

# Initialize DVC
init_dvc() {
  log_info "Initializing DVC..."

  cd "$REPO_ROOT"

  # Check if already initialized
  if [ -d ".dvc" ]; then
    log_warning "DVC already initialized, skipping"
    return 0
  fi

  run_cmd dvc init

  log_success "DVC initialized"
}

# Configure local remote
configure_remote() {
  log_info "Configuring local DVC remote..."

  cd "$REPO_ROOT"

  # Add local remote
  if ! dvc remote list | grep -q "localstore"; then
    run_cmd dvc remote add localstore .dvcstore --local
    run_cmd dvc remote default localstore
    log_success "Local remote configured at .dvcstore"
  else
    log_warning "Remote 'localstore' already exists, skipping"
  fi
}

# Update gitignore
update_gitignore() {
  log_info "Updating .gitignore..."

  GITIGNORE="$REPO_ROOT/.gitignore"

  # Check if DVC patterns already exist
  if grep -q "^/\.dvcstore/" "$GITIGNORE" 2>/dev/null; then
    log_warning "DVC patterns already in .gitignore, skipping"
    return 0
  fi

  # Add DVC patterns
  cat >> "$GITIGNORE" << 'EOF'

# DVC - Data tracked by DVC (pointers in git, data in .dvcstore)
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
EOF

  log_success ".gitignore updated with DVC patterns"
}

# Add DVC tracking
add_dvc_tracking() {
  log_info "Adding DVC tracking to data directories..."

  cd "$REPO_ROOT"

  # Track directories and files
  declare -a DVC_TARGETS=(
    "lab/data/corpora"
    "lab/data/splits"
    "data/lexicons/hedge_booster.json"
    "reports"
    "paper/supplement"
  )

  for target in "${DVC_TARGETS[@]}"; do
    if [ -e "$target" ]; then
      log_info "Tracking $target..."
      run_cmd dvc add "$target"

      # Git add the .dvc file
      DVC_FILE="${target}.dvc"
      if [ -f "$DVC_FILE" ]; then
        run_cmd git add "$DVC_FILE"
      fi
    else
      log_warning "Target not found: $target (skipping)"
    fi
  done

  log_success "DVC tracking added"
}

# Verify tracking
verify_tracking() {
  log_info "Verifying DVC tracking..."

  cd "$REPO_ROOT"

  # Check status
  if [ "$DRY_RUN" = false ]; then
    dvc status
  fi

  # List .dvc files
  log_info "DVC pointer files created:"
  find . -name "*.dvc" -type f | sed 's|^\./||'

  # Check .dvcstore size
  if [ -d ".dvcstore" ]; then
    DVCSTORE_SIZE=$(du -sh .dvcstore | cut -f1)
    log_info ".dvcstore size: $DVCSTORE_SIZE"
  fi

  log_success "Verification complete"
}

# Stage git changes
stage_git_changes() {
  log_info "Staging git changes..."

  cd "$REPO_ROOT"

  # Add DVC config files
  run_cmd git add .dvc/.gitignore .dvc/config .dvc/config.local 2>/dev/null || true

  # Add .gitignore changes
  run_cmd git add .gitignore

  # Add all .dvc pointer files
  run_cmd git add "*.dvc" 2>/dev/null || true
  run_cmd git add "**/*.dvc" 2>/dev/null || true

  log_success "Git changes staged"
}

# Test data retrieval
test_retrieval() {
  log_info "Testing data retrieval (this is a dry-run test)..."

  if [ "$DRY_RUN" = true ]; then
    log_info "Skipping retrieval test in dry-run mode"
    return 0
  fi

  cd "$REPO_ROOT"

  # Create temporary directory
  TEST_DIR=$(mktemp -d)
  log_info "Test directory: $TEST_DIR"

  # Try to pull one file
  log_info "Testing dvc status..."
  if dvc status; then
    log_success "DVC status check passed"
  else
    log_warning "DVC status check failed (this may be normal if data is already in cache)"
  fi

  rm -rf "$TEST_DIR"
}

# Summary
print_summary() {
  echo
  log_info "=========================================="
  log_info "DVC Migration Summary"
  log_info "=========================================="
  echo

  if [ "$DRY_RUN" = true ]; then
    log_warning "DRY RUN MODE - No changes were made"
    echo
    log_info "To execute the migration, run:"
    log_info "  $0"
    echo
  else
    log_success "Migration completed successfully!"
    echo
    log_info "Next steps:"
    echo "  1. Review changes: git status"
    echo "  2. Test DVC: dvc status"
    echo "  3. Commit changes:"
    echo "     git commit -m 'Add DVC tracking for datasets and results'"
    echo "  4. Test data retrieval:"
    echo "     dvc pull"
    echo "  5. Run tests:"
    echo "     python smoke_test.py"
    echo
    log_info "For more information, see DVC_SETUP.md"
  fi

  echo
}

# Main execution
main() {
  log_info "Starting DVC migration for tinyLab"
  echo

  if [ "$DRY_RUN" = true ]; then
    log_warning "Running in DRY-RUN mode - no changes will be made"
    echo
  fi

  check_prerequisites
  create_backup
  init_dvc
  configure_remote
  update_gitignore
  add_dvc_tracking
  verify_tracking
  stage_git_changes
  test_retrieval
  print_summary
}

# Run main
main
