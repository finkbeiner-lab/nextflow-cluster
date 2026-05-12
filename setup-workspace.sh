#!/bin/bash
# =========================================================================
# Setup a new user workspace for the Finkbeiner pipeline.
#
# Usage:
#   bash /gladstone/finkbeiner/steve/work/projects/nextflow-cluster/setup-workspace.sh [directory]
#
# If no directory is given, the current directory is used.
# =========================================================================

INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${1:-.}"

# Resolve to absolute path
TARGET_DIR="$(cd "${TARGET_DIR}" 2>/dev/null && pwd)" || {
    echo "Creating directory: ${1}"
    mkdir -p "${1}" && TARGET_DIR="$(cd "${1}" && pwd)"
}

if [ -f "${TARGET_DIR}/finkbeiner.config" ]; then
    echo "finkbeiner.config already exists in ${TARGET_DIR}"
    echo "Remove it first if you want a fresh copy."
    exit 1
fi

cp "${INSTALL_DIR}/finkbeiner.config.template" "${TARGET_DIR}/finkbeiner.config"

echo "==========================================="
echo "  Workspace ready: ${TARGET_DIR}"
echo "==========================================="
echo ""
echo "  1. Edit your config:"
echo "     vi ${TARGET_DIR}/finkbeiner.config"
echo ""
echo "  2. Run the pipeline:"
echo "     cd ${TARGET_DIR}"
echo "     sbatch ${INSTALL_DIR}/run.sh"
echo ""
echo "  3. Check your log:"
echo "     ls ${TARGET_DIR}/*.out"
echo "==========================================="
