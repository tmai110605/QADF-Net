#!/usr/bin/env bash
# =============================================================================
# QADF-Net: Single Entry Point Script for Full CMIG Paper Pipeline
# Usage:
#   bash cmig_qadf/run_all.sh                     # Default profile = full
#   bash cmig_qadf/run_all.sh --profile smoke     # Fast sanity check (< 15 mins)
#   bash cmig_qadf/run_all.sh --profile fast      # Intermediate draft
#   bash cmig_qadf/run_all.sh --dry-run           # Inspect job plan & GPU hours
# =============================================================================

set -euo pipefail

# Optimize PyTorch CUDA memory allocation to avoid VRAM fragmentation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Change directory to repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

echo "============================================================================="
echo "QADF-Net: Quality-Aware Spatial Dual-View Fusion (CMIG Reproduction Pipeline)"
echo "Working directory: ${ROOT_DIR}"
echo "Current time: $(date)"
echo "============================================================================="

# 1. Check Python environment
PYTHON_CMD="python"
if ! command -v ${PYTHON_CMD} &> /dev/null; then
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        echo "[ERROR] Python not found in PATH!"
        exit 1
    fi
fi

echo "Python interpreter: $(${PYTHON_CMD} -c 'import sys; print(sys.executable)')"
echo "PyTorch version:    $(${PYTHON_CMD} -c "import torch; print(torch.__version__, '| CUDA:', torch.cuda.is_available())")"

# 2. Install optional requirements (if needed)
NO_INSTALL=false
for arg in "$@"; do
    if [ "$arg" == "--no-install" ]; then
        NO_INSTALL=true
    fi
done

if [ "$NO_INSTALL" = false ]; then
    if [ -f "cmig_qadf/requirements_cmig.txt" ]; then
        echo "Checking / installing cmig_qadf/requirements_cmig.txt ..."
        ${PYTHON_CMD} -m pip install -q -r cmig_qadf/requirements_cmig.txt || true
    fi
fi

# 3. Run master orchestrator
echo "Starting Master DAG Orchestrator..."
${PYTHON_CMD} -m cmig_qadf.run_all "$@"
