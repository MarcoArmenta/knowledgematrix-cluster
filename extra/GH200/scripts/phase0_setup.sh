#!/bin/bash
# =============================================================================
# GH200 Environment Setup for KnowledgeMatrix Benchmarks
#
# Run this once on your GH200 node to validate the environment and install
# dependencies. Supports both NGC container and bare-metal venv paths.
#
# Usage: bash extra/GH200/scripts/phase0_setup.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "=== GH200 KnowledgeMatrix Benchmark Setup ==="
echo "Project root: $PROJECT_ROOT"
echo ""

# --- Architecture check ---
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    echo "WARNING: Expected aarch64 (GH200 ARM), got $ARCH"
    echo "  Some checks may not apply on this architecture."
else
    echo "[OK] Architecture: $ARCH"
fi

# --- Page size check ---
PAGE_SIZE=$(getconf PAGESIZE)
if [ "$PAGE_SIZE" -eq 65536 ]; then
    echo "[OK] Page size: ${PAGE_SIZE} (64K pages — required for good UM performance)"
elif [ "$PAGE_SIZE" -eq 4096 ]; then
    echo "[WARN] Page size: ${PAGE_SIZE} (4K pages)"
    echo "  Unified memory bandwidth will be ~10x worse. 64K kernel pages recommended."
    echo "  Check: dpkg -l | grep linux-nvidia-64k"
else
    echo "[INFO] Page size: ${PAGE_SIZE}"
fi

# --- GPU check ---
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    echo "[OK] GPU: $GPU_NAME ($GPU_MEM)"
else
    echo "[WARN] nvidia-smi not found. GPU may not be available."
fi

# --- NUMA check ---
if [ -f /proc/sys/kernel/numa_balancing ]; then
    NUMA_BAL=$(cat /proc/sys/kernel/numa_balancing)
    if [ "$NUMA_BAL" -eq 0 ]; then
        echo "[OK] AutoNUMA: disabled (recommended)"
    else
        echo "[WARN] AutoNUMA: enabled. Consider disabling for consistent performance."
        echo "  sudo echo 0 > /proc/sys/kernel/numa_balancing"
    fi
fi

# --- Container runtime detection ---
echo ""
echo "--- Environment Setup ---"

CONTAINER_RUNTIME=""
if command -v apptainer &>/dev/null; then
    CONTAINER_RUNTIME="apptainer"
elif command -v singularity &>/dev/null; then
    CONTAINER_RUNTIME="singularity"
elif command -v docker &>/dev/null; then
    CONTAINER_RUNTIME="docker"
fi

if [ -n "$CONTAINER_RUNTIME" ]; then
    echo "Container runtime found: $CONTAINER_RUNTIME"
    SIF_PATH="$HOME/pytorch_25.02.sif"

    if [ -f "$SIF_PATH" ]; then
        echo "[OK] NGC container already exists: $SIF_PATH"
    else
        echo "Pulling NGC PyTorch container (this may take a while)..."
        echo "  Image: nvcr.io/nvidia/pytorch:25.02-py3"
        echo "  Destination: $SIF_PATH"

        if [ "$CONTAINER_RUNTIME" = "apptainer" ] || [ "$CONTAINER_RUNTIME" = "singularity" ]; then
            $CONTAINER_RUNTIME pull "$SIF_PATH" docker://nvcr.io/nvidia/pytorch:25.02-py3
        else
            echo "For Docker, pull manually: docker pull nvcr.io/nvidia/pytorch:25.02-py3"
        fi
    fi

    echo ""
    echo "To run inside container:"
    echo "  $CONTAINER_RUNTIME exec --nv --writable-tmpfs --bind $PROJECT_ROOT --pwd $PROJECT_ROOT $SIF_PATH bash"
else
    echo "No container runtime found. Setting up bare-metal venv."
fi

# --- Python venv setup ---
echo ""
echo "--- Python Environment ---"

VENV_DIR="$PROJECT_ROOT/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

echo "Activating venv: $VENV_DIR"
source "$VENV_DIR/bin/activate"

# Install PyTorch with CUDA (ARM64 requires explicit index URL)
echo "Installing PyTorch (with CUDA for ARM64)..."
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install torchvision --index-url https://download.pytorch.org/whl/cu128

# Install RMM for unified memory benchmarks
echo "Installing RMM..."
pip install rmm-cu12

# Install knowledgematrix in editable mode
echo "Installing knowledgematrix..."
pip install -e "$PROJECT_ROOT"

# --- Validation ---
echo ""
echo "--- Validation ---"

python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
"

python3 -c "
from knowledgematrix.neural_net import NN
from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
print('[OK] knowledgematrix imports work')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Submit HBM benchmark:  sbatch extra/GH200/scripts/job_hbm.sh"
echo "  2. Submit UM benchmark:   sbatch extra/GH200/scripts/job_um.sh"
echo "  3. Monitor:               squeue -u \$USER"
