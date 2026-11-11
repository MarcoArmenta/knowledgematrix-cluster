#!/bin/bash
# =============================================================================
# SLURM job: HBM-only benchmark (standard PyTorch allocator)
#
# Uses --mem=32G to constrain memory to HBM only (no unified memory spill).
# Edit --partition to match your GH200 cluster. Submit from the project root.
#
# Usage: sbatch extra/GH200/scripts/job_hbm.sh
# =============================================================================
#SBATCH --partition=<your-gh200-partition>
#SBATCH --job-name=km-bench-hbm
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=extra/GH200/results/km-hbm-%j.out
#SBATCH --error=extra/GH200/results/km-hbm-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}" || { echo "Cannot cd to SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}"; exit 1; }

echo "=== HBM-only Benchmark ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date -u)"
echo ""

# Activate venv if it exists
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# Flush Python stdout immediately (SLURM has no TTY, so Python block-buffers by default)
export PYTHONUNBUFFERED=1

python -u extra/GH200/orchestrator.py --allocator default --resume
