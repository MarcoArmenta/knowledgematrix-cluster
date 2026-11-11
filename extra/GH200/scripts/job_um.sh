#!/bin/bash
# =============================================================================
# SLURM job: Unified Memory benchmark (RMM managed allocator)
#
# Uses --mem=0 to allow access to full system memory via unified memory
# (144 GB HBM3E + LPDDR5X). RMM is initialized inside each subprocess.
# Edit --partition to match your GH200 cluster. Submit from the project root.
#
# Usage: sbatch extra/GH200/scripts/job_um.sh
# =============================================================================
#SBATCH --partition=<your-gh200-partition>
#SBATCH --job-name=km-bench-um
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --output=extra/GH200/results/km-um-%j.out
#SBATCH --error=extra/GH200/results/km-um-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}" || { echo "Cannot cd to SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}"; exit 1; }

echo "=== Unified Memory Benchmark ==="
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

python -u extra/GH200/orchestrator.py --allocator rmm --resume
