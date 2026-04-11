#!/bin/bash
# =============================================================================
# SLURM job: Unified Memory benchmark (RMM managed allocator)
#
# Uses --mem=0 to allow access to full system memory via unified memory
# (144 GB HBM3E + LPDDR5X). RMM is initialized inside each subprocess.
# Edit --partition and --chdir to match your GH200 cluster.
#
# Usage: sbatch extra/GH200/scripts/job_um.sh
# =============================================================================
#SBATCH --partition=<your-gh200-partition>
#SBATCH --job-name=km-bench-um
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --chdir=/path/to/knowledgematrix
#SBATCH --output=extra/GH200/results/km-um-%j.out
#SBATCH --error=extra/GH200/results/km-um-%j.err

set -euo pipefail

echo "=== Unified Memory Benchmark ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date -u)"
echo ""

# Activate venv if it exists
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

python extra/GH200/orchestrator.py --allocator rmm --resume
