# GH200 Knowledge Matrix Benchmark

Benchmark suite for profiling knowledge matrix computation on the NVIDIA GH200 GraceHopper.
Tests how column batch size, model size, input size, number of classes, and memory allocator
affect throughput and memory usage.

## Architecture

The benchmark computes knowledge matrices using the `knowledgematrix` library with custom
parameterized ResNets of varying depth and width. All inputs are synthetic (random tensors)
and all weights are random (no pretrained models).

### Model Ladder

Custom ResNets built with the `NN` base class. `blocks_per_stage` controls depth:
each stage has N residual blocks, and each block has 2 conv layers.

| Label | base_width | blocks_per_stage | Depth | Stage widths |
|-------|-----------|-----------------|-------|-------------|
| R18-w128 | 128 | [2, 2, 2, 2] | 18 | 128, 256, 512, 1024 |
| R18-w256 | 256 | [2, 2, 2, 2] | 18 | 256, 512, 1024, 2048 |
| R34-w128 | 128 | [3, 4, 6, 3] | 34 | 128, 256, 512, 1024 |
| R34-w256 | 256 | [3, 4, 6, 3] | 34 | 256, 512, 1024, 2048 |
| R46-w128 | 128 | [4, 6, 8, 4] | 46 | 128, 256, 512, 1024 |
| R46-w256 | 256 | [4, 6, 8, 4] | 46 | 256, 512, 1024, 2048 |

### Sweep Dimensions

| Parameter | Values |
|-----------|--------|
| Input size | (3,32,32), (3,64,64), (3,128,128) |
| Column batch size | 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 |
| Number of samples | 10, 100, 1000, 10000 |
| Number of classes | 10, 100, 1000 |
| Allocator | default (HBM), rmm (unified memory) |

Total: 3,888 configurations per allocator.

### Metrics Collected

- Wall-clock time per matrix (median, mean, p95) via `torch.cuda.Event` timing
- Peak GPU memory allocated and reserved
- Peak RSS (VmHWM from `/proc/self/status`)
- Derived: columns/sec, matrices/sec
- Sanity check: `||model(x) - mat.sum(1)||` for first sample
- OOM detection with partial results logged
- Optional: `torch.profiler` Chrome traces

## Quick Start

Once you are on your GH200:

**Important:** All GPU workloads must be submitted via `sbatch` (or `srun` for
short interactive tasks). Never run benchmarks directly with `bash` or `python`
on the login node.

### 1. Setup

Run the setup script via `salloc` to get an interactive allocation:

```bash
cd /path/to/knowledgematrix
salloc --partition=<your-gh200-partition> --gres=gpu:1 --mem=32G --time=01:00:00
bash extra/GH200/scripts/phase0_setup.sh
exit  # release the allocation
```

This validates the environment (architecture, page size, GPU, NUMA) and installs
dependencies (PyTorch with CUDA for ARM64, RMM, knowledgematrix).

### 2. Configure SLURM Scripts

Edit the SLURM scripts to match your cluster:

```bash
# Set your partition and project path in both scripts:
vim extra/GH200/scripts/job_hbm.sh
vim extra/GH200/scripts/job_um.sh
```

Replace `<your-gh200-partition>` with your GH200 partition name and
`/path/to/knowledgematrix` with the actual project path.

### 3. Run HBM Benchmark

```bash
sbatch extra/GH200/scripts/job_hbm.sh
```

Uses the standard PyTorch allocator with `--mem=32G` (HBM-only, no unified memory).

### 4. Run Unified Memory Benchmark

```bash
sbatch extra/GH200/scripts/job_um.sh
```

Uses RMM managed allocator with `--mem=0` (full HBM + LPDDR5X via unified memory).

### 5. Monitor Progress

```bash
squeue -u $USER
tail -f extra/GH200/results/km-hbm-*.out
```

### 6. Resume After Interruption

Jobs auto-resume. Just resubmit the same script — completed configurations are
skipped based on the JSONL results file.

```bash
sbatch extra/GH200/scripts/job_hbm.sh  # picks up where it left off
```

### 7. Generate Summary

Submit a short job to generate the summary:

```bash
srun --partition=<your-gh200-partition> --gres=gpu:0 --time=00:05:00 \
    python extra/GH200/summarize.py
cat extra/GH200/results/summary.md
```

### 8. Optional: Profiling Traces

Edit the SLURM script to add `--enable-profiler` to the orchestrator command,
then resubmit:

```bash
sbatch extra/GH200/scripts/job_hbm.sh
```

View traces in Chrome: open `chrome://tracing` and load files from
`extra/GH200/results/traces/`.

## Testing Locally (CPU, No GPU Required)

You can test the orchestrator dry-run mode on a login node or your local machine:

```bash
python extra/GH200/orchestrator.py --allocator default --dry-run
```

This prints all 1,944 configurations without launching subprocesses or using the GPU.

## Results Format

Results are stored as JSONL (one JSON object per line) in
`extra/GH200/results/results.jsonl`. Each line contains:

```json
{
    "timestamp": "2026-04-11T...",
    "model_label": "R18-w128",
    "base_width": 128,
    "blocks_per_stage": [2, 2, 2, 2],
    "num_classes": 100,
    "input_size": [3, 64, 64],
    "column_batch_size": 256,
    "num_samples": 100,
    "allocator": "default",
    "status": "ok",
    "median_matrix_ms": 12.34,
    "mean_matrix_ms": 13.01,
    "p95_matrix_ms": 15.67,
    "total_time_s": 1.301,
    "columns_per_sec": 248576.0,
    "matrices_per_sec": 81.2,
    "peak_mem_allocated_gb": 4.56,
    "peak_mem_reserved_gb": 5.12,
    "peak_rss_gb": 6.78,
    "model_params": 44000000,
    "matrix_shape": [100, 12289],
    "sanity_check_diff": 0.001,
    "samples_completed": 100,
    "oom_at_sample": null
}
```

OOM results have `"status": "oom"` with partial timing data for however many
samples completed before the OOM event.

## Adding New Model Configurations

Edit `extra/GH200/configs.py` to add entries to `MODEL_LADDER`:

```python
MODEL_LADDER.append(ResNetConfig(512, [2, 2, 2, 2]))  # R18-w512
```

Or add new sweep dimensions to `INPUT_SIZES`, `COLUMN_BATCH_SIZES`, etc.

## File Structure

```
extra/GH200/
├── README.md                  # This file
├── __init__.py
├── configs.py                 # Model ladder and sweep constants
├── run_single.py              # Subprocess entry point (handles RMM init)
├── orchestrator.py            # Dispatches all configs as subprocesses
├── summarize.py               # Generates summary.md from results.jsonl
├── utils.py                   # JSONL logging, memory helpers, subprocess runner
├── models/
│   ├── __init__.py
│   └── custom_resnet.py       # Parameterized ResNet builder
├── scripts/
│   ├── phase0_setup.sh        # Environment validation and setup
│   ├── job_hbm.sh             # SLURM job: HBM-only (--mem=32G)
│   └── job_um.sh              # SLURM job: Unified memory (--mem=0)
└── results/
    ├── .gitkeep
    ├── results.jsonl           # Raw benchmark data (generated)
    ├── summary.md              # Summary tables (generated)
    └── traces/                 # torch.profiler Chrome traces (generated)
```
