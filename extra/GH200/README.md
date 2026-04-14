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
cd /path/to/knowledgematrix   # your project root
salloc --partition=<your-gh200-partition> --gres=gpu:1 --mem=32G --time=01:00:00
bash extra/GH200/scripts/phase0_setup.sh
exit  # release the allocation
```

This validates the environment (architecture, page size, GPU, NUMA) and installs
dependencies (PyTorch with CUDA for ARM64, RMM, knowledgematrix).

### 2. Configure SLURM Scripts

Edit the SLURM scripts to match your cluster:

```bash
# Set your partition in both scripts:
vim extra/GH200/scripts/job_hbm.sh
vim extra/GH200/scripts/job_um.sh
```

Replace `<your-gh200-partition>` with your GH200 partition name.
The working directory is detected automatically via `SLURM_SUBMIT_DIR` —
just submit from the project root.

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

## Preliminary Results

> Benchmarks in progress. Only R18-w128 and partial R18-w256 results available so far.
> R34/R46 models pending.

**Hardware:** NVIDIA GH200 144G HBM3e (node gh1305.m)
**Dates:** 2026-04-13 (HBM), 2026-04-14 (Unified Memory)

### Best Achievable Latency

Best median time per knowledge matrix across all class counts and column batch sizes tested.

| Model | Input | HBM ms | ColBatch | HBM Peak GB | UM ms | ColBatch | UM Peak GB |
|-------|-------|-------:|:--------:|------------:|------:|:--------:|-----------:|
| R18-w128 | 3x32x32 | 75 | 4096 | 9.3 | 76 | 4096 | 1.3 |
| R18-w128 | 3x64x64 | 372 | 4096 | 35.7 | 372 | 4096 | 1.3 |
| R18-w128 | 3x128x128 | 5,760 | 1024 | 35.8 | 5,760 | 1024 | 1.35 |
| R18-w256 | 3x32x32 | 218 | 1024 | 9.4 | -- | -- | -- |
| R18-w256 | 3x64x64 | 1,012 | 1024 | 18.4 | -- | -- | -- |
| R18-w256 | 3x128x128 | 16,728 | 64 | 5.3 | -- | -- | -- |

R18-w128 3x128x128 at colbatch=4096 hits OOM on HBM (would need ~142 GB).
R18-w256 3x128x128 only has colbatch 16 and 64 results so far.

### Column Batch Scaling — R18-w128

Median ms per matrix (1000 classes, 10 samples). Shows how increasing column batch
size trades memory for speed.

| ColBatch | 3x32x32 HBM | 3x32x32 UM | 3x64x64 HBM | 3x64x64 UM | 3x128x128 HBM | 3x128x128 UM |
|:--------:|------------:|----------:|------------:|-----------:|--------------:|--------------:|
| 16 | 595 | 730 | 2,484 | 2,838 | 10,882 | 11,622 |
| 64 | 161 | 191 | 638 | 775 | 6,923 | 6,938 |
| 256 | 91 | 92 | 435 | 435 | 5,930 | 5,934 |
| 1024 | 78 | 79 | 379 | 380 | 5,765 | 5,779 |
| 4096 | 76 | 76 | 373 | 373 | OOM | 41,121 |

### Column Batch Scaling — R18-w256

Median ms per matrix (1000 classes, 10 samples). HBM only (UM data pending).

| ColBatch | 3x32x32 | 3x64x64 | 3x128x128 |
|:--------:|--------:|--------:|----------:|
| 16 | 606 | 2,584 | -- |
| 64 | 326 | 1,452 | -- |
| 256 | 232 | 1,071 | -- |
| 1024 | 219 | 1,012 | -- |
| 4096 | 219 | 1,020 | -- |

R18-w256 latency saturates at colbatch=1024 — no benefit from 4096, which uses 4x
more memory (18.8 vs 71.5 GB at 3x64x64).

### Memory: HBM vs Unified Memory

Peak GPU memory (GB) for R18-w128 at 3x64x64, 1000 classes.
HBM grows linearly with column batch; UM stays flat.

| ColBatch | HBM Peak GB | UM Peak GB |
|:--------:|------------:|-----------:|
| 16 | 0.41 | 1.34 |
| 64 | 0.83 | 1.34 |
| 256 | 2.49 | 1.36 |
| 1024 | 9.13 | 1.36 |
| 4096 | 35.71 | 1.35 |

UM reports a constant ~1.3 GB because RMM's managed pool backs allocations with
unified memory that spills transparently between HBM and LPDDR5X. The actual
working set is larger but not reflected in `torch.cuda.max_memory_allocated`.

### Failures

| Failure | Conditions |
|---------|------------|
| Timeout (>30s) | ColBatch=16 with 1000 samples at 64x64+ inputs; most 128x128 configs with >=100 samples |
| OOM | R18-w128 3x128x128 at colbatch=4096 (HBM only, needs ~142 GB) |
| UM page thrashing | R18-w128 3x128x128 at colbatch=4096 — 41s vs 5.8s at 1024 (~7x regression) |
| Not tested | R18-w256 3x128x128 at colbatch >=256; all R34/R46 models |

### Key Findings

- **Input size dominates latency.** Going from 32x32 to 128x128 increases compute
  time ~75x for the same model and column batch.
- **ColBatch 16→256 gives 6-8x speedup.** Beyond 256 the returns diminish sharply;
  1024→4096 gives <2% improvement for R18-w128 and 0% for R18-w256.
- **Number of classes has negligible effect on latency** (<3% variation across
  10/100/1000 classes at the same colbatch).
- **Number of samples does not affect per-sample latency** — medians are stable across
  10, 100, and 1000 samples (as expected for independent computations).
- **Unified memory matches HBM latency** at colbatch >=256 (within 1%) for inputs
  up to 128x128, but is 15-20% slower at colbatch=16 and =64.
- **Unified memory keeps peak memory flat at ~1.3 GB** regardless of column batch,
  versus up to 71.5 GB for HBM. This enables larger models/inputs that would
  otherwise OOM.
- **UM page thrashing at large working sets:** colbatch=4096 at 3x128x128 is ~7x
  slower than colbatch=1024 (41s vs 5.8s) due to page migration overhead between
  HBM and LPDDR5X. Use colbatch=1024 for large inputs with unified memory.
- **Recommended colbatch for R18-w128:** 1024 (best speed/memory tradeoff —
  within 2% of maximum speed at 4x less memory than 4096).
- **Recommended colbatch for R18-w256:** 1024 (latency saturates here; 4096 wastes
  memory with no speed gain).
