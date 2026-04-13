# Debug Session 1: First GH200 Benchmark Run

**Date:** 2026-04-13
**Node:** gh1305 (GH200 144G HBM3e)
**Jobs:** 187899 (km-bench-hbm), 187903 (km-bench-um)

## Timeline

- Jobs submitted to gh-aria partition
- 187899 (HBM) started, ran for 8 hours at 99.9% CPU
- 187903 (UM) stayed pending (Resources) the entire time — only 1 GPU on node
- 187899 hit TIMEOUT (8h wall clock)
- 187903 FAILED with exit code 15 (SIGTERM), 0s wall-clock — never executed

## Issues Found

### 1. No Python output in SLURM .out file

**Cause:** Python block-buffers stdout when no TTY is attached (always under SLURM).
The orchestrator's `print()` calls accumulated in the buffer. The TIMEOUT kill
(SIGTERM then SIGKILL) prevented clean exit, so the buffer was never flushed.
Bash `echo` commands appeared because they flush immediately.

**Fix:** Added `export PYTHONUNBUFFERED=1` and `python -u` to both job scripts.
Added `sys.stdout.flush()` calls in orchestrator.py after each progress row.

### 2. Sweep too large for 8h wall clock

**Cause:** 1,944 configs per allocator (6 models x 3 inputs x 9 batch_sizes x 4 num_samples x 3 classes).
With per-subprocess timeout of 3600s and many configs taking minutes each,
8 hours is insufficient.

**Fix:** Reduced sweep to 810 configs (5 batch sizes, 3 num_samples tiers — dropped
10000-sample tier and intermediate batch sizes). Lowered per-subprocess timeout
from 3600s to 900s.

### 3. UM job never executed (exit code 15)

**Cause:** Most likely bad `--chdir` path in job_um.sh (placeholder not edited,
or typo). SLURM validates `--chdir` at dispatch time, not submission. Job sat
pending for hours, then was killed during prologue when `chdir()` failed.

**Fix:** User must verify both scripts have correct paths on the cluster before
resubmission.

## System Inventory

- GPU: NVIDIA GH200 144G HBM3e
- Node: gh1305
- SLURM memory for HBM job: 32G (host RAM only; GPU HBM not tracked)
- SLURM memory for UM job: 0 (all node memory = 616 GB)
- Host RAM used by orchestrator: 1.63 GB (expected — GPU memory invisible to SLURM)

## What Exists on Cluster

- `extra/GH200/results/km-hbm-187899.out` — bash header only (no Python output)
- `extra/GH200/results/km-hbm-187899.err` — unchecked
- `extra/GH200/results/km-um-187903.err` — unchecked (may not exist if chdir failed)
- `extra/GH200/results/results.jsonl` — likely has partial results from 187899

## Resume Plan

1. Run cluster commands to confirm diagnosis (check .err files, results.jsonl)
2. Pull updated code to cluster
3. Edit `--partition` and `--chdir` in BOTH job scripts
4. Resubmit HBM job — `--resume` will skip already-completed configs
5. After HBM completes, submit UM job
6. Monitor with `tail -f extra/GH200/results/km-hbm-*.out` (now shows live progress)
