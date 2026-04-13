# SLURM Cluster Test for KnowledgeMatrix

## Context

We need to validate that the knowledgematrix library works correctly on a SLURM HPC cluster with H100 GPUs (80GB). This is a first deployment test — no real datasets or pretrained weights, just random inputs and random model weights to verify end-to-end correctness. The test also calibrates optimal `batch_size` (the number of knowledge matrix columns processed simultaneously) for the H100.

## Deliverables

Two files:
- `extra/cluster_test.py` — Python test script with 4 sequential phases
- `extra/cluster_test.sh` — SLURM submission script

## Model

**ResNet18** with random weights (no pretrained, no downloads).
- `input_shape=(3, 32, 32)` for pipeline phases
- Additional `input_shape=(3, 224, 224)` for calibration phase
- `num_classes=10`

## Phase 1: Smoke Test

1. Create `ResNet18(input_shape=(3,32,32), num_classes=10, device="cuda")`
2. `model.eval()` (standalone, returns None)
3. `x = torch.randn(3, 32, 32, device="cuda")`
4. `out = model(x)`
5. `KnowledgeMatrixComputer(model, batch_size=1, device="cuda")`
6. `mat = computer.forward(x)`
7. Assert `torch.norm(out - mat.sum(1)).item() < 0.1`
8. Print PASS/FAIL with the error norm

## Phase 2: Batch Size Calibration

For each input size `(3, 32, 32)` and `(3, 224, 224)`:

1. Create fresh `ResNet18` with corresponding input_shape, move to CUDA
2. Create single random input on CUDA
3. For `batch_size` in `[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]`:
   - `torch.cuda.empty_cache()`
   - Warm-up: 1 KM computation (discard)
   - Timed: 3 KM computations, record average time
   - Catch `RuntimeError` (OOM) → stop doubling, record last successful batch_size
   - Record: `batch_size`, `avg_time_s`, `throughput_cols_per_s` (= input_features / avg_time)
4. Print results table per input size
5. Report optimal batch_size per input size (lowest avg_time)

## Phase 3: DatasetComputer Test

1. Use optimal batch_size from Phase 2 (for 32x32 input)
2. Create dummy dataset: `[(torch.randn(3, 32, 32), i) for i in range(20)]`
3. Create temp output dir via `tempfile.mkdtemp()`
4. `DatasetComputer(model, batch_size=optimal, device="cuda")`
5. `computer.compute(data, output_dir)`
6. Verify: 20 `sample_*.pt` files exist
7. Load each, verify `mat.sum(1) ≈ model(x)` for corresponding input
8. `computer.compress(output_dir)` → verify `.tar.gz` exists
9. Clean up temp dir
10. Print PASS/FAIL

## Phase 4: ExperimentRunner Test

1. Create temp experiment dir with `weights/` and `matrices/` subdirs
2. Save `model.state_dict()` as 3 checkpoints: `epoch_0.pt`, `epoch_10.pt`, `epoch_20.pt`
3. Create dummy dataset: `[(torch.randn(3, 32, 32), i) for i in range(10)]`
4. `ExperimentRunner(model, experiment_dir, batch_size=optimal, device="cuda")`
5. `runner.run(data)`
6. Verify: 3 directories under `matrices/` (`epoch_0/`, `epoch_10/`, `epoch_20/`)
7. Each directory has 10 `sample_*.pt` files + `matrices.tar.gz`
8. Spot-check: load a matrix from each checkpoint, verify reconstruction
9. Clean up temp experiment dir
10. Print PASS/FAIL

## SLURM Script (`extra/cluster_test.sh`)

```bash
#!/bin/bash
#SBATCH --job-name=km-test
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=km-test-%j.out
# #SBATCH --partition=<your-partition>

python3 extra/cluster_test.py
```

## Output Format

- Python `logging` with timestamps, level INFO
- Each phase prints a header: `=== Phase N: <name> ===`
- Calibration prints a formatted table
- Final summary: all phase results + optimal batch_sizes
- Exit code 0 if all pass, 1 if any fail

## Verification

After submitting `sbatch extra/cluster_test.sh`:
1. Check job output: `cat km-test-<jobid>.out`
2. All 4 phases should show PASS
3. Calibration table should show increasing throughput up to the optimal point
4. No OOM errors in the pipeline phases (they use the calibrated batch_size)
