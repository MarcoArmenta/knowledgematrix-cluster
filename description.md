## Summary

Add `DatasetComputer` and `ExperimentRunner` classes to the library, enabling batch computation of knowledge matrices over entire datasets and across multiple weight checkpoints.

- **`DatasetComputer`** (`knowledgematrix/dataset_computer.py`): Wraps `KnowledgeMatrixComputer` to iterate over any dataset (Dataset, Subset, list, generator), compute one knowledge matrix per sample, save each as `sample_{i}.pt`, and compress results to `.tar.gz`. Supports resume by skipping already-computed samples.
- **`ExperimentRunner`** (`knowledgematrix/experiment_runner.py`): Orchestrates multi-checkpoint workflows. Loads weight snapshots from `experiments/<name>/weights/`, delegates per-checkpoint computation to `DatasetComputer`, and compresses outputs. Supports resume at both the checkpoint and sample level. Designed for SLURM cluster pipelines.
- **Unit tests** (`extra/tests/dataset_computer.py`, `extra/tests/experiment_runner.py`): Cover core compute, resume logic, compression, and the full multi-checkpoint pipeline. All tests pass.

## Test plan

- [x] `python3.12 extra/tests/dataset_computer.py` — 5 tests OK
- [x] `python3.12 extra/tests/experiment_runner.py` — 4 tests OK
- [x] Existing tests (`mlp.py`, `smallcnn.py`) still pass
