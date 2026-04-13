# Dataset & Multi-Weight Knowledge Matrix Computation

## Context

The knowledgematrix library currently computes one knowledge matrix at a time via `KnowledgeMatrixComputer.forward(x)`. Real research workflows require computing matrices across entire datasets and multiple weight checkpoints (e.g., snapshots along training). This extension adds two new classes to support these workflows efficiently on SLURM clusters.

The library's `batch_size` parameter controls how many **columns** of the knowledge matrix are processed simultaneously — it is NOT a training batch size. For small inputs, `batch_size` can equal the number of features; for ImageNet-scale inputs, it must be smaller due to GPU memory constraints.

## Architecture

Two new classes, each in its own file:

### 1. `DatasetComputer` (`knowledgematrix/dataset_computer.py`)

Wraps `KnowledgeMatrixComputer`. Iterates over a dataset, computes one knowledge matrix per sample, saves each to disk as a `.pt` file.

**Constructor:**
- `model: NN` — the neural network architecture
- `batch_size: int = 1` — KM column batch size
- `device: str | None = None` — computation device

**Methods:**

- `compute(data, output_dir, resume=True) -> None`
  - `data`: any iterable yielding tensors (Dataset, Subset, list, generator). If items are `(input, label)` tuples, extracts the input.
  - `output_dir`: directory for individual `.pt` files (`sample_0.pt`, `sample_1.pt`, ...)
  - `resume`: if True, skips samples where `sample_{i}.pt` already exists
  - Creates `output_dir` if it doesn't exist
  - Logs progress via Python `logging` module (sample index, time, memory)

- `compress(output_dir, archive_path=None) -> str`
  - Compresses `output_dir` contents into a `.tar.gz` archive
  - `archive_path` defaults to `{output_dir}/matrices.tar.gz`
  - Returns the archive path

### 2. `ExperimentRunner` (`knowledgematrix/experiment_runner.py`)

Orchestrates computation across multiple weight checkpoints. Manages the experiment directory layout.

**Constructor:**
- `model: NN` — the model architecture (weights will be swapped via `load_state_dict`)
- `experiment_dir: str` — path to `experiments/<experiment-name>/`
- `batch_size: int = 1` — KM column batch size
- `device: str | None = None` — computation device

**Methods:**

- `get_weight_paths() -> list[str]`
  - Lists and sorts all `.pt` files in `experiment_dir/weights/`

- `run(data, weight_paths=None, resume=True) -> None`
  - `data`: any iterable yielding tensors
  - `weight_paths`: list of `.pt` file paths; if None, uses all weights from `get_weight_paths()`
  - For each weight checkpoint:
    1. Load `state_dict` into model
    2. Set model to eval mode
    3. Compute matrices for all samples → `experiment_dir/matrices/<weight_name>/sample_i.pt`
    4. Compress to `experiment_dir/matrices/<weight_name>/matrices.tar.gz`
  - `resume`: skips weight checkpoints that already have `matrices.tar.gz`

- `run_single(data, weight_path, resume=True) -> None`
  - Convenience for a single weight checkpoint

**Directory Layout:**
```
experiments/<experiment-name>/
├── weights/
│   ├── epoch_0.pt
│   ├── epoch_10.pt
│   └── epoch_50.pt
└── matrices/
    ├── epoch_0/
    │   ├── sample_0.pt  (compute node only, deleted after compress)
    │   └── matrices.tar.gz
    ├── epoch_10/
    │   └── matrices.tar.gz
    └── epoch_50/
        └── matrices.tar.gz
```

## Data Flow

1. User provides: model architecture + dataset (or Subset) + experiment directory
2. `ExperimentRunner` loads weight checkpoint → swaps model weights
3. `DatasetComputer` iterates samples → calls `KnowledgeMatrixComputer.forward(x)` per sample
4. Each knowledge matrix saved as `sample_i.pt` via `torch.save()`
5. After all samples: compress directory to `.tar.gz`
6. User copies `.tar.gz` to login node (external to library)

## Testing

All tests use `unittest`, `torch.set_default_dtype(torch.float64)`, and follow existing patterns.

### `extra/tests/dataset_computer.py`
- Small MLP model (~3 layers), mock dataset of ~5 random tensors
- Tests `compute()`: verifies each saved `.pt` satisfies `mat.sum(1) ≈ model(x)` (delta=0.1)
- Tests resume: partial compute, restart, verify skipped samples
- Tests `compress()`: `.tar.gz` created with correct contents
- Tests with `torch.utils.data.Subset`
- Uses `tempfile.mkdtemp()` for output directories, cleanup after

### `extra/tests/experiment_runner.py`
- Small MLP, 2-3 weight snapshots saved to temp experiment dir
- Tests `run()`: verifies directory layout and matrix correctness per checkpoint
- Tests resume across weight checkpoints
- Tests `run_single()` for a single weight
- Tests `get_weight_paths()` returns sorted list
- Uses `tempfile.mkdtemp()`, cleanup after

## Conventions

- Type hints on all signatures (modern lowercase: `list[str]`, `str | None`)
- Docstrings on all public methods
- Python `logging` for progress (not print)
- No new dependencies beyond torch
- Git: lowercase imperative commit messages (user handles all commits)
- CLAUDE.md added to `.gitignore`
- All work on a new branch

## Files to Create/Modify

| Action | File |
|--------|------|
| Create | `knowledgematrix/dataset_computer.py` |
| Create | `knowledgematrix/experiment_runner.py` |
| Create | `extra/tests/dataset_computer.py` |
| Create | `extra/tests/experiment_runner.py` |
| Create | `CLAUDE.md` (final step) |
| Modify | `.gitignore` (add CLAUDE.md) |
