# KnowledgeMatrix

Library for computing knowledge matrices of neural networks, based on the theory from arXiv papers 2007.12213, 2109.14589, and 2409.13163.

## Project Structure

```
knowledgematrix/
├── knowledgematrix/
│   ├── __init__.py
│   ├── neural_net.py          # NN base class (layer builders, forward pass, residuals)
│   ├── matrix_computer.py     # KnowledgeMatrixComputer (single sample KM computation)
│   ├── dataset_computer.py    # DatasetComputer (dataset iteration, disk I/O, compression)
│   ├── experiment_runner.py   # ExperimentRunner (multi-weight checkpoint orchestration)
│   └── models/                # Pre-built architectures (AlexNet, ResNet18, VGG11, Transformer)
├── extra/tests/               # Unit tests (one per architecture + dataset/experiment tests)
├── example.py                 # Usage examples
└── setup.py                   # Package config (requires torch>=2.0, torchvision>=0.15)
```

## Key Concepts

**Knowledge matrix:** For a neural network with function Psi(W,f) and input x in R^d, the knowledge matrix M(W,f)(x) decomposes the output as `mat.sum(1) ≈ model(x)`. This is the fundamental invariant verified in all tests.

**batch_size:** This is NOT a training/data batch size. It controls how many **columns** of the knowledge matrix are processed simultaneously during computation. Higher values = faster but more GPU memory. For small inputs (e.g., MNIST 28x28=784), batch_size can equal input features. For large inputs (e.g., ImageNet), it must be smaller.

## Core Classes

- **NN** (`neural_net.py`): Base class for neural networks. Provides builder methods (`linear()`, `conv()`, `relu()`, etc.) and a `forward()` method. When `save=True`, records activations needed for KM computation.
- **KnowledgeMatrixComputer** (`matrix_computer.py`): Computes one knowledge matrix for one input sample. `forward(x)` takes an unbatched tensor matching `model.input_shape`.
- **DatasetComputer** (`dataset_computer.py`): Wraps KnowledgeMatrixComputer. Iterates over datasets/subsets, saves each KM as `sample_i.pt`, supports resume and .tar.gz compression.
- **ExperimentRunner** (`experiment_runner.py`): Orchestrates computation across multiple weight checkpoints from `experiments/<name>/weights/`. Outputs to `experiments/<name>/matrices/<weight_name>/`.

## Important Caveats

- `NN.eval()` returns `None` (unlike PyTorch's `nn.Module.eval()`). Never chain: `model.eval()` must be a standalone statement.
- `KnowledgeMatrixComputer.forward(x)` manages the `save` flag internally. Do not set `model.save` manually when using the computer.
- `KnowledgeMatrixComputer.forward(x)` takes an **unbatched** tensor (shape matches `model.input_shape`, no leading batch dimension).
- For multi-weight runs with `ExperimentRunner.run()`, the data iterable must support re-iteration (Dataset, Subset, list — not single-use generators).

## Experiment Directory Layout

```
experiments/<experiment-name>/
├── weights/
│   ├── epoch_0.pt
│   ├── epoch_10.pt
│   └── epoch_50.pt
└── matrices/
    ├── epoch_0/
    │   ├── sample_0.pt
    │   └── matrices.tar.gz
    └── epoch_10/
        └── matrices.tar.gz
```

## Running Tests

Tests use `unittest`, `torch.float64` precision, and `assertAlmostEqual(delta=0.1)`. Run from project root with Python 3.9+:

```bash
python3.12 extra/tests/mlp.py
python3.12 extra/tests/smallcnn.py
python3.12 extra/tests/dataset_computer.py
python3.12 extra/tests/experiment_runner.py
```

## Code Conventions

- **Type hints** on all function signatures (modern lowercase: `list[str]`, `str | None`)
- **Docstrings** on all public classes and methods
- **unittest** framework, tests run as standalone scripts
- **Python logging** module for progress reporting in DatasetComputer/ExperimentRunner
- **No additional dependencies** beyond torch and torchvision
- **Git commits**: lowercase imperative mood (e.g., "add dataset computer", "fix resume logic")
- Custom models inherit from `NN` and build layers in `__init__`
