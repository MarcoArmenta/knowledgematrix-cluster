# Experiment: training a network *through* its knowledge matrices

This experiment trains a small ReLU MLP by supervising its **knowledge matrix**
`M(W,f)(x)` instead of (only) its output, and compares several such losses against
vanilla cross-entropy under the **same architecture, the same initial weights, and
the same hyper-parameters**.

Recall that `output(x) = M(x).sum(over columns)`, so a loss on `M(x)` also shapes the
prediction — but it can constrain much more than the output alone.

## Losses on the knowledge matrix

**1. `E_ii` loss (the original request).** For an input `x` with label `i`,

```
L_eii(x, i) = || M(W,f)(x) - E_ii ||_F^2
```

where `E_ii` is the matrix unit (same shape as `M(x)`, zero everywhere except a `1`
in entry `(i, i)`). This pins down *every* entry of the local affine decomposition
of the network at `x`.

**2. `off-class` loss (recommended alternative).** `E_ii` is extremely rigid: it
fixes not just *which* class is predicted but the exact per-feature attribution.
A more flexible, and better-behaved, KM loss keeps the cross-class structure but
frees the within-row pattern:

```
L_off(x, i) = ( sum_c M[i, c] - 1 )^2  +  sum_{j != i} || M[j, :] ||^2
```

i.e. **make every wrong-class row of `M(x)` vanish, and make the true class row sum
to 1**. Since `output = M.sum(columns)`, the minimizer still produces a one-hot,
confident, correct output — but the network is free to choose *how* the true logit
is attributed across the input features. This is a strictly larger solution set than
`E_ii`, and it trains much better.

## Differentiable knowledge matrix

The library's `KnowledgeMatrixComputer` runs under `torch.no_grad()` and cannot be
used as a training loss. For a `Flatten/Linear/ReLU` MLP the knowledge matrix has a
closed form that *is* differentiable w.r.t. the weights (with the ReLU gating pattern
held fixed — the correct sub-gradient almost everywhere). It is implemented in
`knowledge_matrix_mlp()` and checked numerically against the library computer at the
start of every run (`abs_diff = 0` to machine precision).

## Data

- **MNIST-1D** (default) — Greydanus, <https://github.com/greydanus/mnist1d>: 40-dim
  inputs, 10 classes, 4000 train / 1000 test. Inputs are standardized with the
  training statistics.
- **blobs** — a self-contained synthetic Gaussian-blob task (useful as a quick,
  clearly-separable sanity check).

## Run

```
pip install -e .
python extra/experiments/download_mnist1d.py         # fetch the dataset (~1.6 MB)
python extra/experiments/knowledge_matrix_training.py --dataset mnist1d --epochs 60 \
       --report extra/experiments/results_mnist1d.md
```

Useful flags: `--dataset {mnist1d,blobs}`, `--hidden`, `--epochs`, `--lr`,
`--batch-size`, `--seed` (and `--d`, `--k`, `--n-train`, `--n-test` for blobs).

## Findings

**MNIST-1D** (hidden=64, 60 epochs, lr 1e-3, seed 0 — see
[`results_mnist1d.md`](results_mnist1d.md)):

| training                                  | train acc | test acc |
|-------------------------------------------|-----------|----------|
| KM loss `‖M(x) − E_ii‖²`                  | 0.15      | 0.17     |
| KM loss  off-class rows→0, true logit→1   | 0.43      | 0.40     |
| vanilla  cross-entropy                    | 0.81      | 0.57     |

On this harder task the rigid `E_ii` target barely clears the 0.10 chance level,
whereas the recommended `off-class` loss learns substantially and closes much of the
gap to cross-entropy — while remaining a genuine loss on the knowledge matrix. On the
easy `blobs` task the `off-class` loss matches vanilla exactly (1.00) and `E_ii`
reaches ~0.87.

Takeaway: you *can* train through the knowledge matrix, and how you shape the target
matters a lot. Constraining the cross-class structure of `M(x)` while leaving the
per-feature attribution free is far more trainable than pinning the entire matrix.
