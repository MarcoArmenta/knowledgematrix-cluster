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

## Additional KM losses (loosest targets)

Two further losses shape only *which* class row of `M(x)` carries the explanatory
mass, via its row norm `r_k = ||M(x)[k, :]||`:

- **`rownorm_ce`** — cross-entropy on the row norms: `CE(softmax(r), i)`.
- **`rownorm_margin`** — a multiclass hinge: `relu(margin + max_{j≠i} r_j − r_i)`.

These never constrain the column sums (the actual output), only the row magnitudes,
so they are the loosest KM targets here — and, empirically, the weakest.

## Findings

All five losses, identical init & hyper-parameters (hidden=64, lr 1e-3, seed 0).

**MNIST-1D** (60 epochs — [`results_mnist1d.md`](results_mnist1d.md)):

| training                                  | train acc | test acc |
|-------------------------------------------|-----------|----------|
| KM `‖M(x) − E_ii‖²`                       | 0.15      | 0.17     |
| KM off-class rows→0, true logit→1         | 0.43      | **0.40** |
| KM cross-entropy on row norms             | 0.27      | 0.27     |
| KM margin on row norms                    | 0.17      | 0.17     |
| vanilla cross-entropy                     | 0.81      | 0.57     |

**blobs** (40 epochs — [`results_blobs.md`](results_blobs.md)):

| training                                  | train acc | test acc |
|-------------------------------------------|-----------|----------|
| KM `‖M(x) − E_ii‖²`                       | 0.75      | 0.74     |
| KM off-class rows→0, true logit→1         | 1.00      | **1.00** |
| KM cross-entropy on row norms             | 0.17      | 0.16     |
| KM margin on row norms                    | 0.21      | 0.20     |
| vanilla cross-entropy                     | 1.00      | 1.00     |

Takeaway: you *can* train through the knowledge matrix, and how you shape the target
matters enormously. The `off-class` loss — constrain the cross-class structure of
`M(x)` while leaving the per-feature attribution free — is by far the best KM target
(it even matches vanilla on `blobs`). The rigid `E_ii` is much harder to optimize,
and the row-norm losses (which ignore the column sums / output entirely) barely beat
chance on both tasks.
