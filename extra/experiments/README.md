# Experiment: training a network *through* its knowledge matrices

This experiment trains a small ReLU MLP by supervising its **knowledge matrix**
instead of its output. For an input `x` with label `i`, the loss is

```
loss(x, i) = || M(W,f)(x) - E_ii ||_F^2
```

where `E_ii` is the matrix unit (same shape as `M(x)`, zero everywhere except a
`1` in entry `(i, i)`). Because `output(x) = M(x).sum(over columns)`, a knowledge
matrix equal to `E_ii` forces a correct one-hot prediction for class `i` — but it
constrains the *entire* local affine decomposition of the network at `x`, not just
the column sums that cross-entropy sees.

The result is compared against vanilla cross-entropy training using the **same
architecture, the same initial weights, and the same hyper-parameters**.

## Differentiable knowledge matrix

The library's `KnowledgeMatrixComputer` runs under `torch.no_grad()` and therefore
cannot be used as a training loss. For a `Flatten/Linear/ReLU` MLP the knowledge
matrix has a closed form that *is* differentiable w.r.t. the weights (with the ReLU
gating pattern held fixed — the correct sub-gradient almost everywhere). It is
implemented in `knowledge_matrix_mlp()` and checked numerically against the library
computer at the start of every run (`abs_diff = 0` to machine precision).

## Run

```
pip install -e .
python extra/experiments/knowledge_matrix_training.py --report extra/experiments/results.md
```

Useful flags: `--d`, `--k`, `--hidden`, `--epochs`, `--lr`, `--batch-size`, `--seed`.

## Findings (default config, seed 0)

| training            | final train acc | final test acc |
|---------------------|-----------------|----------------|
| knowledge matrices  | 0.75            | 0.74           |
| vanilla (cross-ent) | 1.00            | 1.00           |

Knowledge-matrix training genuinely learns (well above the `1/k` chance level) and
generalizes (train ≈ test), but it is a much harder optimization target than
cross-entropy and plateaus below it: pinning every entry of `M(x)` to a fixed sparse
matrix is far more restrictive than only constraining the output. See
[`results.md`](results.md) for a full run log.
