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

## Differentiable knowledge matrix (gradient backend)

The library's `KnowledgeMatrixComputer` runs under `torch.no_grad()` and cannot be
used as a training loss. The shared core (`km_core.py`) computes `M(x)` with the
**gradient × input** method from the repo's `gradient-km` branch
(`knowledgematrix/gradient_matrix.py`, `GradientMatrixComputer`): for a
piecewise-linear network `f(x) = J(x) x + c(x)`, the input columns of `M(x)` are the
per-class input Jacobian scaled by the input, `J(x) ⊙ x`, and the bias column is
`c(x) = f(x) − J(x) x`. The Jacobian is obtained with `torch.func.jacrev`
(reverse-mode autograd), which stays differentiable w.r.t. the weights, so it can
drive a loss. `km_core.grad_knowledge_matrix()` batches this over a minibatch with
`vmap` (and also works for CNNs, unlike the single-sample class). It is checked
against `KnowledgeMatrixComputer` at the start of every run (`rel_diff ≈ 1e-16`, i.e.
exact; a hand-rolled forward-propagation KM is kept as a second cross-check).

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

## Pushing harder: HPO + CNNs (`hpo.py`)

`hpo.py` runs a random hyper-parameter search comparing the **off-class KM loss**
against **vanilla cross-entropy** for both an **MLP** and a **1D-CNN** on MNIST-1D.
Every sampled configuration trains both losses on the *same architecture and the same
initial weights*; model selection is by a held-out validation split (500 of the 4000
train points), and the reported number is test accuracy.

```
python extra/experiments/hpo.py --mlp-trials 12 --cnn-trials 10 \
       --report extra/experiments/results_hpo.md
```

Best configuration found for each family/loss (full search in
[`results_hpo.md`](results_hpo.md)):

| family | loss | best config | train | val | **test** | same-config other loss |
|--------|------|-------------|-------|-----|----------|-------------------------|
| MLP | off-class KM | hidden=256, depth=2, lr=3e-3, bs=256 | 0.64 | 0.48 | **0.46** | vanilla 0.63 |
| MLP | vanilla CE   | hidden=128, depth=2, lr=3e-3, bs=64  | 1.00 | 0.66 | **0.63** | KM 0.39 |
| CNN | off-class KM | ch=(16,32,64), k=5, hid=64, lr=3e-3  | 0.44 | 0.45 | **0.40** | vanilla 0.91 |
| CNN | vanilla CE   | ch=(32,64), k=5, hid=128, lr=3e-3    | 0.99 | 0.93 | **0.93** | KM 0.37 |

Conclusions after tuning:

- **Vanilla wins at every capacity, and the gap widens with the CNN.** Best MLP:
  0.63 vs 0.46; best CNN: **0.93 vs 0.40**. Cross-entropy turns convolutional
  capacity into a +30-point jump (0.63 → 0.93); the off-class KM loss does *not*
  benefit from the CNN at all (~0.40–0.46 regardless of architecture).
- **The KM loss underfits by construction.** Even its best configs reach only
  0.44–0.64 *train* accuracy, versus ~1.00 for vanilla. Its small train↔test gap is
  a symptom of a hard, heavily-constraining optimization target — not of superior
  generalization (its test accuracy is always lower).
- So: training through `M(x)` is real and works, but as a *replacement* for
  cross-entropy it leaves a lot of accuracy on the table, and the shortfall grows
  exactly when the architecture (a CNN) has more structure to exploit. Its natural
  use is as an auxiliary/regularizing term rather than the sole objective.

All knowledge matrices in these experiments are computed with the differentiable
gradient (`jacrev`) backend described above, checked exact against the library.

### Stronger optimization: SGD+momentum, cosine LR, 100 epochs

Re-running the sweep with **SGD (momentum 0.9, Nesterov) + cosine-annealing LR over
100 epochs** (both losses identical; `results_hpo_sgd100.md`):

```
python extra/experiments/hpo.py --mlp-trials 10 --cnn-trials 8 --epochs 100 \
       --optimizer sgd --momentum 0.9 --scheduler cosine \
       --report extra/experiments/results_hpo_sgd100.md
```

| family | loss | best config | train | val | **test** | same-config other loss |
|--------|------|-------------|-------|-----|----------|-------------------------|
| MLP | off-class KM | hidden=256, depth=2, lr=0.1, bs=128 | 0.75 | 0.53 | **0.48** | vanilla 0.66 |
| MLP | vanilla CE   | hidden=256, depth=3, lr=0.1, bs=64  | 1.00 | 0.73 | **0.73** | KM 0.45 |
| CNN | off-class KM | ch=(16,32,64), k=5, hid=64, lr=0.1  | 0.71 | 0.66 | **0.64** | vanilla 0.95 |
| CNN | vanilla CE   | ch=(16,32,64), k=3, hid=32, lr=0.1  | 1.00 | 0.95 | **0.96** | KM 0.56 |

What the stronger regime changes (vs the Adam / shorter run above):

- **The KM loss benefits a lot from more optimization.** Off-class KM on the CNN
  jumps **0.40 → 0.64**, and it now *does* exploit convolutional capacity
  (0.48 MLP → 0.64 CNN, whereas before it was flat ~0.40–0.46). Its train accuracy
  rises to ~0.71–0.75 (from ~0.44–0.64): momentum + a long cosine decay push this
  stiff target much further, so the earlier plateau was largely an optimization
  limit, not a hard ceiling.
- **Vanilla still wins at every capacity, but the CNN gap narrows** from ~0.53 to
  ~0.32 (0.96 vs 0.64). Cross-entropy also improved (0.93 → 0.96) and still fits the
  training set completely (train 1.00).
- Net: better optimization is worth more to the KM loss than to cross-entropy, but
  even tuned hard it remains a harder objective that trails CE — consistent with
  using it as an auxiliary/regularizing term rather than the sole loss.

### Depth × width sweep — which architecture wins (`hpo_arch.py`)

Making **depth and width** the searched axes (fixed shared recipe: SGD momentum 0.9 +
cosine LR, 100 epochs, lr=0.1; all ReLU so the gradient KM applies), a grid over
architectures with both losses trained on identical weights per grid point
(`results_arch.md`):

```
python extra/experiments/hpo_arch.py --report extra/experiments/results_arch.md
```

**MLP — test accuracy (depth × hidden width):**

| off-class KM | 32 | 64 | 128 | 256 | | vanilla CE | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|---|
| **d1** | .30 | .34 | .34 | .37 | | **d1** | .61 | .56 | .58 | .58 |
| **d2** | .32 | .39 | .44 | .48 | | **d2** | .59 | .61 | .64 | .64 |
| **d3** | .32 | .40 | .45 | .49 | | **d3** | .57 | .63 | .70 | .69 |
| **d4** | .28 | .36 | .44 | .49 | | **d4** | .58 | .64 | .71 | **.75** |

**CNN — test accuracy (depth × base channels, channels double per block):**

| off-class KM | 8 | 16 | 32 | | vanilla CE | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|---|
| **d1** | .27 | .30 | .29 | | **d1** | .87 | .90 | .90 |
| **d2** | .32 | .43 | .54 | | **d2** | .93 | .94 | .95 |
| **d3** | .51 | .62 | **.76** | | **d3** | .93 | .95 | **.97** |

**Winning architecture in every case** (selected by validation accuracy):

| family | loss | winning architecture | train | val | **test** |
|--------|------|----------------------|-------|-----|----------|
| MLP | off-class KM | depth=3, width=256 | 0.76 | 0.53 | **0.49** |
| MLP | vanilla CE   | depth=4, width=256 | 1.00 | 0.72 | **0.75** |
| CNN | off-class KM | depth=3, channels=(32,64,128) | 0.83 | 0.76 | **0.76** |
| CNN | vanilla CE   | depth=3, channels=(16,32,64)  | 1.00 | 0.96 | **0.95** |

The answer is consistent across all four cases: **bigger wins — the deepest, widest
network is best (or tied-best) everywhere**, and CNNs beat MLPs for both losses.

- Accuracy is essentially **monotone in depth and width** in every grid, so the
  winner sits at (or one step from) the deep/wide corner in all cases; there is no
  interior sweet spot to discover.
- **The off-class KM loss scales strongly with capacity — more than vanilla does.**
  On the CNN it climbs from 0.27 (d1,w8) to **0.76** (d3,w32); giving it the largest
  architecture is what unlocks it (its earlier ~0.40–0.64 plateaus were capacity- and
  optimization-limited, not a hard ceiling). The KM→vanilla CNN gap shrinks to ~0.19
  at the winning architecture.
- **Vanilla still wins head-to-head at every architecture**, and saturates the
  training set (train ≈ 1.00) while KM does not (train ≤ 0.83) — so the KM loss keeps
  behaving like a hard, capacity-hungry, regularizing objective rather than a drop-in
  replacement for cross-entropy.

#### Generalization gap (train − test)

Gap = train accuracy − test accuracy, across the same grids (`results_arch.md` has
the full train and gap grids):

| MLP KM gap | 32 | 64 | 128 | 256 | | MLP CE gap | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|---|
| **d1** | .03 | .05 | .10 | .12 | | **d1** | .16 | .34 | .42 | .42 |
| **d2** | .05 | .10 | .13 | .26 | | **d2** | .37 | .39 | .37 | .36 |
| **d3** | .04 | .08 | .14 | .27 | | **d3** | .43 | .37 | .30 | .31 |
| **d4** | .04 | .08 | .12 | .21 | | **d4** | .42 | .36 | .29 | .25 |

| CNN KM gap | 8 | 16 | 32 | | CNN CE gap | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|---|
| **d1** | .03 | .04 | .04 | | **d1** | .02 | .04 | .06 |
| **d2** | .03 | .05 | .04 | | **d2** | .07 | .06 | .05 |
| **d3** | .06 | .07 | .08 | | **d3** | .07 | .05 | .03 |

Reading these together with the train grids (vanilla reaches **train ≈ 1.00** for
every MLP and for CNN depth ≥ 2; KM tops out at **train ≈ 0.83**):

- **KM's gap is small mainly because it *underfits*, not because it transfers better.**
  KM's train and test rise together (train never saturates), so they stay close —
  while its test accuracy is lower than vanilla's everywhere. A small gap here is the
  signature of a hard-to-fit objective, not of superior generalization.
- **Vanilla memorizes (train = 1) and its gap is essentially `1 − test`**, so its gap
  *shrinks as capacity grows* (bigger net → higher test → smaller gap); e.g. CNN CE
  gap falls .07 → .03 across depth-3 widths.
- **The "KM regularizes" story does NOT hold at scale.** As width grows the KM loss
  starts to overfit too — MLP-KM gap climbs to **0.27** at width 256 (train 0.76 vs
  test 0.49). At the *winning* (largest) architectures the gaps are comparable or
  actually worse for KM: MLP 0.27 (KM) vs 0.25 (CE); CNN **0.075 (KM) vs 0.048 (CE)**.
  So KM only looks like a regularizer in the small/underfit regime; given enough
  capacity it overfits like anything else, without closing the accuracy gap.

### Regularized HPO: weight decay + LR-scheduler search + mid-training best (`hpo_reg.py`)

Random search adding **weight decay** `{0, 1e-5, 1e-4, 1e-3, 1e-2}` and an **LR-scheduler
search** `{none, cosine, step, exp, onecycle, plateau}` on top of architecture and lr
(SGD momentum 0.9, 100 epochs). Every config is evaluated on val/test **each epoch** and
selected by its **best-by-validation checkpoint** (mid-training best / early stopping),
not the final epoch. Full logs: `results_reg.md`.

Best configuration in every case (test at the best-val checkpoint; final in parens):

| family | loss | best config | best epoch | **best test** (final) |
|--------|------|-------------|-----------|-----------------------|
| MLP | off-class KM | d=2, w=512, lr=0.3, wd=0, sched=step | 74 | **0.46** (0.48) |
| MLP | vanilla CE   | d=1, w=512, lr=0.3, wd=1e-5, sched=plateau | 16 | **0.64** (0.63) |
| CNN | off-class KM | d=3, ch=(32,64,128), lr=0.01, wd=1e-5, sched=onecycle | 97 | **0.56** (0.56) |
| CNN | vanilla CE   | d=2, ch=(32,64), lr=0.03, wd=0, sched=plateau | 36 | **0.94** (0.94) |

Mean best-test per scheduler (noisy — each scheduler sees different random archs/lrs):

| family | loss | none | cosine | step | exp | onecycle | plateau |
|--------|------|------|--------|------|-----|----------|---------|
| CNN | vanilla CE | .73 | **.94** | .80 | .87 | .93 | .75 |
| CNN | off-class KM | .29 | .36 | .41 | .17 | **.56** | .43 |
| MLP | vanilla CE | **.61** | .55 | .57 | .41 | .56 | .58 |
| MLP | off-class KM | **.46** | .25 | .32 | .22 | .37 | .26 |

Findings from adding regularization + scheduler search + mid-training best:

- **Mid-training best matters most for the KM loss.** KM training is unstable and often
  *peaks then degrades*, so the best checkpoint beats the final epoch by up to +0.14
  (e.g. a CNN-KM run 0.32 best vs 0.19 final; an MLP-KM run 0.46 vs 0.41). Early stopping
  / best-checkpoint selection is effectively part of making the KM loss usable. Vanilla is
  steadier but early stopping still rescues the occasional late collapse (one CNN-CE run:
  0.26 best vs 0.10 final).
- **Weight decay does not help — it hurts KM.** Every winning config uses `wd ∈ {0, 1e-5}`;
  larger `wd` (1e-3, 1e-2) consistently lowers accuracy and, combined with high lr, collapses
  KM to chance (0.10). Adding L2 regularization to a loss that already *underfits* is
  counter-productive.
- **KM is far more lr/scheduler-sensitive than vanilla.** `lr=0.3` repeatedly collapses KM
  to chance while vanilla tolerates it; aggressive `exp` decay is the worst scheduler overall;
  `onecycle`/`cosine` are best for the CNN. Vanilla is robust across schedulers.
- **Caveat — random search underperformed the controlled grid for CNN-KM** (0.56 here vs 0.76
  in the depth×width grid): with a 6-way scheduler × 5-way wd × lr × architecture space and
  only 12 CNN trials, it under-sampled the good "big-CNN + cosine + lr=0.1" region the grid
  hit directly. For a stiff, sensitive objective like KM, a controlled sweep beats sparse
  random search. None of this changes the headline: **vanilla still wins every case**, and
  tuning/regularization mostly help KM catch up rather than overtake.

#### Generalization gap (regularized HPO, mid-training best vs final)

Gap = train − test measured *at the same checkpoint* (per-trial gaps for all 28 configs
are in `results_reg.md`). At the winning configs:

| family | loss | best test | **best-ckpt gap** | final test | final gap |
|--------|------|-----------|-------------------|------------|-----------|
| MLP | off-class KM | 0.46 | **0.20** | 0.48 | 0.19 |
| MLP | vanilla CE   | 0.64 (ep 16) | **0.35** | 0.63 | 0.37 |
| CNN | off-class KM | 0.56 | **0.07** | 0.56 | 0.07 |
| CNN | vanilla CE   | 0.94 | **0.06** | 0.94 | 0.06 |

- **At matched architecture the KM gap is systematically smaller than vanilla's** — on the
  MLP trials KM gaps are mostly 0.00–0.15 while vanilla's are 0.10–0.40 (e.g. one config:
  KM 0.01 vs vanilla 0.37; another: KM 0.15 vs vanilla 0.40). Same story as before: KM's
  tighter gap is *underfitting* (lower train **and** lower test), not better transfer.
- **On CNNs both gaps are small** (~0.06): vanilla because it sits near the accuracy ceiling
  (test ≈ 0.94, train ≈ 1.0), KM because it underfits. So a small gap means opposite things
  for the two losses.
- **Mid-training best (early stopping) shrinks the gap — mostly for vanilla.** Selecting the
  best-val checkpoint instead of the final epoch reduces vanilla's gap in most trials (up to
  0.43 → 0.29) and often lifts its test, because the peak precedes full memorization (the MLP
  winner peaks at **epoch 16**). KM's best-ckpt ≈ final gap — its instability is late *collapse*,
  not gradual overfitting, so early stopping rescues accuracy without changing the gap much.
- **Negative gaps** appear for a few KM runs (test ≥ train) — these are the lr-collapsed configs
  sitting at chance, i.e. no fitting at all.