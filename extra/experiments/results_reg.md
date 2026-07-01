# Regularized HPO: off-class KM vs vanilla, MNIST-1D

Data: train=3500, val=500, test=1000, classes=10.
Search: architecture (depth x width), weight_decay [0.0, 1e-05, 0.0001, 0.001, 0.01], LR scheduler ['none', 'cosine', 'step', 'exp', 'onecycle', 'plateau'], lr [0.01, 0.03, 0.1, 0.3]. Optimizer: SGD(momentum=0.9, Nesterov), 100 epochs. All layers ReLU (gradient KM applies).
Both losses share architecture + init per config. Selection is by the **best-by-validation checkpoint over training** (mid-training best); we report that checkpoint's test accuracy and the final-epoch test.

## MLP: 16 random configurations

| # | config | KM best-test | KM final | van best-test | van final |
|---|--------|--------------|----------|---------------|-----------|
| 1 | MLP d=4 w=256 lr=0.01 bs=128 wd=0.01 sched=exp | 0.222 | 0.239 | 0.409 | 0.413 |
| 2 | MLP d=4 w=128 lr=0.3 bs=128 wd=0.01 sched=cosine | 0.102 | 0.102 | 0.555 | 0.566 |
| 3 | MLP d=2 w=128 lr=0.03 bs=64 wd=0.01 sched=step | 0.253 | 0.255 | 0.625 | 0.618 |
| 4 | MLP d=2 w=128 lr=0.01 bs=256 wd=0 sched=plateau | 0.297 | 0.309 | 0.497 | 0.503 |
| 5 | MLP d=3 w=256 lr=0.01 bs=128 wd=0.001 sched=step | 0.337 | 0.342 | 0.537 | 0.546 |
| 6 | MLP d=2 w=512 lr=0.3 bs=128 wd=0.01 sched=step | 0.259 | 0.102 | 0.615 | 0.619 |
| 7 | MLP d=1 w=512 lr=0.01 bs=64 wd=0.001 sched=plateau | 0.392 | 0.373 | 0.574 | 0.577 |
| 8 | MLP d=1 w=512 lr=0.3 bs=128 wd=1e-05 sched=plateau | 0.102 | 0.102 | 0.644 | 0.630 |
| 9 | MLP d=3 w=32 lr=0.03 bs=256 wd=1e-05 sched=cosine | 0.245 | 0.246 | 0.495 | 0.494 |
| 10 | MLP d=2 w=512 lr=0.3 bs=64 wd=0 sched=step | 0.462 | 0.484 | 0.591 | 0.594 |
| 11 | MLP d=4 w=32 lr=0.1 bs=256 wd=0.0001 sched=plateau | 0.252 | 0.241 | 0.613 | 0.614 |
| 12 | MLP d=1 w=512 lr=0.1 bs=256 wd=1e-05 sched=onecycle | 0.370 | 0.344 | 0.559 | 0.568 |
| 13 | MLP d=3 w=256 lr=0.01 bs=256 wd=0.001 sched=step | 0.295 | 0.302 | 0.482 | 0.488 |
| 14 | MLP d=2 w=128 lr=0.03 bs=64 wd=1e-05 sched=none | 0.461 | 0.406 | 0.612 | 0.625 |
| 15 | MLP d=3 w=256 lr=0.01 bs=64 wd=1e-05 sched=cosine | 0.407 | 0.410 | 0.599 | 0.603 |
| 16 | MLP d=1 w=32 lr=0.3 bs=256 wd=0.01 sched=step | 0.295 | 0.287 | 0.557 | 0.563 |

## CNN: 12 random configurations

| # | config | KM best-test | KM final | van best-test | van final |
|---|--------|--------------|----------|---------------|-----------|
| 1 | CNN d=3 ch=(8, 16, 32) k=3 lr=0.3 bs=128 wd=0.001 sched=exp | 0.102 | 0.102 | 0.832 | 0.835 |
| 2 | CNN d=3 ch=(32, 64, 128) k=5 lr=0.01 bs=128 wd=0.01 sched=none | 0.264 | 0.236 | 0.865 | 0.867 |
| 3 | CNN d=2 ch=(32, 64) k=5 lr=0.03 bs=64 wd=0 sched=plateau | 0.527 | 0.493 | 0.938 | 0.938 |
| 4 | CNN d=2 ch=(8, 16) k=3 lr=0.1 bs=64 wd=0.0001 sched=exp | 0.234 | 0.233 | 0.903 | 0.904 |
| 5 | CNN d=1 ch=(8,) k=3 lr=0.03 bs=64 wd=0.01 sched=plateau | 0.224 | 0.183 | 0.361 | 0.364 |
| 6 | CNN d=3 ch=(32, 64, 128) k=3 lr=0.01 bs=64 wd=1e-05 sched=onecycle | 0.557 | 0.556 | 0.934 | 0.936 |
| 7 | CNN d=3 ch=(8, 16, 32) k=5 lr=0.01 bs=128 wd=0 sched=none | 0.478 | 0.419 | 0.910 | 0.909 |
| 8 | CNN d=3 ch=(8, 16, 32) k=3 lr=0.03 bs=64 wd=0.001 sched=cosine | 0.360 | 0.356 | 0.942 | 0.941 |
| 9 | CNN d=3 ch=(8, 16, 32) k=3 lr=0.3 bs=64 wd=0.0001 sched=none | 0.102 | 0.098 | 0.263 | 0.098 |
| 10 | CNN d=1 ch=(8,) k=5 lr=0.1 bs=128 wd=1e-05 sched=none | 0.324 | 0.185 | 0.874 | 0.856 |
| 11 | CNN d=3 ch=(16, 32, 64) k=3 lr=0.01 bs=128 wd=1e-05 sched=step | 0.410 | 0.409 | 0.802 | 0.798 |
| 12 | CNN d=2 ch=(32, 64) k=5 lr=0.03 bs=64 wd=0 sched=plateau | 0.525 | 0.526 | 0.946 | 0.943 |

## Best configuration in every case (selected by mid-training best val)

Gaps are train − test at the *same* checkpoint: `best gap` at the best-val checkpoint, `final gap` at the last epoch.

| family | loss | best config | best epoch | **best test** | best gap | final test | final gap |
|--------|------|-------------|-----------|---------------|----------|------------|-----------|
| MLP | off-class KM | MLP d=2 w=512 lr=0.3 bs=64 wd=0 sched=step | 74 | **0.462** | 0.201 | 0.484 | 0.188 |
| MLP | vanilla CE | MLP d=1 w=512 lr=0.3 bs=128 wd=1e-05 sched=plateau | 16 | **0.644** | 0.349 | 0.630 | 0.370 |
| CNN | off-class KM | CNN d=3 ch=(32, 64, 128) k=3 lr=0.01 bs=64 wd=1e-05 sched=onecycle | 97 | **0.557** | 0.069 | 0.556 | 0.068 |
| CNN | vanilla CE | CNN d=2 ch=(32, 64) k=5 lr=0.03 bs=64 wd=0 sched=plateau | 36 | **0.938** | 0.061 | 0.938 | 0.062 |

## Generalization gap: best-checkpoint vs final for every trial

| family | # | KM best gap | KM final gap | van best gap | van final gap |
|--------|---|-------------|--------------|--------------|---------------|
| MLP | 1 | 0.024 | 0.012 | 0.040 | 0.043 |
| MLP | 2 | -0.004 | -0.004 | 0.118 | 0.139 |
| MLP | 3 | 0.024 | -0.002 | 0.087 | 0.107 |
| MLP | 4 | 0.057 | 0.062 | 0.081 | 0.080 |
| MLP | 5 | 0.111 | 0.108 | 0.155 | 0.163 |
| MLP | 6 | 0.009 | -0.004 | 0.120 | 0.118 |
| MLP | 7 | 0.123 | 0.142 | 0.212 | 0.236 |
| MLP | 8 | -0.004 | -0.004 | 0.349 | 0.370 |
| MLP | 9 | 0.039 | 0.043 | 0.102 | 0.135 |
| MLP | 10 | 0.201 | 0.188 | 0.239 | 0.275 |
| MLP | 11 | 0.009 | 0.020 | 0.366 | 0.377 |
| MLP | 12 | 0.091 | 0.148 | 0.294 | 0.430 |
| MLP | 13 | 0.079 | 0.074 | 0.068 | 0.070 |
| MLP | 14 | 0.148 | 0.109 | 0.349 | 0.375 |
| MLP | 15 | 0.150 | 0.166 | 0.401 | 0.397 |
| MLP | 16 | 0.042 | 0.046 | 0.073 | 0.071 |
| CNN | 1 | -0.004 | -0.004 | 0.033 | 0.033 |
| CNN | 2 | 0.014 | -0.004 | 0.030 | 0.039 |
| CNN | 3 | 0.055 | 0.069 | 0.061 | 0.062 |
| CNN | 4 | 0.041 | 0.035 | 0.066 | 0.083 |
| CNN | 5 | 0.010 | 0.016 | 0.018 | 0.018 |
| CNN | 6 | 0.069 | 0.068 | 0.065 | 0.063 |
| CNN | 7 | 0.061 | 0.022 | 0.052 | 0.066 |
| CNN | 8 | 0.028 | 0.022 | 0.051 | 0.059 |
| CNN | 9 | -0.005 | 0.005 | 0.000 | 0.005 |
| CNN | 10 | 0.001 | -0.006 | 0.025 | 0.039 |
| CNN | 11 | 0.037 | 0.034 | 0.036 | 0.041 |
| CNN | 12 | 0.054 | 0.050 | 0.053 | 0.057 |

## Which LR scheduler is best (mean best-test over its trials)

| family | loss | none | cosine | step | exp | onecycle | plateau |
|--------|------|----|----|----|----|----|----|
| MLP | off-class KM | 0.461 | 0.251 | 0.317 | 0.222 | 0.370 | 0.261 |
| MLP | vanilla CE | 0.612 | 0.550 | 0.568 | 0.409 | 0.559 | 0.582 |
| CNN | off-class KM | 0.292 | 0.360 | 0.410 | 0.168 | 0.557 | 0.425 |
| CNN | vanilla CE | 0.728 | 0.942 | 0.802 | 0.867 | 0.934 | 0.748 |
