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

| family | loss | best config | best epoch | val | **best test** | final test |
|--------|------|-------------|-----------|-----|---------------|------------|
| MLP | off-class KM | MLP d=2 w=512 lr=0.3 bs=64 wd=0 sched=step | 74 | 0.546 | **0.462** | 0.484 |
| MLP | vanilla CE | MLP d=1 w=512 lr=0.3 bs=128 wd=1e-05 sched=plateau | 16 | 0.668 | **0.644** | 0.630 |
| CNN | off-class KM | CNN d=3 ch=(32, 64, 128) k=3 lr=0.01 bs=64 wd=1e-05 sched=onecycle | 97 | 0.584 | **0.557** | 0.556 |
| CNN | vanilla CE | CNN d=2 ch=(32, 64) k=5 lr=0.03 bs=64 wd=0 sched=plateau | 36 | 0.966 | **0.938** | 0.938 |

## Which LR scheduler is best (mean best-test over its trials)

| family | loss | none | cosine | step | exp | onecycle | plateau |
|--------|------|----|----|----|----|----|----|
| MLP | off-class KM | 0.461 | 0.251 | 0.317 | 0.222 | 0.370 | 0.261 |
| MLP | vanilla CE | 0.612 | 0.550 | 0.568 | 0.409 | 0.559 | 0.582 |
| CNN | off-class KM | 0.292 | 0.360 | 0.410 | 0.168 | 0.557 | 0.425 |
| CNN | vanilla CE | 0.728 | 0.942 | 0.802 | 0.867 | 0.934 | 0.748 |
