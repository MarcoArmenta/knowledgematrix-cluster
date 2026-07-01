# Architecture sweep (depth x width): off-class KM vs vanilla, MNIST-1D

Fixed recipe: SGD (momentum=0.9, Nesterov), cosine LR, 100 epochs, lr=0.1, batch=128, wd=0.0001. All layers ReLU/affine/pool (gradient KM applies).
Data: MNIST-1D, train=3500, val=500, test=1000, classes=10. Both losses share architecture and initial weights at every grid point; winners selected by validation accuracy.

## MLP: depth x hidden width grid  (4x4 architectures)

### MLP — off-class KM: test accuracy

| depth \ width | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| **1** | 0.297 | 0.344 | 0.338 | 0.367 |
| **2** | 0.324 | 0.386 | 0.443 | 0.483 |
| **3** | 0.316 | 0.396 | 0.451 | 0.494 |
| **4** | 0.278 | 0.358 | 0.443 | 0.489 |

### MLP — off-class KM: train accuracy

| depth \ width | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| **1** | 0.328 | 0.398 | 0.433 | 0.488 |
| **2** | 0.369 | 0.490 | 0.573 | 0.738 |
| **3** | 0.355 | 0.478 | 0.587 | 0.764 |
| **4** | 0.321 | 0.442 | 0.566 | 0.698 |

### MLP — off-class KM: generalization gap (train − test)

| depth \ width | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| **1** | 0.031 | 0.054 | 0.095 | 0.121 |
| **2** | 0.045 | 0.104 | 0.130 | 0.255 |
| **3** | 0.039 | 0.082 | 0.136 | 0.270 |
| **4** | 0.043 | 0.084 | 0.123 | 0.209 |

### MLP — vanilla CE: test accuracy

| depth \ width | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| **1** | 0.607 | 0.562 | 0.581 | 0.584 |
| **2** | 0.587 | 0.611 | 0.635 | 0.639 |
| **3** | 0.572 | 0.632 | 0.702 | 0.687 |
| **4** | 0.579 | 0.643 | 0.713 | 0.751 |

### MLP — vanilla CE: train accuracy

| depth \ width | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| **1** | 0.762 | 0.899 | 0.999 | 1.000 |
| **2** | 0.953 | 1.000 | 1.000 | 1.000 |
| **3** | 0.999 | 1.000 | 1.000 | 1.000 |
| **4** | 0.995 | 1.000 | 1.000 | 1.000 |

### MLP — vanilla CE: generalization gap (train − test)

| depth \ width | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| **1** | 0.155 | 0.337 | 0.418 | 0.416 |
| **2** | 0.366 | 0.389 | 0.365 | 0.361 |
| **3** | 0.427 | 0.368 | 0.298 | 0.313 |
| **4** | 0.416 | 0.357 | 0.287 | 0.249 |

## CNN: depth x base channels grid  (3x3 architectures)

### CNN — off-class KM: test accuracy

| depth \ width | 8 | 16 | 32 |
|---|---|---|---|
| **1** | 0.265 | 0.301 | 0.289 |
| **2** | 0.320 | 0.425 | 0.538 |
| **3** | 0.511 | 0.623 | 0.759 |

### CNN — off-class KM: train accuracy

| depth \ width | 8 | 16 | 32 |
|---|---|---|---|
| **1** | 0.293 | 0.338 | 0.330 |
| **2** | 0.351 | 0.475 | 0.576 |
| **3** | 0.573 | 0.692 | 0.834 |

### CNN — off-class KM: generalization gap (train − test)

| depth \ width | 8 | 16 | 32 |
|---|---|---|---|
| **1** | 0.028 | 0.037 | 0.041 |
| **2** | 0.031 | 0.050 | 0.038 |
| **3** | 0.062 | 0.069 | 0.075 |

### CNN — vanilla CE: test accuracy

| depth \ width | 8 | 16 | 32 |
|---|---|---|---|
| **1** | 0.866 | 0.897 | 0.899 |
| **2** | 0.929 | 0.939 | 0.953 |
| **3** | 0.930 | 0.952 | 0.967 |

### CNN — vanilla CE: train accuracy

| depth \ width | 8 | 16 | 32 |
|---|---|---|---|
| **1** | 0.881 | 0.940 | 0.962 |
| **2** | 1.000 | 1.000 | 1.000 |
| **3** | 1.000 | 1.000 | 1.000 |

### CNN — vanilla CE: generalization gap (train − test)

| depth \ width | 8 | 16 | 32 |
|---|---|---|---|
| **1** | 0.015 | 0.043 | 0.063 |
| **2** | 0.071 | 0.061 | 0.047 |
| **3** | 0.070 | 0.048 | 0.033 |

## Winning architecture in every case (selected by validation accuracy)

| family | loss | winning architecture | train | val | test | gap (train−test) |
|--------|------|----------------------|-------|-----|------|------------------|
| MLP | off-class KM | depth=3, width=256 | 0.764 | 0.526 | **0.494** | 0.270 |
| MLP | vanilla CE | depth=4, width=256 | 1.000 | 0.720 | **0.751** | 0.249 |
| CNN | off-class KM | depth=3, channels=(32, 64, 128) | 0.834 | 0.764 | **0.759** | 0.075 |
| CNN | vanilla CE | depth=3, channels=(16, 32, 64) | 1.000 | 0.962 | **0.952** | 0.048 |
