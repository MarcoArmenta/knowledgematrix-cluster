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

### MLP — vanilla CE: test accuracy

| depth \ width | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| **1** | 0.607 | 0.562 | 0.581 | 0.584 |
| **2** | 0.587 | 0.611 | 0.635 | 0.639 |
| **3** | 0.572 | 0.632 | 0.702 | 0.687 |
| **4** | 0.579 | 0.643 | 0.713 | 0.751 |

## CNN: depth x base channels grid  (3x3 architectures)

### CNN — off-class KM: test accuracy

| depth \ width | 8 | 16 | 32 |
|---|---|---|---|
| **1** | 0.265 | 0.301 | 0.289 |
| **2** | 0.320 | 0.425 | 0.538 |
| **3** | 0.511 | 0.623 | 0.759 |

### CNN — vanilla CE: test accuracy

| depth \ width | 8 | 16 | 32 |
|---|---|---|---|
| **1** | 0.866 | 0.897 | 0.899 |
| **2** | 0.929 | 0.939 | 0.953 |
| **3** | 0.930 | 0.952 | 0.967 |

## Winning architecture in every case (selected by validation accuracy)

| family | loss | winning architecture | train | val | test |
|--------|------|----------------------|-------|-----|------|
| MLP | off-class KM | depth=3, width=256 | 0.764 | 0.526 | **0.494** |
| MLP | vanilla CE | depth=4, width=256 | 1.000 | 0.720 | **0.751** |
| CNN | off-class KM | depth=3, channels=(32, 64, 128) | 0.834 | 0.764 | **0.759** |
| CNN | vanilla CE | depth=3, channels=(16, 32, 64) | 1.000 | 0.962 | **0.952** |
