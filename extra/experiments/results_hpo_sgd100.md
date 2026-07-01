# HPO: off-class KM loss vs vanilla cross-entropy (MNIST-1D)

Data: MNIST-1D, train=3500, val=500, test=1000, classes=10.
KM (gradient/jacrev) faithfulness vs library (float64): MLP rel=3.0e-16, CNN rel=1.7e-16 (0 == exact).
Optimizer: SGD (momentum=0.9, Nesterov), cosine-annealing LR, 100 epochs.
Every configuration trains both losses on the SAME architecture and the SAME initial weights. Model selection is by validation accuracy; the reported number is test accuracy.

## MLP: 10 random configurations

| # | config | KM val | KM test | vanilla val | vanilla test |
|---|--------|--------|---------|-------------|--------------|
| 1 | MLP hidden=256 depth=2 lr=0.01 bs=128 wd=0.001 | 0.396 | 0.382 | 0.614 | 0.601 |
| 2 | MLP hidden=256 depth=2 lr=0.1 bs=128 wd=0.0001 | 0.528 | 0.482 | 0.658 | 0.661 |
| 3 | MLP hidden=64 depth=3 lr=0.03 bs=128 wd=0 | 0.332 | 0.310 | 0.610 | 0.606 |
| 4 | MLP hidden=32 depth=3 lr=0.1 bs=256 wd=0.001 | 0.278 | 0.262 | 0.582 | 0.599 |
| 5 | MLP hidden=64 depth=2 lr=0.01 bs=256 wd=0 | 0.284 | 0.286 | 0.446 | 0.473 |
| 6 | MLP hidden=128 depth=2 lr=0.01 bs=128 wd=0.0001 | 0.344 | 0.337 | 0.582 | 0.576 |
| 7 | MLP hidden=128 depth=3 lr=0.03 bs=256 wd=0.0001 | 0.350 | 0.336 | 0.596 | 0.608 |
| 8 | MLP hidden=256 depth=3 lr=0.1 bs=64 wd=0.001 | 0.476 | 0.452 | 0.732 | 0.730 |
| 9 | MLP hidden=32 depth=1 lr=0.3 bs=256 wd=0.001 | 0.288 | 0.278 | 0.568 | 0.574 |
| 10 | MLP hidden=32 depth=3 lr=0.3 bs=128 wd=0 | 0.272 | 0.274 | 0.568 | 0.556 |

## CNN: 8 random configurations

| # | config | KM val | KM test | vanilla val | vanilla test |
|---|--------|--------|---------|-------------|--------------|
| 1 | CNN ch=(16, 32) k=3 hid=32 pool=4 lr=0.01 bs=64 wd=0 | 0.426 | 0.402 | 0.916 | 0.914 |
| 2 | CNN ch=(16, 32, 64) k=5 hid=32 pool=2 lr=0.03 bs=128 wd=0 | 0.564 | 0.560 | 0.934 | 0.925 |
| 3 | CNN ch=(16, 32) k=5 hid=128 pool=2 lr=0.1 bs=128 wd=0 | 0.524 | 0.525 | 0.942 | 0.940 |
| 4 | CNN ch=(16, 32, 64) k=5 hid=64 pool=2 lr=0.1 bs=128 wd=0.0001 | 0.664 | 0.635 | 0.946 | 0.952 |
| 5 | CNN ch=(16, 32, 64) k=3 hid=64 pool=2 lr=0.01 bs=64 wd=0 | 0.504 | 0.473 | 0.928 | 0.932 |
| 6 | CNN ch=(16, 32, 64) k=5 hid=64 pool=2 lr=0.01 bs=64 wd=0 | 0.544 | 0.517 | 0.934 | 0.915 |
| 7 | CNN ch=(16,) k=3 hid=128 pool=4 lr=0.1 bs=128 wd=0.0001 | 0.370 | 0.322 | 0.916 | 0.902 |
| 8 | CNN ch=(16, 32, 64) k=3 hid=32 pool=4 lr=0.1 bs=128 wd=0.0001 | 0.578 | 0.563 | 0.952 | 0.956 |

## Best results (selected by validation accuracy)

| family | loss | best config | train | val | test | same-config other-loss test |
|--------|------|-------------|-------|-----|------|-----------------------------|
| MLP | off-class KM | MLP hidden=256 depth=2 lr=0.1 bs=128 wd=0.0001 | 0.753 | 0.528 | **0.482** | 0.661 |
| MLP | vanilla CE | MLP hidden=256 depth=3 lr=0.1 bs=64 wd=0.001 | 1.000 | 0.732 | **0.730** | 0.452 |
| CNN | off-class KM | CNN ch=(16, 32, 64) k=5 hid=64 pool=2 lr=0.1 bs=128 wd=0.0001 | 0.706 | 0.664 | **0.635** | 0.952 |
| CNN | vanilla CE | CNN ch=(16, 32, 64) k=3 hid=32 pool=4 lr=0.1 bs=128 wd=0.0001 | 1.000 | 0.952 | **0.956** | 0.563 |

## Head-to-head at each best configuration

- Best MLP for off-class KM (MLP hidden=256 depth=2 lr=0.1 bs=128 wd=0.0001): off-class KM test=0.482, vanilla test=0.661.
- Best MLP for vanilla CE (MLP hidden=256 depth=3 lr=0.1 bs=64 wd=0.001): vanilla CE test=0.730, off-class KM test=0.452.
- Best CNN for off-class KM (CNN ch=(16, 32, 64) k=5 hid=64 pool=2 lr=0.1 bs=128 wd=0.0001): off-class KM test=0.635, vanilla test=0.952.
- Best CNN for vanilla CE (CNN ch=(16, 32, 64) k=3 hid=32 pool=4 lr=0.1 bs=128 wd=0.0001): vanilla CE test=0.956, off-class KM test=0.563.
