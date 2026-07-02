# HPO: off-class KM loss vs vanilla cross-entropy (MNIST-1D)

Data: MNIST-1D, train=3500, val=500, test=1000, classes=10.
KM (gradient/jacrev) faithfulness vs library (float64): MLP rel=2.6e-16, CNN rel=1.1e-16 (0 == exact).
Every configuration trains both losses on the SAME architecture and the SAME initial weights. Model selection is by validation accuracy; the reported number is test accuracy.

## MLP: 12 random configurations

| # | config | KM val | KM test | vanilla val | vanilla test |
|---|--------|--------|---------|-------------|--------------|
| 1 | MLP hidden=256 depth=2 lr=0.0003 bs=128 wd=0.001 | 0.450 | 0.442 | 0.590 | 0.588 |
| 2 | MLP hidden=256 depth=2 lr=0.001 bs=128 wd=0.0001 | 0.478 | 0.443 | 0.628 | 0.619 |
| 3 | MLP hidden=64 depth=3 lr=0.0003 bs=128 wd=0 | 0.344 | 0.306 | 0.476 | 0.509 |
| 4 | MLP hidden=32 depth=3 lr=0.001 bs=256 wd=0.001 | 0.296 | 0.285 | 0.496 | 0.480 |
| 5 | MLP hidden=64 depth=2 lr=0.0003 bs=256 wd=0 | 0.300 | 0.285 | 0.446 | 0.463 |
| 6 | MLP hidden=128 depth=2 lr=0.003 bs=64 wd=0.0001 | 0.416 | 0.391 | 0.660 | 0.631 |
| 7 | MLP hidden=256 depth=2 lr=0.003 bs=256 wd=0 | 0.482 | 0.458 | 0.634 | 0.629 |
| 8 | MLP hidden=256 depth=2 lr=0.003 bs=128 wd=0 | 0.470 | 0.455 | 0.644 | 0.647 |
| 9 | MLP hidden=32 depth=1 lr=0.003 bs=128 wd=0.001 | 0.282 | 0.267 | 0.552 | 0.568 |
| 10 | MLP hidden=32 depth=3 lr=0.001 bs=128 wd=0 | 0.284 | 0.299 | 0.490 | 0.486 |
| 11 | MLP hidden=128 depth=3 lr=0.0003 bs=64 wd=0.001 | 0.446 | 0.414 | 0.558 | 0.571 |
| 12 | MLP hidden=64 depth=1 lr=0.0003 bs=256 wd=0.0001 | 0.264 | 0.265 | 0.330 | 0.317 |

## CNN: 10 random configurations

| # | config | KM val | KM test | vanilla val | vanilla test |
|---|--------|--------|---------|-------------|--------------|
| 1 | CNN ch=(16,) k=3 hid=64 pool=4 lr=0.001 bs=64 wd=0.0001 | 0.290 | 0.283 | 0.636 | 0.662 |
| 2 | CNN ch=(16, 32, 64) k=5 hid=128 pool=2 lr=0.003 bs=128 wd=0 | 0.392 | 0.343 | 0.926 | 0.921 |
| 3 | CNN ch=(16, 32, 64) k=5 hid=64 pool=2 lr=0.003 bs=128 wd=0.0001 | 0.454 | 0.397 | 0.912 | 0.909 |
| 4 | CNN ch=(16, 32, 64) k=3 hid=64 pool=2 lr=0.0003 bs=64 wd=0 | 0.392 | 0.383 | 0.630 | 0.661 |
| 5 | CNN ch=(16, 32, 64) k=5 hid=64 pool=2 lr=0.0003 bs=64 wd=0 | 0.420 | 0.404 | 0.696 | 0.733 |
| 6 | CNN ch=(16,) k=3 hid=128 pool=4 lr=0.003 bs=128 wd=0.0001 | 0.282 | 0.267 | 0.704 | 0.739 |
| 7 | CNN ch=(16, 32, 64) k=3 hid=32 pool=4 lr=0.003 bs=128 wd=0.0001 | 0.388 | 0.368 | 0.894 | 0.917 |
| 8 | CNN ch=(32, 64) k=5 hid=128 pool=4 lr=0.003 bs=128 wd=0 | 0.418 | 0.370 | 0.928 | 0.927 |
| 9 | CNN ch=(16, 32) k=3 hid=64 pool=4 lr=0.003 bs=128 wd=0 | 0.338 | 0.308 | 0.876 | 0.884 |
| 10 | CNN ch=(32,) k=3 hid=128 pool=3 lr=0.0003 bs=64 wd=0.0001 | 0.300 | 0.314 | 0.546 | 0.581 |

## Best results (selected by validation accuracy)

| family | loss | best config | train | val | test | same-config other-loss test |
|--------|------|-------------|-------|-----|------|-----------------------------|
| MLP | off-class KM | MLP hidden=256 depth=2 lr=0.003 bs=256 wd=0 | 0.636 | 0.482 | **0.458** | 0.629 |
| MLP | vanilla CE | MLP hidden=128 depth=2 lr=0.003 bs=64 wd=0.0001 | 1.000 | 0.660 | **0.631** | 0.391 |
| CNN | off-class KM | CNN ch=(16, 32, 64) k=5 hid=64 pool=2 lr=0.003 bs=128 wd=0.0001 | 0.442 | 0.454 | **0.397** | 0.909 |
| CNN | vanilla CE | CNN ch=(32, 64) k=5 hid=128 pool=4 lr=0.003 bs=128 wd=0 | 0.992 | 0.928 | **0.927** | 0.370 |

## Head-to-head at each best configuration

- Best MLP for off-class KM (MLP hidden=256 depth=2 lr=0.003 bs=256 wd=0): off-class KM test=0.458, vanilla test=0.629.
- Best MLP for vanilla CE (MLP hidden=128 depth=2 lr=0.003 bs=64 wd=0.0001): vanilla CE test=0.631, off-class KM test=0.391.
- Best CNN for off-class KM (CNN ch=(16, 32, 64) k=5 hid=64 pool=2 lr=0.003 bs=128 wd=0.0001): off-class KM test=0.397, vanilla test=0.909.
- Best CNN for vanilla CE (CNN ch=(32, 64) k=5 hid=128 pool=4 lr=0.003 bs=128 wd=0): vanilla CE test=0.927, off-class KM test=0.370.
