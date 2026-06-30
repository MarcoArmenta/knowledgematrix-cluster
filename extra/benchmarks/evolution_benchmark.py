#!/usr/bin/env python
"""
    Wall-clock benchmark comparing two ways of tracking the Knowledge Matrix
    across one vanilla-GD optimizer step on a ReLU MLP:

      Method A (recompute):  apply the optimizer step manually, then call
                             KnowledgeMatrixComputer.forward(x) from scratch.

      Method B (evolve):     call KnowledgeMatrixEvolution.forward(x) to get
                             (dKM_smooth, dKM_cross), apply_step() to mutate
                             parameters, and reconstruct KM_{t+1} from the
                             cached KM_t.

    Both methods end at the same (theta_{t+1}, KM_{t+1}) in exact arithmetic;
    the empirical question is which is faster.

    Usage:
        PYTHONPATH="$(pwd):$PYTHONPATH" python extra/benchmarks/evolution_benchmark.py
"""
import time
from statistics import median
from typing import List, Tuple

import torch

from knowledgematrix.evolution import KnowledgeMatrixEvolution
from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.neural_net import NN


DEVICE = "cpu"
DTYPE = torch.float64
SEED = 0
LR = 1e-3
N_WARMUP = 5
N_MEASURE = 40
SPOT_CHECK_TOL = 1e-5

# Pin to one thread. Multi-threaded BLAS on a noisy laptop swings wall-clock
# by ~2x between runs; single-threaded is stable and the A-vs-B ratio is the
# same quantity either way.
torch.set_num_threads(1)

# (L, n, d, C) configurations. L = number of hidden ReLU blocks, so the MLP
# has L+1 Linear layers and L ReLU activations.
CONFIGS: List[Tuple[int, int, int, int]] = [
    (2, 16, 16, 2),
    (4, 64, 64, 10),
    (4, 128, 128, 10),
    (6, 256, 256, 10),
]


class MLP(NN):
    """Flatten -> (Linear + ReLU) * L -> Linear, matching evolution.py rules."""

    def __init__(self, L: int, n: int, d: int, C: int) -> None:
        super().__init__(input_shape=(1, d, 1), save=False, device=DEVICE)
        self.flatten()
        prev = self.get_input_size()
        for _ in range(L):
            self.linear(in_features=prev, out_features=n)
            self.relu()
            prev = n
        self.linear(in_features=prev, out_features=C)


def _clone_state(model: NN) -> List[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters()]


def _load_state(model: NN, state: List[torch.Tensor]) -> None:
    with torch.no_grad():
        for p, s in zip(model.parameters(), state):
            p.copy_(s)


def _zero_grads(model: NN) -> None:
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()


def _backward(model: NN, x: torch.Tensor, y: torch.Tensor) -> None:
    _zero_grads(model)
    out = model.forward(x)
    loss = 0.5 * ((out.reshape(-1) - y) ** 2).sum()
    loss.backward()


def _method_A_step(
        model: NN,
        computer: KnowledgeMatrixComputer,
        x: torch.Tensor,
    ) -> torch.Tensor:
    """
        Manual vanilla-GD step on model, then recompute KM from scratch.
        Timed region: (optimizer step + full KM compute). Excludes backward.
    """
    with torch.no_grad():
        for p in model.parameters():
            p.add_(p.grad, alpha=-LR)
    km_new = computer.forward(x)
    return km_new


def _method_B_step(
        ev: KnowledgeMatrixEvolution,
        km_cache: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
    """
        Evolution-formula step: forward() -> apply_step() -> KM_{t+1} from cache.
        Timed region: (ev.forward + ev.apply_step + add). Excludes backward.
    """
    delta_smooth, delta_cross = ev.forward(x)
    ev.apply_step()
    return km_cache + delta_smooth + delta_cross


def _time_method_A(L: int, n: int, d: int, C: int) -> float:
    torch.manual_seed(SEED)
    model = MLP(L, n, d, C)
    model.to(DEVICE)
    for p in model.parameters():
        p.data = p.data.to(DTYPE)

    x = torch.randn(1, d, 1, dtype=DTYPE, device=DEVICE)
    y = torch.randn(C, dtype=DTYPE, device=DEVICE)
    computer = KnowledgeMatrixComputer(model, batch_size=max(16, d // 4), device=DEVICE)

    for _ in range(N_WARMUP):
        _backward(model, x, y)
        _ = _method_A_step(model, computer, x)

    per_step: List[float] = []
    for _ in range(N_MEASURE):
        _backward(model, x, y)
        t0 = time.perf_counter()
        _ = _method_A_step(model, computer, x)
        per_step.append(time.perf_counter() - t0)
    return median(per_step)


def _time_method_B(L: int, n: int, d: int, C: int) -> float:
    torch.manual_seed(SEED)
    model = MLP(L, n, d, C)
    model.to(DEVICE)
    for p in model.parameters():
        p.data = p.data.to(DTYPE)

    x = torch.randn(1, d, 1, dtype=DTYPE, device=DEVICE)
    y = torch.randn(C, dtype=DTYPE, device=DEVICE)
    ev = KnowledgeMatrixEvolution(model, optimizer="gd", lr=LR, device=DEVICE)
    km_cache = KnowledgeMatrixComputer(model, batch_size=max(16, d // 4), device=DEVICE).forward(x)

    for _ in range(N_WARMUP):
        _backward(model, x, y)
        km_cache = _method_B_step(ev, km_cache, x)

    per_step: List[float] = []
    for _ in range(N_MEASURE):
        _backward(model, x, y)
        t0 = time.perf_counter()
        km_cache = _method_B_step(ev, km_cache, x)
        per_step.append(time.perf_counter() - t0)
    return median(per_step)


def _spot_check(L: int, n: int, d: int, C: int) -> float:
    """
        Single-step correctness check: both methods should produce the same
        KM(theta_{t+1}, x) up to O(lr^2). Uses a fresh model per call.
    """
    torch.manual_seed(SEED)
    model_A = MLP(L, n, d, C)
    model_A.to(DEVICE)
    for p in model_A.parameters():
        p.data = p.data.to(DTYPE)
    init = _clone_state(model_A)

    torch.manual_seed(SEED)
    model_B = MLP(L, n, d, C)
    model_B.to(DEVICE)
    for p in model_B.parameters():
        p.data = p.data.to(DTYPE)
    _load_state(model_B, init)

    x = torch.randn(1, d, 1, dtype=DTYPE, device=DEVICE)
    y = torch.randn(C, dtype=DTYPE, device=DEVICE)

    # Method A single step.
    computer = KnowledgeMatrixComputer(model_A, batch_size=max(16, d // 4), device=DEVICE)
    _backward(model_A, x, y)
    km_A = _method_A_step(model_A, computer, x)

    # Method B single step from the same init.
    ev = KnowledgeMatrixEvolution(model_B, optimizer="gd", lr=LR, device=DEVICE)
    km_t = KnowledgeMatrixComputer(model_B, batch_size=max(16, d // 4), device=DEVICE).forward(x)
    _backward(model_B, x, y)
    km_B = _method_B_step(ev, km_t, x)

    return float((km_A - km_B).abs().max())


def main() -> int:
    torch.manual_seed(SEED)
    torch.set_default_dtype(DTYPE)

    header = f"| {'config (L, n, d, C)':<22} | {'A: recompute (ms)':>18} | {'B: evolve (ms)':>16} | {'B/A':>6} | {'spot-check':>10} |"
    sep    = "|" + "-" * (len(header) - 2) + "|"
    rows: List[str] = [header, sep]

    exit_code = 0
    for cfg in CONFIGS:
        L, n, d, C = cfg
        tA = _time_method_A(L, n, d, C)
        tB = _time_method_B(L, n, d, C)
        err = _spot_check(L, n, d, C)
        status = "OK" if err < SPOT_CHECK_TOL else "FAIL"
        if err >= SPOT_CHECK_TOL:
            exit_code = 1
        rows.append(
            f"| ({L}, {n:>3}, {n:>3}, {C:>2})       "[:24]
            + f"| {tA * 1e3:>18.3f} | {tB * 1e3:>16.3f} | {tB / tA:>6.2f} | {err:>8.2e} {status} |"
        )

    print()
    for r in rows:
        print(r)
    print()
    print(f"Warmup: {N_WARMUP} steps. Measured: median over {N_MEASURE} steps. "
          f"dtype={DTYPE}, device={DEVICE}, lr={LR}, seed={SEED}.")
    print(f"Spot-check tolerance: {SPOT_CHECK_TOL:.0e} (expected ~O(lr^2) = {LR ** 2:.0e}).")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
