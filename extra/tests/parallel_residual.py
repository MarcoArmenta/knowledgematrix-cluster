#!/usr/bin/env python
import unittest
from functools import reduce
import torch
from torch import nn

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer, compose
from knowledgematrix.neural_net import NN, MultiHeadAttention

torch.set_default_dtype(torch.float64)
V, D, T, H, L = 40, 24, 4, 4, 2


def tiny_neox(seed=0):
    """Minimal parallel-residual transformer: x + attn(ln1(x)) + mlp(ln2(x))."""
    torch.manual_seed(seed)
    m = NN(input_shape=(1, T, D))
    m.embedding(V, D)
    causal = torch.tril(torch.ones(16, 16))
    for _ in range(L):
        start = m.get_num_layers()
        m.layernorm(D)                                        # ln1  (start)
        m.layers.append(MultiHeadAttention(D, H, mask=causal, rotary_pct=0.25))
        mid = m.get_num_layers()
        m.layernorm(D)                                        # ln2  (mid)
        m.linear(D, 4 * D)
        m.gelu()
        m.linear(4 * D, D)
        end = m.get_num_layers()
        m.parallel_blocks[end] = (start, mid)
    m.layernorm(D)
    m.linear(D, V)
    m.eval()
    return m


class TestParallelForward(unittest.TestCase):
    def test_forward_matches_manual(self):
        m = tiny_neox()
        x = torch.randint(0, V, (1, 1, T))
        out = m(x)
        # manual recomputation of block 0 on the embedded stream
        h = m.layers[0](x)
        if h.dim() == 3:
            h = h.unsqueeze(0)
        s, mid, e = 1, 3, 7
        attn = m.layers[2](m.layers[1](h))
        mlp = m.layers[6](m.layers[5](m.layers[4](m.layers[3](h))))
        h1 = h + attn + mlp
        # push through remaining layers using the model itself from stream capture
        m.save = True
        m(x)
        m.save = False
        self.assertTrue(torch.allclose(m.stream[7], h1, atol=1e-12),
                        "parallel merge value wrong at block boundary")
        self.assertEqual(tuple(out.shape[-2:]), (T, V))


class TestParallelKM(unittest.TestCase):
    def run_mode(self, mode):
        m = tiny_neox()
        x = torch.randint(0, V, (1, 1, T))
        mc = KnowledgeMatrixComputer(m, batch_size=16, attention_mode=mode)
        mat = mc.forward(x)
        diff = torch.norm(mc.current_output.reshape(-1) - mat.sum(1)).item()
        self.assertLess(diff, 1e-10, f"[{mode}] identity: {diff}")
        # segment composition across parallel blocks
        cuts = [1, 7, 13, 15]           # block ends + network end (L=2)
        segs = [mc.segment(x, a, b) for a, b in zip(cuts[:-1], cuts[1:])]
        prod = reduce(compose, reversed(segs))
        gap = (prod - mat).abs().max().item()
        self.assertLess(gap, 1e-8, f"[{mode}] composition: {gap}")

    def test_monolithic(self):
        self.run_mode("monolithic")

    def test_frozen_pattern(self):
        self.run_mode("frozen_pattern")


if __name__ == "__main__":
    unittest.main()
