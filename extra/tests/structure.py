#!/usr/bin/env python
import unittest
import torch

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.models.gpt2 import GPT2

torch.set_default_dtype(torch.float64)
V, D, T = 40, 24, 4


def small_gpt2(seed=0):
    torch.manual_seed(seed)
    m = GPT2(vocab_size=V, d_model=D, d_ff=48, num_heads=4, num_layers=2, max_len=16)
    m.eval()
    return m


def position_block(mat, t, s):
    """Linear-part block (t, s): rows = logits at position t, cols = resid dims of position s.
    The trailing bias column is excluded (it is a single shared column)."""
    return mat[t * V:(t + 1) * V, s * D:(s + 1) * D]


class TestMonolithicBlockDiagonal(unittest.TestCase):
    def test_offdiagonal_blocks_exactly_zero(self):
        m = small_gpt2()
        x = torch.randint(0, V, (1, 1, T))
        mat = KnowledgeMatrixComputer(m, batch_size=16).forward(x)
        self.assertEqual(tuple(mat.shape), (T * V, T * D + 1))
        for t in range(T):
            for s in range(T):
                if t != s:
                    self.assertEqual(position_block(mat, t, s).abs().max().item(), 0.0,
                                     f"monolithic block ({t},{s}) must be exactly zero")


class TestFrozenCausalStructure(unittest.TestCase):
    def test_upper_blocks_zero_lower_blocks_mix(self):
        m = small_gpt2()
        x = torch.randint(0, V, (1, 1, T))
        mat = KnowledgeMatrixComputer(m, batch_size=16, attention_mode="frozen_pattern").forward(x)
        for t in range(T):
            for s in range(T):
                blk = position_block(mat, t, s)
                if s > t:
                    self.assertEqual(blk.abs().max().item(), 0.0,
                                     f"frozen block ({t},{s}) violates causality")
        # genuine token mixing: at least one strictly-lower block is materially nonzero
        mix = max(position_block(mat, t, s).norm().item()
                  for t in range(T) for s in range(t))
        self.assertGreater(mix, 1e-8, "frozen presentation shows no token mixing")


if __name__ == "__main__":
    unittest.main()
