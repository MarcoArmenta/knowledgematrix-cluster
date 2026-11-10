#!/usr/bin/env python
import unittest
from functools import reduce
import torch

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer, compose
from knowledgematrix.models.gpt2 import GPT2

torch.set_default_dtype(torch.float64)
V, D, T = 40, 24, 4


def small_gpt2(seed=0):
    torch.manual_seed(seed)
    m = GPT2(vocab_size=V, d_model=D, d_ff=48, num_heads=4, num_layers=2, max_len=16)
    m.eval()
    return m


class TestBlockBoundaries(unittest.TestCase):
    def test_layout(self):
        m = small_gpt2()
        self.assertEqual(m.block_boundaries(), [2, 10, 18, 20])
        self.assertEqual(m.block_boundaries(sub_blocks=True), [2, 5, 10, 13, 18, 20])


class TestSegments(unittest.TestCase):
    def run_mode(self, attention_mode):
        m = small_gpt2()
        x = torch.randint(0, V, (1, 1, T))
        mc = KnowledgeMatrixComputer(m, batch_size=16, attention_mode=attention_mode)
        full = mc.forward(x)                       # (T·V, T·D+1)
        cuts = m.block_boundaries()                # [2, 10, 18, 20]

        segs = [mc.segment(x, a, b) for a, b in zip(cuts[:-1], cuts[1:])]

        # (1) per-segment row-sum identity vs the reference stream
        for (a, b), s in zip(zip(cuts[:-1], cuts[1:]), segs):
            target = mc.current_output if b == m.get_num_layers() else m.stream[b]
            diff = torch.norm(target.reshape(-1) - s.sum(1)).item()
            self.assertLess(diff, 1e-10, f"[{attention_mode}] segment ({a},{b}) identity: {diff}")

        # (2) composition gate: product of segments == full KM
        prod = reduce(compose, reversed(segs))     # segs[-1] ⋄ … ⋄ segs[0]
        gap = (prod - full).abs().max().item()
        self.assertLess(gap, 1e-8, f"[{attention_mode}] composition gate: {gap}")

        # (3) full-range segment == forward
        whole = mc.segment(x, cuts[0], cuts[-1])
        self.assertTrue(torch.allclose(whole, full, atol=1e-10))

        # (4) final_position_only slices the last row block
        fp = mc.segment(x, cuts[0], cuts[-1], final_position_only=True)
        self.assertEqual(tuple(fp.shape), (V, T * D + 1))
        self.assertTrue(torch.allclose(fp, full[(T - 1) * V:, :], atol=1e-10))

    def test_monolithic(self):
        self.run_mode("monolithic")

    def test_frozen_pattern(self):
        self.run_mode("frozen_pattern")

    def test_final_position_only_rejected_midstream(self):
        m = small_gpt2()
        x = torch.randint(0, V, (1, 1, T))
        mc = KnowledgeMatrixComputer(m, batch_size=16)
        with self.assertRaises(ValueError):
            mc.segment(x, 2, 10, final_position_only=True)


if __name__ == "__main__":
    unittest.main()
