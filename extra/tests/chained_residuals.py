#!/usr/bin/env python
"""
    The wiring conventions that meet where the GPT-2 / GPT-NeoX adapters
    join the rest of the library:

    - two residual blocks sharing a layer index (`NN.chained_residuals`):
      the default convention (the second block skips from the tensor BEFORE
      the merge at the shared index) and the chained one GPT2 uses (it
      skips from the merged stream), in the forward pass, the knowledge
      matrix and the reverse-mode rows;
    - the residual stream record (`NN.save_stream`), off unless asked for;
    - segment KMs on a network wired with the identity() idiom (LlamaLike),
      in both mixer modes, and the cuts segment() refuses;
    - device-split cut points of a parallel-residual network.

    Only torch is required.
"""
import unittest
from functools import reduce

import torch

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer, compose
from knowledgematrix.row_computer import KnowledgeRowComputer
from knowledgematrix.neural_net import NN
from knowledgematrix.models.gpt2 import GPT2
from knowledgematrix.models.gpt_neox import GPTNeoX
from knowledgematrix.models.llama import LlamaLike

torch.set_default_dtype(torch.float64)
V, D, T = 40, 24, 5


def two_blocks(chained: bool, seed=0) -> NN:
    """x2 = x0 + F(x0) merged at index 2, then a second block 2 -> 4."""
    torch.manual_seed(seed)
    m = NN(input_shape=(1, 1, 10))
    m.chained_residuals = chained
    m.linear(10, 10)     # 0   } F
    m.tanh()             # 1   }
    m.linear(10, 10)     # 2   } G
    m.tanh()             # 3   }
    m.linear(10, 4)      # 4
    m.residual(0, 2)
    m.residual(2, 4)
    m.eval()
    return m


def small_gpt2(seed=0) -> GPT2:
    torch.manual_seed(seed)
    m = GPT2(vocab_size=V, d_model=D, d_ff=48, num_heads=4, num_layers=2, max_len=16)
    m.eval()
    return m


def small_llama(seed=0) -> LlamaLike:
    torch.manual_seed(seed)
    m = LlamaLike(vocab_size=V, hidden_size=D, intermediate_size=48, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=6)
    m.eval()
    return m


class TestSharedIndexConvention(unittest.TestCase):
    def manual(self, m: NN, x: torch.Tensor, chained: bool) -> torch.Tensor:
        L = m.layers
        x0 = x.unsqueeze(0)
        f = L[1](L[0](x0))            # F(x0), the tensor arriving at index 2
        x2 = x0 + f                   # merged stream at index 2
        g = L[3](L[2](x2))            # G(x2), the tensor arriving at index 4
        skip = x2 if chained else f
        return L[4](g + skip)

    def test_default_skips_from_before_the_merge(self):
        m = two_blocks(chained=False)
        self.assertFalse(NN(input_shape=(1, 1, 10)).chained_residuals)
        x = torch.randn(1, 1, 10)
        self.assertTrue(torch.allclose(m(x), self.manual(m, x, chained=False), atol=1e-12))
        self.assertFalse(torch.allclose(m(x), self.manual(m, x, chained=True), atol=1e-6))

    def test_chained_skips_from_the_merged_stream(self):
        m = two_blocks(chained=True)
        x = torch.randn(1, 1, 10)
        self.assertTrue(torch.allclose(m(x), self.manual(m, x, chained=True), atol=1e-12))
        self.assertFalse(torch.allclose(m(x), self.manual(m, x, chained=False), atol=1e-6))

    def test_matrix_and_rows_follow_the_convention(self):
        for chained in (False, True):
            m = two_blocks(chained=chained)
            x = torch.randn(1, 1, 10)
            out = self.manual(m, x, chained).reshape(-1)
            mat = KnowledgeMatrixComputer(m, batch_size=4).forward(x)
            self.assertEqual(tuple(mat.shape), (4, 11))
            self.assertLess((mat.sum(1) - out).abs().max().item(), 1e-10, f"chained={chained} identity")
            rows = KnowledgeRowComputer(m, mixer_mode="ratio").forward(x, [0, 3])
            self.assertLess((rows - mat[[0, 3]]).abs().max().item(), 1e-10, f"chained={chained} rows")

    def test_gpt2_is_a_pre_norm_chain(self):
        """Block 0 of GPT2 recomputed by hand from the recorded stream:
        h1 = h0 + attn(ln1(h0)), h2 = h1 + mlp(ln2(h1))."""
        m = small_gpt2()
        self.assertTrue(m.chained_residuals)
        x = torch.randint(0, V, (1, 1, T))
        m.save = True
        m(x)
        m.save = False
        L = m.layers
        h0 = m.stream[2]
        h1 = h0 + L[4](L[3](L[2](h0)))
        h2 = h1 + L[9](L[8](L[7](L[6](L[5](h1)))))
        self.assertTrue(torch.allclose(m.stream[5], h1, atol=1e-12))
        self.assertTrue(torch.allclose(m.stream[10], h2, atol=1e-12))


class TestStreamRecord(unittest.TestCase):
    def test_off_by_default_on_for_the_adapters(self):
        self.assertFalse(NN(input_shape=(1, 1, 10)).save_stream)
        self.assertFalse(small_llama().save_stream)
        self.assertTrue(small_gpt2().save_stream)
        torch.manual_seed(0)
        self.assertTrue(GPTNeoX(vocab_size=V, d_model=D, d_ff=48, num_heads=4,
                                num_layers=1, max_len=16).save_stream)

    def test_nothing_recorded_unless_asked(self):
        m = small_llama()
        x = torch.randint(0, V, (1, 1, T))
        KnowledgeMatrixComputer(m, batch_size=16).forward(x)
        self.assertTrue(all(s is None for s in m.stream))

    def test_segment_records_then_restores_the_flag(self):
        m = small_llama()
        x = torch.randint(0, V, (1, 1, T))
        mc = KnowledgeMatrixComputer(m, batch_size=16)
        mc.segment(x, 1, 7)
        self.assertFalse(m.save_stream)
        self.assertIsNotNone(m.stream[7])
        # a later plain pass does not leave a stale stream behind
        mc.forward(x)
        self.assertTrue(all(s is None for s in m.stream))


class TestSegmentsWithIdentityIdiom(unittest.TestCase):
    """LlamaLike: x + attn(rmsnorm(x)), x + swiglu(rmsnorm(x)), each residual
    ending on an identity() layer -- 6 layers per block after the embedding."""

    def test_compose_in_both_mixer_modes(self):
        m = small_llama()
        n = m.get_num_layers()
        cuts = [1, 7, 13, n]
        x = torch.randint(0, V, (1, 1, T))
        for mode in ("ratio", "frozen"):
            mc = KnowledgeMatrixComputer(m, batch_size=16, mixer_mode=mode)
            full = mc.forward(x)
            segs = [mc.segment(x, a, b) for a, b in zip(cuts[:-1], cuts[1:])]
            for (a, b), s in zip(zip(cuts[:-1], cuts[1:]), segs):
                target = mc.current_output if b == n else m.stream[b]
                self.assertLess((target.reshape(-1) - s.sum(1)).abs().max().item(), 1e-10,
                                f"[{mode}] segment ({a},{b}) identity")
            prod = reduce(compose, reversed(segs))
            self.assertLess((prod - full).abs().max().item(), 1e-8, f"[{mode}] composition")
            # a cut ON the merge index (3) and one after it (4) both split block 0
            for c in (3, 4):
                split = compose(mc.segment(x, c, 7), mc.segment(x, 1, c))
                self.assertLess((split - segs[0]).abs().max().item(), 1e-8, f"[{mode}] cut {c}")

    def test_invalid_cuts_rejected(self):
        m = small_llama()
        x = torch.randint(0, V, (1, 1, T))
        mc = KnowledgeMatrixComputer(m, batch_size=16)
        for a, b in ((1, 2), (2, 7), (1, 5)):       # inside a residual block
            with self.assertRaises(ValueError):
                mc.segment(x, a, b)
        # a shared index without chained_residuals: the skip of the second
        # block is upstream of the cut
        m2 = two_blocks(chained=False)
        with self.assertRaises(ValueError):
            KnowledgeMatrixComputer(m2, batch_size=4).segment(torch.randn(1, 1, 10), 2, 5)

    def test_shared_index_cut_with_chained_residuals(self):
        m = two_blocks(chained=True)
        x = torch.randn(1, 1, 10)
        mc = KnowledgeMatrixComputer(m, batch_size=4)
        full = mc.forward(x)
        prod = compose(mc.segment(x, 2, 5), mc.segment(x, 0, 2))
        self.assertLess((prod - full).abs().max().item(), 1e-8)

    def test_concat_networks_not_supported(self):
        torch.manual_seed(0)
        m = NN(input_shape=(3, 6, 6))
        m.conv(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1)
        m.relu()
        m.conv(in_channels=4, out_channels=2, kernel_size=3, stride=1, padding=1)
        m.conv(in_channels=6, out_channels=2, kernel_size=1, stride=1, padding=0)
        m.flatten()
        m.linear(2 * 36, 3)
        m.concat_skip(2, 3)
        m.eval()
        with self.assertRaises(NotImplementedError):
            KnowledgeMatrixComputer(m, batch_size=8).segment(torch.randn(3, 6, 6), 0, 2)


class TestParallelBlocksAndDeviceSplit(unittest.TestCase):
    def test_cut_points_avoid_parallel_blocks(self):
        torch.manual_seed(0)
        m = GPTNeoX(vocab_size=V, d_model=D, d_ff=48, num_heads=4, num_layers=2, max_len=16)
        m.eval()
        # blocks span (1, 7] and (7, 13]; only the cut before the head is free
        self.assertEqual(m.shard_cut_points(), [14])
        x = torch.randint(0, V, (1, 1, T))
        want = m(x)
        m.shard(["cpu", "cpu"])
        self.assertEqual(m.shard_boundaries, [14])
        self.assertTrue(torch.equal(m(x), want))
        m.unshard()


if __name__ == "__main__":
    unittest.main()
