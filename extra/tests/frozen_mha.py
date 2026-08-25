#!/usr/bin/env python
"""
    Tests for the frozen-routing linearization of MultiHeadAttention
    (classic softmax attention), mirroring the invariants tested for the
    Qwen3.5 blocks:

    - frozen_forward(x0, x0) equals forward(x0) exactly (with and without
      GQA/MQA and with a causal mask);
    - the frozen map is linear in x for fixed x0;
    - with a causal mask, cross-position sensitivities above the diagonal
      are exactly zero; without a mask, attribution flows across positions
      (which the elementwise ratio treatment cannot express);
    - KnowledgeMatrixComputer(mixer_mode="frozen") preserves the row-sum
      invariant mat.sum(1) == forward on the Transformer model;
    - KnowledgeRowComputer rows agree with the full matrix rows in frozen
      mode.

    Torch only; float64 for exactness.
"""
import random
import unittest

import torch

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.models.transformer import Transformer
from knowledgematrix.neural_net import MultiHeadAttention
from knowledgematrix.row_computer import KnowledgeRowComputer

DEVICE = "cpu"
torch.set_default_dtype(torch.float64)


def causal_mask(T: int) -> torch.Tensor:
    return torch.tril(torch.ones(T, T))


class TestFrozenForwardMHA(unittest.TestCase):
    def _check_exact_at_x0(self, att: MultiHeadAttention, T: int, d_model: int):
        x0 = torch.randn(1, 1, T, d_model)
        self.assertTrue(torch.allclose(att(x0), att.frozen_forward(x0, x0), atol=1e-12))

    def test_frozen_equals_forward_at_x0(self):
        torch.manual_seed(0)
        self._check_exact_at_x0(MultiHeadAttention(d_model=24, num_heads=4), 7, 24)

    def test_frozen_equals_forward_gqa_mqa(self):
        torch.manual_seed(1)
        self._check_exact_at_x0(
            MultiHeadAttention(d_model=24, num_heads=4, num_kv_heads=2), 6, 24)
        self._check_exact_at_x0(
            MultiHeadAttention(d_model=24, num_heads=4, num_kv_heads=1), 6, 24)

    def test_frozen_equals_forward_with_causal_mask(self):
        torch.manual_seed(2)
        T = 5
        att = MultiHeadAttention(d_model=16, num_heads=2, mask=causal_mask(T))
        self._check_exact_at_x0(att, T, 16)

    def test_frozen_map_is_linear(self):
        torch.manual_seed(3)
        att = MultiHeadAttention(d_model=16, num_heads=2)
        x0 = torch.randn(1, 1, 6, 16)
        x1, x2 = torch.randn_like(x0), torch.randn_like(x0)
        a, b = 0.7, -1.3
        lhs = att.frozen_forward(a * x1 + b * x2, x0, affine=False)
        rhs = (a * att.frozen_forward(x1, x0, affine=False)
               + b * att.frozen_forward(x2, x0, affine=False))
        self.assertTrue(torch.allclose(lhs, rhs, atol=1e-10))

    def _frozen_jacobian_blocks(self, att, T, d_model):
        """(T, T) matrix of block norms ||d out_t / d x_s|| of the frozen map."""
        x0 = torch.randn(1, 1, T, d_model)
        f = lambda xf: att.frozen_forward(
            xf.reshape(1, 1, T, d_model), x0, affine=False).reshape(-1)
        J = torch.func.jacrev(f)(x0.reshape(-1))
        J = J.reshape(T, d_model, T, d_model)
        return torch.linalg.matrix_norm(J.permute(0, 2, 1, 3))

    def test_causal_mask_gives_exact_zeros_above_diagonal(self):
        torch.manual_seed(4)
        T = 5
        att = MultiHeadAttention(d_model=16, num_heads=2, mask=causal_mask(T))
        blocks = self._frozen_jacobian_blocks(att, T, 16)
        upper = torch.triu(blocks, diagonal=1)
        self.assertEqual(float(upper.abs().max()), 0.0)
        self.assertGreater(float(torch.tril(blocks, diagonal=-1).abs().max()), 0.0)

    def test_unmasked_attribution_flows_across_positions(self):
        torch.manual_seed(5)
        T = 5
        att = MultiHeadAttention(d_model=16, num_heads=2)
        blocks = self._frozen_jacobian_blocks(att, T, 16)
        off_diag = blocks - torch.diag(torch.diag(blocks))
        self.assertGreater(float(off_diag.abs().max()), 0.0)


class TestFrozenTransformerInvariant(unittest.TestCase):
    def _random_transformer(self):
        vocab_size = random.randint(30, 50)
        num_heads = random.randint(2, 4)
        d_model = num_heads * random.randint(4, 8)
        model = Transformer(vocab_size=vocab_size, d_model=d_model,
                            d_ff=random.randint(10, 20), num_heads=num_heads)
        model.eval()
        x = torch.randint(0, vocab_size, (1, 1, random.randint(6, 10)))
        return model, x

    def test_row_sum_invariant_frozen(self):
        random.seed(0)
        torch.manual_seed(0)
        for _ in range(3):
            model, x = self._random_transformer()
            with torch.no_grad():
                out = model.forward(x).reshape(-1)
            mat = KnowledgeMatrixComputer(model, mixer_mode="frozen").forward(x)
            self.assertTrue(torch.allclose(mat.sum(1), out, atol=1e-8))

    def test_rows_agree_with_matrix_frozen(self):
        random.seed(1)
        torch.manual_seed(1)
        model, x = self._random_transformer()
        mat = KnowledgeMatrixComputer(model, mixer_mode="frozen").forward(x)
        rows_idx = list(range(0, mat.shape[0], max(1, mat.shape[0] // 17)))
        rows = KnowledgeRowComputer(model, mixer_mode="frozen").forward(x, rows_idx)
        self.assertTrue(torch.allclose(rows, mat[rows_idx], atol=1e-10))

    def test_ratio_mode_unchanged(self):
        """Adding frozen support must not alter the default ratio mode."""
        random.seed(2)
        torch.manual_seed(2)
        model, x = self._random_transformer()
        with torch.no_grad():
            out = model.forward(x).reshape(-1)
        mat = KnowledgeMatrixComputer(model, mixer_mode="ratio").forward(x)
        self.assertTrue(torch.allclose(mat.sum(1), out, atol=1e-8))


if __name__ == "__main__":
    unittest.main()
