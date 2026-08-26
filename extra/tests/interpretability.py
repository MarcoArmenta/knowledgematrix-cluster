#!/usr/bin/env python
"""
    Tests for the interpretability primitives: the frozen-routing
    linearization (mixer_mode="frozen") and reverse-mode row extraction
    (KnowledgeRowComputer). Only torch is required.
"""
import unittest

import torch

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.row_computer import KnowledgeRowComputer
from knowledgematrix.models.qwen3_5 import Qwen3_5

torch.set_default_dtype(torch.float64)


def tiny_model(device: str = "cpu") -> Qwen3_5:
    torch.manual_seed(0)
    model = Qwen3_5(
        vocab_size=50,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=4,  # 3 linear_attention + 1 full_attention
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=6,
        linear_value_head_dim=6,
        device=device,
    )
    model.to(device)
    model.eval()
    return model


class TestFrozenMode(unittest.TestCase):
    def test_invariant(self):
        """
            mat.sum(1) must reproduce the forward pass exactly in frozen
            mode, like in ratio mode.
        """
        model = tiny_model()
        T = 7
        x = torch.randint(0, 50, (1, 1, T))
        forward_pass = model.forward(x)
        model.save = True
        mat = KnowledgeMatrixComputer(model, batch_size=16, mixer_mode="frozen").forward(x)
        diff = torch.norm(forward_pass.reshape(1, -1) - mat.sum(1)).item()
        self.assertAlmostEqual(first=diff, second=0, places=None, delta=1e-8)

    def test_cross_token_blocks(self):
        """
            In frozen mode attribution flows across positions: blocks
            A[t, :, s, :] with s < t are populated, while causality keeps
            blocks with s > t exactly zero. In ratio mode all off-diagonal
            blocks are exactly zero (the mixers act as diagonal maps).
        """
        model = tiny_model()
        T, V, D = 7, 50, 24
        x = torch.randint(0, 50, (1, 1, T))
        model.forward(x)
        model.save = True

        mat = KnowledgeMatrixComputer(model, batch_size=16, mixer_mode="frozen").forward(x)
        A = mat[:, :-1].reshape(T, V, T, D)
        block_max = torch.tensor([[A[t, :, s, :].abs().max() for s in range(T)] for t in range(T)])
        self.assertEqual(block_max.triu(1).max().item(), 0.0, "future blocks must be exactly zero")
        past = max(block_max[t, s].item() for t in range(T) for s in range(t))
        self.assertGreater(past, 0.0, "frozen mode must populate past cross-token blocks")

        mat_r = KnowledgeMatrixComputer(model, batch_size=16, mixer_mode="ratio").forward(x)
        A_r = mat_r[:, :-1].reshape(T, V, T, D)
        off = max(A_r[t, :, s, :].abs().max().item() for t in range(T) for s in range(T) if s != t)
        self.assertEqual(off, 0.0, "ratio mode is block diagonal over positions")

    def test_invalid_mode_raises(self):
        model = tiny_model()
        with self.assertRaises(ValueError):
            KnowledgeMatrixComputer(model, mixer_mode="wrong")
        with self.assertRaises(ValueError):
            KnowledgeRowComputer(model, mixer_mode="wrong")


class TestRowComputer(unittest.TestCase):
    def test_rows_match_matrix(self):
        """
            Reverse-mode rows must equal the corresponding rows of the
            forward-mode matrix, in both modes, for both A and W_eff.
        """
        model = tiny_model()
        T, V = 7, 50
        x = torch.randint(0, 50, (1, 1, T))
        forward_pass = model.forward(x)
        rows = [0, 3 * V + 17, (T - 1) * V + 42]

        for mode in ("frozen", "ratio"):
            model.save = True
            mat = KnowledgeMatrixComputer(model, batch_size=16, mixer_mode=mode).forward(x)
            weff = KnowledgeMatrixComputer(model, batch_size=16, mixer_mode=mode).forward(x, extract_weff=True)

            row_computer = KnowledgeRowComputer(model, mixer_mode=mode)
            R = row_computer.forward(x, rows)
            Rw = row_computer.forward(x, rows, extract_weff=True)

            self.assertLess((R - mat[rows]).abs().max().item(), 1e-10, f"A rows differ in mode {mode}")
            self.assertLess((Rw - weff[rows]).abs().max().item(), 1e-10, f"W_eff rows differ in mode {mode}")
            self.assertLess(
                (R.sum(1) - forward_pass.reshape(-1)[rows]).abs().max().item(),
                1e-10,
                f"row sums must equal the output coordinates in mode {mode}",
            )

    def test_unsupported_model_raises(self):
        from knowledgematrix.neural_net import NN

        class TinyCNN(NN):
            def __init__(self):
                super().__init__(input_shape=(1, 6, 6))
                self.conv(1, 2, (3, 3))
                self.relu()
                self.flatten()
                self.linear(2 * 4 * 4, 3)

        model = TinyCNN()
        model.eval()
        x = torch.randn(1, 6, 6)
        with self.assertRaises(NotImplementedError):
            KnowledgeRowComputer(model).forward(x, rows=[0])


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class TestGPU(unittest.TestCase):
    def test_frozen_invariant_and_rows_on_cuda(self):
        model = tiny_model(device="cuda")
        T, V = 7, 50
        x = torch.randint(0, 50, (1, 1, T), device="cuda")
        forward_pass = model.forward(x)
        model.save = True
        mat = KnowledgeMatrixComputer(model, batch_size=16, mixer_mode="frozen").forward(x)
        diff = torch.norm(forward_pass.reshape(1, -1) - mat.sum(1)).item()
        self.assertAlmostEqual(first=diff, second=0, places=None, delta=1e-8)

        rows = [0, (T - 1) * V + 42]
        R = KnowledgeRowComputer(model, mixer_mode="frozen").forward(x, rows)
        self.assertLess((R - mat[rows]).abs().max().item(), 1e-10)


if __name__ == "__main__":
    unittest.main()
