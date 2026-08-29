#!/usr/bin/env python
"""
    Tests for the Qwen3.5 building blocks (GatedDeltaNet, GatedAttention,
    SwiGLU) and the Qwen3_5 model.

    The tests that compare against the Hugging Face transformers reference
    implementation are skipped when `transformers` is not installed
    (pip install transformers). The knowledge matrix invariant tests only
    need torch.
"""
import unittest
import random

import torch

from knowledgematrix.neural_net import NN, GatedAttention, GatedDeltaNet, SwiGLU
from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.models.qwen3_5 import Qwen3_5

try:
    from transformers.models.qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

DEVICE = "cpu"
torch.set_default_dtype(torch.float64)


def tiny_hf_config():
    return Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,  # 3 linear_attention + 1 full_attention
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        max_position_embeddings=128,
    )


def tiny_nn_model(**kwargs) -> Qwen3_5:
    params = dict(
        vocab_size=50,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=6,
        linear_value_head_dim=6,
    )
    params.update(kwargs)
    return Qwen3_5(**params)


class TestModuleValidation(unittest.TestCase):
    def test_gated_attention_head_validation(self):
        with self.assertRaises(ValueError):
            GatedAttention(d_model=32, num_heads=4, num_kv_heads=3)
        with self.assertRaises(ValueError):
            GatedAttention(d_model=32, num_heads=4, num_kv_heads=0)

    def test_gated_deltanet_head_validation(self):
        with self.assertRaises(ValueError):
            GatedDeltaNet(d_model=32, num_v_heads=4, num_k_heads=3)

    def test_swiglu_shapes(self):
        glu = SwiGLU(d_model=16, d_ff=40)
        x = torch.randn(1, 1, 5, 16)
        self.assertEqual(glu(x).shape, x.shape)

    def test_gated_attention_shapes(self):
        att = GatedAttention(d_model=16, num_heads=2, num_kv_heads=1, head_dim=24)
        x = torch.randn(1, 1, 5, 16)
        self.assertEqual(att(x).shape, x.shape)
        # fused q+gate projection, decoupled head_dim
        self.assertEqual(att.q_proj.weight.shape, (2 * 24 * 2, 16))
        self.assertEqual(att.k_proj.weight.shape, (1 * 24, 16))
        self.assertEqual(att.o_proj.weight.shape, (16, 2 * 24))

    def test_gated_deltanet_shapes(self):
        dn = GatedDeltaNet(d_model=16, num_v_heads=4, num_k_heads=2, head_k_dim=6, head_v_dim=6)
        x = torch.randn(1, 1, 5, 16)
        self.assertEqual(dn(x).shape, x.shape)

    def test_causality_of_gated_attention(self):
        # Changing a later token must not change earlier outputs.
        torch.manual_seed(0)
        att = GatedAttention(d_model=16, num_heads=2, head_dim=8)
        x = torch.randn(1, 1, 6, 16)
        y = att(x)
        x2 = x.clone()
        x2[..., -1, :] += 1.0
        y2 = att(x2)
        self.assertTrue(torch.allclose(y[..., :-1, :], y2[..., :-1, :]))

    def test_causality_of_gated_deltanet(self):
        torch.manual_seed(0)
        dn = GatedDeltaNet(d_model=16, num_v_heads=2, num_k_heads=2, head_k_dim=8, head_v_dim=8)
        x = torch.randn(1, 1, 6, 16)
        y = dn(x)
        x2 = x.clone()
        x2[..., -1, :] += 1.0
        y2 = dn(x2)
        self.assertTrue(torch.allclose(y[..., :-1, :], y2[..., :-1, :]))


class TestQwen3_5Model(unittest.TestCase):
    def test_default_layer_types(self):
        model = tiny_nn_model()
        self.assertEqual(
            model.layer_types,
            ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        )

    def test_layer_types_validation(self):
        with self.assertRaises(ValueError):
            tiny_nn_model(layer_types=["linear_attention"])  # wrong length
        with self.assertRaises(ValueError):
            tiny_nn_model(layer_types=["foo"] * 4)

    def test_knowledge_matrix_invariant(self):
        """
            mat.sum(1) must reproduce the forward pass exactly.
        """
        torch.manual_seed(0)
        random.seed(0)
        for _ in range(3):
            model = tiny_nn_model().to(DEVICE)
            model.eval()
            seq_len = random.randint(5, 12)
            x = torch.randint(0, model.vocab_size, (1, 1, seq_len))
            forward_pass = model.forward(x)
            model.save = True

            computer = KnowledgeMatrixComputer(model, batch_size=16)
            mat = computer.forward(x)
            diff = torch.norm(forward_pass.reshape(1, -1) - mat.sum(1)).item()

            self.assertAlmostEqual(
                first=diff,
                second=0,
                places=None,
                msg=f"mat.sum(1) and forward_pass differ by {diff}.",
                delta=1e-8,
            )

    def test_knowledge_matrix_invariant_without_lm_head(self):
        torch.manual_seed(1)
        model = tiny_nn_model(include_lm_head=False).to(DEVICE)
        model.eval()
        x = torch.randint(0, model.vocab_size, (1, 1, 7))
        forward_pass = model.forward(x)
        model.save = True

        computer = KnowledgeMatrixComputer(model, batch_size=16)
        mat = computer.forward(x)
        diff = torch.norm(forward_pass.reshape(1, -1) - mat.sum(1)).item()
        self.assertAlmostEqual(first=diff, second=0, places=None, delta=1e-8)


@unittest.skipUnless(HAS_TRANSFORMERS, "requires the optional dependency transformers")
class TestHuggingFaceParity(unittest.TestCase):
    """
        The Hugging Face implementation is the oracle: a randomly
        initialized tiny Qwen3.5 is loaded through from_huggingface and
        both models must compute the same network function.
    """

    def test_forward_parity(self):
        torch.manual_seed(0)
        hf = Qwen3_5ForCausalLM(tiny_hf_config()).double().eval()
        model = Qwen3_5.from_huggingface(hf, dtype=torch.float64).to(DEVICE)
        model.eval()

        for seq_len in (1, 5, 11):
            x = torch.randint(0, 64, (1, 1, seq_len))
            with torch.no_grad():
                hf_logits = hf(x[0]).logits
                logits = model.forward(x).reshape(hf_logits.shape)
            diff = (hf_logits - logits).abs().max().item()
            # The reference casts to float32 inside its normalizations and
            # recurrence, so float64 agreement is capped around 1e-7.
            self.assertLess(diff, 1e-5, f"logits differ by {diff} at seq_len={seq_len}.")

    def test_forward_parity_tied_embeddings(self):
        torch.manual_seed(1)
        cfg = tiny_hf_config()
        cfg.tie_word_embeddings = True
        hf = Qwen3_5ForCausalLM(cfg).double().eval()
        model = Qwen3_5.from_huggingface(hf, dtype=torch.float64).to(DEVICE)
        model.eval()

        x = torch.randint(0, 64, (1, 1, 6))
        with torch.no_grad():
            hf_logits = hf(x[0]).logits
            logits = model.forward(x).reshape(hf_logits.shape)
        diff = (hf_logits - logits).abs().max().item()
        self.assertLess(diff, 1e-5, f"logits differ by {diff}.")

    def test_knowledge_matrix_invariant_on_hf_weights(self):
        torch.manual_seed(2)
        hf = Qwen3_5ForCausalLM(tiny_hf_config()).double().eval()
        model = Qwen3_5.from_huggingface(hf, dtype=torch.float64).to(DEVICE)
        model.eval()

        x = torch.randint(0, 64, (1, 1, 9))
        forward_pass = model.forward(x)
        model.save = True

        computer = KnowledgeMatrixComputer(model, batch_size=16)
        mat = computer.forward(x)
        diff = torch.norm(forward_pass.reshape(1, -1) - mat.sum(1)).item()
        self.assertAlmostEqual(first=diff, second=0, places=None, delta=1e-8)


if __name__ == "__main__":
    unittest.main()
