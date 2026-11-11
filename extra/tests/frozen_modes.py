#!/usr/bin/env python
"""
    How the two frozen linearizations relate.

    KnowledgeMatrixComputer(mixer_mode="frozen") and
    KnowledgeMatrixComputer(attention_mode="frozen_pattern") are ONE
    mechanism -- the layer's `frozen_forward` -- with two scopes:

    - For a MultiHeadAttention layer with input x0 (the saved
      pre-activation) and pattern P = softmax(scores(x0)), both apply the
      linear value-path map
          x  ->  O · (P ⊗ V x)          (biases of V and O in the bias pass)
      with P frozen, instead of the elementwise ratio forward(x0) / x0.
      P is the `attn_pattern` the save pass records.
    - "frozen_pattern" freezes MultiHeadAttention ONLY; mixer_mode="frozen"
      also freezes the other linearizable layers (RopeAttention,
      GatedAttention, GatedDeltaNet, SwiGLU, GeGLU).

    So the two matrices coincide exactly on a network whose only
    linearizable layers are MultiHeadAttention (GPT2, GPTNeoX, Transformer)
    and differ on a network that also has, say, a SwiGLU -- there, layer by
    layer, "frozen_pattern" keeps the ratio treatment of the SwiGLU.
    Every mode keeps the invariant M·1 = f(x).

    Only torch is required.
"""
import unittest

import torch

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.row_computer import KnowledgeRowComputer
from knowledgematrix.neural_net import NN, MultiHeadAttention, SwiGLU
from knowledgematrix.models.gpt2 import GPT2
from knowledgematrix.models.gpt_neox import GPTNeoX
from knowledgematrix.models.transformer import Transformer

torch.set_default_dtype(torch.float64)
V, D, T, H = 40, 24, 5, 4


def small_gpt2(seed=0):
    torch.manual_seed(seed)
    m = GPT2(vocab_size=V, d_model=D, d_ff=48, num_heads=H, num_layers=2, max_len=16)
    m.eval()
    return m


def small_neox(seed=0):
    # d_head = 8, rotary_pct = 0.25 -> 2 rotary dimensions per head
    torch.manual_seed(seed)
    m = GPTNeoX(vocab_size=V, d_model=32, d_ff=48, num_heads=H, num_layers=2, max_len=16)
    m.eval()
    return m


def small_transformer(seed=0):
    torch.manual_seed(seed)
    m = Transformer(vocab_size=V, d_model=D, d_ff=48, num_heads=H, max_len=16)
    m.eval()
    return m


def mixed_net(seed=0):
    """Pre-norm blocks x + attn(ln(x)), x + swiglu(ln(x)): a
    MultiHeadAttention AND a SwiGLU, wired with the identity() idiom."""
    torch.manual_seed(seed)
    m = NN(input_shape=(1, T, D))
    m.embedding(V, D)
    causal = torch.tril(torch.ones(16, 16))
    for _ in range(2):
        start = m.get_num_layers()
        m.layernorm(D)
        m.multiheadattention(D, H, mask=causal)
        end = m.get_num_layers()
        m.identity()
        m.residual(start, end)
        start = m.get_num_layers()
        m.layernorm(D)
        m.swiglu(D, 2 * D)
        end = m.get_num_layers()
        m.identity()
        m.residual(start, end)
    m.layernorm(D)
    m.linear(D, V)
    m.eval()
    return m


ATTENTION_ONLY = (("gpt2", small_gpt2), ("gpt_neox", small_neox), ("transformer", small_transformer))


class TestSameLinearization(unittest.TestCase):
    def test_saved_pattern_is_the_frozen_routing(self):
        """The pattern recorded by the save pass is exactly the routing that
        frozen_forward freezes (rotary embedding and causal mask included)."""
        for name, make in ATTENTION_ONLY:
            m = make()
            x = torch.randint(0, V, (1, 1, T))
            m.save = True
            m(x)
            m.save = False
            idxs = [i for i, l in enumerate(m.layers) if isinstance(l, MultiHeadAttention)]
            self.assertGreater(len(idxs), 0)
            for i in idxs:
                layer = m.layers[i]
                self.assertTrue(
                    torch.equal(layer.attn_pattern, layer._attn_weights(m.pre_acts[i])),
                    f"{name} layer {i}: saved pattern is not the frozen routing")

    def test_frozen_forward_is_the_pattern_value_map(self):
        """frozen_forward equals O·(P ⊗ V x) written out from the saved
        pattern P, without (probe pass) and with (bias pass) the biases."""
        for name, make in ATTENTION_ONLY:
            m = make()
            x = torch.randint(0, V, (1, 1, T))
            m.save = True
            m(x)
            m.save = False
            i = next(j for j, l in enumerate(m.layers) if isinstance(l, MultiHeadAttention))
            layer, x0 = m.layers[i], m.pre_acts[i]
            d = layer.d_model
            P = layer.attn_pattern                       # (1, 1, H, T, T)
            B = torch.randn(3, 1, T, d)
            # probe pass: weights only
            v = (layer.V.weight @ B.transpose(-1, -2)).transpose(-1, -2)
            v = v.view(3, 1, T, layer.num_heads, layer.d_head).transpose(2, 3)
            want = (P @ v).transpose(2, 3).contiguous().view(3, 1, T, d)
            want = (layer.O.weight @ want.transpose(-1, -2)).transpose(-1, -2)
            got = layer.frozen_forward(B, x0, affine=False)
            self.assertTrue(torch.allclose(got, want, atol=1e-12), f"{name}: probe map")
            # bias pass: full affine
            v = layer.V(B).view(3, 1, T, layer.num_heads, layer.d_head).transpose(2, 3)
            want = layer.O((P @ v).transpose(2, 3).contiguous().view(3, 1, T, d))
            got = layer.frozen_forward(B, x0, affine=True)
            self.assertTrue(torch.allclose(got, want, atol=1e-12), f"{name}: affine map")
            # and at the actual input the frozen map is the layer itself
            self.assertTrue(torch.allclose(layer.frozen_forward(x0, x0), layer(x0), atol=1e-12))

    def test_modes_agree_when_attention_is_the_only_mixer(self):
        for name, make in ATTENTION_ONLY:
            m = make()
            x = torch.randint(0, V, (1, 1, T))
            out = m(x).reshape(-1)
            pattern = KnowledgeMatrixComputer(m, batch_size=16, attention_mode="frozen_pattern").forward(x)
            frozen = KnowledgeMatrixComputer(m, batch_size=16, mixer_mode="frozen").forward(x)
            ratio = KnowledgeMatrixComputer(m, batch_size=16).forward(x)
            self.assertTrue(torch.equal(pattern, frozen), f"{name}: the two frozen modes differ")
            self.assertFalse(torch.allclose(pattern, ratio), f"{name}: frozen equals ratio")
            for label, mat in (("frozen_pattern", pattern), ("frozen", frozen), ("ratio", ratio)):
                self.assertLess((mat.sum(1) - out).abs().max().item(), 1e-10, f"{name} {label} identity")

    def test_segments_agree_too(self):
        m = small_gpt2()
        x = torch.randint(0, V, (1, 1, T))
        a, b = m.block_boundaries()[:2]
        pattern = KnowledgeMatrixComputer(m, batch_size=16, attention_mode="frozen_pattern").segment(x, a, b)
        frozen = KnowledgeMatrixComputer(m, batch_size=16, mixer_mode="frozen").segment(x, a, b)
        self.assertTrue(torch.equal(pattern, frozen))


class TestWhereTheyDiffer(unittest.TestCase):
    def test_frozen_pattern_leaves_other_mixers_on_the_ratio(self):
        m = mixed_net()
        x = torch.randint(0, V, (1, 1, T))
        out = m(x).reshape(-1)
        computers = {
            "ratio": KnowledgeMatrixComputer(m, batch_size=16),
            "frozen_pattern": KnowledgeMatrixComputer(m, batch_size=16, attention_mode="frozen_pattern"),
            "frozen": KnowledgeMatrixComputer(m, batch_size=16, mixer_mode="frozen"),
        }
        mats = {k: c.forward(x) for k, c in computers.items()}
        for label, mat in mats.items():
            self.assertLess((mat.sum(1) - out).abs().max().item(), 1e-10, f"{label} identity")
        self.assertFalse(torch.allclose(mats["frozen_pattern"], mats["frozen"]))
        self.assertFalse(torch.allclose(mats["frozen_pattern"], mats["ratio"]))

        # Layer by layer: on the attention the two frozen modes apply the same
        # map; on the SwiGLU "frozen_pattern" applies the ratio (a diagonal map)
        # and "frozen" the frozen-gate linear map.
        i_att = next(j for j, l in enumerate(m.layers) if isinstance(l, MultiHeadAttention))
        i_glu = next(j for j, l in enumerate(m.layers) if isinstance(l, SwiGLU))
        B = torch.randn(3, 1, T, D)
        step = {k: c._linear_step for k, c in computers.items()}
        att, glu = m.layers[i_att], m.layers[i_glu]
        self.assertTrue(torch.equal(step["frozen_pattern"](B, i_att, att), step["frozen"](B, i_att, att)))
        self.assertTrue(torch.equal(step["frozen"](B, i_att, att),
                                    att.frozen_forward(B, m.pre_acts[i_att], affine=False)))
        self.assertTrue(torch.equal(step["frozen_pattern"](B, i_glu, glu), step["ratio"](B, i_glu, glu)))
        self.assertTrue(torch.equal(step["frozen"](B, i_glu, glu),
                                    glu.frozen_forward(B, m.pre_acts[i_glu], affine=False)))
        self.assertFalse(torch.allclose(step["frozen_pattern"](B, i_glu, glu), step["frozen"](B, i_glu, glu)))

    def test_mixer_frozen_already_includes_the_pattern_mode(self):
        m = mixed_net()
        x = torch.randint(0, V, (1, 1, T))
        both = KnowledgeMatrixComputer(m, batch_size=16, mixer_mode="frozen",
                                       attention_mode="frozen_pattern").forward(x)
        frozen = KnowledgeMatrixComputer(m, batch_size=16, mixer_mode="frozen").forward(x)
        self.assertTrue(torch.equal(both, frozen))

    def test_invalid_modes_rejected(self):
        m = small_gpt2()
        with self.assertRaises(ValueError):
            KnowledgeMatrixComputer(m, mixer_mode="frozen_pattern")
        with self.assertRaises(ValueError):
            KnowledgeMatrixComputer(m, attention_mode="frozen")


class TestMaskShapes(unittest.TestCase):
    def test_max_len_mask_is_cut_and_other_shapes_are_untouched(self):
        """The routing (forward and frozen alike) cuts a max_len x max_len
        mask to the sequence; a mask that already broadcasts is left alone."""
        torch.manual_seed(0)
        d, n = 16, 5
        x = torch.randn(1, 1, n, d)
        exact = MultiHeadAttention(d, 4, mask=torch.tril(torch.ones(n, n)))
        long = MultiHeadAttention(d, 4, mask=torch.tril(torch.ones(16, 16)))
        long.load_state_dict(exact.state_dict())
        self.assertTrue(torch.equal(exact(x), long(x)))
        self.assertTrue(torch.equal(exact.frozen_forward(x, x), long.frozen_forward(x, x)))
        self.assertTrue(torch.allclose(long.frozen_forward(x, x), long(x), atol=1e-12))
        # a 1-D key mask broadcasts over the queries
        keys = MultiHeadAttention(d, 4, mask=torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0]))
        keys.load_state_dict(exact.state_dict())
        attn = keys._attn_weights(x)
        self.assertEqual(attn[..., 2].abs().max().item(), 0.0)
        self.assertTrue(torch.allclose(attn.sum(-1), torch.ones(1, 1, 4, n), atol=1e-12))


class TestRowComputerAgreement(unittest.TestCase):
    def test_rows_match_the_matrix(self):
        """Reverse-mode rows of the GPT2 / GPTNeoX adapters (chained and
        parallel residuals) equal the rows of the forward-mode matrix."""
        for name, make in ATTENTION_ONLY:
            m = make()
            x = torch.randint(0, V, (1, 1, T))
            n_out = m(x).numel()
            rows = [0, n_out // 2, n_out - 1]
            pattern = KnowledgeMatrixComputer(m, batch_size=16, attention_mode="frozen_pattern").forward(x)
            ratio = KnowledgeMatrixComputer(m, batch_size=16).forward(x)
            got_frozen = KnowledgeRowComputer(m, mixer_mode="frozen").forward(x, rows)
            got_ratio = KnowledgeRowComputer(m, mixer_mode="ratio").forward(x, rows)
            self.assertLess((got_frozen - pattern[rows]).abs().max().item(), 1e-10, f"{name} frozen rows")
            self.assertLess((got_ratio - ratio[rows]).abs().max().item(), 1e-10, f"{name} ratio rows")


if __name__ == "__main__":
    unittest.main()
