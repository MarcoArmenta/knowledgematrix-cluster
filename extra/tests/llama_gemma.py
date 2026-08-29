#!/usr/bin/env python
"""
    Tests for the Llama/Mistral/Qwen2.5/Qwen3 and Gemma-2/3 building blocks
    (RopeAttention, GeGLU) and for the LlamaLike / GemmaLike models.

    Three things are checked, in increasing order of strength:

    1. shape and validation behaviour of the new modules,
    2. the frozen linearization is EXACT at the layer input --
       ``frozen_forward(x0, x0) == forward(x0)`` -- which is the property
       every knowledge matrix of a mixing layer rests on,
    3. the models reproduce the Hugging Face reference implementation, and
       the knowledge matrix row-sum invariant holds on the copied weights.

    The tests that compare against `transformers` are skipped when it is not
    installed (pip install transformers); the invariant tests only need torch.
"""
import unittest

import torch

from knowledgematrix.neural_net import NN, GeGLU, RopeAttention, SwiGLU
from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.row_computer import KnowledgeRowComputer
from knowledgematrix.models.llama import LlamaLike
from knowledgematrix.models.gemma import GemmaLike

try:
    from transformers.models.llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM
    from transformers.models.mistral import MistralConfig
    from transformers.models.mistral.modeling_mistral import MistralForCausalLM
    from transformers.models.qwen2 import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    from transformers.models.qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    from transformers.models.gemma2 import Gemma2Config
    from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

DEVICE = "cpu"
torch.set_default_dtype(torch.float64)

#: HF hardcodes float32 internals (the RoPE inv_freq buffer, and the
#: .to(torch.float32) inside LlamaRMSNorm.forward), so even in float64 the
#: reference carries float32 rounding. The agreement measured with tiny
#: models is ~4e-8 absolute; anything above this tolerance is a real bug,
#: not a precision artefact.
HF_TOL = 1e-6


def tiny_llama_nn(**kwargs) -> LlamaLike:
    params = dict(
        vocab_size=50,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=6,
    )
    params.update(kwargs)
    return LlamaLike(**params)


def tiny_gemma_nn(**kwargs) -> GemmaLike:
    params = dict(
        vocab_size=50,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=6,
        attn_softcap=30.0,
        sliding_window=4,
    )
    params.update(kwargs)
    return GemmaLike(**params)


def _hf_common(**kwargs) -> dict:
    params = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
    )
    params.update(kwargs)
    return params


class TestModuleValidation(unittest.TestCase):
    def test_rope_attention_head_validation(self):
        with self.assertRaises(ValueError):
            RopeAttention(d_model=32, num_heads=4, num_kv_heads=3)
        with self.assertRaises(ValueError):
            RopeAttention(d_model=32, num_heads=4, num_kv_heads=0)

    def test_rope_attention_head_dim_validation(self):
        # 32 is not divisible by 5, and no head_dim was given
        with self.assertRaises(ValueError):
            RopeAttention(d_model=32, num_heads=5)

    def test_partial_rotary_must_be_even(self):
        with self.assertRaises(ValueError):
            RopeAttention(d_model=32, num_heads=4, head_dim=8,
                          partial_rotary_factor=0.375)  # 8 * 0.375 = 3

    def test_rope_attention_shapes(self):
        attn = RopeAttention(d_model=16, num_heads=4, num_kv_heads=2, head_dim=6)
        x = torch.randn(1, 1, 7, 16)
        self.assertEqual(attn(x).shape, x.shape)

    def test_geglu_shapes(self):
        glu = GeGLU(d_model=16, d_ff=40)
        x = torch.randn(1, 1, 5, 16)
        self.assertEqual(glu(x).shape, x.shape)

    def test_geglu_is_not_swiglu(self):
        # same weights, different gate activation => different output
        torch.manual_seed(0)
        glu = GeGLU(d_model=16, d_ff=40)
        swi = SwiGLU(d_model=16, d_ff=40)
        swi.load_state_dict(glu.state_dict())
        x = torch.randn(1, 1, 5, 16)
        self.assertGreater(torch.norm(glu(x) - swi(x)).item(), 1e-6)

    def test_o_bias_is_independent_of_qkv_bias(self):
        # Qwen2.5 has q/k/v biases but no o_proj bias -- the two flags
        # must not be tied together.
        attn = RopeAttention(d_model=16, num_heads=4, head_dim=4,
                             bias=True, o_bias=False)
        self.assertIsNotNone(attn.q_proj.bias)
        self.assertIsNone(attn.o_proj.bias)


class TestFrozenExactness(unittest.TestCase):
    """
        frozen_forward(x, x0) freezes the attention weights at x0, so at
        x = x0 it must reproduce forward(x0) EXACTLY (bit for bit, not just
        to a tolerance) -- the routing is the same tensor object in both
        paths. Any deviation means the two paths compute different things
        and every knowledge matrix through the layer would be wrong.
    """

    CONFIGS = [
        ("plain", {}),
        ("qk_norm", {"qk_norm": True}),
        ("softcap", {"softcap": 30.0}),
        ("sliding_window", {"sliding_window": 3}),
        ("gqa", {"num_kv_heads": 2}),
        ("mqa", {"num_kv_heads": 1}),
        ("partial_rope", {"partial_rotary_factor": 0.5}),
        ("query_scale", {"query_scale": 12.0}),
        ("biases", {"bias": True, "o_bias": True}),
        ("non_causal", {"causal": False}),
        ("rope_float32", {"rope_float32": True}),
    ]

    def test_rope_attention_frozen_equals_forward(self):
        for name, cfg in self.CONFIGS:
            with self.subTest(config=name):
                torch.manual_seed(0)
                attn = RopeAttention(d_model=16, num_heads=4, head_dim=4, **cfg)
                x0 = torch.randn(1, 1, 6, 16)
                gap = (attn.frozen_forward(x0, x0) - attn(x0)).abs().max().item()
                self.assertEqual(gap, 0.0, f"{name}: frozen != forward, gap {gap}")

    def test_rope_attention_frozen_is_linear(self):
        # frozen_forward(., x0) with affine=False must be a genuine linear
        # map in its first argument: f(a + b) == f(a) + f(b).
        torch.manual_seed(0)
        attn = RopeAttention(d_model=16, num_heads=4, head_dim=4, bias=True, o_bias=True)
        x0 = torch.randn(1, 1, 6, 16)
        a, b = torch.randn(1, 1, 6, 16), torch.randn(1, 1, 6, 16)
        lhs = attn.frozen_forward(a + b, x0, affine=False)
        rhs = attn.frozen_forward(a, x0, affine=False) + attn.frozen_forward(b, x0, affine=False)
        self.assertLess((lhs - rhs).abs().max().item(), 1e-12)

    def test_rope_attention_frozen_mixes_positions(self):
        # The whole point of frozen mode: attribution must flow ACROSS
        # positions. Perturbing position 0 has to change the output at a
        # later position (otherwise the linearization is position-diagonal
        # and cross-token attribution is silently lost).
        torch.manual_seed(0)
        attn = RopeAttention(d_model=16, num_heads=4, head_dim=4)
        x0 = torch.randn(1, 1, 6, 16)
        e = torch.zeros_like(x0)
        e[0, 0, 0] = 1.0
        out = attn.frozen_forward(e, x0, affine=False)
        self.assertGreater(out[0, 0, 5].abs().max().item(), 1e-8)

    def test_geglu_frozen_equals_forward(self):
        for bias in (False, True):
            for approximate in ("tanh", "none"):
                with self.subTest(bias=bias, approximate=approximate):
                    torch.manual_seed(0)
                    glu = GeGLU(d_model=16, d_ff=40, bias=bias, approximate=approximate)
                    x0 = torch.randn(1, 1, 5, 16)
                    gap = (glu.frozen_forward(x0, x0) - glu(x0)).abs().max().item()
                    self.assertEqual(gap, 0.0, f"frozen != forward, gap {gap}")

    def test_sliding_window_actually_masks(self):
        # With window W a position must not attend beyond the last W keys:
        # perturbing a token further back than W leaves the output alone.
        torch.manual_seed(0)
        attn = RopeAttention(d_model=16, num_heads=4, head_dim=4, sliding_window=2)
        x0 = torch.randn(1, 1, 6, 16)
        e = torch.zeros_like(x0)
        e[0, 0, 0] = 1.0
        out = attn.frozen_forward(e, x0, affine=False)
        self.assertLess(out[0, 0, 5].abs().max().item(), 1e-12)
        self.assertGreater(out[0, 0, 0].abs().max().item(), 1e-8)

    def test_causal_mask_blocks_the_future(self):
        torch.manual_seed(0)
        attn = RopeAttention(d_model=16, num_heads=4, head_dim=4)
        x0 = torch.randn(1, 1, 6, 16)
        e = torch.zeros_like(x0)
        e[0, 0, 5] = 1.0
        out = attn.frozen_forward(e, x0, affine=False)
        self.assertLess(out[0, 0, 0].abs().max().item(), 1e-12)


class TestModelStructure(unittest.TestCase):
    def test_llama_block_stride(self):
        model = tiny_llama_nn(num_hidden_layers=3, include_lm_head=True)
        # embedding + N blocks + final norm + lm_head
        expected = 1 + 3 * LlamaLike.BLOCK_STRIDE + 2
        self.assertEqual(model.get_num_layers(), expected)

    def test_gemma_block_stride(self):
        model = tiny_gemma_nn(num_hidden_layers=2, include_lm_head=True)
        expected = 1 + 2 * GemmaLike.BLOCK_STRIDE + 2
        self.assertEqual(model.get_num_layers(), expected)

    def test_no_lm_head_ends_at_final_norm(self):
        model = tiny_llama_nn(include_lm_head=False)
        x = torch.randint(0, 50, (1, 1, 5))
        self.assertEqual(model.forward(x).shape[-1], model.hidden_size)

    def test_gemma_alternates_sliding_window(self):
        model = tiny_gemma_nn(num_hidden_layers=4, sliding_window=4,
                              alternate_sliding=True)
        windows = [l.sliding_window for l in model.layers
                   if isinstance(l, RopeAttention)]
        self.assertEqual(windows, [4, None, 4, None])

    def test_gemma_folds_the_embedding_scale(self):
        model = tiny_gemma_nn()
        self.assertAlmostEqual(model.embed_scale, model.hidden_size ** 0.5)

    def test_llama_forward_shape(self):
        model = tiny_llama_nn()
        x = torch.randint(0, 50, (1, 1, 7))
        self.assertEqual(model.forward(x).shape[-1], 50)

    def test_gemma_forward_shape(self):
        model = tiny_gemma_nn()
        x = torch.randint(0, 50, (1, 1, 7))
        self.assertEqual(model.forward(x).shape[-1], 50)


class TestKnowledgeMatrixInvariant(unittest.TestCase):
    """
        The defining property: every row of the knowledge matrix sums to the
        corresponding output coordinate of the forward pass.
    """

    def _check_matrix(self, model, x, mixer_mode="ratio", delta=1e-8):
        model.eval()
        forward_pass = model.forward(x)
        model.save = True
        computer = KnowledgeMatrixComputer(model, batch_size=16, mixer_mode=mixer_mode)
        mat = computer.forward(x)
        diff = torch.norm(forward_pass.reshape(1, -1) - mat.sum(1)).item()
        self.assertAlmostEqual(first=diff, second=0, places=None, delta=delta)

    def test_llama_matrix_invariant(self):
        torch.manual_seed(0)
        model = tiny_llama_nn().to(DEVICE)
        self._check_matrix(model, torch.randint(0, 50, (1, 1, 6)))

    def test_llama_qk_norm_matrix_invariant(self):
        torch.manual_seed(0)
        model = tiny_llama_nn(qk_norm=True, qkv_bias=True).to(DEVICE)
        self._check_matrix(model, torch.randint(0, 50, (1, 1, 6)))

    def test_llama_sliding_window_matrix_invariant(self):
        torch.manual_seed(0)
        model = tiny_llama_nn(sliding_window=3).to(DEVICE)
        self._check_matrix(model, torch.randint(0, 50, (1, 1, 6)))

    def test_gemma_matrix_invariant(self):
        torch.manual_seed(0)
        model = tiny_gemma_nn().to(DEVICE)
        self._check_matrix(model, torch.randint(0, 50, (1, 1, 6)))

    def test_llama_frozen_matrix_invariant(self):
        torch.manual_seed(0)
        model = tiny_llama_nn().to(DEVICE)
        self._check_matrix(model, torch.randint(0, 50, (1, 1, 6)),
                           mixer_mode="frozen")

    def test_gemma_frozen_matrix_invariant(self):
        torch.manual_seed(0)
        model = tiny_gemma_nn().to(DEVICE)
        self._check_matrix(model, torch.randint(0, 50, (1, 1, 6)),
                           mixer_mode="frozen")

    def test_llama_row_invariant(self):
        torch.manual_seed(0)
        model = tiny_llama_nn().to(DEVICE)
        model.eval()
        x = torch.randint(0, 50, (1, 1, 6))
        out = model.forward(x).reshape(-1)
        rows = [3, 17, int(out.shape[0]) - 1]
        computer = KnowledgeRowComputer(model, mixer_mode="frozen")
        A = computer.forward(x, rows)
        for k, j in enumerate(rows):
            self.assertAlmostEqual(A[k].sum().item(), out[j].item(),
                                   places=None, delta=1e-8)

    def test_gemma_row_invariant(self):
        torch.manual_seed(0)
        model = tiny_gemma_nn().to(DEVICE)
        model.eval()
        x = torch.randint(0, 50, (1, 1, 6))
        out = model.forward(x).reshape(-1)
        rows = [1, 42]
        computer = KnowledgeRowComputer(model, mixer_mode="frozen")
        A = computer.forward(x, rows)
        for k, j in enumerate(rows):
            self.assertAlmostEqual(A[k].sum().item(), out[j].item(),
                                   places=None, delta=1e-8)

    def test_rows_agree_with_full_matrix(self):
        # The row computer and the matrix computer must be the same object
        # seen from two directions -- under the SAME mixer_mode (the two
        # classes have different defaults: "frozen" and "ratio").
        for mode in ("frozen", "ratio"):
            with self.subTest(mixer_mode=mode):
                torch.manual_seed(0)
                model = tiny_llama_nn(num_hidden_layers=2, vocab_size=20).to(DEVICE)
                model.eval()
                x = torch.randint(0, 20, (1, 1, 5))
                model.forward(x)
                model.save = True
                mat = KnowledgeMatrixComputer(
                    model, batch_size=16, mixer_mode=mode).forward(x)
                rows = [0, 7, 33]
                A = KnowledgeRowComputer(model, mixer_mode=mode).forward(x, rows)
                for k, j in enumerate(rows):
                    # the matrix computer keeps the bias in its own column too
                    gap = (A[k] - mat[j]).abs().max().item()
                    self.assertLess(gap, 1e-8, f"row {j} differs by {gap}")


@unittest.skipUnless(HAS_TRANSFORMERS, "transformers is not installed")
class TestHuggingFaceAgreement(unittest.TestCase):
    """
        Weight-map round trip: build a tiny reference model with the actual
        Hugging Face implementation, copy its weights in, and require the
        two to compute the same function. This is what catches a permuted
        RoPE convention, a missed q/k norm, a wrong RMSNorm centering, or a
        forgotten embedding scale -- none of which the invariant tests see,
        because a wrongly-wired network is still self-consistent.
    """

    def _compare(self, hf, nn_model, seq_len=7, vocab=64):
        hf = hf.double().eval()
        nn_model = nn_model.to(DEVICE)
        nn_model.eval()
        ids = torch.randint(0, vocab, (1, seq_len))
        with torch.no_grad():
            ref = hf(ids).logits
            got = nn_model.forward(ids.unsqueeze(0))
        gap = (ref.reshape(-1) - got.reshape(-1)).abs().max().item()
        self.assertLess(gap, HF_TOL, f"HF and KM differ by {gap}")
        return gap

    def test_llama(self):
        torch.manual_seed(0)
        hf = LlamaForCausalLM(LlamaConfig(**_hf_common())).double().eval()
        self._compare(hf, LlamaLike.from_huggingface(hf, dtype=torch.float64))

    def test_mistral_sliding_window(self):
        torch.manual_seed(1)
        cfg = MistralConfig(**_hf_common(sliding_window=4))
        hf = MistralForCausalLM(cfg).double().eval()
        model = LlamaLike.from_huggingface(hf, dtype=torch.float64)
        self.assertEqual(
            [l.sliding_window for l in model.layers if isinstance(l, RopeAttention)],
            [4, 4]
        )
        self._compare(hf, model)

    def test_qwen2_has_qkv_bias_but_no_o_bias(self):
        torch.manual_seed(2)
        hf = Qwen2ForCausalLM(Qwen2Config(**_hf_common())).double().eval()
        model = LlamaLike.from_huggingface(hf, dtype=torch.float64)
        attn = [l for l in model.layers if isinstance(l, RopeAttention)][0]
        self.assertIsNotNone(attn.q_proj.bias)
        self.assertIsNone(attn.o_proj.bias)
        self._compare(hf, model)

    def test_qwen3_qk_norm_is_detected(self):
        torch.manual_seed(3)
        hf = Qwen3ForCausalLM(Qwen3Config(**_hf_common())).double().eval()
        model = LlamaLike.from_huggingface(hf, dtype=torch.float64)
        attn = [l for l in model.layers if isinstance(l, RopeAttention)][0]
        self.assertTrue(attn.use_qk_norm)
        self._compare(hf, model)

    def test_gemma2(self):
        torch.manual_seed(4)
        cfg = Gemma2Config(**_hf_common(
            sliding_window=4,
            attn_logit_softcapping=50.0,
            query_pre_attn_scalar=8,
        ))
        cfg.final_logit_softcapping = None   # applied outside the network
        hf = Gemma2ForCausalLM(cfg).double().eval()
        self._compare(hf, GemmaLike.from_huggingface(hf, dtype=torch.float64))

    def test_llama_matrix_invariant_on_hf_weights(self):
        torch.manual_seed(5)
        hf = LlamaForCausalLM(LlamaConfig(**_hf_common())).double().eval()
        model = LlamaLike.from_huggingface(hf, dtype=torch.float64).to(DEVICE)
        model.eval()

        x = torch.randint(0, 64, (1, 1, 6))
        forward_pass = model.forward(x)
        model.save = True
        mat = KnowledgeMatrixComputer(model, batch_size=16).forward(x)
        diff = torch.norm(forward_pass.reshape(1, -1) - mat.sum(1)).item()
        self.assertAlmostEqual(first=diff, second=0, places=None, delta=1e-8)

    def test_gemma_matrix_invariant_on_hf_weights(self):
        torch.manual_seed(6)
        cfg = Gemma2Config(**_hf_common(sliding_window=4,
                                        attn_logit_softcapping=50.0))
        cfg.final_logit_softcapping = None
        hf = Gemma2ForCausalLM(cfg).double().eval()
        model = GemmaLike.from_huggingface(hf, dtype=torch.float64).to(DEVICE)
        model.eval()

        x = torch.randint(0, 64, (1, 1, 6))
        forward_pass = model.forward(x)
        model.save = True
        mat = KnowledgeMatrixComputer(model, batch_size=16).forward(x)
        diff = torch.norm(forward_pass.reshape(1, -1) - mat.sum(1)).item()
        self.assertAlmostEqual(first=diff, second=0, places=None, delta=1e-8)

    def test_hidden_states_agree_without_lm_head(self):
        # The alignment pipeline runs with include_lm_head=False and applies
        # the head itself, so the headless network has to match HF's final
        # hidden state, not just the logits.
        torch.manual_seed(7)
        hf = LlamaForCausalLM(LlamaConfig(**_hf_common())).double().eval()
        model = LlamaLike.from_huggingface(
            hf, include_lm_head=False, dtype=torch.float64).to(DEVICE)
        model.eval()
        ids = torch.randint(0, 64, (1, 7))
        with torch.no_grad():
            ref = hf.model(ids).last_hidden_state
            got = model.forward(ids.unsqueeze(0))
        gap = (ref.reshape(-1) - got.reshape(-1)).abs().max().item()
        self.assertLess(gap, HF_TOL, f"hidden states differ by {gap}")


if __name__ == "__main__":
    unittest.main()
