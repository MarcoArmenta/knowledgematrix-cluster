#!/usr/bin/env python
"""
    Tests for the Qwen3.5 building blocks (GatedDeltaNet, GatedAttention,
    SwiGLU) and the Qwen3_5 model.

    A rebuilt network is always self-consistent (its knowledge matrix
    rows sum to ITS forward pass), so only a comparison with the Hugging
    Face reference catches a wiring error. The comparison is made on the
    PUBLISHED architectures -- extra/tests/fixtures holds the config.json
    of Qwen3.5-4B / 27B (hf_configs) and 0.8B / 2B / 9B (qwen3_5_siblings)
    -- with the widths scaled down and random weights, and at full size
    without weights (the "meta" device).

    Two forms of the reference are used:

    - as shipped. transformers computes its normalizations, RoPE tables
      and delta rule in float32 whatever the dtype of the model, so a
      float64 model agrees with exact arithmetic only to ~1e-5 on these
      random networks (bound TOL_REF32);
    - `reference_in_float64`: the same code with those hard-coded
      float32 round trips redirected to float64. Only dtypes change, and
      our float64 network then agrees to ~1e-14 (bound TOL_EXACT). This is
      the sharp test: it resolves an epsilon in the wrong place, which
      float32 noise would hide.

    The tests that compare against `transformers` are skipped when it is
    not installed (pip install transformers). The knowledge matrix
    invariant tests on hand-built networks only need torch. Nothing is
    downloaded: the tests on real checkpoints run on what is already in
    the local Hugging Face cache and are skipped otherwise; the ones that
    read gigabytes are opt-in (KM_SLOW_TESTS=1).
"""
import contextlib
import copy
import gc
import json
import os
import random
import tempfile
import types
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from knowledgematrix.neural_net import NN, GatedAttention, GatedDeltaNet, RMSNorm, SwiGLU
from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.row_computer import KnowledgeRowComputer
from knowledgematrix.models.qwen3_5 import Qwen3_5

try:
    import safetensors
    import safetensors.torch
    import transformers
    from transformers.models.qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as REF
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForCausalLM,
        Qwen3_5ForConditionalGeneration,
        Qwen3_5TextModel,
    )
    from knowledgematrix.models import _hf
    HAS_TRANSFORMERS = True
    transformers.utils.logging.disable_progress_bar()   # save_pretrained
except ImportError:
    HAS_TRANSFORMERS = False

DEVICE = "cpu"
torch.set_default_dtype(torch.float64)

_FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
#: the published dense Qwen3.5 checkpoints: name -> (repository, parameters
#: of the text decoder without the LM head)
PUBLISHED = {
    "0.8B": ("Qwen/Qwen3.5-0.8B", 752_393_024),
    "2B": ("Qwen/Qwen3.5-2B", 1_881_825_088),
    "4B": ("Qwen/Qwen3.5-4B", 4_205_751_296),
    "9B": ("Qwen/Qwen3.5-9B", 7_936_684_544),
    "27B": ("Qwen/Qwen3.5-27B", 25_624_600_064),
}

#: max |ours - reference| / max |reference| allowed against the reference
#: run in true float64 (measured: <= 8e-14 up to 200 tokens)
TOL_EXACT = 1e-10
#: ... and wherever float32 is involved: our float64 network against the
#: reference as shipped (float32 inside; measured <= 7e-5), or a float32
#: network against anything (measured <= 9e-5). This is float32 round-off
#: amplified by eight random layers, not a difference of function.
TOL_REF32 = 5e-4
#: a deliberately wrong network must miss by at least this much
#: (measured: >= 0.17), i.e. 8 orders of magnitude above TOL_EXACT
MISS = 1e-2

SLOW = os.environ.get("KM_SLOW_TESTS") == "1"


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


### Helpers for the comparison with transformers ###

def published_json(name: str) -> dict:
    """
        The config.json of Qwen/Qwen3.5-<name>, as published: the 4B and
        27B from the fixtures of the supported checkpoints, their siblings
        from fixtures/qwen3_5_siblings.
    """
    for directory in ("hf_configs", "qwen3_5_siblings"):
        path = os.path.join(_FIXTURES, directory, f"Qwen__Qwen3.5-{name}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError(f"no config fixture for Qwen3.5-{name}")


def published_config(name: str):
    """The transformers config of a published checkpoint, at full size."""
    with tempfile.TemporaryDirectory() as d:
        raw = published_json(name)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(raw, f)
        return transformers.AutoConfig.from_pretrained(d)


def scaled_config(name: str, real_head_dim: bool = True, layers: int = 8, **text_overrides):
    """
        The published config `name` with only the sizes scaled down.

        Kept from the real config: the layer pattern (3 linear + 1 full,
        repeated), the RoPE parameters (base 1e7, partial rotary factor
        0.25, the mrope sections), query heads : KV heads, linear value
        heads : key heads (1 for the 0.8B / 2B, 2 for the 4B / 9B, 3 for
        the 27B), heads * head_dim != hidden_size, the tied or untied
        head, and every flag (attn_output_gate, mlp_only_layers, ...).
        With real_head_dim the attention head stays 256 wide, so the
        rotated slice (64) and the mrope section layout are the real ones.
    """
    raw = published_json(name)
    t = raw["text_config"]
    kv_ratio = t["num_attention_heads"] // t["num_key_value_heads"]
    v_ratio = t["linear_num_value_heads"] // t["linear_num_key_heads"]
    t.update(
        vocab_size=96,
        hidden_size=40,
        intermediate_size=72,
        num_hidden_layers=layers,
        layer_types=t["layer_types"][:layers],
        num_key_value_heads=2,
        num_attention_heads=2 * kv_ratio,
        linear_num_key_heads=2,
        linear_num_value_heads=2 * v_ratio,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        max_position_embeddings=512,
        eos_token_id=1,
    )
    if not real_head_dim:
        t["head_dim"] = 16
    t.update(text_overrides)
    raw["vision_config"].update(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                                out_hidden_size=t["hidden_size"], num_position_embeddings=16)
    raw.update(image_token_id=92, video_token_id=93, vision_start_token_id=94, vision_end_token_id=95)
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(raw, f)
        return transformers.AutoConfig.from_pretrained(d)


def randomize(model, seed: int = 0):
    """
        Random weights at a scale where every nonlinearity is exercised.
        The default initialization (std 0.02, norm weights 0, dt_bias 1)
        leaves the gates near-linear and the zero-centered norms at their
        fixed point; the ranges of A_log / dt_bias are those of the
        published Qwen3.5-4B tensors.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("A_log"):
                p.uniform_(-4.2, 4.3, generator=g)
            elif name.endswith("dt_bias"):
                p.normal_(-2.0, 3.0, generator=g)
            elif "linear_attn.norm" in name:          # plain weight, around 1
                p.normal_(0.95, 0.3, generator=g)
            elif "norm" in name and p.dim() == 1:     # zero-centered weight
                p.normal_(0.1, 0.3, generator=g)
            elif "embed_tokens" in name:
                p.normal_(0.0, 1.0, generator=g)
            elif p.dim() >= 2:
                p.normal_(0.0, 1.5 / p[0].numel() ** 0.5, generator=g)
            else:                                     # biases
                p.normal_(0.0, 0.5, generator=g)
    return model


def build_reference(cls, config, dtype=torch.float64, seed: int = 1):
    """A randomized transformers model of class `cls` on a scaled config."""
    cfg = config.text_config if cls is not Qwen3_5ForConditionalGeneration else config
    return randomize(cls(cfg), seed).to(dtype).eval()


class _Float64Torch:
    """The `torch` module with float32 spelled float64 (see below)."""
    float32 = torch.float64
    float = torch.float64

    def __getattr__(self, name):
        return getattr(torch, name)


@contextlib.contextmanager
def reference_in_float64(hf=None):
    """
        Run the transformers reference in true float64.

        Its RMSNorm (`x.float()`), gated norm, RoPE tables, softplus gate
        and delta rule (`.to(torch.float32)`) round-trip through float32
        whatever the model dtype. Inside this context the name `torch`
        seen by modeling_qwen3_5 maps float32 to float64, `Tensor.float()`
        is `Tensor.double()`, and the RoPE frequencies of `hf` are
        recomputed (by the reference's own function) in float64. No line
        of the reference is replaced: only the precision it runs in.
    """
    saved_torch, saved_float = REF.torch, torch.Tensor.float
    rotaries = [] if hf is None else [
        m for m in hf.modules() if isinstance(m, REF.Qwen3_5TextRotaryEmbedding)]
    saved_freqs = [m.inv_freq for m in rotaries]
    REF.torch = _Float64Torch()
    torch.Tensor.float = lambda self, *args, **kwargs: self.to(torch.float64)
    try:
        for m in rotaries:
            m.inv_freq, _ = m.compute_default_rope_parameters(m.config)
        yield
    finally:
        REF.torch, torch.Tensor.float = saved_torch, saved_float
        for m, inv_freq in zip(rotaries, saved_freqs):
            m.inv_freq = inv_freq


def token_ids(length: int, vocab: int = 90, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(1000 * seed + length)
    return torch.randint(0, vocab, (1, length), generator=g)


def reference_logits(hf, ids, **kwargs):
    with torch.no_grad():
        return hf(input_ids=ids, **kwargs).logits


def reference_hidden(hf, ids, **kwargs):
    """Final hidden states (after the last norm), without the LM head."""
    base = hf.model if hasattr(hf, "model") else hf
    with torch.no_grad():
        return base(input_ids=ids, **kwargs).last_hidden_state


def ours(net, ids):
    net.eval()
    with torch.no_grad():
        return net.forward(ids.unsqueeze(0))[0]      # (1, T, V or D)


def rel_gap(a: torch.Tensor, b: torch.Tensor) -> float:
    """max |a - b| relative to the scale of b."""
    return ((a.double() - b.double()).abs().max() / b.double().abs().max()).item()


def write_published_layout(hf, directory: str) -> dict:
    """
        Save `hf` the way the published repositories are laid out: two
        safetensors shards behind a model.safetensors.index.json, with the
        tensors of a multi-token-prediction head (`mtp.*`, which no
        transformers class loads) next to the decoder. Returns the tensors.
    """
    hf.save_pretrained(directory)
    single = os.path.join(directory, "model.safetensors")
    tensors = safetensors.torch.load_file(single)
    os.remove(single)
    D = hf.config.text_config.hidden_size if hasattr(hf.config, "text_config") else hf.config.hidden_size
    g = torch.Generator().manual_seed(7)
    tensors["mtp.fc.weight"] = torch.randn(D, 2 * D, generator=g)
    tensors["mtp.norm.weight"] = torch.randn(D, generator=g)
    tensors["mtp.layers.0.input_layernorm.weight"] = torch.randn(D, generator=g)
    tensors["mtp.layers.0.self_attn.q_proj.weight"] = torch.randn(4 * D, D, generator=g)
    tensors["mtp.layers.0.mlp.gate_proj.weight"] = torch.randn(2 * D, D, generator=g)
    write_shards(tensors, directory)
    return tensors


def write_shards(tensors: dict, directory: str) -> None:
    keys = sorted(tensors)
    half = len(keys) // 2
    weight_map = {}
    for n, part in enumerate((keys[:half], keys[half:]), start=1):
        name = f"model.safetensors-{n:05d}-of-00002.safetensors"
        safetensors.torch.save_file({k: tensors[k] for k in part}, os.path.join(directory, name))
        weight_map.update({k: name for k in part})
    with open(os.path.join(directory, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)


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

    def test_block_layout(self):
        """embedding, then BLOCK_STRIDE layers per block, final norm, head."""
        model = tiny_nn_model()
        self.assertEqual(model.get_num_layers(), 1 + 4 * Qwen3_5.BLOCK_STRIDE + 2)
        self.assertIsInstance(model.layers[1], RMSNorm)
        self.assertIsInstance(model.layers[2], GatedDeltaNet)
        self.assertIsInstance(model.layers[5], SwiGLU)
        self.assertIsInstance(model.layers[1 + 3 * Qwen3_5.BLOCK_STRIDE + 1], GatedAttention)
        headless = tiny_nn_model(include_lm_head=False)
        self.assertEqual(headless.get_num_layers(), model.get_num_layers() - 1)
        self.assertIsInstance(headless.layers[-1], RMSNorm)

    def test_construction_evaluates_nothing(self):
        """
            The residuals are registered without shape probes, so the
            network can be built without storage (the "meta" device), which
            is how from_huggingface builds a 27B-parameter model for free.
        """
        with mock.patch.object(NN, "shape_at_layer", side_effect=AssertionError("probed")):
            model = tiny_nn_model()
        self.assertEqual(len(model.residuals), 2 * 4)
        with torch.device("meta"):
            empty = tiny_nn_model(num_hidden_layers=8)
        self.assertTrue(all(p.is_meta for p in empty.parameters()))

    def test_attention_bias_keyword(self):
        model = tiny_nn_model(attention_bias=True)
        attention = [m for m in model.layers if isinstance(m, GatedAttention)]
        self.assertEqual(len(attention), 1)
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            self.assertIsNotNone(getattr(attention[0], proj).bias)
        self.assertIsNone(tiny_nn_model().layers[1 + 3 * Qwen3_5.BLOCK_STRIDE + 1].q_proj.bias)

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


@unittest.skipUnless(HAS_TRANSFORMERS, "requires the optional dependency transformers")
class TestReferenceComponents(unittest.TestCase):
    """
        The pieces of the reference that depend on the config, one at a
        time: the multimodal rotary table on text positions, and the
        chunked delta rule against our recurrence.
    """

    def _reference_table(self, config, position_ids, dtype):
        rotary = REF.Qwen3_5TextRotaryEmbedding(config=config)
        x = torch.zeros(1, position_ids.shape[-1], 1, dtype=dtype)
        if dtype == torch.float64:
            with reference_in_float64(rotary):
                cos, sin = rotary(x, position_ids)
        else:
            cos, sin = rotary(x, position_ids)
        return cos[0], sin[0]

    def _text_positions(self, T):
        # Qwen3_5TextModel.forward for text: one arange on four axes
        # (text, temporal, height, width); the last three reach the table.
        return torch.arange(T).view(1, 1, -1).expand(4, 1, -1)[1:]

    def test_text_rope_is_the_plain_table(self):
        """
            Published settings (head 256, factor 0.25, base 1e7, interleaved
            sections [11, 11, 10]): for text the three position axes are
            equal, and the interleaved M-RoPE table is the 1-D table of
            GatedAttention -- bit for bit in float32, where both evaluate
            float32(position) * float32(frequency).
        """
        for name in ("4B", "27B"):
            config = published_config(name).text_config
            rope = config.rope_parameters
            self.assertEqual((rope["rope_theta"], rope["partial_rotary_factor"]), (10_000_000, 0.25))
            attention = GatedAttention(d_model=8, num_heads=1, head_dim=config.head_dim,
                                       rope_theta=rope["rope_theta"],
                                       partial_rotary_factor=rope["partial_rotary_factor"])
            self.assertEqual(attention.rotary_dim, 64)
            self.assertEqual(2 * sum(rope["mrope_section"]), attention.rotary_dim)
            T = 3000
            for dtype in (torch.float32, torch.float64):
                with self.subTest(config=name, dtype=dtype):
                    ref_cos, ref_sin = self._reference_table(config, self._text_positions(T), dtype)
                    cos, sin = attention._rope(T, "cpu", dtype)
                    self.assertEqual(cos.shape, ref_cos.shape)
                    if dtype == torch.float32:
                        self.assertTrue(torch.equal(cos, ref_cos) and torch.equal(sin, ref_sin))
                    else:
                        self.assertLess((cos - ref_cos).abs().max().item(), 1e-12)
                        self.assertLess((sin - ref_sin).abs().max().item(), 1e-12)

    def test_the_reduction_needs_equal_axes(self):
        """
            Negative control of the above, and the limit of this class:
            on image tokens (height / width positions that differ from the
            temporal one) the interleaved table is NOT the 1-D table.
        """
        config = published_config("4B").text_config
        T = 50
        positions = self._text_positions(T).clone()
        positions[1] = torch.arange(T).flip(0)            # another height axis
        ref_cos, _ = self._reference_table(config, positions, torch.float64)
        attention = GatedAttention(d_model=8, num_heads=1, head_dim=256, rope_theta=1e7)
        cos, _ = attention._rope(T, "cpu", torch.float64)
        self.assertGreater((cos - ref_cos).abs().max().item(), 0.5)

    @staticmethod
    def _delta_inputs(T, seed, H=6, dk=16, dv=16):
        """Recurrence inputs in the regime of the published decay tensors."""
        g = torch.Generator().manual_seed(seed)
        q = torch.randn(1, T, H, dk, generator=g)
        k = torch.randn(1, T, H, dk, generator=g)
        v = torch.randn(1, T, H, dv, generator=g)
        A = torch.exp(torch.empty(H).uniform_(-4.2, 4.3, generator=g))
        decay = -A * F.softplus(3.0 * torch.randn(1, T, H, generator=g) - 2.0)
        beta = torch.sigmoid(2.0 * torch.randn(1, T, H, generator=g))
        return q, k, v, decay, beta

    def test_chunked_delta_rule_is_our_recurrence(self):
        """
            The reference evaluates the delta rule 64 tokens at a time,
            in a different algebraic form (a triangular solve inside each
            chunk); ours is the step-by-step recurrence. Same function:
            1e-12 in float64 on sequences shorter than, equal to, longer
            than and not a multiple of the chunk.
        """
        layer = GatedDeltaNet(d_model=8, num_v_heads=6, num_k_heads=6, head_k_dim=16, head_v_dim=16)
        for T in (1, 5, 63, 64, 65, 70, 128, 130, 200):
            with self.subTest(T=T):
                q, k, v, decay, beta = self._delta_inputs(T, seed=T)
                exact = layer._delta_rule(q, k, v, decay, beta, torch.float64)
                with reference_in_float64():
                    chunked, _ = REF.torch_chunk_gated_delta_rule(
                        q, k, v, g=decay, beta=beta, use_qk_l2norm_in_kernel=True)
                self.assertEqual(chunked.dtype, torch.float64)
                self.assertLess(rel_gap(exact, chunked), 1e-12)

    def test_float32_delta_rule(self):
        """
            In float32 (the dtype the reference always uses here) our
            recurrence is bit-identical to the reference's own recurrent
            kernel (the one it uses for single-token decoding) and stays
            at float32 round-off of the exact result (measured <= 4e-7),
            while the chunked kernel, which takes differences of
            cumulative decays, loses digits on long sequences (measured
            up to 4e-5 with decays in the published range).
        """
        layer = GatedDeltaNet(d_model=8, num_v_heads=6, num_k_heads=6, head_k_dim=16, head_v_dim=16)
        for T in (5, 64, 70, 130, 200):
            with self.subTest(T=T):
                inputs = self._delta_inputs(T, seed=T)
                exact = layer._delta_rule(*inputs, torch.float64)
                q, k, v, decay, beta = [t.float() for t in inputs]
                mine = layer._delta_rule(q, k, v, decay, beta, torch.float32)
                recurrent, _ = REF.torch_recurrent_gated_delta_rule(
                    q, k, v, g=decay, beta=beta, initial_state=None,
                    output_final_state=False, use_qk_l2norm_in_kernel=True)
                chunked, _ = REF.torch_chunk_gated_delta_rule(
                    q, k, v, g=decay, beta=beta, use_qk_l2norm_in_kernel=True)
                self.assertEqual(mine.dtype, torch.float32)
                self.assertTrue(torch.equal(mine, recurrent))
                self.assertLess(rel_gap(mine, exact), 2e-6)
                self.assertLess(rel_gap(chunked, exact), TOL_REF32)
                self.assertLess(rel_gap(mine, chunked), TOL_REF32)

    def test_the_patched_reference_is_the_reference(self):
        """
            reference_in_float64 changes precision, not the function: the
            patched and the shipped reference agree to float32 round-off,
            and the patch is undone on exit.
        """
        hf = build_reference(Qwen3_5ForConditionalGeneration, scaled_config("4B"))
        ids = token_ids(70)
        shipped = reference_logits(hf, ids)
        rotary = hf.model.language_model.rotary_emb
        rounded = rotary.inv_freq          # float32 values (in a float64 buffer)
        self.assertTrue(torch.equal(rounded, rounded.to(torch.float32).to(torch.float64)))
        with reference_in_float64(hf):
            self.assertFalse(torch.equal(rotary.inv_freq, rounded))
            self.assertLess((rotary.inv_freq - rounded).abs().max().item(), 1e-7)
            exact = reference_logits(hf, ids)
        gap = rel_gap(shipped, exact)
        self.assertLess(gap, TOL_REF32)
        self.assertGreater(gap, 1e-9)   # ... and it does change the precision
        self.assertEqual(torch.ones(1).float().dtype, torch.float32)
        self.assertIs(REF.torch, torch)
        self.assertIs(rotary.inv_freq, rounded)
        self.assertTrue(torch.equal(reference_logits(hf, ids), shipped))


@unittest.skipUnless(HAS_TRANSFORMERS, "requires the optional dependency transformers")
class TestPublishedConfigParity(unittest.TestCase):
    """
        Qwen3_5.from_huggingface against the real transformers classes on
        the published architectures (sizes scaled down, random float64
        weights): logits with the head, final hidden states without,
        loaded from the model in memory and streamed from disk, on
        sequences up to several delta-rule chunks (64 tokens each).
    """

    LENGTHS = (1, 7, 70, 130)

    def _compare(self, hf, source, lengths, label):
        with_head = Qwen3_5.from_huggingface(source, dtype=torch.float64)
        headless = Qwen3_5.from_huggingface(source, dtype=torch.float64, include_lm_head=False)
        self.assertEqual(with_head.layers[0].weight.dtype, torch.float64)
        for T in lengths:
            ids = token_ids(T)
            with reference_in_float64(hf):
                exact_logits = reference_logits(hf, ids)
                exact_hidden = reference_hidden(hf, ids)
            logits, hidden = ours(with_head, ids), ours(headless, ids)
            self.assertEqual(logits.shape, exact_logits.shape)
            self.assertEqual(hidden.shape, exact_hidden.shape)
            for what, gap in (("logits", rel_gap(logits, exact_logits)),
                              ("hidden states", rel_gap(hidden, exact_hidden))):
                self.assertLess(gap, TOL_EXACT, f"{label}, T={T}: {what} differ by {gap:.2e}")
            # the reference as shipped (float32 inside), with and without
            # its cache (which changes how the convolution is padded)
            for use_cache in (True, False):
                gap = rel_gap(logits, reference_logits(hf, ids, use_cache=use_cache))
                self.assertLess(gap, TOL_REF32, f"{label}, T={T}, use_cache={use_cache}: {gap:.2e}")

    def _run(self, name, cls):
        config = scaled_config(name)
        text = config.text_config
        full = published_config(name).text_config
        # the scaling kept what distinguishes the published architecture
        self.assertEqual(text.num_hidden_layers, 8)
        self.assertEqual(text.layer_types, (["linear_attention"] * 3 + ["full_attention"]) * 2)
        self.assertEqual(text.rope_parameters, full.rope_parameters)
        self.assertEqual(text.head_dim, 256)
        self.assertNotEqual(text.num_attention_heads * text.head_dim, text.hidden_size)
        self.assertEqual(text.linear_num_value_heads * full.linear_num_key_heads,
                         full.linear_num_value_heads * text.linear_num_key_heads)
        self.assertEqual(text.num_attention_heads * full.num_key_value_heads,
                         full.num_attention_heads * text.num_key_value_heads)
        self.assertEqual(config.tie_word_embeddings, published_config(name).tie_word_embeddings)

        hf = build_reference(cls, config)
        tied = hf.lm_head.weight is hf.get_input_embeddings().weight
        self.assertEqual(tied, name in ("0.8B", "2B", "4B"))
        self._compare(hf, hf, self.LENGTHS, f"{name} {cls.__name__} in memory")
        with tempfile.TemporaryDirectory() as d:
            hf.save_pretrained(d)
            with safetensors.safe_open(os.path.join(d, "model.safetensors"), framework="pt") as f:
                stored = set(f.keys())
            # a tied head is not stored: the loader must use the embedding
            self.assertEqual("lm_head.weight" in stored, not tied)
            self._compare(hf, d, (7, 130), f"{name} {cls.__name__} from disk")

    def test_4b_causal_lm(self):
        self._run("4B", Qwen3_5ForCausalLM)

    def test_4b_conditional_generation(self):
        self._run("4B", Qwen3_5ForConditionalGeneration)

    def test_27b_causal_lm(self):
        self._run("27B", Qwen3_5ForCausalLM)

    def test_27b_conditional_generation(self):
        self._run("27B", Qwen3_5ForConditionalGeneration)

    def test_siblings(self):
        """The 0.8B / 2B (one value head per key head, tied) and the 9B."""
        for name in ("0.8B", "2B", "9B"):
            with self.subTest(config=name):
                config = scaled_config(name)
                ratio = config.text_config.linear_num_value_heads // config.text_config.linear_num_key_heads
                self.assertEqual(ratio, 1 if name in ("0.8B", "2B") else 2)
                hf = build_reference(Qwen3_5ForConditionalGeneration, config)
                self._compare(hf, hf, (7, 70), f"{name} in memory")

    def test_published_layout_on_disk(self):
        """
            The layout of the real repositories: sharded safetensors behind
            an index, the decoder under model.language_model., no stored
            head when tied, and the vision tower and the mtp head next to
            it -- which must be ignored.
        """
        for name in ("4B", "27B"):
            with self.subTest(config=name):
                hf = build_reference(Qwen3_5ForConditionalGeneration, scaled_config(name))
                with tempfile.TemporaryDirectory() as d:
                    tensors = write_published_layout(hf, d)
                    self.assertTrue(any(k.startswith("model.visual.") for k in tensors))
                    self.assertTrue(any(k.startswith("mtp.layers.0.") for k in tensors))
                    self.assertIn("model.language_model.embed_tokens.weight", tensors)
                    self.assertEqual("lm_head.weight" in tensors, name == "27B")
                    self.assertFalse(os.path.exists(os.path.join(d, "model.safetensors")))
                    self._compare(hf, d, (7, 70), f"{name} published layout")

    def test_attention_bias(self):
        """attention_bias is false in every published config; true works too."""
        hf = build_reference(Qwen3_5ForCausalLM, scaled_config("4B", attention_bias=True))
        self.assertIsNotNone(hf.model.layers[3].self_attn.q_proj.bias)
        self._compare(hf, hf, (7, 70), "attention_bias")
        with tempfile.TemporaryDirectory() as d:
            hf.save_pretrained(d)
            self._compare(hf, d, (7,), "attention_bias from disk")

    def test_config_values_are_read_not_assumed(self):
        """
            Every architecture value that is the same in all published
            configs, set to something else: the loader must follow the
            config (another RoPE base and rotated fraction, epsilon,
            convolution kernel, key head width different from the value
            head width, another layer pattern).
        """
        rope = {"rope_type": "default", "rope_theta": 50_000.0, "partial_rotary_factor": 0.5,
                "mrope_section": [3, 3, 2], "mrope_interleaved": True}
        config = scaled_config(
            "27B", real_head_dim=False, layers=6, head_dim=24, rope_parameters=rope, rms_norm_eps=1e-4,
            linear_conv_kernel_dim=3, linear_key_head_dim=12, linear_value_head_dim=20,
            layer_types=["full_attention", "linear_attention", "linear_attention",
                         "full_attention", "full_attention", "linear_attention"])
        hf = build_reference(Qwen3_5ForConditionalGeneration, config)
        net = Qwen3_5.from_huggingface(hf)
        attention, delta = net.layers[2], net.layers[8]
        self.assertEqual((attention.rope_theta, attention.rotary_dim), (50_000.0, 12))
        self.assertEqual((delta.conv_kernel_size, delta.head_k_dim, delta.head_v_dim, delta.norm.eps),
                         (3, 12, 20, 1e-4))
        self.assertEqual(net.layers[1].eps, 1e-4)
        self.assertEqual(net.layer_types, config.text_config.layer_types)
        self._compare(hf, hf, (7, 70), "non-default config values")

    def test_bare_text_model(self):
        """A Qwen3_5TextModel has no LM head: hidden states only."""
        hf = build_reference(Qwen3_5TextModel, scaled_config("27B"))
        net = Qwen3_5.from_huggingface(hf, include_lm_head=False)
        ids = token_ids(70)
        with reference_in_float64(hf):
            exact = reference_hidden(hf, ids)
        self.assertLess(rel_gap(ours(net, ids), exact), TOL_EXACT)
        with self.assertRaisesRegex(ValueError, "no LM head"):
            Qwen3_5.from_huggingface(hf)             # untied config: there is no head
        tied = build_reference(Qwen3_5TextModel, scaled_config("4B"))
        self.assertTrue(tied.config.tie_word_embeddings)
        net = Qwen3_5.from_huggingface(tied)         # tied config: the embedding is the head
        with reference_in_float64(tied):
            exact = reference_hidden(tied, ids) @ tied.embed_tokens.weight.T
        self.assertLess(rel_gap(ours(net, ids), exact), TOL_EXACT)

    def test_float32_networks(self):
        """
            The production dtype. Our float32 network against the float32
            reference, and both against the exact (float64) function. All
            three gaps are float32 round-off of the same size (measured
            over 8 seeds at 70 - 200 tokens: medians 1e-5 for ours, 2e-5
            for the reference, maxima 6e-5 and 9e-5): neither network is
            systematically the closer one.
        """
        for name in ("4B", "27B"):
            with self.subTest(config=name):
                hf64 = build_reference(Qwen3_5ForConditionalGeneration, scaled_config(name))
                hf32 = copy.deepcopy(hf64).float()
                net32 = Qwen3_5.from_huggingface(hf32)
                self.assertEqual(net32.layers[0].weight.dtype, torch.float32)
                for T in (7, 70, 130):
                    ids = token_ids(T)
                    with reference_in_float64(hf64):
                        exact = reference_logits(hf64, ids)
                    mine, shipped = ours(net32, ids), reference_logits(hf32, ids)
                    self.assertEqual(mine.dtype, torch.float32)
                    self.assertLess(rel_gap(mine, shipped), TOL_REF32)
                    self.assertLess(rel_gap(mine, exact), TOL_REF32)
                    self.assertLess(rel_gap(shipped, exact), TOL_REF32)


@unittest.skipUnless(HAS_TRANSFORMERS, "requires the optional dependency transformers")
class TestNegativeControls(unittest.TestCase):
    """
        A parity test that cannot fail proves nothing. Each control loads
        the network correctly, breaks ONE convention, and must then miss
        the reference by MISS (1e-2) or more, where the correct network is
        within TOL_EXACT (1e-10). The fine controls move an epsilon and
        must still be resolved, a thousand times above the tolerance.
    """

    T = 70

    @classmethod
    def setUpClass(cls):
        cls.cases = {}
        for name in ("4B", "27B"):
            hf = build_reference(Qwen3_5ForConditionalGeneration, scaled_config(name))
            ids = token_ids(cls.T)
            with reference_in_float64(hf):
                cls.cases[name] = (hf, ids, reference_logits(hf, ids))

    def _gap(self, name, break_it=None):
        hf, ids, exact = self.cases[name]
        net = Qwen3_5.from_huggingface(hf)
        if break_it is not None:
            with torch.no_grad():
                break_it(net)
        return rel_gap(ours(net, ids), exact)

    def _must_miss(self, break_it, names=("4B", "27B"), at_least=MISS):
        for name in names:
            with self.subTest(config=name):
                self.assertLess(self._gap(name), TOL_EXACT)
                gap = self._gap(name, break_it)
                self.assertGreater(gap, at_least, f"the broken network is only {gap:.2e} away")

    @staticmethod
    def _each(net, kind):
        return [m for m in net.layers if isinstance(m, kind)]

    def test_wrong_rope_base(self):
        """10000 -- the default of the constructor and of the config class."""
        def break_it(net):
            for m in self._each(net, GatedAttention):
                self.assertEqual(m.rope_theta, 1e7)
                m.rope_theta = 10000.0
        self._must_miss(break_it)

    def test_wrong_partial_rotary_factor(self):
        for factor in (1.0, 0.5, 0.0):
            def break_it(net):
                for m in self._each(net, GatedAttention):
                    self.assertEqual(m.rotary_dim, m.d_head // 4)
                    m.rotary_dim = int(m.d_head * factor)
            with self.subTest(factor=factor):
                self._must_miss(break_it)

    def test_without_the_zero_centered_fold(self):
        """Layer norms, final norm and q/k norms scale by 1 + weight."""
        def unfolded(net):
            for m in self._each(net, RMSNorm):
                m.weight -= 1.0
            for m in self._each(net, GatedAttention):
                m.q_norm.weight -= 1.0
                m.k_norm.weight -= 1.0

        def unfolded_qk_only(net):
            for m in self._each(net, GatedAttention):
                m.q_norm.weight -= 1.0
                m.k_norm.weight -= 1.0

        def unfolded_final_only(net):
            self._each(net, RMSNorm)[-1].weight -= 1.0

        for break_it in (unfolded, unfolded_qk_only, unfolded_final_only):
            with self.subTest(control=break_it.__name__):
                self._must_miss(break_it)

    def test_folding_the_gated_norm(self):
        """... but the gated norm of the DeltaNet scales by its weight."""
        def break_it(net):
            for m in self._each(net, GatedDeltaNet):
                m.norm.weight += 1.0
        self._must_miss(break_it)

    def test_wrong_value_to_key_head_expansion(self):
        """
            Value head j reads key head j // ratio (repeat_interleave), not
            j % num_key_heads (tiling). Only visible when the ratio is not
            1: 2 in the 4B, 3 in the 27B.
        """
        def tiled(self, h, cdt, qkv=None):
            T = h.shape[1]
            if qkv is None:
                qkv = self._conv_silu_qkv(h)
            ratio = self.num_v_heads // self.num_k_heads
            query = qkv[..., :self.key_dim].reshape(-1, T, self.num_k_heads, self.head_k_dim)
            key = qkv[..., self.key_dim:2 * self.key_dim].reshape(-1, T, self.num_k_heads, self.head_k_dim)
            beta = self.in_proj_b(h).sigmoid()
            decay = -self.A_log.to(cdt).exp() * F.softplus(self.in_proj_a(h).to(cdt) + self.dt_bias.to(cdt))
            return query.repeat(1, 1, ratio, 1), key.repeat(1, 1, ratio, 1), decay, beta

        def break_it(net):
            for m in self._each(net, GatedDeltaNet):
                self.assertGreater(m.num_v_heads // m.num_k_heads, 1)
                m._routing = types.MethodType(tiled, m)
        self._must_miss(break_it)

    def test_wrong_key_value_head_groups(self):
        """Grouped-query attention: query head h reads KV head h // ratio."""
        def break_it(net):
            for m in self._each(net, GatedAttention):
                self.assertEqual(m.num_kv_heads, 2)
                for proj in (m.k_proj, m.v_proj):       # swap the two KV heads
                    proj.weight.copy_(proj.weight.view(2, m.d_head, -1).flip(0).reshape(proj.weight.shape))
        self._must_miss(break_it)

    def test_wrong_query_gate_split(self):
        """q_proj is split per head into (query, gate), not into two halves."""
        def break_it(net):
            for m in self._each(net, GatedAttention):
                per_head = m.q_proj.weight.view(m.num_heads, 2, m.d_head, -1)
                m.q_proj.weight.copy_(per_head.transpose(0, 1).reshape(m.q_proj.weight.shape))
        self._must_miss(break_it)

    def test_reversed_convolution_kernel(self):
        def break_it(net):
            for m in self._each(net, GatedDeltaNet):
                m.conv1d.weight.copy_(m.conv1d.weight.flip(-1))
        self._must_miss(break_it)

    def test_embedding_as_untied_head(self):
        """The 27B has its own head; the 4B's head IS the embedding."""
        def break_it(net):
            net.layers[-1].weight.copy_(net.layers[0].weight)
        self._must_miss(break_it, names=("27B",))
        self.assertLess(self._gap("4B", break_it), TOL_EXACT)

    def test_wrong_layer_pattern(self):
        """A config that disagrees with the weights cannot even be loaded."""
        hf = copy.deepcopy(self.cases["4B"][0])
        hf.config.text_config.layer_types = ["full_attention"] * 8
        with self.assertRaisesRegex(KeyError, "self_attn"):
            Qwen3_5.from_huggingface(hf)

    def test_epsilons_are_resolved(self):
        """
            The fine controls: the 1e-6 inside the l2 normalization of
            q and k (measured effect 1.5e-6), and the epsilon of the
            layer norms (1e-5 instead of 1e-6: 1e-4 .. 3e-4). Both sit
            under the float32 noise of the shipped reference (TOL_REF32),
            and 1e3 or more above what the float64 comparison tolerates.
            The epsilon of the gated norm is a coarse control: the delta
            rule output is small, so that epsilon is part of the function.
        """
        def no_l2_eps(net):
            for m in self._each(net, GatedDeltaNet):
                m._l2norm = lambda x, eps=0.0: x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True))

        def rms_eps(net):
            for m in self._each(net, RMSNorm):
                m.eps = 1e-5

        def gated_rms_eps(net):
            for m in self._each(net, GatedDeltaNet):
                m.norm.eps = 1e-5

        for break_it, at_least in ((no_l2_eps, 1e-7), (rms_eps, 1e-5), (gated_rms_eps, MISS)):
            with self.subTest(control=break_it.__name__):
                self._must_miss(break_it, at_least=at_least)


@unittest.skipUnless(HAS_TRANSFORMERS, "requires the optional dependency transformers")
class TestKnowledgeMatrixOnLoadedNetworks(unittest.TestCase):
    """
        The knowledge matrix of the LOADED networks (published layer
        pattern and head ratios, random weights): M(x) . 1 = f(x) in both
        linearization modes, rows by reverse mode equal to the matrix rows,
        and exactness of the frozen linearization of both mixers.
    """

    T = 6

    @classmethod
    def setUpClass(cls):
        cls.nets = {}
        for name, hf_cls in (("4B", Qwen3_5ForConditionalGeneration), ("27B", Qwen3_5ForCausalLM)):
            hf = build_reference(hf_cls, scaled_config(name, real_head_dim=False))
            cls.nets[name] = hf

    def _net(self, name, **kwargs):
        net = Qwen3_5.from_huggingface(self.nets[name], **kwargs)
        net.eval()
        return net

    def _matrix(self, net, x, mode):
        net.save = True
        return KnowledgeMatrixComputer(net, batch_size=64, mixer_mode=mode).forward(x)

    def test_rows_sum_to_the_forward_pass(self):
        for name in ("4B", "27B"):
            for mode in ("ratio", "frozen"):
                for head in (True, False):
                    with self.subTest(config=name, mixer_mode=mode, lm_head=head):
                        net = self._net(name, include_lm_head=head)
                        x = token_ids(self.T).unsqueeze(0)
                        out = net.forward(x)
                        mat = self._matrix(net, x, mode)
                        self.assertEqual(mat.shape, (out.numel(), self.T * net.hidden_size + 1))
                        diff = torch.norm(out.reshape(-1) - mat.sum(1)).item()
                        self.assertLess(diff, 1e-8 * max(1.0, torch.norm(out).item()))

    def test_the_matrix_is_of_the_reference_function(self):
        """Row sums equal the logits of the transformers model itself."""
        for name in ("4B", "27B"):
            with self.subTest(config=name):
                hf = self.nets[name]
                ids = token_ids(self.T)
                with reference_in_float64(hf):
                    exact = reference_logits(hf, ids)
                mat = self._matrix(self._net(name), ids.unsqueeze(0), "frozen")
                self.assertLess(rel_gap(mat.sum(1).reshape(exact.shape), exact), TOL_EXACT)

    def test_rows_agree_with_the_matrix(self):
        """KnowledgeRowComputer under the same mixer_mode as the matrix."""
        for name in ("4B", "27B"):
            for mode in ("ratio", "frozen"):
                with self.subTest(config=name, mixer_mode=mode):
                    net = self._net(name)
                    x = token_ids(self.T).unsqueeze(0)
                    out = net.forward(x).reshape(-1)
                    mat = self._matrix(net, x, mode)
                    rows = [0, 95, 96 * 3 + 17, out.numel() - 1]
                    A = KnowledgeRowComputer(net, mixer_mode=mode).forward(x, rows)
                    scale = mat.abs().max().item()
                    for k, j in enumerate(rows):
                        self.assertLess((A[k] - mat[j]).abs().max().item(), 1e-9 * scale,
                                        f"row {j} differs")
                        self.assertLess(abs(A[k].sum().item() - out[j].item()), 1e-9 * scale)

    def test_frozen_mode_mixes_positions_causally(self):
        """
            In frozen mode attribution crosses positions through both
            mixers, and never from the future; in ratio mode the matrix is
            block diagonal over positions.
        """
        net = self._net("27B")
        x = token_ids(self.T).unsqueeze(0)
        T, V, D = self.T, net.vocab_size, net.hidden_size
        frozen = self._matrix(net, x, "frozen")[:, :-1].reshape(T, V, T, D)
        ratio = self._matrix(net, x, "ratio")[:, :-1].reshape(T, V, T, D)
        for t in range(T):
            for s in range(T):
                block = frozen[t, :, s, :].abs().max().item()
                if s > t:
                    self.assertEqual(block, 0.0)
                else:
                    self.assertGreater(block, 0.0)
                if s != t:
                    self.assertEqual(ratio[t, :, s, :].abs().max().item(), 0.0)

    def test_frozen_linearization_of_each_mixer(self):
        """
            At the actual layer input x0 of every GatedDeltaNet,
            GatedAttention and SwiGLU of the loaded network:
            frozen_forward(x0, x0) is forward(x0), and x -> frozen_forward(x,
            x0) is linear.
        """
        for name in ("4B", "27B"):
            net = self._net(name)
            net.save = True
            with torch.no_grad():
                net.forward(token_ids(self.T).unsqueeze(0))
            net.save = False
            seen = set()
            g = torch.Generator().manual_seed(0)
            for i, layer in enumerate(net.layers):
                if not isinstance(layer, (GatedDeltaNet, GatedAttention, SwiGLU)):
                    continue
                seen.add(type(layer))
                x0 = net.pre_acts[i]
                with self.subTest(config=name, layer=i, kind=type(layer).__name__):
                    with torch.no_grad():
                        out = layer(x0)
                        self.assertLess(rel_gap(layer.frozen_forward(x0, x0), out), 1e-12)
                        x1 = torch.randn(x0.shape, generator=g)
                        x2 = torch.randn(x0.shape, generator=g)
                        combined = layer.frozen_forward(2.0 * x1 - 3.0 * x2, x0)
                        separate = 2.0 * layer.frozen_forward(x1, x0) - 3.0 * layer.frozen_forward(x2, x0)
                        self.assertLess(rel_gap(combined, separate), 1e-11)
            self.assertEqual(seen, {GatedDeltaNet, GatedAttention, SwiGLU})


@unittest.skipUnless(HAS_TRANSFORMERS, "requires the optional dependency transformers")
class TestFullSizeDryRun(unittest.TestCase):
    """
        The published architectures at FULL size, without weights: the
        real transformers class is instantiated on the "meta" device
        (shapes, no storage) and loaded with device="meta", which locates
        and shape-checks every tensor of the decoder.
    """

    def _dry_run(self, name):
        config = published_config(name)
        with torch.device("meta"):
            hf = Qwen3_5ForConditionalGeneration(config)
        return config, hf, Qwen3_5.from_huggingface(hf, device="meta")

    def test_published_architectures(self):
        for name, (_, text_parameters) in PUBLISHED.items():
            with self.subTest(config=name):
                config, hf, net = self._dry_run(name)
                text = config.text_config
                self.assertTrue(all(p.is_meta for p in net.parameters()))
                self.assertEqual(net.device, "meta")
                self.assertEqual(net.get_num_layers(), 1 + 6 * text.num_hidden_layers + 2)

                count = lambda module: sum(p.numel() for p in module.parameters())  # noqa: E731
                head = text.vocab_size * text.hidden_size
                decoder = count(hf.model.language_model)
                self.assertEqual(decoder, text_parameters)
                # ours = the text decoder + a head matrix (also when the
                # checkpoint ties it to the embedding and stores none)
                self.assertEqual(count(net), decoder + head)
                headless = Qwen3_5.from_huggingface(hf, device="meta", include_lm_head=False)
                self.assertEqual(count(headless), decoder)
                # transformers = decoder + vision tower (ignored by us)
                # + the head when it is not tied
                tied = hf.lm_head.weight is hf.model.language_model.embed_tokens.weight
                self.assertEqual(tied, config.tie_word_embeddings)
                vision = count(hf.model.visual)
                self.assertGreater(vision, 100_000_000)
                self.assertEqual(count(hf), decoder + vision + (0 if tied else head))

    def test_full_size_shapes(self):
        """Spot checks of the shapes that are easy to get wrong."""
        _, _, net = self._dry_run("27B")
        delta, attention = net.layers[2], net.layers[1 + 3 * 6 + 1]
        self.assertIsInstance(delta, GatedDeltaNet)
        self.assertIsInstance(attention, GatedAttention)
        # 48 value heads on 16 key heads, 128 wide: 2 * 2048 + 6144
        self.assertEqual(tuple(delta.in_proj_qkv.weight.shape), (10240, 5120))
        self.assertEqual(tuple(delta.conv1d.weight.shape), (10240, 1, 4))
        self.assertEqual(tuple(delta.A_log.shape), (48,))
        self.assertEqual(tuple(delta.out_proj.weight.shape), (5120, 6144))
        # 24 heads of 256 on a 5120 stream: query + gate, 4 KV heads
        self.assertEqual(tuple(attention.q_proj.weight.shape), (2 * 24 * 256, 5120))
        self.assertEqual(tuple(attention.k_proj.weight.shape), (4 * 256, 5120))
        self.assertEqual(tuple(attention.o_proj.weight.shape), (5120, 24 * 256))
        self.assertEqual((attention.rope_theta, attention.rotary_dim), (1e7, 64))
        self.assertEqual(tuple(net.layers[-1].weight.shape), (248320, 5120))

    def test_dry_run_catches_a_wrong_shape(self):
        config = published_config("4B")
        with torch.device("meta"):
            hf = Qwen3_5ForConditionalGeneration(config)
            hf.model.language_model.layers[5].linear_attn.A_log = torch.nn.Parameter(torch.empty(31))
        with self.assertRaisesRegex(ValueError, "shape mismatch for layer 32 A_log"):
            Qwen3_5.from_huggingface(hf, device="meta")

    @unittest.skipUnless(SLOW, "builds full-width blocks (up to 3 GB): set KM_SLOW_TESTS=1")
    def test_full_width_blocks(self):
        """
            One linear and one full-attention block at the REAL widths and
            head counts (4B: 2560 wide, 32 value heads on 16 key heads, 16
            query heads on 4; 27B: 5120 wide, 48 on 16, 24 on 4), random
            weights, 130 tokens. Only the depth and the vocabulary are cut.
            The 4B blocks run in float64 against the float64 reference,
            the 27B blocks (3 GB) in float32 against the shipped one.
        """
        for name, dtype in (("4B", torch.float64), ("27B", torch.float32)):
            with self.subTest(config=name):
                config = published_config(name).text_config
                config.vocab_size = 96
                config.num_hidden_layers = 2
                config.layer_types = ["linear_attention", "full_attention"]
                torch.set_default_dtype(dtype)
                try:
                    hf = randomize(Qwen3_5ForCausalLM(config), seed=1).eval()
                finally:
                    torch.set_default_dtype(torch.float64)
                self.assertEqual({p.dtype for p in hf.parameters()}, {dtype})
                net = Qwen3_5.from_huggingface(hf, copy=False)
                delta, attention = net.layers[2], net.layers[8]
                self.assertEqual((delta.num_v_heads, delta.num_k_heads, delta.head_v_dim),
                                 (config.linear_num_value_heads, 16, 128))
                self.assertEqual((attention.num_heads, attention.num_kv_heads, attention.d_head),
                                 (config.num_attention_heads, 4, 256))
                ids = token_ids(130)
                if dtype == torch.float64:
                    with reference_in_float64(hf):      # measured: 2.7e-15
                        expected, tolerance = reference_logits(hf, ids), TOL_EXACT
                else:                                   # measured: 9.4e-6
                    expected, tolerance = reference_logits(hf, ids), TOL_REF32
                gap = rel_gap(ours(net, ids), expected)
                self.assertLess(gap, tolerance, f"{name}: {gap:.2e}")
                # control at full width: the constructor's default RoPE base
                # in the single attention block (measured: 1.7e-2, 7.1e-2)
                attention.rope_theta = 10000.0
                self.assertGreater(rel_gap(ours(net, ids), expected), 10 * TOL_REF32)
                del hf, net
                gc.collect()

    def test_sharded_layout_at_full_size(self):
        """Three stages for the 27B: cuts between blocks, balanced."""
        config = published_config("27B")
        with torch.device("meta"):
            hf = Qwen3_5ForConditionalGeneration(config)
            net = Qwen3_5.from_huggingface(hf, device="meta")
        layer_devices, boundaries = net.shard_layout(["cuda:0", "cuda:1", "cuda:2"])
        self.assertEqual(len(boundaries), 2)
        per_stage = [0, 0, 0]
        for layer, device in zip(net.layers, layer_devices):
            per_stage[int(device[-1])] += sum(p.numel() for p in layer.parameters())
        total = sum(per_stage)
        for share in per_stage:
            self.assertLess(abs(share / total - 1 / 3), 0.03)


@unittest.skipUnless(HAS_TRANSFORMERS, "requires the optional dependency transformers")
class TestLoaderMechanics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tied = build_reference(Qwen3_5ForConditionalGeneration, scaled_config("4B", real_head_dim=False))
        cls.untied = build_reference(Qwen3_5ForConditionalGeneration, scaled_config("27B", real_head_dim=False))
        cls.ids = token_ids(9)

    def test_signature_is_the_family_signature(self):
        """One calling convention for every loader (the model registry relies on it)."""
        import inspect
        from knowledgematrix.models.llama import LlamaLike
        mine = inspect.signature(Qwen3_5.from_huggingface).parameters
        theirs = inspect.signature(LlamaLike.from_huggingface).parameters
        self.assertEqual([(p.name, p.default, p.kind) for p in mine.values()],
                         [(p.name, p.default, p.kind) for p in theirs.values()])

    def test_sharded_equals_unsharded(self):
        for hf in (self.tied, self.untied):
            plain = Qwen3_5.from_huggingface(hf)
            self.assertIsNone(plain.layer_devices)
            expected = ours(plain, self.ids)
            sharded = Qwen3_5.from_huggingface(hf, device=["cpu", "cpu"])
            self.assertEqual(len(sharded.layer_devices), sharded.get_num_layers())
            self.assertEqual(len(sharded.shard_boundaries), 1)
            self.assertIn(sharded.shard_boundaries[0], sharded.shard_cut_points())
            self.assertTrue(torch.equal(ours(sharded, self.ids), expected))
            cut = sharded.shard_cut_points()[3]
            chosen = Qwen3_5.from_huggingface(hf, device=["cpu", "cpu", "cpu"], boundaries=[cut, cut + 12])
            self.assertEqual(chosen.shard_boundaries, [cut, cut + 12])
            self.assertTrue(torch.equal(ours(chosen, self.ids), expected))
            with self.assertRaises(ValueError):      # a cut inside a residual block
                Qwen3_5.from_huggingface(hf, device=["cpu", "cpu"], boundaries=[cut + 1])
            # rows by reverse mode on the sharded network
            rows = [0, 500]
            x = self.ids.unsqueeze(0)
            A = KnowledgeRowComputer(plain).forward(x, rows)
            B = KnowledgeRowComputer(sharded).forward(x, rows)
            self.assertLess((A - B).abs().max().item(), 1e-12 * A.abs().max().item())

    def test_copy_false_shares_storage(self):
        hf = copy.deepcopy(self.untied)
        text = hf.model.language_model
        shared = Qwen3_5.from_huggingface(hf, copy=False)
        copied = Qwen3_5.from_huggingface(hf, copy=True)
        pairs = [
            (0, "weight", text.embed_tokens.weight),
            (2, "in_proj_qkv.weight", text.layers[0].linear_attn.in_proj_qkv.weight),
            (2, "A_log", text.layers[0].linear_attn.A_log),
            (5, "down_proj.weight", text.layers[0].mlp.down_proj.weight),
            (1 + 3 * 6 + 1, "q_proj.weight", text.layers[3].self_attn.q_proj.weight),
            (shared.get_num_layers() - 1, "weight", hf.lm_head.weight),
        ]
        for layer, attr, source in pairs:
            get = lambda net: _hf.WeightLoader._owner(  # noqa: E731
                types.SimpleNamespace(net=net), layer, attr)
            module, name = get(shared)
            self.assertEqual(module._parameters[name].data_ptr(), source.data_ptr(), attr)
            module, name = get(copied)
            self.assertNotEqual(module._parameters[name].data_ptr(), source.data_ptr(), attr)
        # the folded norm weights (1 + w) are new tensors either way
        self.assertNotEqual(shared.layers[1].weight.data_ptr(), text.layers[0].input_layernorm.weight.data_ptr())
        before = ours(copied, self.ids)
        with torch.no_grad():
            text.layers[0].mlp.down_proj.weight.mul_(2.0)
        self.assertTrue(torch.equal(ours(copied, self.ids), before))      # independent
        self.assertFalse(torch.equal(ours(shared, self.ids), before))     # aliased

    def test_tied_head_is_stored_once(self):
        """
            A checkpoint with a tied head stores no lm_head.weight; read
            from disk, the head of our network IS its embedding tensor
            (one (V, D) table in memory, not two).
        """
        with tempfile.TemporaryDirectory() as d:
            write_published_layout(self.tied, d)
            net = Qwen3_5.from_huggingface(d, dtype=torch.float64)
            self.assertEqual(net.layers[-1].weight.data_ptr(), net.layers[0].weight.data_ptr())
            sharded = Qwen3_5.from_huggingface(d, dtype=torch.float64, device=["cpu", "cpu"])
            self.assertTrue(torch.equal(ours(sharded, self.ids), ours(net, self.ids)))
            # a head on ANOTHER device than the embedding cannot alias it:
            # the table is read again and placed there (the second stage is
            # the storage-free device here, to have two devices on any machine)
            split = Qwen3_5.from_huggingface(d, dtype=torch.float64, device=["cpu", "meta"])
            self.assertEqual(split.layer_devices[0], "cpu")
            self.assertEqual(split.layer_devices[-1], "meta")
            self.assertFalse(split.layers[0].weight.is_meta)
            self.assertTrue(split.layers[-1].weight.is_meta)
            self.assertEqual(split.layers[-1].weight.shape, split.layers[0].weight.shape)
            write_published_layout(self.untied, d)
            net = Qwen3_5.from_huggingface(d, dtype=torch.float64)
            self.assertNotEqual(net.layers[-1].weight.data_ptr(), net.layers[0].weight.data_ptr())
            self.assertFalse(torch.equal(net.layers[-1].weight, net.layers[0].weight))

    def test_default_dtype(self):
        """float32 from disk (the files are a storage format); the model's own in memory."""
        with tempfile.TemporaryDirectory() as d:
            copy.deepcopy(self.tied).to(torch.bfloat16).save_pretrained(d)
            net = Qwen3_5.from_huggingface(d)
            self.assertEqual({p.dtype for p in net.parameters()}, {torch.float32})
            half = Qwen3_5.from_huggingface(d, dtype=torch.bfloat16)
            self.assertEqual({p.dtype for p in half.parameters()}, {torch.bfloat16})
            # the fold 1 + w is evaluated before the cast, not in bfloat16
            stored = safetensors.torch.load_file(os.path.join(d, "model.safetensors"))
            w = stored["model.language_model.layers.0.input_layernorm.weight"]
            self.assertTrue(torch.equal(net.layers[1].weight, 1.0 + w.float()))
            self.assertTrue(torch.equal(half.layers[1].weight, (1.0 + w.float()).to(torch.bfloat16)))
        self.assertEqual({p.dtype for p in Qwen3_5.from_huggingface(self.tied).parameters()}, {torch.float64})

    def _on_disk(self, hf, edit):
        """Load from a directory whose tensors were edited by `edit`."""
        with tempfile.TemporaryDirectory() as d:
            hf.save_pretrained(d)
            path = os.path.join(d, "model.safetensors")
            tensors = safetensors.torch.load_file(path)
            edit(tensors)
            safetensors.torch.save_file(tensors, path)
            return Qwen3_5.from_huggingface(d, dtype=torch.float64)

    def test_extra_tensors_outside_the_decoder_are_ignored(self):
        def edit(tensors):
            tensors["mtp.layers.0.self_attn.q_proj.weight"] = torch.randn(3, 5)
            tensors["mtp.fc.weight"] = torch.randn(2, 2)
            tensors["model.visual.blocks.7.attn.qkv.weight"] = torch.randn(4, 4)
            tensors["some_other_head.weight"] = torch.randn(2)
        net = self._on_disk(self.untied, edit)
        self.assertTrue(torch.equal(ours(net, self.ids), ours(Qwen3_5.from_huggingface(self.untied), self.ids)))

    def test_missing_tensor_raises(self):
        p = "model.language_model."
        for key in (p + "layers.1.linear_attn.A_log", p + "layers.3.self_attn.k_norm.weight",
                    p + "layers.6.mlp.up_proj.weight", p + "norm.weight", p + "embed_tokens.weight"):
            with self.subTest(key=key):
                with self.assertRaises((KeyError, ValueError)) as caught:
                    self._on_disk(self.untied, lambda tensors: tensors.pop(key))
                if key != p + "embed_tokens.weight":       # (without it the decoder is not found)
                    self.assertIn(key, str(caught.exception))

    def test_missing_untied_head_raises(self):
        """No stored head and no tying: transformers would leave it random."""
        with self.assertRaisesRegex(ValueError, "no LM head"):
            self._on_disk(self.untied, lambda tensors: tensors.pop("lm_head.weight"))
        with tempfile.TemporaryDirectory() as d:        # ... but the hidden states are defined
            self.untied.save_pretrained(d)
            Qwen3_5.from_huggingface(d, dtype=torch.float64, include_lm_head=False)

    def test_misshaped_tensor_raises(self):
        p = "model.language_model."

        def transposed(key):
            def edit(tensors):
                tensors[key] = tensors[key].T.contiguous()
            return edit

        def truncated(key):
            def edit(tensors):
                tensors[key] = tensors[key][:-1].contiguous()
            return edit

        for edit in (transposed(p + "layers.0.mlp.gate_proj.weight"),
                     truncated(p + "layers.2.linear_attn.dt_bias"),
                     truncated(p + "layers.7.self_attn.q_proj.weight"),
                     truncated("lm_head.weight")):
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                self._on_disk(self.untied, edit)

    def test_unplaced_decoder_tensor_raises(self):
        """A decoder tensor the network has no place for is an architecture we do not have."""
        p = "model.language_model."
        for key in (p + "layers.3.self_attn.q_proj.bias", p + "layers.0.linear_attn.conv1d.bias",
                    p + "layers.4.extra_norm.weight"):
            with self.subTest(key=key):
                def edit(tensors):
                    tensors[key] = torch.zeros(4)
                with self.assertRaisesRegex(ValueError, "no counterpart"):
                    self._on_disk(self.untied, edit)

    def test_unsupported_configs_are_refused(self):
        """Every config value that would change the function, and that we do not implement."""
        def refused(pattern, **text_overrides):
            hf = copy.deepcopy(self.untied)
            for key, value in text_overrides.items():
                setattr(hf.config.text_config, key, value)
            with self.assertRaisesRegex(NotImplementedError, pattern):
                Qwen3_5.from_huggingface(hf)
            with self.assertRaisesRegex(NotImplementedError, pattern):
                Qwen3_5.from_huggingface(hf, device="meta")

        refused("attn_output_gate", attn_output_gate=False)
        refused("mlp_only_layers", mlp_only_layers=[1])
        refused("mamba_ssm_dtype", mamba_ssm_dtype="bfloat16")
        refused("hidden_act", hidden_act="gelu")
        rope = dict(self.untied.config.text_config.rope_parameters)
        refused("RoPE", rope_parameters={**rope, "rope_type": "yarn", "factor": 2.0})
        refused("RoPE", rope_parameters={**rope, "rope_type": "linear", "factor": 2.0})
        refused("layer_types", layer_types=["linear_attention", "sliding_attention"] * 4)
        refused("linear_num_value_heads", linear_num_value_heads=3)
        # the published flags themselves are accepted
        config = self.untied.config.text_config
        self.assertEqual((config.attn_output_gate, config.mlp_only_layers, config.mamba_ssm_dtype,
                          config.hidden_act), (True, [], "float32", "silu"))

    def test_other_checkpoints_are_refused(self):
        class InMemory(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config

        text = self.untied.config.text_config
        moe = types.SimpleNamespace(model_type="qwen3_5_moe", text_config=types.SimpleNamespace(
            model_type="qwen3_5_moe_text", num_experts=256))
        with self.assertRaisesRegex(NotImplementedError, "MoE"):
            Qwen3_5.from_huggingface(InMemory(moe))
        with self.assertRaisesRegex(NotImplementedError, "qwen3_5_text"):
            Qwen3_5.from_huggingface(InMemory(types.SimpleNamespace(model_type="llama")))
        quantized = types.SimpleNamespace(model_type="qwen3_5", text_config=text,
                                          quantization_config={"quant_method": "fp8"})
        with self.assertRaisesRegex(NotImplementedError, "quantized"):
            Qwen3_5.from_huggingface(InMemory(quantized))


_SAFETENSORS_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64}


def cached_repository(repository: str, tokenizer: bool = False) -> bool:
    """Whether the whole checkpoint is in the local Hugging Face cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    index = try_to_load_from_cache(repository, "model.safetensors.index.json")
    if not isinstance(index, str):
        return False
    with open(index) as f:
        names = set(json.load(f)["weight_map"].values())
    names |= {"config.json"} | ({"tokenizer.json", "tokenizer_config.json"} if tokenizer else set())
    return all(isinstance(try_to_load_from_cache(repository, name), str) for name in names)


@unittest.skipUnless(HAS_TRANSFORMERS, "requires the optional dependency transformers")
class TestRealCheckpoints(unittest.TestCase):
    """
        The published files themselves, when they are already in the local
        Hugging Face cache (local_files_only: nothing is ever downloaded).
    """

    def _header_only(self):
        """Serve every tensor as its header (shape, dtype): no data is read."""
        def tensor(checkpoint, key):
            view = checkpoint._handle(checkpoint._key_to_file[key]).get_slice(key)
            return torch.empty(view.get_shape(), dtype=_SAFETENSORS_DTYPES[view.get_dtype()], device="meta")
        return mock.patch.object(_hf._SafetensorsCheckpoint, "tensor", tensor)

    def _check_counts(self, net, name):
        _, text_parameters = PUBLISHED[name]
        head = net.vocab_size * net.hidden_size
        self.assertEqual(sum(p.numel() for p in net.parameters()), text_parameters + head)

    def test_real_file_layout(self):
        """
            Every tensor our loader wants exists in the real files with the
            shape of our full-size network, and no decoder tensor is left
            over (read from the safetensors headers: fast).
        """
        cached = [name for name, (repository, _) in PUBLISHED.items() if cached_repository(repository)]
        if not cached:
            self.skipTest("no published Qwen3.5 checkpoint in the local Hugging Face cache")
        for name in cached:
            repository, _ = PUBLISHED[name]
            with self.subTest(checkpoint=repository):
                checkpoint = _hf.open_checkpoint(repository, local_files_only=True)
                keys = list(checkpoint.keys())
                self.assertEqual(checkpoint.text_prefix, "model.language_model.")
                self.assertTrue(any(k.startswith("model.visual.") for k in keys))
                self.assertTrue(any(k.startswith("mtp.layers.") for k in keys))
                tied = checkpoint.config.tie_word_embeddings
                self.assertEqual("lm_head.weight" in keys, not tied)
                checkpoint.close()
                with self._header_only():
                    net = Qwen3_5.from_huggingface(repository, device="meta", local_files_only=True)
                self._check_counts(net, name)

    @unittest.skipUnless(SLOW, "reads the whole checkpoint from disk: set KM_SLOW_TESTS=1")
    def test_real_4b_dry_run(self):
        """Every decoder tensor of the real 4B is read, shape-checked and discarded."""
        repository, _ = PUBLISHED["4B"]
        if not cached_repository(repository):
            self.skipTest(f"{repository} is not in the local Hugging Face cache")
        net = Qwen3_5.from_huggingface(repository, device="meta", local_files_only=True)
        self.assertTrue(all(p.is_meta for p in net.parameters()))
        self._check_counts(net, "4B")

    @unittest.skipUnless(SLOW, "loads two 0.8B models in float32 (one at a time): set KM_SLOW_TESTS=1")
    def test_real_0_8b_matches_transformers(self):
        """
            Real weights, real text: our float32 network streamed from the
            safetensors file against the transformers model, on a short
            prompt and on one that spans three delta-rule chunks.
        """
        repository, _ = PUBLISHED["0.8B"]
        if not cached_repository(repository, tokenizer=True):
            self.skipTest(f"{repository} (weights and tokenizer) is not in the local Hugging Face cache")
        torch.set_default_dtype(torch.float32)
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(repository, local_files_only=True)
            prompts = {name: tokenizer(text, return_tensors="pt").input_ids for name, text in REAL_PROMPTS.items()}
            self.assertGreater(prompts["long"].shape[1], 2 * 64)
            hf = transformers.AutoModelForImageTextToText.from_pretrained(
                repository, local_files_only=True, dtype=torch.float32).eval()
            self.assertIsInstance(hf, Qwen3_5ForConditionalGeneration)
            with torch.no_grad():
                reference = {name: hf(input_ids=ids).logits[0].clone() for name, ids in prompts.items()}
            del hf
            gc.collect()

            net = Qwen3_5.from_huggingface(repository, local_files_only=True)
            self.assertEqual({p.dtype for p in net.parameters()}, {torch.float32})
            for name, ids in prompts.items():
                logits = ours(net, ids)[0]
                gap = rel_gap(logits, reference[name])
                # measured: 1.9e-6 (5 tokens) and 2.0e-6 (163 tokens) of the
                # logit scale (max |logit| 18.6 and 24.4)
                self.assertLess(gap, 1e-4, f"{name}: {gap:.2e}")
                agree = (logits.argmax(-1) == reference[name].argmax(-1)).double().mean().item()
                self.assertEqual(agree, 1.0, f"{name}: top-1 agreement {agree}")
        finally:
            torch.set_default_dtype(torch.float64)


REAL_PROMPTS = {
    "short": "The capital of France is",
    "long": (
        "A knowledge matrix records, for one input, how every output coordinate of a network "
        "is assembled from the coordinates of that input. Each row sums to one output of the "
        "forward pass, so nothing is approximated: the matrix is the network at that input, "
        "written as a linear map. For a language model the rows are logits and the columns are "
        "the coordinates of the token embeddings, position by position. Attention layers and "
        "linear-attention recurrences move information between positions, and the frozen "
        "linearization keeps that routing fixed at the values it actually took, so that the "
        "contribution of every earlier token to the next-token prediction can be read off the "
        "matrix. The first step is always the same: rebuild the network layer by layer and check "
        "that it computes exactly what the reference implementation computes, on short inputs "
        "and on long ones."
    ),
}


if __name__ == "__main__":
    unittest.main()
