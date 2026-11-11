#!/usr/bin/env python
"""
    Tests for Gemma-2 and Gemma-3 (models/gemma.py::GemmaLike) and for the
    SoftCap layer that carries Gemma-2's final logit soft-capping.

    A rebuilt network is self-consistent whatever its wiring -- the rows
    of its knowledge matrix sum to ITS forward pass -- so the only thing
    that catches a wiring error is parity against the reference
    implementation. The tests are organised around that:

    1. PARITY, for each of the eight published configs
       (fixtures/hf_configs/google__gemma-*.json): the real config with
       only its sizes scaled down, instantiated with the real transformers
       class in float64, must give the same logits and the same final
       hidden states as our network -- loaded from the model in memory
       and streamed from a `save_pretrained` directory.
    2. NEGATIVE CONTROLS: for each mechanism (RoPE base per layer type,
       RoPE rescaling, q/k norm, block pattern, window extent, soft-caps,
       query scalar, embedding scale, ...) the same comparison FAILS by
       orders of magnitude when that one mechanism is wrong. A parity
       test that cannot fail proves nothing.
    3. KNOWLEDGE MATRIX invariants on the loaded networks, in both mixer
       modes and through the soft-cap layer.
    4. FULL-SIZE dry runs: the eight published architectures at their real
       size on the "meta" device (shapes, no memory).
    5. REAL WEIGHTS: google/gemma-3-270m (via the ungated mirror
       unsloth/gemma-3-270m) when it is in the local Hugging Face cache;
       skipped otherwise -- the tests never download.

    The reference is always transformers with attn_implementation="eager":
    under its default ("sdpa") transformers drops Gemma-2's attention
    soft-capping (see TestReferenceQuirks).

    The tests of the soft-cap layer, of exactly-zero pre-activations and
    of the constructor need torch only; everything else needs transformers
    and safetensors and is skipped without them.
"""
import contextlib
import json
import math
import os
import shutil
import tempfile
import unittest

import torch

from knowledgematrix.neural_net import (
    NN,
    ACTIVATION_LAYERS,
    LINEARIZABLE_LAYERS,
    GeGLU,
    RMSNorm,
    RopeAttention,
    SoftCap,
)
from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.row_computer import KnowledgeRowComputer
from knowledgematrix.models.gemma import GemmaLike, FULL, SLIDING
from knowledgematrix.models._hf import empty_network, open_checkpoint

try:
    import transformers
    from safetensors.torch import load_file, save_file
    transformers.utils.logging.set_verbosity_error()
    transformers.utils.logging.disable_progress_bar()
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

torch.set_default_dtype(torch.float64)

#: Same convention and same reason as in llama_gemma.py: transformers
#: hardcodes float32 internals (x.float() inside the RMSNorm, the softmax
#: of the eager attention, the RoPE tables), so even a float64 reference
#: carries float32 rounding. Here the comparison is RELATIVE to the
#: largest output, because these tests use weights that make every
#: mechanism matter and the logits are then of order 30, not 0.3. The
#: agreement measured over the eight configs and five seeds is at most
#: 3.3e-7; anything above the tolerance is a real bug.
HF_TOL = 1e-6

#: A negative control must miss the reference by at least this much
#: (relative), i.e. by three orders of magnitude more than the tolerance.
CONTROL_MIN = 1e-3

#: Three effects are small by nature: the bfloat16 rounding of
#: sqrt(hidden_size) (a 0.2% change of one factor; measured gap >= 1.0e-3),
#: a cap of 50 on attention scores of a few units (>= 5.6e-4), and the
#: tanh approximation of GELU, which is accurate to ~5e-4 (>= 7.8e-5).
#: Their controls must still miss by 300x, resp. 30x, the tolerance.
SMALL_CONTROL_MIN = 3e-4
GELU_CONTROL_MIN = 3e-5

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "hf_configs")

#: The sizes of the scaled-down models. Everything else comes from the
#: published config. What is kept on purpose:
#:   - heads / key-value heads in the published ratio (270m and 1b: 4 / 1),
#:   - num_heads * head_dim != hidden_size, as in every published Gemma,
#:   - query_pre_attn_scalar untouched, hence != head_dim,
#:   - enough blocks for the pattern (Gemma-2: local first, alternating;
#:     Gemma-3: five local, one global -- twice over in 4b and 27b),
#:   - a window shorter than the test sequence.
TINY = {
    "gemma-2-2b":     dict(hidden_size=36, intermediate_size=72, num_hidden_layers=5, num_attention_heads=4,
                           num_key_value_heads=2, head_dim=8, vocab_size=61, sliding_window=5),
    "gemma-2-9b":     dict(hidden_size=28, intermediate_size=56, num_hidden_layers=6, num_attention_heads=4,
                           num_key_value_heads=2, head_dim=8, vocab_size=53, sliding_window=4),
    "gemma-2-27b":    dict(hidden_size=40, intermediate_size=80, num_hidden_layers=4, num_attention_heads=8,
                           num_key_value_heads=4, head_dim=6, vocab_size=47, sliding_window=6),
    "gemma-3-270m":   dict(hidden_size=40, intermediate_size=64, num_hidden_layers=7, num_attention_heads=4,
                           num_key_value_heads=1, head_dim=8, vocab_size=59, sliding_window=4),
    "gemma-3-1b-it":  dict(hidden_size=36, intermediate_size=72, num_hidden_layers=8, num_attention_heads=4,
                           num_key_value_heads=1, head_dim=8, vocab_size=43, sliding_window=5),
    "gemma-3-4b-it":  dict(hidden_size=40, intermediate_size=80, num_hidden_layers=12, num_attention_heads=4,
                           num_key_value_heads=2, head_dim=8, vocab_size=67, sliding_window=6),
    "gemma-3-12b-it": dict(hidden_size=28, intermediate_size=56, num_hidden_layers=7, num_attention_heads=4,
                           num_key_value_heads=2, head_dim=8, vocab_size=41, sliding_window=4),
    "gemma-3-27b-it": dict(hidden_size=40, intermediate_size=80, num_hidden_layers=13, num_attention_heads=8,
                           num_key_value_heads=4, head_dim=6, vocab_size=37, sliding_window=5),
}
#: the vision tower of the multimodal checkpoints (not on the text path)
TINY_VISION = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                   num_attention_heads=2, image_size=28, patch_size=14)
SIZE_KEYS = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
             "num_key_value_heads", "head_dim", "vocab_size", "sliding_window")

NAMES = list(TINY)
GEMMA2 = [n for n in NAMES if n.startswith("gemma-2")]
GEMMA3 = [n for n in NAMES if n.startswith("gemma-3")]
MULTIMODAL = ["gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it"]

#: What the published configs say, as normalized by transformers -- the
#: facts the loader has to get right. (rope: (base, rescaling) of the
#: local and of the global blocks.)
_LINEAR8 = {"rope_type": "linear", "factor": 8.0}
PUBLISHED = {
    "gemma-2-2b":     dict(cls="Gemma2ForCausalLM", layers=26, hidden=2304, head_dim=256, heads=(8, 4), window=4096,
                           attn_cap=50.0, final_cap=30.0, scalar=256, local=(1e4, None), glob=(1e4, None), period=2,
                           decoder_params=2614341888),
    "gemma-2-9b":     dict(cls="Gemma2ForCausalLM", layers=42, hidden=3584, head_dim=256, heads=(16, 8), window=4096,
                           attn_cap=50.0, final_cap=30.0, scalar=256, local=(1e4, None), glob=(1e4, None), period=2,
                           decoder_params=9241705984),
    "gemma-2-27b":    dict(cls="Gemma2ForCausalLM", layers=46, hidden=4608, head_dim=128, heads=(32, 16), window=4096,
                           attn_cap=50.0, final_cap=30.0, scalar=144, local=(1e4, None), glob=(1e4, None), period=2,
                           decoder_params=27227128320),
    "gemma-3-270m":   dict(cls="Gemma3ForCausalLM", layers=18, hidden=640, head_dim=256, heads=(4, 1), window=512,
                           attn_cap=None, final_cap=None, scalar=256, local=(1e4, None), glob=(1e6, None), period=6,
                           decoder_params=268098176),
    "gemma-3-1b-it":  dict(cls="Gemma3ForCausalLM", layers=26, hidden=1152, head_dim=256, heads=(4, 1), window=512,
                           attn_cap=None, final_cap=None, scalar=256, local=(1e4, None), glob=(1e6, None), period=6,
                           decoder_params=999885952),
    "gemma-3-4b-it":  dict(cls="Gemma3ForConditionalGeneration", layers=34, hidden=2560, head_dim=256, heads=(8, 4),
                           window=1024, attn_cap=None, final_cap=None, scalar=256, local=(1e4, None),
                           glob=(1e6, _LINEAR8), period=6, decoder_params=3880263168),
    "gemma-3-12b-it": dict(cls="Gemma3ForConditionalGeneration", layers=48, hidden=3840, head_dim=256, heads=(16, 8),
                           window=1024, attn_cap=None, final_cap=None, scalar=256, local=(1e4, None),
                           glob=(1e6, _LINEAR8), period=6, decoder_params=11766034176),
    "gemma-3-27b-it": dict(cls="Gemma3ForConditionalGeneration", layers=62, hidden=5376, head_dim=128, heads=(32, 16),
                           window=1024, attn_cap=None, final_cap=None, scalar=168, local=(1e4, None),
                           glob=(1e6, _LINEAR8), period=6, decoder_params=27009346304),
}

#: length of the parity sequences: several windows long, so the local and
#: the global blocks see different contexts at most positions
SEQ_LEN = 24
#: length of the sequences whose full knowledge matrix is computed
KM_SEQ_LEN = 9


### Building the references ###

def published_config(name):
    """The published config.json of google/<name>, as transformers normalizes it."""
    directory = tempfile.mkdtemp()
    try:
        shutil.copy(os.path.join(FIXTURE_DIR, f"google__{name}.json"),
                    os.path.join(directory, "config.json"))
        return transformers.AutoConfig.from_pretrained(directory)
    finally:
        shutil.rmtree(directory)


def tiny_config(name, **overrides):
    """
        The published config of google/<name> with ONLY its sizes scaled
        down (TINY), read back through transformers.AutoConfig exactly as
        the real one would be. `overrides` set further keys of the text
        config (for the tests of optional features).
    """
    with open(os.path.join(FIXTURE_DIR, f"google__{name}.json")) as f:
        raw = json.load(f)
    text = raw.get("text_config", raw)
    text.update(TINY[name])
    text.update(overrides)
    if "layer_types" in text:
        text["layer_types"] = text["layer_types"][: text["num_hidden_layers"]]
    if "vision_config" in raw:
        raw["vision_config"].update(TINY_VISION)
        raw["mm_tokens_per_image"] = 4
    directory = tempfile.mkdtemp()
    try:
        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump(raw, f)
        return transformers.AutoConfig.from_pretrained(directory)
    finally:
        shutil.rmtree(directory)


def text_config(hf):
    return getattr(hf.config, "text_config", None) or hf.config


def text_decoder(hf):
    """The text decoder of a causal LM or of the multimodal wrapper."""
    return getattr(hf.model, "language_model", hf.model)


def enliven(hf, seed, qk_gain=1.0, weight_gain=1.0, embed_gain=1.0):
    """
        Replace the freshly initialized weights of the text decoder by
        ones that make every mechanism count. The transformers
        initialization would hide most errors: its norm weights are all
        zero (scale exactly 1, so a swapped or un-centered norm weight
        goes unnoticed) and its attention scores are ~0.01 (uniform
        attention, on which RoPE, soft-capping, the query scalar and the
        q/k norms have no visible effect).

        - norm weights w ~ U(-0.5, 1), i.e. scales 1 + w in (0.5, 2),
        - q/k norm weights ~ U(1, 4), giving attention scores of a few
          units after the division by sqrt(query_pre_attn_scalar),
        - matrices ~ N(0, 1 / fan_in), the q and k projections times
          `qk_gain` (Gemma-2 has no q/k norm: this is what brings its
          scores to the range where the cap at 50 matters),
        - biases, where present, ~ U(-0.5, 1).
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, prm in text_decoder(hf).named_parameters():
            shape, kw = prm.shape, dict(generator=g, dtype=torch.float64)
            if name.endswith("embed_tokens.weight"):
                new = torch.randn(shape, **kw) * embed_gain
            elif name.endswith("q_norm.weight") or name.endswith("k_norm.weight"):
                new = torch.rand(shape, **kw) * 3.0 + 1.0
            elif prm.dim() == 1:
                new = torch.rand(shape, **kw) * 1.5 - 0.5
            else:
                gain = qk_gain if (".q_proj." in name or ".k_proj." in name) else 1.0
                new = torch.randn(shape, **kw) * gain * weight_gain / shape[1] ** 0.5
            prm.copy_(new.to(prm.dtype))
    return hf


_REFERENCES = {}


def reference(name, seed=0, direct=False, weight_gain=1.0, embed_gain=1.0, attn="eager", **overrides):
    """
        The reference model: the actual transformers class of the
        checkpoint (Gemma2ForCausalLM, Gemma3ForCausalLM or the multimodal
        Gemma3ForConditionalGeneration) on the scaled-down published
        config, in float64, with lively weights. Cached: the tests never
        modify a reference.

        It is built with `_from_config(dtype=float64)`, the way
        `from_pretrained(dtype=...)` instantiates a model, so every
        sub-model is created under float64. `direct=True` calls the
        constructor instead; the text decoder of a multimodal model is
        then created under the dtype its sub-config names (bfloat16),
        which leaves a bfloat16-rounded embedding factor behind (see
        TestEmbeddingScale).
    """
    key = (name, seed, direct, weight_gain, embed_gain, attn, tuple(sorted(overrides.items())))
    if key not in _REFERENCES:
        config = tiny_config(name, **overrides)
        cls = getattr(transformers, config.architectures[0])
        torch.manual_seed(seed)
        if direct:
            config._attn_implementation = attn
            hf = cls(config).double()
        else:
            hf = cls._from_config(config, dtype=torch.float64, attn_implementation=attn)
        hf.eval()
        enliven(hf, seed + 1000, qk_gain=4.0 if name.startswith("gemma-2") else 1.0,
                weight_gain=weight_gain, embed_gain=embed_gain)
        _REFERENCES[key] = hf
    return _REFERENCES[key]


def token_ids(hf, length=SEQ_LEN, seed=1):
    """Random token ids, with the padding id (0) and a repeated token among them."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, text_config(hf).vocab_size, (1, length), generator=g)
    ids[0, 3] = 0
    ids[0, length - 2] = ids[0, 1]
    return ids


def reference_logits(hf, ids):
    with torch.no_grad():
        return hf(ids).logits


def reference_hidden(hf, ids):
    """Final hidden states (after the last norm), through the real base model."""
    with torch.no_grad():
        return hf.model(input_ids=ids).last_hidden_state


def run(net, ids):
    net.eval()
    with torch.no_grad():
        return net.forward(ids.unsqueeze(0))


def rel_gap(ref, got):
    """Largest deviation, relative to the largest reference value."""
    ref, got = ref.reshape(-1), got.reshape(-1).to(ref.dtype)
    return ((ref - got).abs().max() / ref.abs().max()).item()


def attention_layers(net):
    return [layer for layer in net.layers if isinstance(layer, RopeAttention)]


@contextlib.contextmanager
def patched(obj, **attrs):
    """Temporarily set attributes on a config (restored on exit)."""
    missing = object()
    old = {k: getattr(obj, k, missing) for k in attrs}
    try:
        for k, v in attrs.items():
            setattr(obj, k, v)
        yield obj
    finally:
        for k, v in old.items():
            if v is missing:
                delattr(obj, k)
            else:
                setattr(obj, k, v)


def second_device():
    """Two distinct devices, when the machine has them (as in model_split.py)."""
    if torch.cuda.device_count() >= 2:
        return ["cuda:0", "cuda:1"]
    if torch.cuda.is_available():
        return ["cpu", "cuda:0"]
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return ["cpu", "mps"]
    return None


def tiny_gemma(**kwargs):
    """A GemmaLike built directly, with the keywords a caller would use."""
    params = dict(
        vocab_size=50,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=6,
        attn_softcap=50.0,
        query_pre_attn_scalar=9,
        sliding_window=4,
    )
    params.update(kwargs)
    return GemmaLike(**params)


def randomize(net, seed=0):
    """Lively weights for a directly built network (norm scales away from 1)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for prm in net.parameters():
            if prm.dim() == 1:
                prm.copy_(torch.rand(prm.shape, generator=g) * 1.5 + 0.5)
            else:
                prm.copy_(torch.randn(prm.shape, generator=g) * 2.0 / prm.shape[1] ** 0.5)
    return net


### 1. The soft-cap layer ###

class TestSoftCapLayer(unittest.TestCase):
    """
        Gemma-2's final logit soft-capping as a layer of the network:
        SoftCap(c)(x) = c * tanh(x / c), an elementwise activation that
        the knowledge matrix linearizes by its post/pre ratio.
    """

    def test_formula_and_bounds(self):
        cap = SoftCap(30.0)
        x = torch.linspace(-200, 200, 4001)
        y = cap(x)
        self.assertTrue(torch.equal(y, torch.tanh(x / 30.0) * 30.0))
        self.assertLessEqual(y.abs().max().item(), 30.0)
        # the identity to first order at 0: c * tanh(x / c) = x - x^3 / (3 c^2) + ...
        small = torch.tensor([1e-3, -2e-3])
        self.assertLess((cap(small) - (small - small ** 3 / 2700.0)).abs().max().item(), 1e-15)
        self.assertEqual(cap(torch.zeros(3)).abs().max().item(), 0.0)

    def test_cap_must_be_positive(self):
        for bad in (0.0, -1.0, float("nan")):
            with self.assertRaises(ValueError):
                SoftCap(bad)

    def test_builder_and_registration(self):
        net = NN(input_shape=(1, 1, 4))
        net.linear(4, 4)
        net.softcap(2.5)
        self.assertIsInstance(net.layers[-1], SoftCap)
        self.assertEqual(net.layers[-1].cap, 2.5)
        # an activation (ratio), not a mixer: there is no routing to
        # freeze, so "frozen" mode treats it by the ratio as well
        self.assertIn(SoftCap, ACTIVATION_LAYERS)
        self.assertNotIn(SoftCap, LINEARIZABLE_LAYERS)
        self.assertFalse(hasattr(SoftCap, "frozen_forward"))
        self.assertEqual(sum(p.numel() for p in net.layers[-1].parameters()), 0)

    def test_ratio_reproduces_the_layer_at_the_input(self):
        # the linearization the matrix computers apply: x -> x * (post / pre)
        cap = SoftCap(30.0)
        x0 = torch.randn(1, 1, 5, 40) * 40
        ratio = cap(x0) / x0
        self.assertLess((x0 * ratio - cap(x0)).abs().max().item(), 1e-13)
        self.assertTrue(((ratio > 0) & (ratio <= 1)).all())

    CAP = 5.0

    def _capped_and_raw(self, seed=0, **kwargs):
        """The same network with and without a final soft-cap."""
        torch.manual_seed(seed)
        capped = randomize(tiny_gemma(final_softcap=self.CAP, **kwargs), seed)
        raw = tiny_gemma(**kwargs)
        raw.load_state_dict(capped.state_dict())
        return capped, raw

    def test_network_outputs_the_capped_logits(self):
        capped, raw = self._capped_and_raw()
        x = torch.randint(0, 50, (1, 1, 7))
        z, y = run(raw, x[0]), run(capped, x[0])
        self.assertGreater(z.abs().max().item(), self.CAP)   # the cap is active
        self.assertLess((y - self.CAP * torch.tanh(z / self.CAP)).abs().max().item(), 1e-12)
        self.assertLess(y.abs().max().item(), self.CAP)
        self.assertGreater((y - z).abs().max().item(), 1.0)

    def test_matrix_rows_sum_to_the_capped_logits(self):
        capped, raw = self._capped_and_raw()
        x = torch.randint(0, 50, (1, 1, 7))
        y = run(capped, x[0]).reshape(-1)
        z = run(raw, x[0]).reshape(-1)
        for mode in ("ratio", "frozen"):
            with self.subTest(mixer_mode=mode):
                mat = KnowledgeMatrixComputer(capped, batch_size=64, mixer_mode=mode).forward(x)
                self.assertEqual(tuple(mat.shape), (7 * 50, 7 * 24 + 1))
                self.assertLess((mat.sum(1) - y).abs().max().item(), 1e-9)
                # ... and the soft-cap acts on the matrix as the diagonal
                # map it is: row j of the capped network is row j of the
                # uncapped one times cap(z_j) / z_j.
                mat_raw = KnowledgeMatrixComputer(raw, batch_size=64, mixer_mode=mode).forward(x)
                self.assertLess((mat_raw.sum(1) - z).abs().max().item(), 1e-9)
                expected = mat_raw * (y / z).unsqueeze(1)
                self.assertLess((mat - expected).abs().max().item(), 1e-9)

    def test_row_computer_goes_through_the_cap(self):
        capped, _ = self._capped_and_raw()
        x = torch.randint(0, 50, (1, 1, 7))
        y = run(capped, x[0]).reshape(-1)
        rows = [0, 17, 173, 7 * 50 - 1]
        for mode in ("ratio", "frozen"):
            with self.subTest(mixer_mode=mode):
                mat = KnowledgeMatrixComputer(capped, batch_size=64, mixer_mode=mode).forward(x)
                got = KnowledgeRowComputer(capped, mixer_mode=mode).forward(x, rows)
                self.assertLess((got - mat[rows]).abs().max().item(), 1e-9)
                self.assertLess((got.sum(1) - y[rows]).abs().max().item(), 1e-9)

    def test_a_logit_of_exactly_zero(self):
        # post / pre is 0 / 0 there; the computers put 0, and the row
        # still sums to the output (0).
        net = NN(input_shape=(1, 1, 3))
        net.flatten()
        net.linear(3, 3, bias=False)
        net.softcap(5.0)
        with torch.no_grad():
            net.layers[1].weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, -3.0, 9.0]]))
        x = torch.tensor([[[4.0, 1.0, -2.0]]])
        y = net.forward(x).reshape(-1)
        self.assertEqual(y[1].item(), 0.0)
        self.assertLess((y - 5.0 * torch.tanh(torch.tensor([4.0, 0.0, -13.0]) / 5.0)).abs().max().item(), 1e-15)
        mat = KnowledgeMatrixComputer(net, batch_size=4).forward(x)
        self.assertTrue(torch.isfinite(mat).all())
        self.assertLess((mat.sum(1) - y).abs().max().item(), 1e-12)


class TestZeroPreActivation(unittest.TestCase):
    """
        Trained Gemma-3 checkpoints contain norm weights of exactly -1,
        i.e. a scale 1 + w of exactly 0 (143 of them in gemma-3-270m; one
        coordinate of the pre-feedforward norm is dead in every block).
        The mixing layer behind such a norm receives exactly 0 in that
        coordinate while its output there is not 0, so the post/pre ratio
        of "ratio" mode does not exist. The output is then a constant of
        the linearization and belongs to the bias column; dropping it
        (ratio := 0) breaks M(x) . 1 = f(x) in those coordinates.
    """

    def _network(self, include_lm_head=True):
        net = randomize(tiny_gemma(qk_norm=True, include_lm_head=include_lm_head), seed=3)
        stride = GemmaLike.BLOCK_STRIDE
        with torch.no_grad():
            net.layers[1 + 4].weight[5] = 0.0             # pre-feedforward norm, block 0
            net.layers[1 + stride].weight[7] = 0.0        # input norm, block 1
            net.layers[1 + stride + 4].weight[[5, 11]] = 0.0
            net.layers[1 + stride + 1].k_norm.weight[2] = 0.0
        return net

    def test_the_mixers_do_receive_exact_zeros(self):
        net = self._network()
        x = torch.randint(0, 50, (1, 1, 7))
        net.save = True
        with torch.no_grad():
            net.forward(x)
        net.save = False
        for index, coordinates in ((1 + 5, [5]), (1 + 8 + 1, [7]), (1 + 8 + 5, [5, 11])):
            pre, post = net.pre_acts[index], net.acts[index]
            self.assertEqual(pre[..., coordinates].abs().max().item(), 0.0)
            self.assertGreater(post[..., coordinates].abs().min().item(), 1e-6)

    def test_ratio_rows_still_sum_to_the_output(self):
        for head in (True, False):
            with self.subTest(include_lm_head=head):
                net = self._network(include_lm_head=head)
                x = torch.randint(0, 50, (1, 1, 7))
                out = run(net, x[0]).reshape(-1)
                mat = KnowledgeMatrixComputer(net, batch_size=64, mixer_mode="ratio").forward(x)
                self.assertLess((mat.sum(1) - out).abs().max().item(), 1e-9)
                # the row computer agrees, bias entry included
                rows = list(range(0, out.numel(), 7))
                got = KnowledgeRowComputer(net, mixer_mode="ratio").forward(x, rows)
                self.assertLess((got - mat[rows]).abs().max().item(), 1e-9)

    def test_the_lost_output_is_in_the_bias_column(self):
        # without the head the ratio-mode matrix is diagonal over the
        # coordinates: all that reaches a dead coordinate is bias
        net = self._network(include_lm_head=False)
        x = torch.randint(0, 50, (1, 1, 7))
        mat = KnowledgeMatrixComputer(net, batch_size=64, mixer_mode="ratio").forward(x)
        self.assertGreater(mat[:, -1].abs().max().item(), 1e-3)
        # ... and the network has no other source of bias
        clean = randomize(tiny_gemma(qk_norm=True, include_lm_head=False), seed=3)
        mat = KnowledgeMatrixComputer(clean, batch_size=64, mixer_mode="ratio").forward(x)
        self.assertEqual(mat[:, -1].abs().max().item(), 0.0)

    def test_frozen_mode(self):
        net = self._network()
        x = torch.randint(0, 50, (1, 1, 7))
        out = run(net, x[0]).reshape(-1)
        mat = KnowledgeMatrixComputer(net, batch_size=64, mixer_mode="frozen").forward(x)
        self.assertLess((mat.sum(1) - out).abs().max().item(), 1e-9)
        self.assertEqual(mat[:, -1].abs().max().item(), 0.0)    # nothing is lost there

    def test_activation_with_a_nonzero_value_at_zero(self):
        # the same rule for an ordinary activation: sigmoid(0) = 1/2
        net = NN(input_shape=(1, 1, 3))
        net.flatten()
        net.linear(3, 3)
        net.sigmoid()
        net.linear(3, 2)
        with torch.no_grad():
            net.layers[1].weight[1] = 0.0
            net.layers[1].bias[1] = 0.0
        x = torch.tensor([[[0.3, -1.2, 2.0]]])
        y = net.forward(x).reshape(-1)
        mat = KnowledgeMatrixComputer(net, batch_size=4).forward(x)
        self.assertLess((mat.sum(1) - y).abs().max().item(), 1e-12)


### 2. The constructor ###

class TestConstructor(unittest.TestCase):
    """The existing keywords keep their meaning; the new ones default to it."""

    OLD_KEYWORDS = dict(
        vocab_size=50, hidden_size=24, intermediate_size=48, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=6, rope_theta=10000.0,
        rms_norm_eps=1e-6, attn_softcap=50.0, query_pre_attn_scalar=9, sliding_window=4,
        alternate_sliding=True, include_lm_head=True, save=False, device="cpu",
    )

    def test_old_keywords_build_the_gemma2_network(self):
        net = GemmaLike(**self.OLD_KEYWORDS)
        self.assertEqual(net.get_num_layers(), 1 + 4 * GemmaLike.BLOCK_STRIDE + 2)
        self.assertIsInstance(net.layers[-1], torch.nn.Linear)       # no cap unless asked for
        self.assertIsNone(net.final_softcap)
        self.assertEqual(net.layer_types, [SLIDING, FULL, SLIDING, FULL])
        self.assertAlmostEqual(net.embed_scale, 24 ** 0.5)
        for attn, window in zip(attention_layers(net), [4, None, 4, None]):
            self.assertEqual(attn.sliding_window, window)
            self.assertEqual(attn.rope_theta, 10000.0)
            self.assertIsNone(attn.rope_scaling)
            self.assertFalse(attn.use_qk_norm)
            self.assertEqual(attn.softcap, 50.0)
            self.assertAlmostEqual(attn.query_scale, 3.0)
            self.assertIsNone(attn.q_proj.bias)
        self.assertEqual(run(net, torch.randint(0, 50, (1, 7))).shape, (1, 1, 7, 50))

    def test_old_sliding_options(self):
        every = GemmaLike(**dict(self.OLD_KEYWORDS, alternate_sliding=False))
        self.assertEqual([a.sliding_window for a in attention_layers(every)], [4, 4, 4, 4])
        never = GemmaLike(**dict(self.OLD_KEYWORDS, sliding_window=None))
        self.assertEqual([a.sliding_window for a in attention_layers(never)], [None] * 4)
        self.assertEqual(never.layer_types, [FULL] * 4)

    def test_query_scalar_defaults_to_head_dim(self):
        net = GemmaLike(**dict(self.OLD_KEYWORDS, query_pre_attn_scalar=None))
        self.assertAlmostEqual(attention_layers(net)[0].query_scale, 6 ** 0.5)

    def test_final_softcap_is_a_layer_after_the_head(self):
        net = GemmaLike(**dict(self.OLD_KEYWORDS, final_softcap=30.0))
        self.assertEqual(net.get_num_layers(), 1 + 4 * GemmaLike.BLOCK_STRIDE + 3)
        self.assertIsInstance(net.layers[-2], torch.nn.Linear)
        self.assertIsInstance(net.layers[-1], SoftCap)
        self.assertEqual(net.layers[-1].cap, 30.0)
        self.assertEqual(net.final_softcap, 30.0)

    def test_final_softcap_needs_the_head(self):
        net = GemmaLike(**dict(self.OLD_KEYWORDS, include_lm_head=False, final_softcap=30.0))
        self.assertIsInstance(net.layers[-1], RMSNorm)
        self.assertIsNone(net.final_softcap)
        self.assertEqual(run(net, torch.randint(0, 50, (1, 5))).shape[-1], 24)

    def test_gemma3_options(self):
        types = [SLIDING] * 5 + [FULL] + [SLIDING]
        net = GemmaLike(**dict(
            self.OLD_KEYWORDS, num_hidden_layers=7, attn_softcap=None, layer_types=types,
            qk_norm=True, rope_theta=1e6, rope_scaling={"rope_type": "linear", "factor": 8.0},
            rope_local_theta=1e4,
        ))
        self.assertEqual(net.layer_types, types)
        for attn, kind in zip(attention_layers(net), types):
            self.assertTrue(attn.use_qk_norm)
            if kind == SLIDING:
                self.assertEqual((attn.sliding_window, attn.rope_theta, attn.rope_scaling), (4, 1e4, None))
            else:
                self.assertEqual((attn.sliding_window, attn.rope_theta), (None, 1e6))
                self.assertEqual(attn.rope_scaling, {"rope_type": "linear", "factor": 8.0})

    def test_one_rope_for_all_blocks_unless_a_local_base_is_given(self):
        scaling = {"rope_type": "linear", "factor": 2.0}
        net = GemmaLike(**dict(self.OLD_KEYWORDS, rope_theta=5e4, rope_scaling=scaling))
        for attn in attention_layers(net):
            self.assertEqual((attn.rope_theta, attn.rope_scaling), (5e4, scaling))

    def test_attention_bias_and_activation(self):
        net = GemmaLike(**dict(self.OLD_KEYWORDS, attention_bias=True, hidden_activation="gelu"))
        attn = attention_layers(net)[0]
        for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
            self.assertIsNotNone(proj.bias)
        self.assertEqual([l.approximate for l in net.layers if isinstance(l, GeGLU)], ["none"] * 4)
        default = GemmaLike(**self.OLD_KEYWORDS)
        self.assertEqual([l.approximate for l in default.layers if isinstance(l, GeGLU)], ["tanh"] * 4)

    def test_validation(self):
        with self.assertRaises(ValueError):       # one entry per block
            GemmaLike(**dict(self.OLD_KEYWORDS, layer_types=[SLIDING, FULL]))
        with self.assertRaises(NotImplementedError):
            GemmaLike(**dict(self.OLD_KEYWORDS, layer_types=[SLIDING, FULL, "chunked_attention", FULL]))
        with self.assertRaises(ValueError):       # local blocks need a window
            GemmaLike(**dict(self.OLD_KEYWORDS, sliding_window=None, layer_types=[SLIDING, FULL] * 2))
        with self.assertRaises(ValueError):
            GemmaLike(**dict(self.OLD_KEYWORDS, rope_local_scaling={"rope_type": "linear", "factor": 8.0}))
        with self.assertRaises(NotImplementedError):
            GemmaLike(**dict(self.OLD_KEYWORDS, hidden_activation="silu"))
        with self.assertRaises(NotImplementedError):
            GemmaLike(**dict(self.OLD_KEYWORDS, rope_scaling={"rope_type": "yarn", "factor": 8.0}))

    def test_builds_without_storage(self):
        # the constructor must not evaluate the network (the loaders
        # build it on the "meta" device)
        net = empty_network(GemmaLike, **dict(self.OLD_KEYWORDS, final_softcap=30.0, qk_norm=True))
        self.assertTrue(all(p.is_meta for p in net.parameters()))
        self.assertEqual(len(net.residuals), 2 * 4)

    def test_frozen_linearization_is_exact_with_the_gemma_options(self):
        # frozen_forward(x0, x0) == forward(x0), bit for bit, for the
        # attention as Gemma-2 and as Gemma-3 configure it
        configs = {
            "gemma-2 local": dict(softcap=50.0, query_scale=3.0, sliding_window=3, num_kv_heads=2),
            "gemma-3 local": dict(qk_norm=True, query_scale=16.0, sliding_window=3, num_kv_heads=1, rope_theta=1e4),
            "gemma-3 global": dict(qk_norm=True, query_scale=16.0, num_kv_heads=2, rope_theta=1e6,
                                   rope_scaling={"rope_type": "linear", "factor": 8.0}),
            "biases": dict(qk_norm=True, bias=True, o_bias=True),
        }
        for name, cfg in configs.items():
            with self.subTest(config=name):
                torch.manual_seed(0)
                attn = RopeAttention(d_model=20, num_heads=4, head_dim=6, **cfg)
                with torch.no_grad():
                    for prm in attn.parameters():
                        prm.mul_(4.0)
                x0 = torch.randn(1, 1, 9, 20)
                gap = (attn.frozen_forward(x0, x0) - attn(x0)).abs().max().item()
                self.assertEqual(gap, 0.0)


### 3. Parity with the reference implementation ###

@unittest.skipUnless(HAS_TRANSFORMERS, "transformers / safetensors are not installed")
class TestTinyConfigsAreFaithful(unittest.TestCase):
    """The scaled-down configs differ from the published ones in their sizes only."""

    def test_published_switches(self):
        # what the loader has to reproduce, read from the fixtures at
        # their real size
        for name in NAMES:
            with self.subTest(checkpoint=name):
                config = published_config(name)
                text = getattr(config, "text_config", None) or config
                pub = PUBLISHED[name]
                self.assertEqual(config.architectures, [pub["cls"]])
                self.assertEqual(text.num_hidden_layers, pub["layers"])
                self.assertEqual(text.hidden_size, pub["hidden"])
                self.assertEqual(text.head_dim, pub["head_dim"])
                self.assertEqual((text.num_attention_heads, text.num_key_value_heads), pub["heads"])
                self.assertEqual(text.sliding_window, pub["window"])
                self.assertEqual(text.attn_logit_softcapping, pub["attn_cap"])
                self.assertEqual(text.final_logit_softcapping, pub["final_cap"])
                self.assertEqual(text.query_pre_attn_scalar, pub["scalar"])
                self.assertEqual(text.hidden_activation, "gelu_pytorch_tanh")
                self.assertEqual(text.rms_norm_eps, 1e-6)
                self.assertFalse(text.attention_bias)
                self.assertFalse(text.use_bidirectional_attention)
                self.assertTrue(text.tie_word_embeddings)
                expected = [FULL if (n + 1) % pub["period"] == 0 else SLIDING for n in range(pub["layers"])]
                self.assertEqual(list(text.layer_types), expected)
                # the old attribute is gone: reading it with a default of
                # 10000 was a silent bug
                self.assertFalse(hasattr(text, "rope_theta"))

    def test_only_the_sizes_were_scaled(self):
        for name in NAMES:
            with self.subTest(checkpoint=name):
                full, tiny = published_config(name), tiny_config(name)
                self.assertIs(type(tiny), type(full))
                self.assertEqual(tiny.architectures, full.architectures)
                full_text = (getattr(full, "text_config", None) or full).to_dict()
                tiny_text = (getattr(tiny, "text_config", None) or tiny).to_dict()
                self.assertEqual(set(full_text), set(tiny_text))
                for key, value in full_text.items():
                    if key in SIZE_KEYS:
                        self.assertEqual(tiny_text[key], TINY[name][key])
                    elif key == "layer_types":
                        self.assertEqual(tiny_text[key], value[: TINY[name]["num_hidden_layers"]])
                    elif key == "_name_or_path":
                        pass    # where the file was read from
                    else:
                        self.assertEqual(tiny_text[key], value, key)

    def test_the_scaled_models_exercise_every_mechanism(self):
        for name in NAMES:
            with self.subTest(checkpoint=name):
                text = text_config(reference(name))
                pub, tiny = PUBLISHED[name], TINY[name]
                # GQA / MQA ratio of the published model
                self.assertEqual(text.num_attention_heads // text.num_key_value_heads,
                                 pub["heads"][0] // pub["heads"][1])
                self.assertNotEqual(text.num_attention_heads * text.head_dim, text.hidden_size)
                self.assertNotEqual(text.query_pre_attn_scalar, text.head_dim)
                # the sequence is several windows long, and both kinds of
                # block are there, the local one first
                self.assertGreater(SEQ_LEN, 3 * text.sliding_window)
                self.assertGreater(KM_SEQ_LEN, text.sliding_window)
                self.assertEqual(text.layer_types[0], SLIDING)
                self.assertIn(FULL, text.layer_types)
                self.assertGreaterEqual(tiny["num_hidden_layers"], pub["period"] + 1)


@unittest.skipUnless(HAS_TRANSFORMERS, "transformers / safetensors are not installed")
class TestParity(unittest.TestCase):
    """
        Our network against the actual transformers class, for the eight
        published configs: logits (with the LM head -- for Gemma-2 the
        soft-capped ones) and final hidden states (without).
    """

    def assertParity(self, ref, got, what=""):
        gap = rel_gap(ref, got)
        self.assertLess(gap, HF_TOL, f"{what}: differs from the reference by {gap:.3e} (relative)")
        return gap

    def test_logits_from_the_model_in_memory(self):
        for name in NAMES:
            with self.subTest(checkpoint=name):
                hf = reference(name)
                self.assertEqual(type(hf).__name__, PUBLISHED[name]["cls"])
                ids = token_ids(hf)
                net = GemmaLike.from_huggingface(hf)
                got = run(net, ids)
                self.assertEqual(got.dtype, torch.float64)
                self.assertEqual(tuple(got.shape), (1, 1, SEQ_LEN, TINY[name]["vocab_size"]))
                self.assertParity(reference_logits(hf, ids), got, name)

    def test_hidden_states_from_the_model_in_memory(self):
        for name in NAMES:
            with self.subTest(checkpoint=name):
                hf = reference(name)
                ids = token_ids(hf)
                net = GemmaLike.from_huggingface(hf, include_lm_head=False)
                self.assertIsInstance(net.layers[-1], RMSNorm)
                got = run(net, ids)
                self.assertEqual(tuple(got.shape), (1, 1, SEQ_LEN, TINY[name]["hidden_size"]))
                self.assertParity(reference_hidden(hf, ids), got, name)

    def test_streamed_from_a_save_pretrained_directory(self):
        # the network is built from the safetensors file alone; the
        # reference is what transformers makes of the same directory
        for name in NAMES:
            with self.subTest(checkpoint=name), tempfile.TemporaryDirectory() as directory:
                hf = reference(name)
                ids = token_ids(hf)
                hf.save_pretrained(directory)
                reloaded = type(hf).from_pretrained(
                    directory, dtype=torch.float64, attn_implementation="eager").eval()

                net = GemmaLike.from_huggingface(directory, dtype=torch.float64)
                self.assertTrue(all(p.dtype == torch.float64 for p in net.parameters()))
                got = run(net, ids)
                self.assertParity(reference_logits(reloaded, ids), got, f"{name} / reloaded")
                self.assertParity(reference_logits(hf, ids), got, f"{name} / in memory")
                # same network as from memory, to the last bit
                self.assertTrue(torch.equal(got, run(GemmaLike.from_huggingface(hf), ids)))

                headless = GemmaLike.from_huggingface(directory, dtype=torch.float64, include_lm_head=False)
                self.assertParity(reference_hidden(reloaded, ids), run(headless, ids), f"{name} / hidden")

    def test_from_disk_the_default_dtype_is_float32(self):
        with tempfile.TemporaryDirectory() as directory:
            hf = reference("gemma-3-270m")
            hf.save_pretrained(directory)
            net = GemmaLike.from_huggingface(directory)
            self.assertTrue(all(p.dtype == torch.float32 for p in net.parameters()))
            ids = token_ids(hf)
            got = run(net, ids)
            self.assertEqual(got.dtype, torch.float32)
            # float32 arithmetic against the float32 reference
            ref32 = type(hf).from_pretrained(directory, dtype=torch.float32, attn_implementation="eager").eval()
            self.assertLess(rel_gap(reference_logits(ref32, ids), got), 2e-5)

    def test_multimodal_checkpoints_in_both_key_layouts(self):
        # transformers writes the text decoder of Gemma3ForConditionalGeneration
        # as "language_model.model.*" (the layout of the published
        # checkpoints) and holds it in memory as "model.language_model.*";
        # some versions wrote the latter to disk as well.
        for name in MULTIMODAL:
            with self.subTest(checkpoint=name), tempfile.TemporaryDirectory() as root:
                hf = reference(name)
                ids = token_ids(hf)
                ref = reference_logits(hf, ids)

                published = os.path.join(root, "published")
                hf.save_pretrained(published)
                keys = set(load_file(os.path.join(published, "model.safetensors")))
                self.assertIn("language_model.model.embed_tokens.weight", keys)
                self.assertTrue(any(k.startswith("vision_tower.") for k in keys))
                self.assertEqual(open_checkpoint(published).text_prefix, "language_model.model.")

                in_memory = os.path.join(root, "in_memory_names")
                os.mkdir(in_memory)
                hf.config.save_pretrained(in_memory)
                state = {k: v.detach().clone() for k, v in hf.state_dict().items()}
                self.assertIn("model.language_model.embed_tokens.weight", state)
                self.assertIn("lm_head.weight", state)
                save_file(state, os.path.join(in_memory, "model.safetensors"))
                self.assertEqual(open_checkpoint(in_memory).text_prefix, "model.language_model.")

                untied = os.path.join(root, "without_head")
                os.mkdir(untied)
                hf.config.save_pretrained(untied)
                del state["lm_head.weight"]            # tied: the table is the head
                save_file(state, os.path.join(untied, "model.safetensors"))

                for directory in (published, in_memory, untied):
                    net = GemmaLike.from_huggingface(directory, dtype=torch.float64)
                    self.assertParity(ref, run(net, ids), f"{name} / {os.path.basename(directory)}")

    def test_the_vision_tower_is_not_loaded(self):
        hf = reference("gemma-3-4b-it")
        net = GemmaLike.from_huggingface(hf)
        decoder = sum(p.numel() for p in text_decoder(hf).parameters())
        self.assertEqual(sum(p.numel() for p in net.parameters()), decoder + hf.lm_head.weight.numel())
        self.assertGreater(sum(p.numel() for p in hf.parameters()), decoder)

    def test_bare_text_model_as_source(self):
        # the decoder without its causal-LM shell: no head is stored, the
        # (tied) embedding table is the head
        for name in ("gemma-2-9b", "gemma-3-1b-it", "gemma-3-12b-it"):
            with self.subTest(checkpoint=name):
                hf = reference(name)
                ids = token_ids(hf)
                bare = text_decoder(hf)
                self.assertEqual(open_checkpoint(bare).text_prefix, "")
                headless = GemmaLike.from_huggingface(bare, include_lm_head=False)
                self.assertParity(reference_hidden(hf, ids), run(headless, ids), name)
                if name not in MULTIMODAL:
                    self.assertParity(reference_logits(hf, ids), run(GemmaLike.from_huggingface(bare), ids), name)

    def test_config_without_layer_types(self):
        # older transformers versions did not store the block pattern
        for name in ("gemma-2-2b", "gemma-3-4b-it"):
            with self.subTest(checkpoint=name):
                hf = reference(name)
                ids = token_ids(hf)
                expected = list(text_config(hf).layer_types)
                with patched(text_config(hf), layer_types=None):
                    net = GemmaLike.from_huggingface(hf)
                self.assertEqual(net.layer_types, expected)
                self.assertParity(reference_logits(hf, ids), run(net, ids), name)

    def test_optional_features_of_the_reference(self):
        # switches no published checkpoint uses, but the reference
        # implements and the network reproduces
        cases = {
            "attention biases (gemma-2)": ("gemma-2-2b", dict(attention_bias=True)),
            "attention biases (gemma-3)": ("gemma-3-270m", dict(attention_bias=True)),
            "exact GELU": ("gemma-3-1b-it", dict(hidden_activation="gelu")),
            "no attention cap (gemma-2)": ("gemma-2-9b", dict(attn_logit_softcapping=None)),
            "no final cap (gemma-2)": ("gemma-2-27b", dict(final_logit_softcapping=None)),
            "final cap on a text-only gemma-3": ("gemma-3-270m", dict(final_logit_softcapping=25.0)),
        }
        for label, (name, overrides) in cases.items():
            with self.subTest(case=label):
                hf = reference(name, **overrides)
                ids = token_ids(hf)
                net = GemmaLike.from_huggingface(hf)
                self.assertParity(reference_logits(hf, ids), run(net, ids), label)
                headless = GemmaLike.from_huggingface(hf, include_lm_head=False)
                self.assertParity(reference_hidden(hf, ids), run(headless, ids), label)
        net = GemmaLike.from_huggingface(reference("gemma-3-270m", final_logit_softcapping=25.0))
        self.assertEqual(net.final_softcap, 25.0)
        self.assertIsNone(GemmaLike.from_huggingface(
            reference("gemma-2-27b", final_logit_softcapping=None)).final_softcap)

    def test_eps_regime(self):
        # With ordinary weights the mean square entering a norm is ~1 and
        # eps = 1e-6 moves the result by ~5e-7: below the float32 noise
        # of the reference, so the parity tests above would not see a
        # wrong eps, nor one added outside the square root. Scaled-down
        # weights put every norm (the four of a block, the q/k norms, the
        # final one) AT the scale of eps, where it decides the output.
        for name in ("gemma-2-2b", "gemma-3-4b-it"):
            with self.subTest(checkpoint=name):
                hf = reference(name, weight_gain=1e-3, embed_gain=1e-4)
                ids = token_ids(hf)
                ref, ref_h = reference_logits(hf, ids), reference_hidden(hf, ids)
                self.assertParity(ref, run(GemmaLike.from_huggingface(hf), ids), name)
                self.assertParity(ref_h, run(GemmaLike.from_huggingface(hf, include_lm_head=False), ids), name)

                # control 1: another eps
                wrong = GemmaLike.from_huggingface(hf)
                for module in wrong.modules():
                    if isinstance(module, RMSNorm):
                        module.eps = 1e-5
                self.assertGreater(rel_gap(ref, run(wrong, ids)), 0.1)

                # control 2: eps outside the square root, x / (rms + eps)
                # -- at this scale that is x / rms to 1e-3, i.e. no eps
                wrong = GemmaLike.from_huggingface(hf)
                for module in wrong.modules():
                    if isinstance(module, RMSNorm):
                        module.eps = 0.0
                self.assertGreater(rel_gap(ref, run(wrong, ids)), 0.1)


@unittest.skipUnless(HAS_TRANSFORMERS, "transformers / safetensors are not installed")
class TestFinalSoftCap(unittest.TestCase):
    """Gemma-2's final logit soft-capping: on by default, removable, absent without the head."""

    def test_default_reproduces_the_reference_logits(self):
        for name in GEMMA2:
            with self.subTest(checkpoint=name):
                hf = reference(name)
                ids = token_ids(hf)
                net = GemmaLike.from_huggingface(hf)
                self.assertIsInstance(net.layers[-1], SoftCap)
                self.assertEqual(net.layers[-1].cap, 30.0)
                self.assertEqual(net.final_softcap, 30.0)
                ref = reference_logits(hf, ids)
                self.assertLess(rel_gap(ref, run(net, ids)), HF_TOL)
                self.assertLess(ref.abs().max().item(), 30.0)

    def test_uncapped_logits_on_request(self):
        for name in GEMMA2:
            with self.subTest(checkpoint=name):
                hf = reference(name)
                ids = token_ids(hf)
                net = GemmaLike.from_huggingface(hf, final_softcap=False)
                self.assertIsInstance(net.layers[-1], torch.nn.Linear)
                self.assertIsNone(net.final_softcap)
                with torch.no_grad():
                    uncapped = hf.lm_head(hf.model(input_ids=ids).last_hidden_state)
                got = run(net, ids)
                self.assertLess(rel_gap(uncapped, got), HF_TOL)
                # the cap is far from the identity on these logits: the
                # uncapped network is NOT the reference
                self.assertGreater(uncapped.abs().max().item(), 30.0)
                self.assertGreater(rel_gap(reference_logits(hf, ids), got), 0.1)

    def test_without_the_head_nothing_changes(self):
        hf = reference("gemma-2-2b")
        ids = token_ids(hf)
        a = GemmaLike.from_huggingface(hf, include_lm_head=False)
        b = GemmaLike.from_huggingface(hf, include_lm_head=False, final_softcap=False)
        for net in (a, b):
            self.assertIsInstance(net.layers[-1], RMSNorm)
            self.assertIsNone(net.final_softcap)
        self.assertTrue(torch.equal(run(a, ids), run(b, ids)))

    def test_gemma3_has_no_cap(self):
        for name in GEMMA3:
            with self.subTest(checkpoint=name):
                hf = reference(name)
                for flag in (True, False):
                    net = GemmaLike.from_huggingface(hf, final_softcap=flag)
                    self.assertIsInstance(net.layers[-1], torch.nn.Linear)
                    self.assertIsNone(net.final_softcap)


@unittest.skipUnless(HAS_TRANSFORMERS, "transformers / safetensors are not installed")
class TestNegativeControls(unittest.TestCase):
    """
        Each mechanism, broken alone in an otherwise correct network,
        must be SEEN by the parity comparison -- by a wide margin. The
        network is loaded correctly and then one attribute is changed,
        so every control isolates one mechanism. Several of them are the
        behaviour of the loader before this work (marked "old loader").
    """

    def control(self, name, mutate=None, minimum=CONTROL_MIN, **load_kwargs):
        hf = reference(name)
        ids = token_ids(hf)
        ref = reference_logits(hf, ids)
        net = GemmaLike.from_huggingface(hf, **load_kwargs)
        if mutate is not None:
            mutate(net)
        gap = rel_gap(ref, run(net, ids))
        self.assertGreater(gap, minimum, f"{name}: the control only misses by {gap:.3e}")
        # ... while the untouched network is within tolerance
        self.assertLess(rel_gap(ref, run(GemmaLike.from_huggingface(hf), ids)), HF_TOL)
        return gap

    def each(self, names, mutate, minimum=CONTROL_MIN):
        for name in names:
            with self.subTest(checkpoint=name):
                self.control(name, mutate, minimum)

    # -- RoPE --

    def test_rope_base_of_the_local_blocks(self):
        # old loader: one base for every block
        def one_base(net):
            for attn in attention_layers(net):
                attn.rope_theta = 1e6
        self.each(GEMMA3, one_base)

    def test_rope_base_of_the_global_blocks(self):
        # old loader under transformers 5: config.rope_theta is gone, the
        # default 10000 was used everywhere
        def one_base(net):
            for attn in attention_layers(net):
                attn.rope_theta = 1e4
        self.each(GEMMA3, one_base)

    def test_rope_bases_swapped(self):
        def swap(net):
            for attn in attention_layers(net):
                attn.rope_theta = 1e4 if attn.rope_theta == 1e6 else 1e6
        self.each(GEMMA3, swap)

    def test_rope_linear_rescaling(self):
        def no_rescaling(net):
            for attn in attention_layers(net):
                attn.rope_scaling = None
        self.each(MULTIMODAL, no_rescaling)

    def test_rope_rescaling_belongs_to_the_global_blocks_only(self):
        def rescale_all(net):
            for attn in attention_layers(net):
                attn.rope_scaling = {"rope_type": "linear", "factor": 8.0}
        self.each(MULTIMODAL, rescale_all)

    def test_rope_base_gemma2(self):
        def other_base(net):
            for attn in attention_layers(net):
                attn.rope_theta = 1e6
        self.each(GEMMA2, other_base)

    # -- queries and keys --

    def test_qk_norm(self):
        # old loader: no q/k normalization at all
        def no_norm(net):
            for attn in attention_layers(net):
                attn.use_qk_norm = False
        self.each(GEMMA3, no_norm)

    def test_qk_norm_weights_are_zero_centered(self):
        def uncentered(net):
            for attn in attention_layers(net):
                attn.q_norm.weight.data -= 1.0
                attn.k_norm.weight.data -= 1.0
        self.each(GEMMA3, uncentered)

    def test_q_and_k_norms_are_distinct(self):
        def swapped(net):
            for attn in attention_layers(net):
                attn.q_norm, attn.k_norm = attn.k_norm, attn.q_norm
        self.each(GEMMA3, swapped)

    def test_query_scalar(self):
        def head_dim_scale(net):
            for attn in attention_layers(net):
                attn.query_scale = math.sqrt(attn.d_head)
        self.each(NAMES, head_dim_scale)

    # -- local and global blocks --

    def test_block_pattern_gemma3(self):
        # old loader: Gemma-2's alternation for every Gemma
        def alternate(net):
            window = next(a.sliding_window for a in attention_layers(net) if a.sliding_window)
            for n, attn in enumerate(attention_layers(net)):
                attn.sliding_window = window if n % 2 == 0 else None
        self.each(GEMMA3, alternate)

    def test_block_pattern_gemma2_starts_with_a_local_block(self):
        def shifted(net):
            window = next(a.sliding_window for a in attention_layers(net) if a.sliding_window)
            for n, attn in enumerate(attention_layers(net)):
                attn.sliding_window = window if n % 2 == 1 else None
        self.each(GEMMA2, shifted)

    def test_window_extent_is_exact(self):
        # a window of W = the position itself and the W - 1 before it
        for delta in (+1, -1):
            def off_by_one(net, delta=delta):
                for attn in attention_layers(net):
                    if attn.sliding_window:
                        attn.sliding_window += delta
            with self.subTest(delta=delta):
                self.each(NAMES, off_by_one)

    def test_window_is_applied(self):
        def no_window(net):
            for attn in attention_layers(net):
                attn.sliding_window = None
        self.each(NAMES, no_window)

    def test_mask_semantics_of_the_reference(self):
        # the reference's local mask, read off its masking utility: query
        # q sees key k iff q - W < k <= q -- exactly what RopeAttention
        # builds
        from transformers.masking_utils import sliding_window_causal_mask_function
        window, length = 4, 11
        allowed = sliding_window_causal_mask_function(window)
        q = torch.arange(length).view(-1, 1)
        k = torch.arange(length).view(1, -1)
        expected = allowed(0, 0, q, k)
        ours = ~RopeAttention(d_model=8, num_heads=2, sliding_window=window)._mask(length, "cpu")
        self.assertTrue(torch.equal(ours, expected))
        self.assertEqual(ours.sum(1).max().item(), window)

    # -- soft-capping --

    def test_attention_softcap(self):
        def no_cap(net):
            for attn in attention_layers(net):
                attn.softcap = None
        self.each(GEMMA2, no_cap)

    def test_attention_softcap_value(self):
        def final_value(net):
            for attn in attention_layers(net):
                attn.softcap = 30.0
        self.each(GEMMA2, final_value)

    def test_final_softcap(self):
        # old loader: the cap was left out of the network
        for name in GEMMA2:
            with self.subTest(checkpoint=name):
                self.control(name, final_softcap=False)

    def test_final_softcap_value(self):
        def attention_value(net):
            net.layers[-1].cap = 50.0
        self.each(GEMMA2, attention_value)

    def test_gemma3_is_not_capped(self):
        def cap_attention(net):
            for attn in attention_layers(net):
                attn.softcap = 50.0
        self.each(GEMMA3, cap_attention, minimum=SMALL_CONTROL_MIN)

    # -- embedding, norms, MLP --

    def test_embedding_scale(self):
        def unscaled(net):
            net.layers[0].weight.data /= net.embed_scale
        self.each(NAMES, unscaled)

    def test_embedding_scale_rounding(self):
        # the bfloat16 rounding of sqrt(hidden_size), which is what the
        # reference multiplies by when it is loaded in bfloat16: 0.1-0.2%
        # off, and visible. (hidden_size 36 is skipped: sqrt = 6 exactly.)
        def bf16_factor(net):
            rounded = float(torch.tensor(net.embed_scale).to(torch.bfloat16))
            net.layers[0].weight.data *= rounded / net.embed_scale
        names = [n for n in NAMES if TINY[n]["hidden_size"] != 36]
        self.assertGreaterEqual(len(names), 5)
        self.each(names, bf16_factor, minimum=SMALL_CONTROL_MIN)

    def test_norm_weights_are_zero_centered(self):
        def uncentered(net):
            for layer in net.layers:
                if isinstance(layer, RMSNorm):
                    layer.weight.data -= 1.0
        self.each(NAMES, uncentered)

    def test_each_of_the_four_block_norms(self):
        # dropping any one of them (scale 1) is seen
        for offset, label in ((0, "input"), (2, "post-attention"), (4, "pre-feedforward"), (6, "post-feedforward")):
            def identity_scale(net, offset=offset):
                for start in range(1, 1 + net.num_hidden_layers * GemmaLike.BLOCK_STRIDE, GemmaLike.BLOCK_STRIDE):
                    net.layers[start + offset].weight.data.fill_(1.0)
            with self.subTest(norm=label):
                self.each(("gemma-2-9b", "gemma-3-4b-it"), identity_scale)

    def test_gelu_variant(self):
        def exact_gelu(net):
            for layer in net.layers:
                if isinstance(layer, GeGLU):
                    layer.approximate = "none"
        self.each(NAMES, exact_gelu, minimum=GELU_CONTROL_MIN)

    def test_tied_head_is_the_unscaled_table(self):
        def scaled_head(net):
            net.layers[-1].weight.data *= net.embed_scale
        self.each(GEMMA3, scaled_head)


@unittest.skipUnless(HAS_TRANSFORMERS, "transformers / safetensors are not installed")
class TestEmbeddingScale(unittest.TestCase):
    """
        The reference multiplies the embeddings by sqrt(hidden_size) held
        in a tensor created under the dtype the model is instantiated in,
        cast to the dtype of the table at every call. The factor is
        therefore sqrt(hidden_size) ROUNDED TO THAT DTYPE, and in bfloat16
        it is 0.1-0.2% off (gemma-3-270m: 25.25 for 25.298). The network
        folds the factor into its table and has to round the same way.
    """

    def test_embeddings_are_bitwise_those_of_the_reference(self):
        for name in ("gemma-2-9b", "gemma-3-270m", "gemma-3-4b-it"):
            with tempfile.TemporaryDirectory() as directory:
                hf = reference(name)
                ids = token_ids(hf)
                hf.save_pretrained(directory)
                hidden = TINY[name]["hidden_size"]
                for dtype in (torch.float64, torch.float32, torch.bfloat16):
                    with self.subTest(checkpoint=name, dtype=dtype):
                        loaded = type(hf).from_pretrained(directory, dtype=dtype, attn_implementation="eager")
                        embed = loaded.get_input_embeddings()
                        expected = float(torch.tensor(hidden ** 0.5, dtype=torch.float64).to(dtype))
                        self.assertEqual(float(embed.embed_scale), expected)

                        net = GemmaLike.from_huggingface(directory, dtype=dtype, include_lm_head=False)
                        self.assertEqual(net.embed_scale, expected)
                        with torch.no_grad():
                            self.assertTrue(torch.equal(net.layers[0](ids), embed(ids)))
                # the three factors are three different numbers
                rounded = {float(torch.tensor(hidden ** 0.5, dtype=torch.float64).to(d))
                           for d in (torch.float64, torch.float32, torch.bfloat16)}
                self.assertEqual(len(rounded), 3)

    def test_a_model_in_memory_keeps_its_own_factor(self):
        # Built by its constructor, the text decoder of a multimodal
        # Gemma-3 is created under the dtype of its sub-config (bfloat16)
        # and keeps the bfloat16-rounded factor after .double(): that is
        # the function this model computes, and the one the network has
        # to reproduce.
        for name in MULTIMODAL:
            with self.subTest(checkpoint=name):
                hf = reference(name, direct=True)
                hidden = TINY[name]["hidden_size"]
                rounded = float(torch.tensor(hidden ** 0.5).to(torch.bfloat16))
                self.assertEqual(float(text_decoder(hf).embed_tokens.embed_scale), rounded)
                self.assertNotEqual(rounded, hidden ** 0.5)

                ids = token_ids(hf)
                net = GemmaLike.from_huggingface(hf)
                self.assertEqual(net.embed_scale, rounded)
                ref = reference_logits(hf, ids)
                self.assertLess(rel_gap(ref, run(net, ids)), HF_TOL)
                # with the exact factor instead, the gap is 1e-3
                exact = GemmaLike.from_huggingface(hf, embed_scale=hidden ** 0.5)
                self.assertEqual(exact.embed_scale, hidden ** 0.5)
                self.assertGreater(rel_gap(ref, run(exact, ids)), SMALL_CONTROL_MIN)

    def test_the_factor_can_be_pinned(self):
        # e.g. a float64 network carrying the bfloat16 factor
        with tempfile.TemporaryDirectory() as directory:
            hf = reference("gemma-3-270m")
            hf.save_pretrained(directory)
            ids = token_ids(hf)
            net = GemmaLike.from_huggingface(directory, dtype=torch.float64, embed_scale=6.3125)
            self.assertEqual(net.embed_scale, 6.3125)
            default = GemmaLike.from_huggingface(directory, dtype=torch.float64)
            self.assertEqual(default.embed_scale, 40 ** 0.5)
            ratio = net.layers[0].weight / default.layers[0].weight
            self.assertLess((ratio - 6.3125 / 40 ** 0.5).abs().max().item(), 1e-14)
            self.assertGreater(rel_gap(run(default, ids), run(net, ids)), SMALL_CONTROL_MIN)


@unittest.skipUnless(HAS_TRANSFORMERS, "transformers / safetensors are not installed")
class TestReferenceQuirks(unittest.TestCase):
    """Behaviours of transformers that decide what "the reference" is."""

    def test_sdpa_attention_drops_the_gemma2_softcap(self):
        # Under transformers' default attention implementation the
        # `softcap` argument never reaches torch's
        # scaled_dot_product_attention: Gemma-2 then runs WITHOUT its
        # attention soft-capping. The network follows the architecture
        # (the "eager" implementation), and reproduces the "sdpa" numbers
        # once its own attention caps are removed -- which pins the
        # difference on that one mechanism.
        for name in GEMMA2:
            with self.subTest(checkpoint=name):
                eager, sdpa = reference(name), reference(name, attn="sdpa")
                self.assertEqual(sdpa.config._attn_implementation, "sdpa")
                ids = token_ids(eager)
                ref_eager, ref_sdpa = reference_logits(eager, ids), reference_logits(sdpa, ids)
                self.assertGreater(rel_gap(ref_eager, ref_sdpa), CONTROL_MIN)

                net = GemmaLike.from_huggingface(sdpa)
                self.assertLess(rel_gap(ref_eager, run(net, ids)), HF_TOL)
                for attn in attention_layers(net):
                    attn.softcap = None
                self.assertLess(rel_gap(ref_sdpa, run(net, ids)), HF_TOL)

    def test_gemma3_is_the_same_under_both_implementations(self):
        for name in ("gemma-3-270m", "gemma-3-12b-it"):
            with self.subTest(checkpoint=name):
                eager, sdpa = reference(name), reference(name, attn="sdpa")
                ids = token_ids(eager)
                self.assertLess(rel_gap(reference_logits(eager, ids), reference_logits(sdpa, ids)), 2 * HF_TOL)
                self.assertLess(rel_gap(reference_logits(sdpa, ids),
                                        run(GemmaLike.from_huggingface(sdpa), ids)), HF_TOL)


@unittest.skipUnless(HAS_TRANSFORMERS, "transformers / safetensors are not installed")
class TestRefusals(unittest.TestCase):
    """Whatever the network does not reproduce is refused, not approximated."""

    def refuses(self, hf, error, **kwargs):
        with self.assertRaises(error):
            GemmaLike.from_huggingface(hf, **kwargs)

    def test_bidirectional_attention(self):
        for name in ("gemma-2-2b", "gemma-3-270m", "gemma-3-4b-it"):
            with self.subTest(checkpoint=name):
                hf = reference(name)
                with patched(text_config(hf), use_bidirectional_attention=True):
                    self.refuses(hf, NotImplementedError)
                with patched(text_config(hf), is_causal=False):
                    self.refuses(hf, NotImplementedError)
                GemmaLike.from_huggingface(hf)   # the config is intact again

    def test_unsupported_rope(self):
        hf = reference("gemma-2-2b")
        with patched(hf.config, rope_parameters={"rope_type": "yarn", "rope_theta": 1e4, "factor": 4.0}):
            self.refuses(hf, NotImplementedError)
        with patched(hf.config, rope_parameters={"rope_type": "default", "rope_theta": 1e4,
                                                 "partial_rotary_factor": 0.5}):
            self.refuses(hf, NotImplementedError)
        hf = reference("gemma-3-4b-it")
        params = {SLIDING: {"rope_type": "default", "rope_theta": 1e4},
                  FULL: {"rope_type": "dynamic", "rope_theta": 1e6, "factor": 8.0}}
        with patched(text_config(hf), rope_parameters=params):
            self.refuses(hf, NotImplementedError)

    def test_other_activation(self):
        hf = reference("gemma-2-2b")
        for activation in ("silu", "relu", "gelu_new"):
            with patched(hf.config, hidden_activation=activation):
                self.refuses(hf, NotImplementedError)

    def test_unknown_layer_type(self):
        hf = reference("gemma-2-2b")
        types = list(hf.config.layer_types)
        types[1] = "chunked_attention"
        with patched(hf.config, layer_types=types):
            self.refuses(hf, NotImplementedError)

    def test_local_blocks_without_a_window(self):
        hf = reference("gemma-2-2b")
        with patched(hf.config, sliding_window=None):
            self.refuses(hf, ValueError)

    def test_softcaps_the_reference_ignores(self):
        # transformers' Gemma3Attention never applies attn_logit_softcapping
        # and Gemma3ForConditionalGeneration never applies
        # final_logit_softcapping: a config that sets them means
        # something the reference does not compute.
        hf = reference("gemma-3-270m")
        with patched(hf.config, attn_logit_softcapping=50.0):
            self.refuses(hf, NotImplementedError)
        hf = reference("gemma-3-4b-it")
        with patched(text_config(hf), final_logit_softcapping=30.0):
            self.refuses(hf, NotImplementedError)

    def test_other_architectures(self):
        from transformers.models.llama import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaForCausalLM
        llama = LlamaForCausalLM(LlamaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2))
        self.refuses(llama, NotImplementedError)
        # a Gemma decoder inside another wrapper (PaliGemma attends
        # bidirectionally over its prefix)
        hf = reference("gemma-3-4b-it")
        with patched(hf.config, model_type="paligemma"):
            self.refuses(hf, NotImplementedError)

    def test_checkpoint_with_a_tensor_the_network_has_no_place_for(self):
        with tempfile.TemporaryDirectory() as directory:
            hf = reference("gemma-2-2b")
            hf.save_pretrained(directory)
            path = os.path.join(directory, "model.safetensors")
            state = load_file(path)
            state["model.layers.0.mlp.gate_proj.bias"] = torch.zeros(72)
            save_file(state, path)
            self.refuses(directory, NotImplementedError)

    def test_checkpoint_that_does_not_match_its_config(self):
        with tempfile.TemporaryDirectory() as directory:
            hf = reference("gemma-3-270m")
            hf.save_pretrained(directory)
            path = os.path.join(directory, "model.safetensors")
            state = load_file(path)
            # attention biases the config does not announce
            biased = dict(state)
            biased["model.layers.0.self_attn.q_proj.bias"] = torch.zeros(32)
            save_file(biased, path)
            self.refuses(directory, ValueError)
            # a Gemma-3 without its q/k norms
            bare = {k: v for k, v in state.items() if "q_norm" not in k and "k_norm" not in k}
            save_file(bare, path)
            self.refuses(directory, ValueError)
            # a missing tensor
            short = {k: v for k, v in state.items() if k != "model.layers.3.mlp.up_proj.weight"}
            save_file(short, path)
            self.refuses(directory, KeyError)
            # a tensor of the wrong shape
            wrong = dict(state)
            wrong["model.norm.weight"] = torch.zeros(41)
            save_file(wrong, path)
            self.refuses(directory, ValueError)

    def test_untied_head_that_is_not_stored(self):
        hf = reference("gemma-2-2b")
        bare = text_decoder(hf)
        with patched(bare.config, tie_word_embeddings=False):
            self.refuses(bare, ValueError)
            GemmaLike.from_huggingface(bare, include_lm_head=False)


### 4. Knowledge matrices of the loaded networks ###

@unittest.skipUnless(HAS_TRANSFORMERS, "transformers / safetensors are not installed")
class TestKnowledgeMatrix(unittest.TestCase):
    """
        M(x) . 1 = f(x) on the networks loaded from the reference models,
        in both mixer modes; the rows of KnowledgeRowComputer are the rows
        of the matrix; and f(x) is the reference's output, so the rows sum
        to the logits transformers computes (for Gemma-2: the capped ones).
    """

    TOL = 1e-9   # relative; float64 throughout

    def test_row_sums_and_rows(self):
        for name in NAMES:
            hf = reference(name)
            ids = token_ids(hf, KM_SEQ_LEN)
            ref = reference_logits(hf, ids).reshape(-1)
            scale = ref.abs().max().item()
            net = GemmaLike.from_huggingface(hf)
            self.assertEqual(isinstance(net.layers[-1], SoftCap), name in GEMMA2)
            x = ids.unsqueeze(0)
            out = run(net, ids).reshape(-1)
            vocab, hidden = TINY[name]["vocab_size"], TINY[name]["hidden_size"]
            rows = [0, vocab + 3, (KM_SEQ_LEN // 2) * vocab + 11, KM_SEQ_LEN * vocab - 1]
            matrices = {}
            for mode in ("ratio", "frozen"):
                with self.subTest(checkpoint=name, mixer_mode=mode):
                    mat = KnowledgeMatrixComputer(net, batch_size=64, mixer_mode=mode).forward(x)
                    matrices[mode] = mat
                    self.assertEqual(tuple(mat.shape), (KM_SEQ_LEN * vocab, KM_SEQ_LEN * hidden + 1))
                    self.assertLess((mat.sum(1) - out).abs().max().item() / scale, self.TOL)
                    # ... which is the reference's output
                    self.assertLess((mat.sum(1) - ref).abs().max().item() / scale, HF_TOL)

                    got = KnowledgeRowComputer(net, mixer_mode=mode).forward(x, rows)
                    self.assertEqual(tuple(got.shape), (len(rows), KM_SEQ_LEN * hidden + 1))
                    self.assertLess((got - mat[rows]).abs().max().item() / mat[rows].abs().max().item(), self.TOL)
                    self.assertLess((got.sum(1) - out[rows]).abs().max().item() / scale, self.TOL)
            with self.subTest(checkpoint=name):
                # the two modes are different matrices with the same row sums
                self.assertGreater((matrices["ratio"] - matrices["frozen"]).abs().max().item(), 1e-3)

    def test_hidden_state_matrix(self):
        for name in ("gemma-2-27b", "gemma-3-12b-it"):
            hf = reference(name)
            ids = token_ids(hf, KM_SEQ_LEN)
            net = GemmaLike.from_huggingface(hf, include_lm_head=False)
            out = run(net, ids).reshape(-1)
            scale = out.abs().max().item()
            hidden = TINY[name]["hidden_size"]
            for mode in ("ratio", "frozen"):
                with self.subTest(checkpoint=name, mixer_mode=mode):
                    mat = KnowledgeMatrixComputer(net, batch_size=64, mixer_mode=mode).forward(ids.unsqueeze(0))
                    self.assertEqual(tuple(mat.shape), (KM_SEQ_LEN * hidden, KM_SEQ_LEN * hidden + 1))
                    self.assertLess((mat.sum(1) - out).abs().max().item() / scale, self.TOL)

    def test_frozen_attribution_respects_the_masks(self):
        # In frozen mode a logit at position t receives attribution from
        # the tokens it can see and from no other: nothing from the
        # future, whatever the block pattern.
        for name in ("gemma-2-2b", "gemma-3-270m"):
            with self.subTest(checkpoint=name):
                hf = reference(name)
                ids = token_ids(hf, KM_SEQ_LEN)
                net = GemmaLike.from_huggingface(hf)
                vocab, hidden = TINY[name]["vocab_size"], TINY[name]["hidden_size"]
                t = 4
                rows = [t * vocab + 5]
                row = KnowledgeRowComputer(net, mixer_mode="frozen").forward(ids.unsqueeze(0), rows)[0]
                per_token = row[:-1].reshape(KM_SEQ_LEN, hidden).abs().sum(1)
                self.assertEqual(per_token[t + 1:].max().item(), 0.0)
                self.assertGreater(per_token[: t + 1].min().item(), 0.0)

    def test_frozen_linearization_is_exact_on_the_loaded_layers(self):
        # frozen_forward(x0, x0) == forward(x0) for every mixer of the
        # loaded networks, at the input it actually receives
        for name in ("gemma-2-9b", "gemma-3-1b-it", "gemma-3-27b-it"):
            with self.subTest(checkpoint=name):
                hf = reference(name)
                ids = token_ids(hf, KM_SEQ_LEN)
                net = GemmaLike.from_huggingface(hf)
                net.eval()
                net.save = True
                with torch.no_grad():
                    net.forward(ids.unsqueeze(0))
                net.save = False
                checked = 0
                for i, layer in enumerate(net.layers):
                    if isinstance(layer, LINEARIZABLE_LAYERS):
                        x0 = net.pre_acts[i]
                        with torch.no_grad():
                            gap = (layer.frozen_forward(x0, x0) - layer(x0)).abs().max().item()
                            self.assertEqual(gap, 0.0, f"layer {i} ({type(layer).__name__})")
                            self.assertTrue(torch.equal(layer(x0), net.acts[i]))
                        checked += 1
                self.assertEqual(checked, 2 * TINY[name]["num_hidden_layers"])
                if name in GEMMA2:
                    # the soft-cap was recorded as an activation
                    self.assertIsNotNone(net.pre_acts[len(net.layers) - 1])


### 5. Loading options ###

@unittest.skipUnless(HAS_TRANSFORMERS, "transformers / safetensors are not installed")
class TestLoadingOptions(unittest.TestCase):

    def test_sharded_equals_unsharded(self):
        for name in ("gemma-2-2b", "gemma-3-4b-it"):
            with self.subTest(checkpoint=name), tempfile.TemporaryDirectory() as directory:
                hf = reference(name)
                ids = token_ids(hf)
                whole = GemmaLike.from_huggingface(hf)
                self.assertIsNone(whole.layer_devices)
                expected = run(whole, ids)

                sharded = GemmaLike.from_huggingface(hf, device=["cpu", "cpu"])
                self.assertEqual(len(sharded.layer_devices), sharded.get_num_layers())
                self.assertEqual(len(sharded.shard_boundaries), 1)
                self.assertIn(sharded.shard_boundaries[0], sharded.shard_cut_points())
                self.assertTrue(torch.equal(run(sharded, ids), expected))

                # explicit cut at a block boundary, three stages, from disk
                hf.save_pretrained(directory)
                cuts = [1 + GemmaLike.BLOCK_STRIDE, 1 + 3 * GemmaLike.BLOCK_STRIDE]
                staged = GemmaLike.from_huggingface(directory, dtype=torch.float64,
                                                    device=["cpu", "cpu", "cpu"], boundaries=cuts)
                self.assertEqual(staged.shard_boundaries, cuts)
                self.assertTrue(torch.equal(run(staged, ids), expected))

                # the row computer follows the pipeline
                short = token_ids(hf, KM_SEQ_LEN).unsqueeze(0)
                rows = [3, 2 * TINY[name]["vocab_size"] + 1]
                a = KnowledgeRowComputer(whole, mixer_mode="frozen").forward(short, rows)
                b = KnowledgeRowComputer(sharded, mixer_mode="frozen").forward(short, rows)
                self.assertTrue(torch.equal(a, b))

    def test_two_real_devices(self):
        # each tensor goes straight to the device of its layer; the last
        # layer of a Gemma-2 network (the soft-cap) has no parameter
        devices = second_device()
        if devices is None:
            self.skipTest("one device only")
        for name in ("gemma-2-2b", "gemma-3-4b-it"):
            with self.subTest(checkpoint=name, devices=devices), tempfile.TemporaryDirectory() as root:
                hf = reference(name)
                ids = token_ids(hf)
                # MPS has no float64: go through a float32 checkpoint
                full, single = os.path.join(root, "float64"), os.path.join(root, "float32")
                hf.save_pretrained(full)
                type(hf).from_pretrained(full, dtype=torch.float32).save_pretrained(single)

                whole = GemmaLike.from_huggingface(single)
                split = GemmaLike.from_huggingface(single, device=devices)
                self.assertEqual(split.device, devices[0])
                self.assertEqual(set(split.layer_devices), set(devices))
                self.assertEqual(split.layers[0].weight.device.type, torch.device(devices[0]).type)
                self.assertEqual(split.layers[-2].weight.device.type, torch.device(devices[-1]).type)

                expected = run(whole, ids)
                got = run(split, ids.to(devices[0])).cpu()
                self.assertLess(rel_gap(expected, got), 1e-4)

                short = token_ids(hf, KM_SEQ_LEN).unsqueeze(0)
                rows = [3, 2 * TINY[name]["vocab_size"] + 1]
                a = KnowledgeRowComputer(whole, mixer_mode="frozen").forward(short, rows)
                b = KnowledgeRowComputer(split, mixer_mode="frozen").forward(short.to(devices[0]), rows).cpu()
                self.assertLess((a - b).abs().max().item() / a.abs().max().item(), 1e-3)

    def test_a_cut_inside_a_block_is_refused(self):
        hf = reference("gemma-2-2b")
        with self.assertRaises(ValueError):
            GemmaLike.from_huggingface(hf, device=["cpu", "cpu"], boundaries=[3])

    def test_copy_false_shares_storage(self):
        for name in ("gemma-2-9b", "gemma-3-270m", "gemma-3-12b-it"):
            with self.subTest(checkpoint=name):
                hf = reference(name)
                ids = token_ids(hf)
                pointers = {p.data_ptr() for p in hf.parameters()}

                shared = GemmaLike.from_huggingface(hf, copy=False)
                aliased = sum(p.numel() for p in shared.parameters() if p.data_ptr() in pointers)
                total = sum(p.numel() for p in shared.parameters())
                # everything but the rescaled embedding table and the
                # (1 + w) norm weights is the model's own storage
                table = shared.layers[0].weight
                norms = sum(p.numel() for p in shared.parameters() if p.dim() == 1)
                self.assertEqual(total - aliased, table.numel() + norms)
                self.assertNotIn(table.data_ptr(), pointers)
                # in particular the seven matrices of a block, and the head
                decoder = text_decoder(hf)
                block, attn, mlp = decoder.layers[0], shared.layers[2], shared.layers[6]
                for ours, theirs in (
                    (attn.q_proj, block.self_attn.q_proj), (attn.k_proj, block.self_attn.k_proj),
                    (attn.v_proj, block.self_attn.v_proj), (attn.o_proj, block.self_attn.o_proj),
                    (mlp.gate_proj, block.mlp.gate_proj), (mlp.up_proj, block.mlp.up_proj),
                    (mlp.down_proj, block.mlp.down_proj),
                ):
                    self.assertEqual(ours.weight.data_ptr(), theirs.weight.data_ptr())
                head = [l for l in shared.layers if isinstance(l, torch.nn.Linear)][-1]
                self.assertEqual(head.weight.data_ptr(), hf.lm_head.weight.data_ptr())

                copied = GemmaLike.from_huggingface(hf)   # copy=True is the default
                self.assertFalse(any(p.data_ptr() in pointers for p in copied.parameters()))
                # same function either way
                self.assertTrue(torch.equal(run(shared, ids), run(copied, ids)))

    def test_cast_on_load(self):
        hf = reference("gemma-3-1b-it")
        ids = token_ids(hf)
        net = GemmaLike.from_huggingface(hf, dtype=torch.float32)
        self.assertTrue(all(p.dtype == torch.float32 for p in net.parameters()))
        self.assertLess(rel_gap(reference_logits(hf, ids), run(net, ids)), 2e-5)

    def test_half_precision_network_runs(self):
        # bfloat16 is a storage format (see WeightLoader); the network
        # still has to load and run in it
        with tempfile.TemporaryDirectory() as directory:
            hf = reference("gemma-2-9b")
            hf.save_pretrained(directory)
            ids = token_ids(hf)
            net = GemmaLike.from_huggingface(directory, dtype=torch.bfloat16)
            self.assertTrue(all(p.dtype == torch.bfloat16 for p in net.parameters()))
            got = run(net, ids)
            self.assertEqual(got.dtype, torch.bfloat16)
            self.assertTrue(torch.isfinite(got.float()).all())
            # measured: 1.6e-2 (transformers' own bfloat16 model: 2.1e-2)
            self.assertLess(rel_gap(reference_logits(hf, ids), got.double()), 0.1)

    def test_save_flag(self):
        hf = reference("gemma-2-2b")
        self.assertTrue(GemmaLike.from_huggingface(hf, save=True).save)
        self.assertFalse(GemmaLike.from_huggingface(hf).save)


### 6. The published architectures at full size ###

@unittest.skipUnless(HAS_TRANSFORMERS, "transformers / safetensors are not installed")
class TestFullSizeDryRun(unittest.TestCase):
    """
        The eight published configs at their REAL size: the transformers
        model is instantiated on the "meta" device (shapes, no memory) and
        loaded with device="meta", which locates every tensor and checks
        its shape against the network. Proves that the real checkpoints
        fit, and that the real configs produce the right switches.
    """

    def test_all_published_checkpoints(self):
        for name in NAMES:
            with self.subTest(checkpoint=name):
                config = published_config(name)
                pub = PUBLISHED[name]
                cls = getattr(transformers, config.architectures[0])
                with torch.device("meta"):
                    hf = cls(config)
                self.assertTrue(all(p.is_meta for p in hf.parameters()))

                net = GemmaLike.from_huggingface(hf, device="meta")
                self.assertTrue(all(p.is_meta for p in net.parameters()))
                self.assertEqual(net.device, "meta")

                # parameter count: the text decoder, plus the head, which
                # the reference ties to the embedding table and the
                # network holds as a second matrix
                decoder = text_decoder(hf)
                self.assertIs(hf.lm_head.weight, decoder.embed_tokens.weight)
                n_decoder = sum(p.numel() for p in decoder.parameters())
                self.assertEqual(n_decoder, pub["decoder_params"])
                n_head = hf.lm_head.weight.numel()
                self.assertEqual(sum(p.numel() for p in net.parameters()), n_decoder + n_head)
                headless = GemmaLike.from_huggingface(hf, device="meta", include_lm_head=False)
                self.assertEqual(sum(p.numel() for p in headless.parameters()), n_decoder)

                # structure
                layers = pub["layers"]
                capped = pub["final_cap"] is not None
                self.assertEqual(net.get_num_layers(), 1 + layers * GemmaLike.BLOCK_STRIDE + 2 + int(capped))
                self.assertEqual(isinstance(net.layers[-1], SoftCap), capped)
                self.assertEqual(net.final_softcap, pub["final_cap"])
                self.assertEqual(headless.get_num_layers(), 1 + layers * GemmaLike.BLOCK_STRIDE + 1)
                self.assertEqual(net.layers[0].weight.shape, (config.get_text_config().vocab_size, pub["hidden"]))

                # the switches of every block
                attention = attention_layers(net)
                self.assertEqual(len(attention), layers)
                for n, attn in enumerate(attention):
                    is_global = (n + 1) % pub["period"] == 0
                    theta, scaling = pub["glob"] if is_global else pub["local"]
                    self.assertEqual(attn.sliding_window, None if is_global else pub["window"])
                    self.assertEqual(attn.rope_theta, theta)
                    self.assertEqual(attn.rope_scaling, scaling)
                    self.assertEqual(attn.softcap, pub["attn_cap"])
                    self.assertEqual(attn.query_scale, float(pub["scalar"]) ** 0.5)
                    self.assertEqual(attn.use_qk_norm, name in GEMMA3)
                    self.assertEqual((attn.num_heads, attn.num_kv_heads), pub["heads"])
                    self.assertEqual(attn.d_head, pub["head_dim"])
                    self.assertEqual(attn.rotary_dim, pub["head_dim"])
                    self.assertTrue(attn.causal)
                    self.assertIsNone(attn.q_proj.bias)
                self.assertEqual(sum(a.sliding_window is None for a in attention), layers // pub["period"])
                self.assertTrue(all(l.approximate == "tanh" for l in net.layers if isinstance(l, GeGLU)))
                self.assertTrue(all(m.eps == 1e-6 for m in net.modules() if isinstance(m, RMSNorm)))

    def test_dry_run_of_a_sharded_load(self):
        config = published_config("gemma-3-27b-it")
        with torch.device("meta"):
            hf = transformers.Gemma3ForConditionalGeneration(config)
        net = GemmaLike.from_huggingface(hf, device="meta")
        # the placement a 4-device pipeline would get: cuts at block boundaries
        devices, cuts = net.shard_layout(["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
        self.assertEqual(len(cuts), 3)
        self.assertEqual(len(devices), net.get_num_layers())
        for cut in cuts:
            self.assertEqual((cut - 1) % (GemmaLike.BLOCK_STRIDE // 2), 0)


### 7. Real weights ###

#: google/gemma-3-270m through its ungated mirror. More checkpoints can be
#: verified the same way by listing them (hub ids or directories) in
#: KM_GEMMA_CHECKPOINTS, comma separated; they must be in the local cache.
REAL_CHECKPOINTS = ["unsloth/gemma-3-270m"] + [
    c.strip() for c in os.environ.get("KM_GEMMA_CHECKPOINTS", "").split(",") if c.strip()
]

PARAGRAPH = (
    "The knowledge matrix of a network at an input is a matrix whose rows sum to the outputs of the forward pass. "
    "Each column belongs to one input coordinate, and one more column collects the biases. "
    "For a language model the input coordinates are the embedding coordinates of every token of the prompt, "
    "so a row tells how much each token contributed to one logit. "
)
PROMPTS = {
    "short": "The capital of France is",
    # ~700 tokens: longer than the 512-token window of gemma-3-270m
    "long": " ".join(PARAGRAPH + f"Paragraph {i} ends here." for i in range(9)),
}


def cached_checkpoint(source):
    """`source` if its weights and tokenizer are available locally, else None."""
    if os.path.isdir(source):
        return source
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None

    def cached(filename):
        path = try_to_load_from_cache(source, filename)
        return isinstance(path, str) and os.path.isfile(path)

    has_weights = cached("model.safetensors") or cached("model.safetensors.index.json")
    return source if (has_weights and cached("config.json") and cached("tokenizer.json")) else None


@unittest.skipUnless(HAS_TRANSFORMERS, "transformers / safetensors are not installed")
class TestRealWeights(unittest.TestCase):
    """
        The float32 network streamed from the checkpoint against
        transformers (float32, eager attention) on real text, for the
        checkpoints found in the local cache -- nothing is downloaded.

        Measured on gemma-3-270m: the logits agree to 2.2e-6 of their
        largest value on the 6-token prompt and on the 694-token one
        alike (the window is 512), the hidden states to 1.4e-6.
    """

    #: relative to the largest value of the reference; float32 on both
    #: sides, ten times the measured gap
    TOL = 2e-5

    def test_real_checkpoints(self):
        available = [c for c in REAL_CHECKPOINTS if cached_checkpoint(c)]
        if not available:
            self.skipTest(f"none of {REAL_CHECKPOINTS} is in the local Hugging Face cache")
        for source in available:
            with self.subTest(checkpoint=source):
                self.check(source)

    def check(self, source):
        hub = {} if os.path.isdir(source) else {"local_files_only": True}
        tokenizer = transformers.AutoTokenizer.from_pretrained(source, **hub)
        ids = {k: tokenizer(v, return_tensors="pt").input_ids for k, v in PROMPTS.items()}
        config = transformers.AutoConfig.from_pretrained(source, **hub)
        text = config.get_text_config()
        self.assertGreater(ids["long"].shape[1], text.sliding_window)
        self.assertLess(ids["short"].shape[1], text.sliding_window)

        # 1. the reference, then released: one model in memory at a time
        hf = transformers.AutoModelForCausalLM.from_pretrained(
            source, dtype=torch.float32, attn_implementation="eager", **hub).eval()
        self.assertEqual(type(hf).__name__, config.architectures[0])
        factor = float(hf.get_input_embeddings().embed_scale)
        with torch.no_grad():
            ref = {k: hf(v).logits[0] for k, v in ids.items()}
            ref_hidden = hf.model(input_ids=ids["short"]).last_hidden_state[0]
        del hf

        # 2. logits of the network streamed from disk
        net = GemmaLike.from_huggingface(source, **hub)
        self.assertTrue(all(p.dtype == torch.float32 for p in net.parameters()))
        self.assertEqual(net.embed_scale, factor)
        print()
        for k in PROMPTS:
            out = run(net, ids[k])[0, 0]
            agree = (ref[k].argmax(-1) == out.argmax(-1)).double().mean().item()
            scale, std = ref[k].abs().max().item(), ref[k].std().item()
            if k == "short":
                short = out.clone()
            gap = out.sub_(ref[k]).abs_().max().item()
            del out
            print(f"    {source} [{k}, {ids[k].shape[1]} tokens]: max |logit gap| {gap:.3e} = "
                  f"{gap / scale:.3e} of the largest logit ({scale:.1f}), {gap / std:.3e} of the "
                  f"logit std; argmax agreement {agree:.4f}")
            self.assertLess(gap / scale, self.TOL)
            self.assertGreater(agree, 0.99)

        # 3. negative controls on the real weights. The window only
        # matters where the context is longer than it: removing it, or
        # getting its extent wrong by one, changes the long prompt and
        # leaves the short one untouched.
        attention = attention_layers(net)
        windows = [attn.sliding_window for attn in attention]

        def with_windows(change):
            for attn, window in zip(attention, windows):
                attn.sliding_window = change(window) if window else None
            unchanged = torch.equal(run(net, ids["short"])[0, 0], short)
            out = run(net, ids["long"])[0, 0]
            gap = out.sub_(ref["long"]).abs_().max().item() / ref["long"].abs().max().item()
            for attn, window in zip(attention, windows):
                attn.sliding_window = window
            return unchanged, gap

        for label, change, minimum in (("no window", lambda w: None, 0.1),
                                       ("window + 1", lambda w: w + 1, 1e-3),
                                       ("window - 1", lambda w: w - 1, 1e-3)):
            unchanged, gap = with_windows(change)
            print(f"    control, {label}: {gap:.3e} on the long prompt")
            self.assertTrue(unchanged, label)
            self.assertGreater(gap, minimum, label)
        del net, ref, short

        # 4. hidden states, and the knowledge matrix of the hidden states
        # (the matrix of the logits would have 262144 rows per token)
        headless = GemmaLike.from_huggingface(source, include_lm_head=False, **hub)
        out = run(headless, ids["short"]).reshape(-1)
        scale = ref_hidden.abs().max().item()
        gap = (out - ref_hidden.reshape(-1)).abs().max().item() / scale
        print(f"    hidden states [short]: {gap:.3e} of the largest ({scale:.1f})")
        self.assertLess(gap, self.TOL)

        prefix = ids["short"][:, :4]            # 4 tokens keep the matrix cheap
        x = prefix.unsqueeze(0)
        out = run(headless, prefix).reshape(-1)
        scale = out.abs().max().item()
        size = prefix.shape[1] * text.hidden_size
        rows = [0, size // 3, size // 2 + 1, size - 1]
        # frozen: float32 rounding only. ratio: the ratios reach 1e5
        # where a pre-activation is small, which costs digits in float32
        # (the identity is exact to 1e-15 in float64) -- and gemma-3-270m
        # has norm scales of exactly 0, see TestZeroPreActivation.
        for mode, tolerance in (("frozen", 1e-4), ("ratio", 1e-2)):
            mat = KnowledgeMatrixComputer(headless, batch_size=256, mixer_mode=mode).forward(x)
            self.assertEqual(tuple(mat.shape), (size, size + 1))
            sums = (mat.sum(1) - out).abs().max().item() / scale
            got = KnowledgeRowComputer(headless, mixer_mode=mode).forward(x, rows)
            agreement = (got - mat[rows]).abs().max().item() / mat[rows].abs().max().item()
            print(f"    knowledge matrix [{mode}, {prefix.shape[1]} tokens]: row sums off by {sums:.3e} "
                  f"of the largest output, row computer by {agreement:.3e}")
            self.assertLess(sums, tolerance)
            self.assertLess(agreement, 1e-3)
            del mat


if __name__ == "__main__":
    unittest.main()
