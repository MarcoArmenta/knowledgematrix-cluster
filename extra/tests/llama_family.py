#!/usr/bin/env python
"""
    Tests that LlamaLike.from_huggingface builds, for each supported
    checkpoint of the Llama family, a network computing EXACTLY the
    function of the Hugging Face reference implementation:

        meta-llama/Llama-3.1-8B, Llama-3.2-1B, Llama-3.2-3B,
        meta-llama/Llama-3.3-70B-Instruct,
        HuggingFaceTB/SmolLM2-135M, SmolLM2-360M, SmolLM2-1.7B,
        Qwen/Qwen3-8B, Qwen3-14B

    (plus Mistral, Qwen2.5 and the siblings whose head dimension is not
    hidden_size / heads). A mis-wired network is still self-consistent --
    its knowledge matrix still sums to its own forward pass -- so parity
    against the reference is the only thing that catches a wiring error,
    and a parity test that cannot fail proves nothing. Hence, in order:

    1. PUBLISHED ARCHITECTURES -- what the loader reads from the nine
       published config.json files (extra/tests/fixtures/hf_configs),
       pinned against the published facts; the RoPE frequency tables are
       bit-identical to the ones transformers builds, at full size.
    2. PARITY -- for every one of the nine configs, scaled down in size
       only: the real transformers class with random float64 weights
       against our network, loaded from the model in memory and streamed
       from a `save_pretrained` directory, with and without the LM head.
    3. NEGATIVE CONTROLS -- the same comparison with one thing wrong at a
       time (RoPE base, llama3 rescaling, q/k norm, head grouping, norm
       epsilon, norm weights, LM head) must fail by a wide margin.
    4. KNOWLEDGE MATRIX -- on the loaded networks, rows sum to the forward
       pass in "ratio" and "frozen" mode, and KnowledgeRowComputer returns
       the rows of KnowledgeMatrixComputer.
    5. FULL SIZE -- a dry run of each published architecture against the
       real transformers class on the "meta" device (no memory): every
       tensor is located and shape-checked, parameter counts agree.
    6. LOADER MECHANICS -- sharding, copy / share, dtypes, damaged
       checkpoints, hub arguments; half precision (bfloat16) is as
       accurate as the bfloat16 reference.
    7. REFUSALS -- configs the network does not reproduce raise.
    8. REAL WEIGHTS -- when a checkpoint is in the local Hugging Face
       cache (never downloaded here): our float32 network streamed from
       disk against the transformers model on real text.

    Everything except 8 is offline and needs no weights. The real-weights
    tests run for the checkpoints found in the local cache, one model in
    memory at a time (up to ~5 GB for Llama-3.2-1B); set
    KM_SKIP_REAL_WEIGHTS=1 to skip them. Set KM_REPORT=1 to print the
    measured gaps.

    Requires transformers and safetensors (pip install knowledgematrix[hf]);
    without them the whole file is skipped.
"""
import contextlib
import copy
import gc
import json
import math
import os
import tempfile
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

from knowledgematrix.neural_net import (
    RMSNorm,
    RopeAttention,
    rope_inverse_frequencies,
)
from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.row_computer import KnowledgeRowComputer
from knowledgematrix.models._hf import open_checkpoint, rope_settings
from knowledgematrix.models.llama import (
    SUPPORTED_MODEL_TYPES,
    LlamaLike,
    llama_architecture,
)

try:
    import huggingface_hub
    import safetensors.torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM
    transformers.logging.set_verbosity_error()
    transformers.utils.logging.disable_progress_bar()
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

torch.set_default_dtype(torch.float64)

#: HF hardcodes float32 internals (the RoPE inv_freq buffer and the cos /
#: sin tables, and the .to(torch.float32) inside LlamaRMSNorm.forward), so
#: even in float64 the reference carries float32 rounding. With the O(1)
#: weights used here that rounding is ~2e-7 on outputs of scale ~0.3;
#: anything above this tolerance is a real bug, not a precision artefact.
HF_TOL = 1e-6

#: The same comparison with those two float32 computations of the
#: reference carried out in float64 instead (`in_its_own_dtype`): then
#: nothing but float64 rounding separates the two implementations
#: (measured: ~1e-14), which also proves that the HF_TOL gap above is the
#: reference's float32 and nothing else.
EXACT_TOL = 1e-11

#: A negative control "fails by a wide margin" when it is at least this
#: far from the reference: 1000 x HF_TOL, on outputs of scale ~0.3.
WIDE = 1e-3

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "hf_configs")

LLAMA3 = "llama3"

#: The supported checkpoints, with the published facts the loader has to
#: get right (written down here independently of the fixture files) and
#: the head counts of the scaled-down copy used by the parity tests
#: (`tiny`: query heads, key/value heads -- same grouping ratio).
CHECKPOINTS = {
    "meta-llama/Llama-3.1-8B": dict(
        layers=32, hidden=4096, mlp=14336, heads=32, kv=8, head_dim=128, vocab=128256,
        theta=500000.0, llama3_factor=8.0, eps=1e-5, tied=False, qk_norm=False,
        params=8_030_261_248, tiny=(8, 2)),
    "meta-llama/Llama-3.2-1B": dict(
        layers=16, hidden=2048, mlp=8192, heads=32, kv=8, head_dim=64, vocab=128256,
        theta=500000.0, llama3_factor=32.0, eps=1e-5, tied=True, qk_norm=False,
        params=1_235_814_400, tiny=(8, 2)),
    "meta-llama/Llama-3.2-3B": dict(
        layers=28, hidden=3072, mlp=8192, heads=24, kv=8, head_dim=128, vocab=128256,
        theta=500000.0, llama3_factor=32.0, eps=1e-5, tied=True, qk_norm=False,
        params=3_212_749_824, tiny=(6, 2)),
    "meta-llama/Llama-3.3-70B-Instruct": dict(
        layers=80, hidden=8192, mlp=28672, heads=64, kv=8, head_dim=128, vocab=128256,
        theta=500000.0, llama3_factor=8.0, eps=1e-5, tied=False, qk_norm=False,
        params=70_553_706_496, tiny=(16, 2)),
    # SmolLM2 does not state head_dim: it is hidden / heads = 64, with an
    # odd number of heads grouped three to one in the two small models
    "HuggingFaceTB/SmolLM2-135M": dict(
        layers=30, hidden=576, mlp=1536, heads=9, kv=3, head_dim=64, vocab=49152,
        theta=100000.0, llama3_factor=None, eps=1e-5, tied=True, qk_norm=False,
        params=134_515_008, tiny=(9, 3)),
    "HuggingFaceTB/SmolLM2-360M": dict(
        layers=32, hidden=960, mlp=2560, heads=15, kv=5, head_dim=64, vocab=49152,
        theta=100000.0, llama3_factor=None, eps=1e-5, tied=True, qk_norm=False,
        params=361_821_120, tiny=(15, 5)),
    "HuggingFaceTB/SmolLM2-1.7B": dict(
        layers=24, hidden=2048, mlp=8192, heads=32, kv=32, head_dim=64, vocab=49152,
        theta=130000.0, llama3_factor=None, eps=1e-5, tied=True, qk_norm=False,
        params=1_711_376_384, tiny=(4, 4)),
    "Qwen/Qwen3-8B": dict(
        layers=36, hidden=4096, mlp=12288, heads=32, kv=8, head_dim=128, vocab=151936,
        theta=1000000.0, llama3_factor=None, eps=1e-6, tied=False, qk_norm=True,
        params=8_190_735_360, tiny=(8, 2)),
    "Qwen/Qwen3-14B": dict(
        layers=40, hidden=5120, mlp=17408, heads=40, kv=8, head_dim=128, vocab=151936,
        theta=1000000.0, llama3_factor=None, eps=1e-6, tied=False, qk_norm=True,
        params=14_768_307_200, tiny=(10, 2)),
}

LLAMA3_CHECKPOINTS = [n for n, s in CHECKPOINTS.items() if s["llama3_factor"]]
QWEN3_CHECKPOINTS = [n for n, s in CHECKPOINTS.items() if s["qk_norm"]]
UNTIED_CHECKPOINTS = [n for n, s in CHECKPOINTS.items() if not s["tied"]]
GROUPED_CHECKPOINTS = [n for n, s in CHECKPOINTS.items() if s["heads"] != s["kv"]]

# Sizes of the scaled-down copies. They are pairwise different (and the
# vocabulary is prime), so that no tensor fits a slot it does not belong to.
TINY_HEAD_DIM = 8
TINY_LAYERS = 3
TINY_VOCAB = 97
#: `original_max_position_embeddings` of the scaled-down llama3 configs
#: (published: 8192). With head_dim 8 and base 500000 the four RoPE
#: wavelengths are 6.3, 167, 4443 and 118155 positions; the llama3 rule
#: keeps those below L/4, slows down those above L and blends in between,
#: so L = 256 puts one frequency in the kept band, one in the blended band
#: and two in the slowed band -- and SEQ_LEN positions are enough for the
#: blended and the slowed frequencies to turn by a visible angle (0.5 rad
#: and 0.05 rad more than without the rescaling). With the published 8192
#: nothing but the slowest frequency would move, and only at positions in
#: the thousands.
TINY_ORIGINAL_LENGTH = 256
SEQ_LEN = 40
#: sequence length of the knowledge matrix tests (the full matrix needs
#: one forward pass per input coordinate)
KM_SEQ_LEN = 6

_MEASURED = []


def note(test: str, what: str, value: float, scale: float = float("nan")) -> None:
    _MEASURED.append((test, what, value, scale))


def tearDownModule():
    if os.environ.get("KM_REPORT") and _MEASURED:
        print("\n\nmeasured (test | case | gap | output scale)")
        for test, what, value, scale in _MEASURED:
            print(f"  {test:34s} | {what:58s} | {value:9.2e} | {scale:8.3g}")


### Building the reference models ###

def fixture(name: str) -> dict:
    """The published config.json of a checkpoint."""
    with open(os.path.join(FIXTURES, name.replace("/", "__") + ".json")) as f:
        return json.load(f)


def as_config(raw: dict):
    """A config.json dictionary, normalized by the installed transformers
    exactly as AutoConfig.from_pretrained does on a checkpoint."""
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(raw, f)
        return AutoConfig.from_pretrained(d)


def tiny_raw(name: str) -> dict:
    """
        The published config.json with ONLY its sizes scaled down: widths,
        depth, head counts (same grouping ratio), head dimension and
        vocabulary -- and, for the llama3 RoPE, the pretraining length (see
        TINY_ORIGINAL_LENGTH). Every architectural switch is the published
        one: model type, RoPE base and rescaling factors, epsilons, bias
        flags, head tying, sliding-window settings, the old-format
        rope_theta / rope_scaling keys.
    """
    raw = fixture(name)
    spec = CHECKPOINTS[name]
    heads, kv = spec["tiny"]
    assert heads * spec["kv"] == kv * spec["heads"], "the grouping ratio must be kept"
    raw["num_attention_heads"] = heads
    raw["num_key_value_heads"] = kv
    raw["hidden_size"] = heads * TINY_HEAD_DIM
    if "head_dim" in raw:
        raw["head_dim"] = TINY_HEAD_DIM
    raw["intermediate_size"] = int(round(raw["hidden_size"] * spec["mlp"] / spec["hidden"]))
    raw["num_hidden_layers"] = TINY_LAYERS
    if "max_window_layers" in raw:        # Qwen3: published = num_hidden_layers
        raw["max_window_layers"] = TINY_LAYERS
    raw["vocab_size"] = TINY_VOCAB
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if raw.get(key) is not None:      # the pad token indexes the embedding
            raw[key] = raw[key] % TINY_VOCAB
    if raw.get("rope_scaling"):
        raw["rope_scaling"]["original_max_position_embeddings"] = TINY_ORIGINAL_LENGTH
    return raw


def randomize_(hf, seed: int) -> None:
    """
        Random float64 weights of realistic gain, in place.

        transformers initializes every norm weight to 1, every bias to 0
        and every matrix with standard deviation 0.02. On such a model two
        swapped norm weights or a dropped bias change nothing, and the
        attention is so flat that a wrong position encoding barely shows.
        Here every parameter is random: matrices with unit gain, norm
        weights 1 +- 0.3, biases +- 0.3, and a small embedding (standard
        deviation 0.02, as in real checkpoints) so that the epsilon of the
        first normalization matters. The final norm is scaled down so
        that hidden states and logits are ~0.3: the float32 rounding
        inside the reference is relative, and HF_TOL is absolute.
    """
    g = torch.Generator().manual_seed(seed)

    def randn(p):
        return torch.randn(p.shape, generator=g, dtype=torch.float64).to(p.dtype)

    with torch.no_grad():
        for key, p in hf.named_parameters():
            if key.endswith("model.norm.weight"):
                p.copy_(0.2 * (1.0 + 0.3 * randn(p)))
            elif key.endswith(("layernorm.weight", "q_norm.weight", "k_norm.weight")):
                p.copy_(1.0 + 0.3 * randn(p))
            elif key.endswith("embed_tokens.weight"):
                p.copy_(0.02 * randn(p))
            elif key.endswith(".bias"):
                p.copy_(0.3 * randn(p))
            elif key.endswith("lm_head.weight"):
                p.copy_(0.5 * randn(p) / math.sqrt(p.shape[1]))
            elif p.dim() == 2:
                p.copy_(randn(p) / math.sqrt(p.shape[1]))
            else:
                raise AssertionError(f"randomize_: unexpected parameter {key} {tuple(p.shape)}")


def build_reference(raw, seed: int = 0):
    """The real transformers class for a config.json dictionary (or a
    config object), in float64 with random weights."""
    config = as_config(raw) if isinstance(raw, dict) else raw
    hf = AutoModelForCausalLM.from_config(config).double().eval()
    randomize_(hf, seed)
    return hf


def token_ids(seq_len: int = SEQ_LEN, vocab: int = TINY_VOCAB, seed: int = 1) -> torch.Tensor:
    """(1, seq_len) token ids."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (1, seq_len), generator=g)


def reference_frequencies(config) -> torch.Tensor:
    """
        The RoPE frequencies of a transformers config in float64, written
        out from the published formulas with Python floats -- independent
        of knowledgematrix.neural_net.rope_inverse_frequencies and of the
        float32 arithmetic of transformers.modeling_rope_utils.
    """
    params = config.rope_parameters
    dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    base = float(params["rope_theta"])
    freqs = [base ** (-i / dim) for i in range(0, dim, 2)]
    if params["rope_type"] == LLAMA3:
        factor = float(params["factor"])
        low, high = float(params["low_freq_factor"]), float(params["high_freq_factor"])
        length = float(params["original_max_position_embeddings"])
        out = []
        for w in freqs:
            wavelength = 2 * math.pi / w
            if wavelength < length / high:          # fast: kept
                out.append(w)
            elif wavelength > length / low:         # slow: slowed down
                out.append(w / factor)
            else:                                   # in between: blended
                smooth = (length / wavelength - low) / (high - low)
                out.append((1 - smooth) * w / factor + smooth * w)
        freqs = out
    elif params["rope_type"] == "linear":
        freqs = [w / float(params["factor"]) for w in freqs]
    elif params["rope_type"] != "default":
        raise AssertionError(f"reference_frequencies: {params['rope_type']}")
    return torch.tensor(freqs, dtype=torch.float64)


def llama3_bands(config) -> tuple:
    """How many RoPE frequencies of a llama3 config are (kept, blended, slowed)."""
    params = config.rope_parameters
    dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    length = params["original_max_position_embeddings"]
    kept = blended = slowed = 0
    for i in range(0, dim, 2):
        wavelength = 2 * math.pi * float(params["rope_theta"]) ** (i / dim)
        if wavelength < length / params["high_freq_factor"]:
            kept += 1
        elif wavelength > length / params["low_freq_factor"]:
            slowed += 1
        else:
            blended += 1
    return kept, blended, slowed


@contextlib.contextmanager
def in_its_own_dtype(hf):
    """
        Evaluate the reference model without its two hardcoded float32
        computations, i.e. entirely in the dtype of its weights:

        - *RMSNorm.forward casts its input to float32 before normalizing
          (`hidden_states.to(torch.float32)`); here the same three lines
          run without the cast, on the module's own weight and epsilon;
        - *RotaryEmbedding.forward builds cos / sin from a float32
          frequency buffer in float32; here they are built the same way in
          float64 from `reference_frequencies`, after checking that those
          round to the buffer the reference computed for itself.

        Nothing else is touched (the attention itself runs in the model's
        dtype with the default sdpa implementation).
    """
    patched = []
    try:
        for module in hf.modules():
            if type(module).__name__.endswith("RMSNorm"):
                def forward(x, _m=module):
                    variance = x.pow(2).mean(-1, keepdim=True)
                    return _m.weight * (x * torch.rsqrt(variance + _m.variance_epsilon))
                module.forward = forward
                patched.append(module)
        rotary = hf.model.rotary_emb
        freqs = reference_frequencies(hf.config)
        own = rotary.inv_freq.to(torch.float64)     # float32 values
        if not torch.allclose(freqs, own, rtol=1e-6, atol=0.0):
            raise AssertionError("reference_frequencies is not the table of the reference")

        def rotary_forward(x, position_ids):
            angles = position_ids[:, :, None].to(freqs.dtype) * freqs[None, None, :]
            emb = torch.cat((angles, angles), dim=-1)
            return ((emb.cos() * rotary.attention_scaling).to(x.dtype),
                    (emb.sin() * rotary.attention_scaling).to(x.dtype))
        rotary.forward = rotary_forward
        patched.append(rotary)
        yield hf
    finally:
        for module in patched:
            del module.forward


def reference_outputs(hf, ids: torch.Tensor) -> dict:
    """Logits and final hidden states of the reference: as it is ("logits",
    "hidden") and in its own dtype ("exact_logits", "exact_hidden")."""
    with torch.no_grad():
        out = {"logits": hf(ids).logits, "hidden": hf.model(ids).last_hidden_state}
        with in_its_own_dtype(hf):
            out["exact_logits"] = hf(ids).logits
            out["exact_hidden"] = hf.model(ids).last_hidden_state
    return out


def run(net, ids: torch.Tensor) -> torch.Tensor:
    """Our network on (1, T) token ids -> (T, out)."""
    net.eval()
    with torch.no_grad():
        return net.forward(ids.unsqueeze(0))[0, 0]


def gap(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.reshape(-1).to(torch.float64) - b.reshape(-1).to(torch.float64)).abs().max().item()


def attention_layers(net):
    return [layer for layer in net.layers if isinstance(layer, RopeAttention)]


_TINY = {}


def tiny(name: str):
    """(reference model, token ids, reference outputs) of the scaled-down
    copy of a checkpoint -- built once."""
    if name not in _TINY:
        seed = list(CHECKPOINTS).index(name)       # the same whatever tests run
        hf = build_reference(tiny_raw(name), seed=seed)
        ids = token_ids(seed=100 + seed)
        _TINY[name] = (hf, ids, reference_outputs(hf, ids))
    return _TINY[name]


def per_checkpoint(cls):
    """
        Class decorator: every method `check_<x>(self, name)` becomes one
        test per checkpoint, `test_<x>__<checkpoint>`, so that a failure
        names the checkpoint. A method restricts itself with a
        `checkpoints` attribute.
    """
    for attr, fn in list(vars(cls).items()):
        if not attr.startswith("check_"):
            continue
        for name in getattr(fn, "checkpoints", CHECKPOINTS):
            def test(self, _fn=fn, _name=name):
                _fn(self, _name)
            slug = name.split("/")[1].replace("-", "_").replace(".", "_")
            test.__name__ = f"test_{attr[len('check_'):]}__{slug}"
            test.__doc__ = fn.__doc__
            setattr(cls, test.__name__, test)
    return cls


def only(names):
    def mark(fn):
        fn.checkpoints = list(names)
        return fn
    return mark


### 1. Published architectures ###

@unittest.skipUnless(HAS_TRANSFORMERS, "transformers / safetensors are not installed")
@per_checkpoint
class TestPublishedArchitectures(unittest.TestCase):
    """
        What the loader reads from the published config.json files. No
        weights involved: `llama_architecture` is the complete description
        of the network that from_huggingface builds.
    """

    def check_architecture_is_the_published_one(self, name):
        spec = CHECKPOINTS[name]
        arch = llama_architecture(as_config(fixture(name)))
        scaling = None
        if spec["llama3_factor"]:
            scaling = {"rope_type": LLAMA3, "factor": spec["llama3_factor"],
                       "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                       "original_max_position_embeddings": 8192.0}
        self.assertEqual(arch, dict(
            vocab_size=spec["vocab"],
            hidden_size=spec["hidden"],
            intermediate_size=spec["mlp"],
            num_hidden_layers=spec["layers"],
            num_attention_heads=spec["heads"],
            num_key_value_heads=spec["kv"],
            head_dim=spec["head_dim"],
            rope_theta=spec["theta"],
            rope_scaling=scaling,
            rms_norm_eps=spec["eps"],
            qkv_bias=False,
            o_bias=False,
            qk_norm=spec["qk_norm"],
            mlp_bias=False,
            sliding_window=[None] * spec["layers"],
        ))

    def check_rope_base_is_not_the_default(self, name):
        """
            REGRESSION. transformers 5 moved the RoPE settings from
            `config.rope_theta` / `config.rope_scaling` into
            `config.rope_parameters`; the loader used to read
            `getattr(config, "rope_theta", 10000.0)` and therefore built
            every one of these checkpoints with base 10000 and without
            rescaling -- a different function, and no error.
        """
        spec = CHECKPOINTS[name]
        self.assertNotEqual(spec["theta"], 10000.0)
        for raw in (fixture(name), tiny_raw(name)):
            rope = rope_settings(as_config(raw))
            self.assertEqual(rope["theta"], spec["theta"])
            self.assertEqual(rope["scaling"] is not None, spec["llama3_factor"] is not None)
        hf, _, _ = tiny(name)
        net = LlamaLike.from_huggingface(hf)
        layers = attention_layers(net)
        self.assertEqual(len(layers), TINY_LAYERS)
        for layer in layers:
            self.assertEqual(layer.rope_theta, spec["theta"])
            if spec["llama3_factor"]:
                self.assertEqual(layer.rope_scaling["factor"], spec["llama3_factor"])
                self.assertEqual(layer.rope_scaling["rope_type"], LLAMA3)
            else:
                self.assertIsNone(layer.rope_scaling)

    def check_rope_table_is_the_reference_table(self, name):
        """At full size: our frequencies in float32 are bit for bit the
        buffer transformers computes, and in float64 they are the
        published formula."""
        config = as_config(fixture(name))
        module = {"llama": transformers.models.llama.modeling_llama.LlamaRotaryEmbedding,
                  "qwen3": transformers.models.qwen3.modeling_qwen3.Qwen3RotaryEmbedding}
        theirs = module[config.model_type](config=config).inv_freq
        self.assertEqual(theirs.dtype, torch.float32)
        arch = llama_architecture(config)
        ours32 = rope_inverse_frequencies(arch["head_dim"], arch["rope_theta"],
                                          arch["rope_scaling"], dtype=torch.float32)
        self.assertEqual(ours32.shape, (CHECKPOINTS[name]["head_dim"] // 2,))
        self.assertTrue(torch.equal(ours32, theirs),
                        f"max difference {gap(ours32, theirs)}")
        ours64 = rope_inverse_frequencies(arch["head_dim"], arch["rope_theta"],
                                          arch["rope_scaling"], dtype=torch.float64)
        exact = reference_frequencies(config)
        self.assertLess(((ours64 - exact).abs() / exact).max().item(), 1e-14)
        # and float32 arithmetic stays within a few ulp of the exact table
        self.assertLess(((theirs.double() - exact).abs() / exact).max().item(), 1e-6)

    @only(LLAMA3_CHECKPOINTS)
    def check_llama3_bands(self, name):
        """The llama3 rescaling has three regimes; the published table and
        the scaled-down one used by the parity tests both populate all of
        them, and the rescaled table differs from the plain one."""
        for raw in (fixture(name), tiny_raw(name)):
            config = as_config(raw)
            kept, blended, slowed = llama3_bands(config)
            self.assertGreater(kept, 0)
            self.assertGreater(blended, 0)
            self.assertGreater(slowed, 0)
            arch = llama_architecture(config)
            plain = rope_inverse_frequencies(arch["head_dim"], arch["rope_theta"])
            scaled = rope_inverse_frequencies(arch["head_dim"], arch["rope_theta"],
                                              arch["rope_scaling"])
            self.assertTrue(torch.equal(scaled[:kept], plain[:kept]))
            self.assertTrue((scaled[kept:] < plain[kept:]).all())
            factor = CHECKPOINTS[name]["llama3_factor"]
            self.assertTrue(torch.allclose(scaled[kept + blended:] * factor,
                                           plain[kept + blended:], rtol=1e-12, atol=0.0))

    def check_tiny_config_keeps_every_switch(self, name):
        """The scaled-down config differs from the published one in sizes
        only."""
        sizes = {"vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                 "num_attention_heads", "num_key_value_heads", "head_dim", "sliding_window"}
        full = llama_architecture(as_config(fixture(name)))
        small = llama_architecture(as_config(tiny_raw(name)))
        for key in full:
            if key in sizes:
                continue
            if key == "rope_scaling" and full[key] is not None:
                expected = dict(full[key], original_max_position_embeddings=float(TINY_ORIGINAL_LENGTH))
                self.assertEqual(small[key], expected)
            else:
                self.assertEqual(small[key], full[key], key)
        spec = CHECKPOINTS[name]
        self.assertEqual(small["num_attention_heads"] * spec["kv"],
                         small["num_key_value_heads"] * spec["heads"])
        self.assertEqual(small["sliding_window"], [None] * TINY_LAYERS)
        raw_full, raw_small = fixture(name), tiny_raw(name)
        self.assertEqual(set(raw_full), set(raw_small))
        self.assertEqual(raw_full["tie_word_embeddings"], spec["tied"])
        self.assertEqual(raw_small["tie_word_embeddings"], spec["tied"])


### 2. Parity ###

@unittest.skipUnless(HAS_TRANSFORMERS, "transformers / safetensors are not installed")
@per_checkpoint
class TestParity(unittest.TestCase):
    """
        The published architecture, scaled down in size only, with random
        float64 weights: the transformers class against our network.
    """

    def _compare(self, name, label, source, outputs, ids, **kwargs):
        for head, which in ((True, "logits"), (False, "hidden")):
            net = LlamaLike.from_huggingface(source, include_lm_head=head,
                                             dtype=torch.float64, **kwargs)
            got = run(net, ids)
            self.assertEqual(got.shape, outputs[which].shape[1:])
            scale = outputs[which].abs().max().item()
            as_is = gap(got, outputs[which])
            exact = gap(got, outputs["exact_" + which])
            note("parity", f"{name} {label} {which}", as_is, scale)
            note("parity (own dtype)", f"{name} {label} {which}", exact, scale)
            self.assertGreater(scale, 0.05)
            self.assertLess(as_is, HF_TOL, f"{label} {which}: differs from the reference by {as_is}")
            self.assertLess(exact, EXACT_TOL,
                            f"{label} {which}: differs from the float64 reference by {exact}")

    def check_from_memory(self, name):
        """Loaded from the transformers model in memory."""
        hf, ids, outputs = tiny(name)
        self.assertIs(type(hf).__name__ == "Qwen3ForCausalLM", CHECKPOINTS[name]["qk_norm"])
        self.assertEqual(hf.lm_head.weight is hf.model.embed_tokens.weight,
                         CHECKPOINTS[name]["tied"])
        self._compare(name, "memory", hf, outputs, ids)

    def check_from_disk(self, name):
        """Streamed from a save_pretrained directory whose config.json is
        in the PUBLISHED format (rope_theta / rope_scaling keys of
        transformers 4), as on the hub -- and from the directory exactly
        as transformers 5 writes it (rope_parameters)."""
        hf, ids, outputs = tiny(name)
        tied = CHECKPOINTS[name]["tied"]
        with tempfile.TemporaryDirectory() as d:
            hf.save_pretrained(d)
            stored = safetensors.torch.load_file(os.path.join(d, "model.safetensors"))
            # a tied head is not stored: the embedding table has to serve
            self.assertEqual("lm_head.weight" in stored, not tied)
            self.assertEqual(stored["model.embed_tokens.weight"].dtype, torch.float64)
            with open(os.path.join(d, "config.json")) as f:
                written = json.load(f)
            self.assertIn("rope_parameters", written)
            self._compare(name, "disk (transformers 5 config)", d, outputs, ids)

            published = tiny_raw(name)
            self.assertIn("rope_theta", published)
            self.assertNotIn("rope_parameters", published)
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump(published, f)
            self._compare(name, "disk (published config)", d, outputs, ids)

    def check_sequence_reaches_the_rescaled_frequencies(self, name):
        """A parity test on a short sequence would not see the RoPE
        rescaling at all: the two tables only part ways with the position.
        Over SEQ_LEN positions the frequencies of the scaled-down config
        turn by clearly different angles with and without it."""
        config = as_config(tiny_raw(name))
        arch = llama_architecture(config)
        plain = rope_inverse_frequencies(arch["head_dim"], arch["rope_theta"])
        scaled = rope_inverse_frequencies(arch["head_dim"], arch["rope_theta"],
                                          arch["rope_scaling"])
        kept, blended, slowed = llama3_bands(config)
        turn = (plain - scaled) * (SEQ_LEN - 1)        # radians, at the last position
        self.assertEqual(turn[:kept].abs().max().item(), 0.0)
        self.assertGreater(turn[kept:kept + blended].min().item(), 0.3)
        self.assertGreater(turn[kept + blended:].max().item(), 0.03)
    check_sequence_reaches_the_rescaled_frequencies.checkpoints = LLAMA3_CHECKPOINTS


### 3. Negative controls ###

def _set_theta(theta):
    def mutate(net):
        for layer in attention_layers(net):
            layer.rope_theta = theta
    return mutate


def _no_rescaling(net):
    for layer in attention_layers(net):
        layer.rope_scaling = None


def _other_factor(net):
    # the factor of the other Llama-3 generation (3.1 / 3.3: 8, 3.2: 32)
    for layer in attention_layers(net):
        layer.rope_scaling = dict(layer.rope_scaling,
                                  factor=8.0 if layer.rope_scaling["factor"] == 32.0 else 32.0)


def _published_length(net):
    # the published pretraining length on the scaled-down table: the
    # frequencies land in other bands
    for layer in attention_layers(net):
        layer.rope_scaling = dict(layer.rope_scaling, original_max_position_embeddings=8192.0)


def _no_qk_norm(net):
    for layer in attention_layers(net):
        layer.use_qk_norm = False


def _swap_qk_norm(net):
    for layer in attention_layers(net):
        layer.q_norm, layer.k_norm = layer.k_norm, layer.q_norm


def _roll_kv_heads(net):
    # every query head reads the NEXT key/value head
    with torch.no_grad():
        for layer in attention_layers(net):
            for proj in (layer.k_proj, layer.v_proj):
                w = proj.weight.view(layer.num_kv_heads, layer.d_head, -1)
                proj.weight.copy_(w.roll(1, 0).reshape(proj.weight.shape))


def _tile_kv_heads(net):
    # key/value heads repeated as a block (k0 k1 k0 k1) instead of
    # interleaved (k0 k0 k1 k1). Query head h then reads kv head h % KV
    # instead of h // r; relabelling the query heads by
    # sigma(h) = h // r + KV * (h % r) expresses exactly that with the
    # interleaved layer.
    with torch.no_grad():
        for layer in attention_layers(net):
            H, KV, r, d = layer.num_heads, layer.num_kv_heads, layer.kv_repeat, layer.d_head
            sigma = torch.tensor([h // r + KV * (h % r) for h in range(H)])
            q = layer.q_proj.weight.view(H, d, -1)[sigma]
            o = layer.o_proj.weight.view(-1, H, d)[:, sigma]
            layer.q_proj.weight.copy_(q.reshape(H * d, -1))
            layer.o_proj.weight.copy_(o.reshape(-1, H * d))


def _swap_heads_and_head_dim(net):
    # twice the heads of half the size: every projection keeps its shape
    for layer in attention_layers(net):
        layer.num_heads *= 2
        layer.num_kv_heads *= 2
        layer.d_head //= 2
        layer.rotary_dim = layer.d_head
        layer.query_scale = math.sqrt(layer.d_head)


def _other_eps(net):
    for module in net.modules():
        if isinstance(module, RMSNorm):
            module.eps = 1e-6 if module.eps == 1e-5 else 1e-5


def _zero_centered_norms(net):
    # the convention of Gemma and Qwen3.5: scale by 1 + weight
    with torch.no_grad():
        for layer in net.layers:
            if isinstance(layer, RMSNorm):
                layer.weight.add_(1.0)


def _swap_block_norms(net):
    # input_layernorm <-> post_attention_layernorm
    with torch.no_grad():
        for i in range(1, 1 + TINY_LAYERS * LlamaLike.BLOCK_STRIDE, LlamaLike.BLOCK_STRIDE):
            a, b = net.layers[i].weight, net.layers[i + 3].weight
            tmp = a.clone()
            a.copy_(b)
            b.copy_(tmp)


def _embedding_as_head(net):
    with torch.no_grad():
        net.layers[-1].weight.copy_(net.layers[0].weight)


def _window(width):
    def mutate(net):
        for layer in attention_layers(net):
            layer.sliding_window = width
    return mutate


def _no_causal_mask(net):
    for layer in attention_layers(net):
        layer.causal = False


@unittest.skipUnless(HAS_TRANSFORMERS, "transformers / safetensors are not installed")
@per_checkpoint
class TestNegativeControls(unittest.TestCase):
    """
        The parity test of TestParity with ONE thing wrong in our network.
        Each of these is a mistake a loader can make without any shape
        check noticing; each has to move the logits away from the
        (unmodified) reference by more than WIDE = 1000 x HF_TOL.
    """

    def _control(self, name, label, mutate, floor=WIDE):
        hf, ids, outputs = tiny(name)
        net = LlamaLike.from_huggingface(hf, dtype=torch.float64)
        self.assertLess(gap(run(net, ids), outputs["logits"]), HF_TOL)
        mutate(net)
        off = gap(run(net, ids), outputs["logits"])
        note("control", f"{name} {label}", off, outputs["logits"].abs().max().item())
        self.assertGreater(off, floor, f"{label}: the parity test does not see it ({off})")

    def check_rope_base_10000(self, name):
        """The original bug: every checkpoint rotated with base 10000."""
        self._control(name, "RoPE base 10000", _set_theta(10000.0))

    def check_rope_base_of_another_family(self, name):
        other = 1000000.0 if CHECKPOINTS[name]["theta"] != 1000000.0 else 500000.0
        self._control(name, f"RoPE base {other:g}", _set_theta(other))

    @only(LLAMA3_CHECKPOINTS)
    def check_without_llama3_rescaling(self, name):
        self._control(name, "no llama3 rescaling", _no_rescaling)

    @only(LLAMA3_CHECKPOINTS)
    def check_llama3_factor_of_the_other_generation(self, name):
        self._control(name, "llama3 factor 8 <-> 32", _other_factor)

    @only(LLAMA3_CHECKPOINTS)
    def check_llama3_wrong_pretraining_length(self, name):
        self._control(name, "llama3 original length 8192", _published_length)

    @only(QWEN3_CHECKPOINTS)
    def check_without_qk_norm(self, name):
        self._control(name, "no q/k norm", _no_qk_norm)

    @only(QWEN3_CHECKPOINTS)
    def check_qk_norm_weights_swapped(self, name):
        self._control(name, "q_norm <-> k_norm", _swap_qk_norm)

    def check_kv_heads_rolled(self, name):
        self._control(name, "key/value heads rolled by one", _roll_kv_heads)

    @only(GROUPED_CHECKPOINTS)
    def check_kv_heads_tiled_instead_of_interleaved(self, name):
        self._control(name, "kv heads tiled, not interleaved", _tile_kv_heads)

    @only([n for n in CHECKPOINTS if n not in QWEN3_CHECKPOINTS])
    def check_heads_and_head_dim_swapped(self, name):
        self._control(name, "2x heads of half the dimension", _swap_heads_and_head_dim)

    def check_norm_eps(self, name):
        self._control(name, "RMSNorm eps 1e-5 <-> 1e-6", _other_eps)

    def check_zero_centered_norm_weights(self, name):
        self._control(name, "norm weights as 1 + w", _zero_centered_norms)

    def check_block_norms_swapped(self, name):
        self._control(name, "input <-> post-attention norm", _swap_block_norms)

    @only(UNTIED_CHECKPOINTS)
    def check_embedding_used_as_head(self, name):
        self._control(name, "embedding as LM head", _embedding_as_head)

    def check_sliding_window(self, name):
        self._control(name, "sliding window of 8", _window(8))

    def check_no_causal_mask(self, name):
        self._control(name, "no causal mask", _no_causal_mask)


### 4. Knowledge matrix invariants ###

@unittest.skipUnless(HAS_TRANSFORMERS, "transformers / safetensors are not installed")
@per_checkpoint
class TestKnowledgeMatrix(unittest.TestCase):
    """
        The defining identity M(x) 1 = f(x) on the networks the loader
        builds, in both linearizations, and the agreement of the two
        computers (forward-mode columns, reverse-mode rows).
    """

    def _check(self, name, include_lm_head):
        hf, _, _ = tiny(name)
        net = LlamaLike.from_huggingface(hf, include_lm_head=include_lm_head,
                                         dtype=torch.float64)
        net.eval()
        x = token_ids(KM_SEQ_LEN, seed=7).unsqueeze(0)          # (1, 1, T)
        with torch.no_grad():
            out = net.forward(x).reshape(-1)
        width = out.numel() // KM_SEQ_LEN
        rows = [0, width + 3, out.numel() - 1]
        for mode in ("ratio", "frozen"):
            mat = KnowledgeMatrixComputer(net, batch_size=128, mixer_mode=mode).forward(x)
            self.assertEqual(tuple(mat.shape),
                             (out.numel(), KM_SEQ_LEN * net.hidden_size + 1))
            sums = gap(mat.sum(1), out)
            note("knowledge matrix", f"{name} head={include_lm_head} {mode}: row sums", sums,
                 out.abs().max().item())
            self.assertLess(sums, 1e-9, f"{mode}: rows do not sum to the forward pass")
            got = KnowledgeRowComputer(net, mixer_mode=mode).forward(x, rows)
            same = gap(got, mat[rows])
            note("knowledge matrix", f"{name} head={include_lm_head} {mode}: rows", same)
            self.assertLess(same, 1e-9, f"{mode}: row computer and matrix computer disagree")
            if mode == "frozen":
                # frozen routing lets attribution cross positions: the
                # last position must depend on the first token's embedding
                last = mat[-1, :net.hidden_size]
                self.assertGreater(last.abs().max().item(), 1e-8)
        # and the network is unchanged by the computation
        with torch.no_grad():
            self.assertEqual(gap(net.forward(x), out), 0.0)

    def check_logits(self, name):
        self._check(name, include_lm_head=True)

    def check_hidden_states(self, name):
        self._check(name, include_lm_head=False)


### 5. Full-size dry runs ###

@unittest.skipUnless(HAS_TRANSFORMERS, "transformers / safetensors are not installed")
@per_checkpoint
class TestFullSizeDryRun(unittest.TestCase):
    """
        The published architectures at their real size, without a single
        weight in memory: the transformers class is instantiated on the
        "meta" device, and the loader runs against it with device="meta",
        which locates every tensor and checks its shape.
    """

    def check_dry_run(self, name):
        spec = CHECKPOINTS[name]
        config = as_config(fixture(name))
        with torch.device("meta"):
            hf = AutoModelForCausalLM.from_config(config)
        self.assertTrue(all(p.is_meta for p in hf.parameters()))
        reference_count = sum(p.numel() for p in hf.parameters())
        self.assertEqual(reference_count, spec["params"])
        embedding = spec["vocab"] * spec["hidden"]
        tied = hf.lm_head.weight is hf.model.embed_tokens.weight
        self.assertEqual(tied, spec["tied"])

        net = LlamaLike.from_huggingface(hf, device="meta")
        self.assertEqual(net.device, "meta")
        self.assertTrue(all(p.is_meta for p in net.parameters()))
        self.assertEqual(net.get_num_layers(),
                         1 + spec["layers"] * LlamaLike.BLOCK_STRIDE + 2)
        ours = sum(p.numel() for p in net.parameters())
        # transformers counts a tied head once; our network holds the
        # embedding table and the head as two layers
        self.assertEqual(ours, reference_count + (embedding if tied else 0))

        headless = LlamaLike.from_huggingface(hf, include_lm_head=False, device="meta")
        self.assertEqual(sum(p.numel() for p in headless.parameters()),
                         reference_count - (0 if tied else embedding))

        # every tensor of the reference went somewhere, with its shape
        shapes = sorted(tuple(p.shape) for p in net.parameters())
        expected = sorted(tuple(t.shape) for t in hf.state_dict().values())
        self.assertEqual(shapes, expected)

        # a pipeline over several devices can be laid out at full size
        layer_devices, boundaries = net.shard_layout(["cuda:0", "cuda:1", "cuda:2"])
        self.assertEqual(len(layer_devices), net.get_num_layers())
        self.assertEqual(len(boundaries), 2)

    @only(["meta-llama/Llama-3.2-1B", "Qwen/Qwen3-8B"])
    def check_dry_run_refuses_another_architecture(self, name):
        """The dry run is a real check: the same checkpoint against a
        config that is off by one switch does not go through."""
        with torch.device("meta"):
            hf = AutoModelForCausalLM.from_config(as_config(fixture(name)))
        hf.config.num_key_value_heads = hf.config.num_attention_heads
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            LlamaLike.from_huggingface(hf, device="meta")
        hf.config.num_key_value_heads = CHECKPOINTS[name]["kv"]
        hf.config.num_hidden_layers -= 1
        with self.assertRaisesRegex(RuntimeError, "no counterpart"):
            LlamaLike.from_huggingface(hf, device="meta")


### 6. Loader mechanics ###

def _rewrite(directory: str, change) -> None:
    """Apply `change(tensors)` to the stored tensors of a checkpoint."""
    path = os.path.join(directory, "model.safetensors")
    tensors = safetensors.torch.load_file(path)
    change(tensors)
    safetensors.torch.save_file(tensors, path, metadata={"format": "pt"})


@unittest.skipUnless(HAS_TRANSFORMERS, "transformers / safetensors are not installed")
class TestLoaderMechanics(unittest.TestCase):
    TIED = "meta-llama/Llama-3.2-1B"
    UNTIED = "Qwen/Qwen3-8B"

    # -- sharding --

    def test_sharded_equals_unsharded(self):
        for name in (self.TIED, self.UNTIED):
            hf, ids, outputs = tiny(name)
            with tempfile.TemporaryDirectory() as d:
                hf.save_pretrained(d)
                for source in (hf, d):
                    whole = LlamaLike.from_huggingface(source, dtype=torch.float64)
                    self.assertIsNone(whole.layer_devices)
                    expected = run(whole, ids)
                    stride = LlamaLike.BLOCK_STRIDE
                    for boundaries in (None, [1 + stride], [4, 1 + 2 * stride]):
                        devices = ["cpu"] * (2 if boundaries is None else len(boundaries) + 1)
                        with self.subTest(name=name, disk=source is d, boundaries=boundaries):
                            net = LlamaLike.from_huggingface(
                                source, dtype=torch.float64, device=devices,
                                boundaries=boundaries)
                            self.assertEqual(net.layer_devices, ["cpu"] * net.get_num_layers())
                            if boundaries is not None:
                                self.assertEqual(net.shard_boundaries, boundaries)
                            else:
                                self.assertEqual(len(net.shard_boundaries), 1)
                                self.assertIn(net.shard_boundaries[0], net.shard_cut_points())
                            self.assertEqual(gap(run(net, ids), expected), 0.0)
                            self.assertLess(gap(run(net, ids), outputs["logits"]), HF_TOL)

    def test_sharded_knowledge_rows(self):
        hf, _, _ = tiny(self.UNTIED)
        x = token_ids(KM_SEQ_LEN, seed=3).unsqueeze(0)
        whole = LlamaLike.from_huggingface(hf, dtype=torch.float64)
        split = LlamaLike.from_huggingface(hf, dtype=torch.float64, device=["cpu", "cpu"])
        rows = [2, 3 * TINY_VOCAB + 5]
        for mode in ("ratio", "frozen"):
            a = KnowledgeRowComputer(whole, mixer_mode=mode).forward(x, rows)
            b = KnowledgeRowComputer(split, mixer_mode=mode).forward(x, rows)
            self.assertLess(gap(a, b), 1e-12)

    def test_inadmissible_boundaries_are_refused(self):
        hf, _, _ = tiny(self.TIED)
        with self.assertRaises(ValueError):      # inside a residual span
            LlamaLike.from_huggingface(hf, device=["cpu", "cpu"], boundaries=[3])
        with self.assertRaises(ValueError):      # one device, but a cut
            LlamaLike.from_huggingface(hf, device="cpu", boundaries=[7])
        with self.assertRaises(ValueError):      # two devices, two cuts
            LlamaLike.from_huggingface(hf, device=["cpu", "cpu"], boundaries=[7, 13])

    # -- copy or share --

    def _pairs(self, hf, net):
        """(our parameter, the tensor of the model it was loaded from)."""
        state = hf.state_dict()
        attn, mlp = net.layers[2], net.layers[5]
        return [
            (net.layers[0].weight, state["model.embed_tokens.weight"]),
            (net.layers[1].weight, state["model.layers.0.input_layernorm.weight"]),
            (attn.q_proj.weight, state["model.layers.0.self_attn.q_proj.weight"]),
            (mlp.down_proj.weight, state["model.layers.0.mlp.down_proj.weight"]),
            (net.layers[-2].weight, state["model.norm.weight"]),
            (net.layers[-1].weight, state["lm_head.weight"]),
        ]

    def test_copy_true_is_independent_of_the_model(self):
        for name in (self.TIED, self.UNTIED):
            hf = build_reference(tiny_raw(name), seed=11)
            ids = token_ids()
            net = LlamaLike.from_huggingface(hf)             # copy=True is the default
            for ours, theirs in self._pairs(hf, net):
                self.assertNotEqual(ours.data_ptr(), theirs.data_ptr())
                self.assertTrue(torch.equal(ours, theirs))
            before = run(net, ids)
            with torch.no_grad():
                for p in hf.parameters():
                    p.mul_(1.5)
            self.assertEqual(gap(run(net, ids), before), 0.0)

    def test_copy_false_shares_the_storage(self):
        for name in (self.TIED, self.UNTIED):
            hf = build_reference(tiny_raw(name), seed=12)
            ids = token_ids()
            net = LlamaLike.from_huggingface(hf, copy=False)
            for ours, theirs in self._pairs(hf, net):
                self.assertEqual(ours.data_ptr(), theirs.data_ptr())
            every = {p.data_ptr() for p in hf.parameters()}
            self.assertTrue(all(p.data_ptr() in every for p in net.parameters()))
            with torch.no_grad():
                self.assertLess(gap(run(net, ids), hf(ids).logits), HF_TOL)
                hf.model.layers[1].mlp.up_proj.weight.mul_(-1.0)     # edit the model ...
                self.assertLess(gap(run(net, ids), hf(ids).logits), HF_TOL)   # ... we follow

    def test_copy_false_with_a_cast_cannot_share(self):
        hf = build_reference(tiny_raw(self.UNTIED), seed=13)
        net = LlamaLike.from_huggingface(hf, copy=False, dtype=torch.float32)
        for ours, theirs in self._pairs(hf, net):
            self.assertEqual(ours.dtype, torch.float32)
            self.assertNotEqual(ours.data_ptr(), theirs.data_ptr())

    # -- dtypes --

    def test_dtype_of_a_model_in_memory(self):
        hf, ids, outputs = tiny(self.TIED)
        net = LlamaLike.from_huggingface(hf)                 # default: the model's
        self.assertEqual({p.dtype for p in net.parameters()}, {torch.float64})
        net32 = LlamaLike.from_huggingface(hf, dtype=torch.float32)
        self.assertEqual({p.dtype for p in net32.parameters()}, {torch.float32})
        out = run(net32, ids)
        self.assertEqual(out.dtype, torch.float32)
        off = gap(out, outputs["logits"])
        note("loader", "float32 network of a float64 model", off)
        self.assertLess(off, 1e-4)
        self.assertGreater(off, 0.0)

    def test_dtype_of_a_checkpoint_on_disk(self):
        """Stored in float64 or bfloat16, a checkpoint read from disk
        comes out in float32 unless a dtype is asked for."""
        ids = token_ids()
        for name in (self.TIED, self.UNTIED):
            for stored in (torch.float64, torch.bfloat16):
                hf = build_reference(tiny_raw(name), seed=14).to(stored)
                with tempfile.TemporaryDirectory() as d:
                    hf.save_pretrained(d)
                    on_disk = safetensors.torch.load_file(os.path.join(d, "model.safetensors"))
                    self.assertEqual({t.dtype for t in on_disk.values()}, {stored})
                    with self.subTest(name=name, stored=stored):
                        net = LlamaLike.from_huggingface(d)
                        self.assertEqual({p.dtype for p in net.parameters()}, {torch.float32})
                        # the stored values, exactly (bfloat16 -> float32 is exact)
                        expected = on_disk["model.layers.1.self_attn.v_proj.weight"].to(torch.float32)
                        self.assertTrue(torch.equal(net.layers[1 + 6 + 1].v_proj.weight, expected))
                        out32 = run(net, ids)
                        self.assertEqual(out32.dtype, torch.float32)
                        self.assertTrue(torch.isfinite(out32).all())

                        net64 = LlamaLike.from_huggingface(d, dtype=torch.float64)
                        self.assertEqual({p.dtype for p in net64.parameters()}, {torch.float64})
                        off = gap(out32, run(net64, ids))
                        note("loader", f"{name} stored {stored}: float32 vs float64 network", off)
                        self.assertLess(off, 1e-4)

                        half = LlamaLike.from_huggingface(d, dtype=torch.bfloat16)
                        self.assertEqual({p.dtype for p in half.parameters()}, {torch.bfloat16})
                        out16 = run(half, ids)
                        self.assertEqual(out16.dtype, torch.bfloat16)
                        self.assertTrue(torch.isfinite(out16).all())
                        if stored == torch.bfloat16:
                            # same weights, half-precision arithmetic: close
                            # to the float32 network, not equal to it
                            off = gap(out16, out32)
                            note("loader", f"{name} bfloat16 vs float32 network, same weights",
                                 off, out32.abs().max().item())
                            self.assertLess(off, 0.1 * out32.abs().max().item())

    # -- damaged or mismatching checkpoints --

    def _saved(self, name, seed=15, **overrides):
        raw = tiny_raw(name)
        raw.update(overrides)
        hf = build_reference(raw, seed=seed)
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        hf.save_pretrained(d.name)
        return hf, d.name

    def test_missing_tensor(self):
        for key in ("model.layers.1.mlp.up_proj.weight",
                    "model.layers.2.post_attention_layernorm.weight",
                    "model.layers.0.self_attn.k_norm.weight",
                    "model.norm.weight"):
            _, d = self._saved(self.UNTIED)
            _rewrite(d, lambda tensors: tensors.pop(key))
            with self.subTest(key=key):
                with self.assertRaisesRegex(KeyError, key.replace(".", r"\.")):
                    LlamaLike.from_huggingface(d)
                with self.assertRaises(KeyError):
                    LlamaLike.from_huggingface(d, device="meta")

    def test_missing_embedding(self):
        _, d = self._saved(self.UNTIED)
        _rewrite(d, lambda tensors: tensors.pop("model.embed_tokens.weight"))
        with self.assertRaisesRegex(ValueError, "could not locate the text decoder"):
            LlamaLike.from_huggingface(d)

    def test_misshaped_tensor(self):
        def transposed(key):
            def change(tensors):
                tensors[key] = tensors[key].T.contiguous()
            return change

        def other_tensor(key, source):
            def change(tensors):
                tensors[key] = tensors[source].clone()
            return change

        def truncated(key):
            def change(tensors):
                tensors[key] = tensors[key][:-1].clone()
            return change

        cases = [
            transposed("model.layers.0.mlp.gate_proj.weight"),
            other_tensor("model.layers.1.self_attn.k_proj.weight",
                         "model.layers.1.self_attn.q_proj.weight"),
            truncated("model.embed_tokens.weight"),
            truncated("model.layers.2.input_layernorm.weight"),
        ]
        for i, change in enumerate(cases):
            _, d = self._saved(self.TIED)
            _rewrite(d, change)
            with self.subTest(case=i):
                with self.assertRaisesRegex(ValueError, "shape mismatch"):
                    LlamaLike.from_huggingface(d)
                with self.assertRaisesRegex(ValueError, "shape mismatch"):
                    LlamaLike.from_huggingface(d, device="meta")

    def test_missing_head_of_an_untied_checkpoint(self):
        """transformers would initialize the missing head at random; the
        embedding table must not be used in its place."""
        _, d = self._saved(self.UNTIED)
        _rewrite(d, lambda tensors: tensors.pop("lm_head.weight"))
        with self.assertRaisesRegex(KeyError, "no lm_head"):
            LlamaLike.from_huggingface(d)
        net = LlamaLike.from_huggingface(d, include_lm_head=False)   # hidden states: fine
        self.assertIsInstance(net.layers[-1], RMSNorm)

    def test_untied_head_stored_next_to_a_tied_config(self):
        """Both tensors stored and different, config says tied:
        transformers does NOT tie in that case and uses the stored head."""
        hf, d = self._saved(self.TIED)
        ids = token_ids()
        head = torch.randn(TINY_VOCAB, hf.config.hidden_size, generator=torch.Generator().manual_seed(5)) * 0.05

        def add_head(tensors):
            tensors["lm_head.weight"] = head.clone()
        _rewrite(d, add_head)
        reference = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float64).eval()
        self.assertIsNot(reference.lm_head.weight, reference.model.embed_tokens.weight)
        with torch.no_grad():
            expected = reference(ids).logits
        net = LlamaLike.from_huggingface(d, dtype=torch.float64)
        self.assertTrue(torch.equal(net.layers[-1].weight, head))
        self.assertLess(gap(run(net, ids), expected), HF_TOL)

    def test_leftover_tensor(self):
        """A decoder tensor the architecture has no slot for means the
        network is not the checkpoint's function."""
        def bias(tensors):
            tensors["model.layers.1.self_attn.q_proj.bias"] = torch.zeros(
                tensors["model.layers.1.self_attn.q_proj.weight"].shape[0])

        def extra_layer(tensors):
            for key in [k for k in tensors if k.startswith("model.layers.2.")]:
                tensors[key.replace("layers.2.", "layers.3.")] = tensors[key].clone()

        def unknown(tensors):
            tensors["model.layers.0.self_attn.lambda_q1"] = torch.zeros(TINY_HEAD_DIM)

        for change, pattern in ((bias, r"layers\.1\.self_attn\.q_proj\.bias"),
                                (extra_layer, r"layers\.3\."),
                                (unknown, "lambda_q1")):
            _, d = self._saved(self.TIED)
            _rewrite(d, change)
            with self.subTest(change=change.__name__):
                with self.assertRaisesRegex(RuntimeError, pattern):
                    LlamaLike.from_huggingface(d)

    def test_old_rope_buffer_is_ignored(self):
        """Checkpoints saved by old transformers versions carry the RoPE
        table as a buffer; it is recomputed, not a parameter."""
        hf, d = self._saved(self.TIED)
        ids = token_ids()

        def buffer(tensors):
            tensors["model.layers.0.self_attn.rotary_emb.inv_freq"] = torch.zeros(TINY_HEAD_DIM // 2)
        _rewrite(d, buffer)
        net = LlamaLike.from_huggingface(d, dtype=torch.float64)
        with torch.no_grad():
            self.assertLess(gap(run(net, ids), hf(ids).logits), HF_TOL)

    def test_config_and_weights_disagree_on_biases(self):
        # the config announces biases the checkpoint does not have
        hf, d = self._saved(self.TIED)
        with open(os.path.join(d, "config.json")) as f:
            config = json.load(f)
        for flag, pattern in (("attention_bias", r"q_proj\.bias"), ("mlp_bias", r"gate_proj\.bias")):
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump(dict(config, **{flag: True}), f)
            with self.assertRaisesRegex(KeyError, pattern):
                LlamaLike.from_huggingface(d)
        # the checkpoint has biases the config does not announce
        hf, d = self._saved(self.TIED, attention_bias=True, mlp_bias=True)
        with open(os.path.join(d, "config.json")) as f:
            config = json.load(f)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(dict(config, attention_bias=False, mlp_bias=False), f)
        with self.assertRaisesRegex(RuntimeError, "no counterpart"):
            LlamaLike.from_huggingface(d)

    def test_no_safetensors_weights(self):
        _, d = self._saved(self.TIED)
        os.remove(os.path.join(d, "model.safetensors"))
        with self.assertRaises(FileNotFoundError):
            LlamaLike.from_huggingface(d)

    def test_checkpoint_split_over_several_files(self):
        hf, ids, outputs = tiny(self.UNTIED)
        with tempfile.TemporaryDirectory() as d:
            hf.save_pretrained(d, max_shard_size="30KB")
            files = [f for f in os.listdir(d) if f.endswith(".safetensors")]
            self.assertGreater(len(files), 2)
            self.assertIn("model.safetensors.index.json", os.listdir(d))
            net = LlamaLike.from_huggingface(d, dtype=torch.float64)
            self.assertLess(gap(run(net, ids), outputs["logits"]), HF_TOL)
            os.remove(os.path.join(d, sorted(files)[1]))
            with self.assertRaises(FileNotFoundError):
                LlamaLike.from_huggingface(d, dtype=torch.float64)

    # -- the hub --

    def _hub(self, directory, config):
        """Patches standing in for the hub: `directory` is the repository."""
        calls = []

        def from_pretrained(source, **kwargs):
            calls.append(("config", source, None, kwargs))
            return config

        def download(repo_id, filename, **kwargs):
            calls.append(("file", repo_id, filename, kwargs))
            path = os.path.join(directory, filename)
            if not os.path.isfile(path):
                raise huggingface_hub.errors.EntryNotFoundError(filename)
            return path

        return calls, mock.patch.object(AutoConfig, "from_pretrained", side_effect=from_pretrained), \
            mock.patch.object(huggingface_hub, "hf_hub_download", side_effect=download)

    def test_hub_arguments_reach_the_hub(self):
        hf, ids, outputs = tiny(self.TIED)
        asked = dict(revision="0123abc", local_files_only=True, token="hf_secret",
                     cache_dir="/somewhere/else")
        for shards in (None, "30KB"):
            with tempfile.TemporaryDirectory() as d, self.subTest(shards=shards):
                hf.save_pretrained(d, **({"max_shard_size": shards} if shards else {}))
                calls, patch_config, patch_download = self._hub(d, hf.config)
                with patch_config, patch_download:
                    net = LlamaLike.from_huggingface("some-org/some-model",
                                                     dtype=torch.float64, **asked)
                self.assertLess(gap(run(net, ids), outputs["logits"]), HF_TOL)
                self.assertEqual(calls[0][0], "config")
                files = [c[2] for c in calls if c[0] == "file"]
                self.assertEqual(files[0], "model.safetensors.index.json")
                if shards:
                    self.assertGreater(len(files), 3)
                else:
                    self.assertEqual(files[1:], ["model.safetensors"])
                for kind, repo, _, kwargs in calls:
                    self.assertEqual(repo, "some-org/some-model")
                    self.assertEqual(kwargs, asked, kind)

    def test_hub_defaults(self):
        hf, _, _ = tiny(self.TIED)
        with tempfile.TemporaryDirectory() as d:
            hf.save_pretrained(d)
            calls, patch_config, patch_download = self._hub(d, hf.config)
            with patch_config, patch_download:
                LlamaLike.from_huggingface("some-org/some-model")
            for _, _, _, kwargs in calls:
                self.assertEqual(kwargs, dict(revision=None, local_files_only=False,
                                              token=None, cache_dir=None))

    def test_unknown_arguments_are_not_swallowed(self):
        hf, _, _ = tiny(self.TIED)
        with self.assertRaises(TypeError):
            LlamaLike.from_huggingface(hf, trust_remote_code=True)
        with self.assertRaises(TypeError):
            LlamaLike.from_huggingface(hf, subfolder="decoder")

    def test_local_directory_never_asks_the_hub(self):
        hf, ids, outputs = tiny(self.TIED)
        with tempfile.TemporaryDirectory() as d:
            hf.save_pretrained(d)
            with mock.patch.object(huggingface_hub, "hf_hub_download",
                                   side_effect=AssertionError("the hub was contacted")):
                net = LlamaLike.from_huggingface(d, dtype=torch.float64,
                                                 revision="ignored-for-a-directory")
            self.assertLess(gap(run(net, ids), outputs["logits"]), HF_TOL)

    def test_open_checkpoint_views(self):
        hf, _, _ = tiny(self.TIED)
        memory = open_checkpoint(hf)
        self.assertTrue(memory.in_memory)
        self.assertEqual(memory.text_prefix, "model.")
        self.assertEqual(memory.lm_head_key, "lm_head.weight")
        with tempfile.TemporaryDirectory() as d:
            hf.save_pretrained(d)
            disk = open_checkpoint(d)
            self.assertFalse(disk.in_memory)
            self.assertEqual(disk.text_prefix, "model.")
            self.assertEqual(disk.lm_head_key, "model.embed_tokens.weight")   # tied
            self.assertEqual(set(disk.keys()), set(memory.keys()) - {"lm_head.weight"})
            for key in disk.keys():
                self.assertTrue(torch.equal(disk.tensor(key), memory.tensor(key)), key)
            disk.close()


### Half precision ###

@unittest.skipUnless(HAS_TRANSFORMERS, "transformers / safetensors are not installed")
class TestHalfPrecision(unittest.TestCase):
    """
        bfloat16 is the dtype the large checkpoints have to run in. Half
        precision cannot agree with float32 beyond a few percent, but our
        network must not be LESS accurate than the transformers model in
        the same dtype. The one place where the two differed is RMSNorm:
        the reference normalizes in float32 whatever the dtype, and ours
        accumulated the mean square in bfloat16 (on real SmolLM2-135M
        weights that doubled the worst logit error against float32, 0.21
        of the logit scale instead of the reference's 0.10).
    """

    def test_rmsnorm_is_the_reference_in_half_precision(self):
        from transformers.models.llama.modeling_llama import LlamaRMSNorm
        g = torch.Generator().manual_seed(0)
        for dtype in (torch.bfloat16, torch.float16):
            for eps in (1e-5, 1e-6):
                ours, theirs = RMSNorm(96, eps=eps), LlamaRMSNorm(96, eps=eps)
                weight = 1.0 + 0.3 * torch.randn(96, generator=g)
                with torch.no_grad():
                    ours.weight.copy_(weight)
                    theirs.weight.copy_(weight)
                ours, theirs = ours.to(dtype), theirs.to(dtype)
                x = (4.0 * torch.randn(3, 7, 96, generator=g)).to(dtype)
                with torch.no_grad():
                    out = ours(x)
                    self.assertEqual(out.dtype, dtype)
                    self.assertTrue(torch.equal(out, theirs(x)), f"{dtype} eps={eps}")
                    # the denominator kept for the knowledge matrix: the
                    # exact one rounded once (half an ulp: 2^-9 in
                    # bfloat16), where bfloat16 arithmetic loses more
                    exact = torch.sqrt((x.double() ** 2).mean(-1, keepdim=True) + eps)
                    kept = ours.rms(x)
                    self.assertEqual(kept.dtype, dtype)
                    self.assertLess(((kept.double() - exact) / exact).abs().max().item(), 2.0 ** -8)
                    naive = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
                    self.assertLessEqual(((kept.double() - exact) / exact).abs().max().item(),
                                         ((naive.double() - exact) / exact).abs().max().item())

    def test_rmsnorm_is_unchanged_in_single_and_double_precision(self):
        g = torch.Generator().manual_seed(1)
        for dtype in (torch.float32, torch.float64):
            norm = RMSNorm(96, eps=1e-5).to(dtype)
            with torch.no_grad():
                norm.weight.copy_(1.0 + 0.3 * torch.randn(96, generator=g))
            x = torch.randn(3, 7, 96, generator=g).to(dtype)
            rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-5)
            with torch.no_grad():
                self.assertTrue(torch.equal(norm.rms(x), rms))
                self.assertTrue(torch.equal(norm(x), x * norm.weight / rms))

    def test_half_precision_network_is_as_accurate_as_the_reference(self):
        worst = 0.0
        for name in CHECKPOINTS:
            hf, ids, outputs = tiny(name)
            truth = outputs["exact_logits"]
            scale = truth.abs().max().item()
            half = copy.deepcopy(hf).to(torch.bfloat16)
            with torch.no_grad():
                theirs = half(ids).logits
            net = LlamaLike.from_huggingface(half)       # keeps the model's dtype
            self.assertEqual({p.dtype for p in net.parameters()}, {torch.bfloat16})
            ours = run(net, ids)
            self.assertEqual(ours.dtype, torch.bfloat16)
            reference_error = gap(theirs, truth) / scale
            our_error = gap(ours, truth) / scale
            note("half precision", f"{name}: transformers bfloat16 vs float64", reference_error, scale)
            note("half precision", f"{name}: ours bfloat16 vs float64", our_error, scale)
            note("half precision", f"{name}: ours vs transformers, both bfloat16",
                 gap(ours, theirs) / scale, scale)
            with self.subTest(name=name):
                self.assertLess(reference_error, 0.1)
                self.assertLess(our_error, 1.5 * reference_error)
            worst = max(worst, our_error / reference_error)
        note("half precision", "worst ratio of our error to the reference's", worst)


### 7. Family variants and refusals ###

def _variant(model_type: str, **kwargs) -> dict:
    """config.json of a small model of one of the four families."""
    raw = dict(
        model_type=model_type,
        vocab_size=TINY_VOCAB,
        hidden_size=48,
        intermediate_size=112,
        num_hidden_layers=TINY_LAYERS,
        num_attention_heads=6,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        rope_theta=1000000.0,
        tie_word_embeddings=False,
    )
    raw.update(kwargs)
    return raw


@unittest.skipUnless(HAS_TRANSFORMERS, "transformers / safetensors are not installed")
class TestFamilyVariants(unittest.TestCase):
    """
        The config switches that the nine checkpoints above do not use but
        their siblings and the other two families do, each against the
        transformers class it belongs to.
    """

    def _parity(self, label, raw, seq_len=SEQ_LEN, seed=20, disk=True):
        hf = build_reference(raw, seed=seed)
        ids = token_ids(seq_len, seed=seed)
        outputs = reference_outputs(hf, ids)
        with tempfile.TemporaryDirectory() as d:
            hf.save_pretrained(d)
            nets = []
            for where, source in (("memory", hf), ("disk", d))[:2 if disk else 1]:
                net = LlamaLike.from_huggingface(source, dtype=torch.float64)
                got = run(net, ids)
                as_is, exact = gap(got, outputs["logits"]), gap(got, outputs["exact_logits"])
                note("variant", f"{label} {where}", as_is, outputs["logits"].abs().max().item())
                note("variant (own dtype)", f"{label} {where}", exact)
                self.assertLess(as_is, HF_TOL, f"{label} {where}")
                self.assertLess(exact, EXACT_TOL, f"{label} {where}")
                nets.append(net)
        return hf, nets[0], ids, outputs

    def _windows(self, net):
        return [layer.sliding_window for layer in attention_layers(net)]

    def _control(self, label, net, ids, outputs, mutate):
        mutate(net)
        off = gap(run(net, ids), outputs["logits"])
        note("variant control", label, off, outputs["logits"].abs().max().item())
        self.assertGreater(off, WIDE, label)

    # -- Llama --

    def test_llama_with_biases(self):
        """attention_bias puts a bias on q, k, v AND o; mlp_bias on the
        three MLP projections."""
        hf, net, ids, outputs = self._parity(
            "llama attention_bias + mlp_bias",
            _variant("llama", attention_bias=True, mlp_bias=True))
        attn, mlp = net.layers[2], net.layers[5]
        for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj,
                     mlp.gate_proj, mlp.up_proj, mlp.down_proj):
            self.assertIsNotNone(proj.bias)

        def drop(net):
            with torch.no_grad():
                net.layers[2].o_proj.bias.zero_()
        self._control("llama: o_proj bias dropped", net, ids, outputs, drop)

    def test_llama_biases_in_the_knowledge_matrix(self):
        net = LlamaLike.from_huggingface(
            build_reference(_variant("llama", attention_bias=True, mlp_bias=True), seed=21),
            dtype=torch.float64)
        x = token_ids(KM_SEQ_LEN, seed=8).unsqueeze(0)
        with torch.no_grad():
            out = net.forward(x).reshape(-1)
        rows = [1, out.numel() - 2]
        for mode in ("ratio", "frozen"):
            mat = KnowledgeMatrixComputer(net, batch_size=128, mixer_mode=mode).forward(x)
            self.assertLess(gap(mat.sum(1), out), 1e-9)
            got = KnowledgeRowComputer(net, mixer_mode=mode).forward(x, rows)
            self.assertLess(gap(got, mat[rows]), 1e-9)

    def test_llama_single_kv_head_and_no_grouping(self):
        for kv in (1, 6):
            self._parity(f"llama {kv} kv head(s)", _variant("llama", num_key_value_heads=kv))

    def test_llama_linear_rope_scaling(self):
        hf, net, ids, outputs = self._parity(
            "llama linear RoPE scaling",
            _variant("llama", rope_scaling={"rope_type": "linear", "factor": 4.0}))
        self.assertEqual(net.layers[2].rope_scaling, {"rope_type": "linear", "factor": 4.0})
        self._control("llama: linear scaling dropped", net, ids, outputs, _no_rescaling)

    def test_llama_ignores_a_stray_sliding_window(self):
        """LlamaForCausalLM has no sliding window: an entry of that name
        in a Llama config.json does nothing (the loader used to window
        every block with it)."""
        hf, net, ids, outputs = self._parity(
            "llama + stray sliding_window", _variant("llama", sliding_window=4))
        self.assertEqual(hf.config.sliding_window, 4)
        self.assertEqual(self._windows(net), [None] * TINY_LAYERS)
        self._control("llama: windowed anyway", net, ids, outputs, _window(4))

    def test_swish_is_silu(self):
        self._parity("llama hidden_act=swish", _variant("llama", hidden_act="swish"))

    # -- Mistral --

    def test_mistral_windows_every_block(self):
        hf, net, ids, outputs = self._parity(
            "mistral sliding_window", _variant("mistral", sliding_window=8, rope_theta=10000.0))
        self.assertEqual(self._windows(net), [8] * TINY_LAYERS)
        self.assertEqual(net.layers[2].rope_theta, 10000.0)
        self._control("mistral: window 9 instead of 8", net, ids, outputs, _window(9))

    def test_mistral_window_extent(self):
        """A window of W keeps the last W keys, the query's own position
        included -- pinned on the mask of the reference itself."""
        from transformers.masking_utils import create_sliding_window_causal_mask
        hf = build_reference(_variant("mistral", sliding_window=3), seed=22)
        hf.config._attn_implementation = "eager"
        allowed = create_sliding_window_causal_mask(
            config=hf.config, inputs_embeds=torch.zeros(1, 7, 48),
            attention_mask=None, past_key_values=None)[0, 0] == 0
        ours = ~LlamaLike.from_huggingface(hf).layers[2]._mask(7, "cpu")
        self.assertTrue(torch.equal(ours, allowed))
        self.assertEqual(allowed[6].nonzero().flatten().tolist(), [4, 5, 6])

    def test_mistral_without_window(self):
        hf, net, ids, outputs = self._parity(
            "mistral sliding_window=None", _variant("mistral", sliding_window=None))
        self.assertEqual(self._windows(net), [None] * TINY_LAYERS)

    LAYER_TYPES = ["full_attention", "sliding_attention", "full_attention"]

    def test_mistral_class_ignores_layer_types(self):
        """MistralForCausalLM windows every block whatever `layer_types`
        says (it only logs a warning)."""
        raw = _variant("mistral", sliding_window=8, layer_types=self.LAYER_TYPES)
        del raw["model_type"]
        config = transformers.MistralConfig(head_dim=8, **raw)
        self.assertEqual(config.layer_types, self.LAYER_TYPES)
        hf, net, ids, outputs = self._parity("MistralConfig + layer_types", config, disk=False)
        self.assertEqual(type(hf).__name__, "MistralForCausalLM")
        self.assertEqual(self._windows(net), [8] * TINY_LAYERS)
        # Saved and loaded back through the Auto classes, the SAME files
        # are a Ministral model -- another function. The loader follows
        # transformers in both cases: the class in memory, Auto on disk.
        with tempfile.TemporaryDirectory() as d:
            hf.save_pretrained(d)
            auto = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float64).eval()
            self.assertEqual(type(auto).__name__, "MinistralForCausalLM")
            with torch.no_grad():
                reloaded = auto(ids).logits
            self.assertGreater(gap(reloaded, outputs["logits"]), WIDE)
            from_disk = LlamaLike.from_huggingface(d, dtype=torch.float64)
            self.assertEqual(self._windows(from_disk), [None, 8, None])
            self.assertLess(gap(run(from_disk, ids), reloaded), HF_TOL)

        def per_layer_type(net):
            layers = attention_layers(net)
            layers[0].sliding_window = layers[2].sliding_window = None
        self._control("mistral: layer_types obeyed", net, ids, outputs, per_layer_type)

    def test_mistral_checkpoint_with_layer_types_is_ministral(self):
        """A config.json of model_type "mistral" WITH layer_types is
        resolved to Ministral by transformers' Auto classes, whose blocks
        are windowed per layer type -- the Ministral-8B layout."""
        raw = _variant("mistral", sliding_window=8, head_dim=8, layer_types=self.LAYER_TYPES)
        hf, net, ids, outputs = self._parity("mistral config.json + layer_types", raw)
        self.assertEqual(type(hf).__name__, "MinistralForCausalLM")
        self.assertEqual(hf.config.model_type, "ministral")
        self.assertEqual(self._windows(net), [None, 8, None])
        attn = net.layers[2]
        for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
            self.assertIsNone(proj.bias)
        self._control("ministral: every block windowed", net, ids, outputs, _window(8))

    def test_ministral_default_windows_every_block(self):
        hf, net, ids, outputs = self._parity(
            "ministral without layer_types",
            _variant("ministral", sliding_window=8, head_dim=8))
        self.assertEqual(self._windows(net), [8] * TINY_LAYERS)

    def test_mistral_nemo_head_dim(self):
        """Mistral-Nemo: head_dim (128) is not hidden / heads (5120 / 32)."""
        hf, net, ids, outputs = self._parity(
            "mistral head_dim != hidden / heads",
            _variant("mistral", sliding_window=None, hidden_size=40, num_attention_heads=4,
                     num_key_value_heads=2, head_dim=8))
        self.assertEqual(net.layers[2].d_head, 8)
        self.assertEqual(tuple(net.layers[2].q_proj.weight.shape), (32, 40))

    # -- Qwen2.5 --

    def test_qwen2_biases(self):
        """Biases on q, k, v -- not on o, not in the MLP."""
        hf, net, ids, outputs = self._parity("qwen2", _variant("qwen2"))
        attn = net.layers[2]
        for proj in (attn.q_proj, attn.k_proj, attn.v_proj):
            self.assertIsNotNone(proj.bias)
        self.assertIsNone(attn.o_proj.bias)
        self.assertIsNone(net.layers[5].gate_proj.bias)
        self.assertFalse(attn.use_qk_norm)
        self.assertEqual(self._windows(net), [None] * TINY_LAYERS)

        def swap(net):
            with torch.no_grad():
                a = net.layers[2]
                tmp = a.k_proj.bias.clone()
                a.k_proj.bias.copy_(a.v_proj.bias)
                a.v_proj.bias.copy_(tmp)
        self._control("qwen2: k and v biases swapped", net, ids, outputs, swap)

    def test_qwen2_tied(self):
        self._parity("qwen2 tied", _variant("qwen2", tie_word_embeddings=True))

    def test_qwen2_sliding_layers(self):
        """use_sliding_window windows the blocks from max_window_layers on."""
        hf, net, ids, outputs = self._parity(
            "qwen2 sliding layers",
            _variant("qwen2", use_sliding_window=True, sliding_window=8, max_window_layers=1))
        self.assertEqual(hf.config.layer_types,
                         ["full_attention", "sliding_attention", "sliding_attention"])
        self.assertEqual(self._windows(net), [None, 8, 8])
        self._control("qwen2: every block windowed", net, ids, outputs, _window(8))

    def test_qwen2_window_switched_off(self):
        """The published Qwen2.5 configs: a sliding_window value with
        use_sliding_window=false -- no block is windowed."""
        hf, net, ids, outputs = self._parity(
            "qwen2 use_sliding_window=False",
            _variant("qwen2", use_sliding_window=False, sliding_window=8, max_window_layers=1))
        self.assertEqual(self._windows(net), [None] * TINY_LAYERS)

    # -- Qwen3 --

    def test_qwen3_small_sibling_head_dim(self):
        """Qwen3-0.6B: head_dim 128 with hidden 1024 and 16 heads, tied."""
        hf, net, ids, outputs = self._parity(
            "qwen3 head_dim != hidden / heads, tied",
            _variant("qwen3", hidden_size=40, num_attention_heads=8, num_key_value_heads=4,
                     head_dim=8, rms_norm_eps=1e-6, tie_word_embeddings=True))
        attn = net.layers[2]
        self.assertTrue(attn.use_qk_norm)
        self.assertEqual(tuple(attn.q_proj.weight.shape), (64, 40))
        self.assertEqual(tuple(attn.q_norm.weight.shape), (8,))
        self.assertEqual(attn.q_norm.eps, 1e-6)

    def test_qwen3_qk_norm_epsilon(self):
        """The q/k norms use the config's rms_norm_eps. With q and k of
        ordinary size an epsilon of 1e-6 or 1e-5 is the same number to
        seven digits; shrink the projections and it is not."""
        raw = _variant("qwen3", head_dim=8, rms_norm_eps=1e-6)
        hf = build_reference(raw, seed=23)
        with torch.no_grad():
            for layer in hf.model.layers:
                layer.self_attn.q_proj.weight.mul_(2e-3)
                layer.self_attn.k_proj.weight.mul_(2e-3)
        ids = token_ids(seed=23)
        outputs = reference_outputs(hf, ids)
        net = LlamaLike.from_huggingface(hf, dtype=torch.float64)
        self.assertLess(gap(run(net, ids), outputs["logits"]), HF_TOL)
        self.assertLess(gap(run(net, ids), outputs["exact_logits"]), EXACT_TOL)

        def other_eps(net):
            for layer in attention_layers(net):
                layer.q_norm.eps = layer.k_norm.eps = 1e-5
        self._control("qwen3: q/k norm eps 1e-5", net, ids, outputs, other_eps)

    def test_qwen3_attention_bias(self):
        hf, net, ids, outputs = self._parity(
            "qwen3 attention_bias", _variant("qwen3", head_dim=8, attention_bias=True))
        attn = net.layers[2]
        for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
            self.assertIsNotNone(proj.bias)

    def test_qwen3_sliding_layers(self):
        hf, net, ids, outputs = self._parity(
            "qwen3 layer_types",
            _variant("qwen3", head_dim=8, use_sliding_window=True, sliding_window=8,
                     layer_types=["sliding_attention", "full_attention", "sliding_attention"]))
        self.assertEqual(self._windows(net), [8, None, 8])


@unittest.skipUnless(HAS_TRANSFORMERS, "transformers / safetensors are not installed")
class TestRefusals(unittest.TestCase):
    """
        What the network does not reproduce must raise -- before any
        weight is read -- instead of loading into a network that computes
        something else.
    """

    def test_supported_model_types(self):
        self.assertEqual(SUPPORTED_MODEL_TYPES,
                         ("llama", "mistral", "ministral", "qwen2", "qwen3"))

    def test_lookalike_architectures(self):
        """Same tensor names, another function. Each of these used to load
        without an error (measured logit gap against its own reference:
        0.6 to 4.8)."""
        seen = 0
        for model_type in ("diffllama", "granite", "doge", "gemma2", "smollm3", "olmo2"):
            if model_type not in transformers.CONFIG_MAPPING:
                continue
            with self.subTest(model_type=model_type):
                config = transformers.CONFIG_MAPPING[model_type](
                    vocab_size=TINY_VOCAB, hidden_size=32, intermediate_size=64,
                    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                    pad_token_id=0, bos_token_id=1, eos_token_id=2)
                with self.assertRaisesRegex(NotImplementedError, model_type):
                    llama_architecture(config)
                try:
                    hf = AutoModelForCausalLM.from_config(config)
                except Exception:        # these sizes do not suit that class
                    continue
                with self.assertRaisesRegex(NotImplementedError, model_type):
                    LlamaLike.from_huggingface(hf)
                seen += 1
        self.assertGreaterEqual(seen, 3)

    def _refused(self, raw, error, pattern):
        config = as_config(raw)
        with self.assertRaisesRegex(error, pattern):
            llama_architecture(config)
        return config

    def test_bidirectional_attention(self):
        """is_causal=False switches the reference to bidirectional masks."""
        raw = _variant("llama", is_causal=False)
        self._refused(raw, NotImplementedError, "is_causal")
        hf = build_reference(raw)
        with self.assertRaisesRegex(NotImplementedError, "is_causal"):
            LlamaLike.from_huggingface(hf)
        # ... and it does: the causal network is far from that reference
        ids = token_ids()
        causal = LlamaLike.from_huggingface(build_reference(_variant("llama")), dtype=torch.float64)
        with torch.no_grad():
            self.assertGreater(gap(run(causal, ids), hf(ids).logits), WIDE)

    def test_activation(self):
        for act in ("gelu", "relu", "gelu_pytorch_tanh"):
            self._refused(_variant("llama", hidden_act=act), NotImplementedError, act)

    def test_rope_variants(self):
        for scaling in ({"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 1024},
                        {"rope_type": "dynamic", "factor": 4.0}):
            self._refused(_variant("llama", rope_scaling=scaling), NotImplementedError,
                          scaling["rope_type"])

    def test_partial_rotation(self):
        raw = _variant("llama")
        del raw["rope_theta"]
        raw["rope_parameters"] = {"rope_type": "default", "rope_theta": 10000.0,
                                  "partial_rotary_factor": 0.5}
        self._refused(raw, NotImplementedError, "partial_rotary_factor")

    def test_sliding_layers_without_a_window(self):
        """layer_types asks for a window that use_sliding_window=False
        removed: transformers cannot run this model either."""
        raw = _variant("qwen3", head_dim=8, use_sliding_window=False, sliding_window=8,
                       layer_types=["full_attention", "sliding_attention", "full_attention"])
        self._refused(raw, ValueError, "sliding_window")

    def test_unknown_layer_type(self):
        for model_type in ("qwen2", "qwen3"):
            raw = _variant(model_type, use_sliding_window=True, sliding_window=8,
                           layer_types=["full_attention", "chunked_attention", "full_attention"])
            self._refused(raw, NotImplementedError, "chunked_attention")

    def test_too_few_layer_types(self):
        # (transformers validates the length when the config is created,
        # not when it is edited afterwards)
        config = as_config(_variant("qwen3", head_dim=8))
        config.layer_types = ["full_attention"]
        with self.assertRaisesRegex(ValueError, "layer_types"):
            llama_architecture(config)

    def test_config_without_layer_types(self):
        """A Qwen config object of a transformers version that predates
        `layer_types`: full attention when no window is asked for, a
        refusal when one is (those versions windowed in some attention
        backends only)."""
        class Old:
            model_type = "qwen2"
            vocab_size, hidden_size, intermediate_size = 97, 48, 112
            num_hidden_layers, num_attention_heads, num_key_value_heads = 3, 6, 2
            hidden_act, rms_norm_eps, rope_theta = "silu", 1e-6, 1000000.0
            use_sliding_window, sliding_window, max_window_layers = False, 8, 1
        arch = llama_architecture(Old())
        self.assertEqual(arch["sliding_window"], [None, None, None])
        self.assertEqual(arch["rope_theta"], 1000000.0)
        self.assertTrue(arch["qkv_bias"])
        Old.use_sliding_window = True
        with self.assertRaisesRegex(NotImplementedError, "layer_types"):
            llama_architecture(Old())

    def test_rope_settings_do_not_fall_back(self):
        """rope_settings has no default base: a config it cannot read
        raises (the original bug was a silent 10000)."""
        class Bare:
            pass
        with self.assertRaises(ValueError):
            rope_settings(Bare())
        old = Bare()                     # the attributes of transformers 4
        old.rope_theta = 500000.0
        old.rope_scaling = {"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0,
                            "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}
        new = as_config(fixture("meta-llama/Llama-3.2-1B"))
        self.assertEqual(rope_settings(old), rope_settings(new))

    def test_not_a_transformers_model(self):
        with self.assertRaises(ValueError):
            LlamaLike.from_huggingface(torch.nn.Linear(3, 3))


### 8. Real weights ###

SHORT_TEXT = "The capital of France is"

#: public domain (A. Lincoln, 1863; T. Jefferson et al., 1776)
LONG_TEXT = (
    "Four score and seven years ago our fathers brought forth on this continent, a new "
    "nation, conceived in Liberty, and dedicated to the proposition that all men are "
    "created equal. Now we are engaged in a great civil war, testing whether that nation, "
    "or any nation so conceived and so dedicated, can long endure. We are met on a great "
    "battle-field of that war. We have come to dedicate a portion of that field, as a "
    "final resting place for those who here gave their lives that that nation might live. "
    "It is altogether fitting and proper that we should do this. But, in a larger sense, "
    "we can not dedicate -- we can not consecrate -- we can not hallow -- this ground. The "
    "brave men, living and dead, who struggled here, have consecrated it, far above our "
    "poor power to add or detract. The world will little note, nor long remember what we "
    "say here, but it can never forget what they did here. It is for us the living, "
    "rather, to be dedicated here to the unfinished work which they who fought here have "
    "thus far so nobly advanced. It is rather for us to be here dedicated to the great "
    "task remaining before us -- that from these honored dead we take increased devotion "
    "to that cause for which they gave the last full measure of devotion -- that we here "
    "highly resolve that these dead shall not have died in vain -- that this nation, under "
    "God, shall have a new birth of freedom -- and that government of the people, by the "
    "people, for the people, shall not perish from the earth.\n\n"
    "When in the Course of human events, it becomes necessary for one people to dissolve "
    "the political bands which have connected them with another, and to assume among the "
    "powers of the earth, the separate and equal station to which the Laws of Nature and "
    "of Nature's God entitle them, a decent respect to the opinions of mankind requires "
    "that they should declare the causes which impel them to the separation. We hold these "
    "truths to be self-evident, that all men are created equal, that they are endowed by "
    "their Creator with certain unalienable Rights, that among these are Life, Liberty and "
    "the pursuit of Happiness. That to secure these rights, Governments are instituted "
    "among Men, deriving their just powers from the consent of the governed, That whenever "
    "any Form of Government becomes destructive of these ends, it is the Right of the "
    "People to alter or to abolish it, and to institute new Government, laying its "
    "foundation on such principles and organizing its powers in such form, as to them "
    "shall seem most likely to effect their Safety and Happiness."
)

#: float32 against float32 on real weights: the two implementations round
#: differently, and the difference grows with depth and sequence length
#: (measured values: KM_REPORT=1). A wiring error is orders above -- see
#: the controls.
REAL_TOL = 2e-4


def _cached(repo: str, *filenames: str) -> bool:
    try:
        return all(isinstance(huggingface_hub.try_to_load_from_cache(repo, f), str)
                   for f in filenames)
    except Exception:
        return False


@unittest.skipUnless(HAS_TRANSFORMERS, "transformers / safetensors are not installed")
@unittest.skipIf(os.environ.get("KM_SKIP_REAL_WEIGHTS"), "KM_SKIP_REAL_WEIGHTS is set")
class TestRealWeights(unittest.TestCase):
    """
        Our float32 network, streamed from the safetensors file in the
        local cache, against the transformers model on real text. One
        model in memory at a time: the reference outputs are computed and
        kept, the reference is freed, then ours is loaded.
    """

    def _require(self, repo):
        if not _cached(repo, "config.json", "model.safetensors", "tokenizer.json"):
            self.skipTest(f"{repo} is not in the local Hugging Face cache")

    def _ids(self, repo, text):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(repo, local_files_only=True)
        return tokenizer(text, return_tensors="pt").input_ids

    def _reference(self, repo, prompts):
        hf = AutoModelForCausalLM.from_pretrained(repo, local_files_only=True,
                                                  dtype=torch.float32).eval()
        self.assertEqual({p.dtype for p in hf.parameters()}, {torch.float32})
        outputs = {}
        with torch.no_grad():
            for label, ids in prompts.items():
                hidden = hf.model(ids).last_hidden_state
                logits = hf.lm_head(hidden)
                if ids.shape[1] < 32:      # the head applied here is the model's own forward
                    self.assertTrue(torch.equal(hf(ids).logits, logits))
                outputs[label] = (hidden.clone(), logits.clone())
        tied = hf.lm_head.weight is hf.model.embed_tokens.weight
        count = sum(p.numel() for p in hf.parameters())
        del hf
        gc.collect()
        return outputs, tied, count

    def _check(self, repo, wrong, with_head=True):
        """
            wrong: {label: mutation of the attention layers} -- negative
            controls on the real weights, evaluated on the long prompt
            (each one applied to the correct network, then undone).
        """
        self._require(repo)
        prompts = {"short": self._ids(repo, SHORT_TEXT), "long": self._ids(repo, LONG_TEXT)}
        self.assertGreater(prompts["long"].shape[1], 300)
        reference, tied, count = self._reference(repo, prompts)

        net = LlamaLike.from_huggingface(repo, include_lm_head=with_head, local_files_only=True)
        self.assertEqual({p.dtype for p in net.parameters()}, {torch.float32})
        embedding = net.layers[0].weight.numel()
        self.assertEqual(sum(p.numel() for p in net.parameters()),
                         count + (embedding if tied and with_head else 0)
                         - (embedding if not tied and not with_head else 0))

        def logits_of(ids):
            got = run(net, ids)
            if with_head:
                return None, got
            # the tied head is the embedding table: apply it here
            self.assertTrue(tied)
            return got, F.linear(got, net.layers[0].weight)

        for label, ids in prompts.items():
            hidden, logits = reference[label]
            T = ids.shape[1]
            got_hidden, got = logits_of(ids)
            if got_hidden is not None:
                off = gap(got_hidden, hidden) / hidden.abs().max().item()
                note("real weights", f"{repo} {label} (T={T}) hidden states, relative", off,
                     hidden.abs().max().item())
                self.assertLess(off, REAL_TOL)
            scale = logits.abs().max().item()
            off = gap(got, logits) / scale
            note("real weights", f"{repo} {label} (T={T}) logits, relative", off, scale)
            self.assertLess(off, REAL_TOL, f"{repo} {label}: logits differ by {off} of their scale")
            same = (got.argmax(-1) == logits[0].argmax(-1)).double().mean().item()
            note("real weights", f"{repo} {label} (T={T}) top-1 agreement", same)
            self.assertGreater(same, 0.99)

        ids = prompts["long"]
        logits = reference["long"][1][0]
        scale = logits.abs().max().item()
        baseline = gap(logits_of(ids)[1], logits) / scale
        layers = attention_layers(net)
        for label, mutate in wrong.items():
            saved = [(layer.rope_theta, layer.rope_scaling, layer.use_qk_norm) for layer in layers]
            mutate(net)
            per_position = (logits_of(ids)[1] - logits).abs().amax(-1) / scale
            for layer, (theta, scaling, qk) in zip(layers, saved):
                layer.rope_theta, layer.rope_scaling, layer.use_qk_norm = theta, scaling, qk
            off = per_position.max().item()
            note("real weights control", f"{repo} {label}: logits, relative", off, scale)
            for level in (1e-3, 1e-2, 1e-1):
                above = (per_position > level).nonzero()
                note("real weights control", f"{repo} {label}: first position off by > {level:g}",
                     float(above[0]) if len(above) else float("nan"))
            self.assertGreater(off, 100 * REAL_TOL, f"{repo}: {label} is not visible")
            self.assertGreater(off, 100 * baseline)
        self.assertLess(gap(logits_of(ids)[1], logits) / scale, REAL_TOL)    # undone
        del net
        gc.collect()

    def test_smollm2_135m(self):
        """Tied head, 9 query heads on 3 key/value heads, RoPE base 1e5."""
        self._check("HuggingFaceTB/SmolLM2-135M", {"RoPE base 10000": _set_theta(10000.0)})

    def test_smollm2_135m_knowledge_matrix(self):
        """The knowledge matrix of the final hidden states of a real
        checkpoint (float64, so that the identities are sharp)."""
        repo = "HuggingFaceTB/SmolLM2-135M"
        self._require(repo)
        net = LlamaLike.from_huggingface(repo, include_lm_head=False,
                                         local_files_only=True, dtype=torch.float64)
        net.eval()
        D = net.hidden_size
        x = self._ids(repo, SHORT_TEXT)[:, :2].unsqueeze(0)             # two tokens
        with torch.no_grad():
            out = net.forward(x).reshape(-1)
        rows = [3, D + 10, 2 * D - 1]
        for mode in ("frozen", "ratio"):
            mat = KnowledgeMatrixComputer(net, batch_size=2 * D, mixer_mode=mode).forward(x)
            self.assertEqual(tuple(mat.shape), (2 * D, 2 * D + 1))
            sums = gap(mat.sum(1), out)
            note("real weights", f"{repo} knowledge matrix {mode} (float64, T=2): row sums", sums,
                 out.abs().max().item())
            self.assertLess(sums, 1e-8 * out.abs().max().item())
            got = KnowledgeRowComputer(net, mixer_mode=mode).forward(x, rows)
            same = gap(got, mat[rows])
            note("real weights", f"{repo} knowledge matrix {mode} (float64, T=2): rows", same)
            self.assertLess(same, 1e-8 * mat[rows].abs().max().item())
        del net
        gc.collect()

    def test_qwen3_0_6b(self):
        """Same architecture as Qwen3-8B / 14B (q/k norm, RoPE base 1e6),
        with head_dim 128 on a hidden size of 1024."""
        self._check("Qwen/Qwen3-0.6B", {"RoPE base 10000": _set_theta(10000.0),
                                        "no q/k norm": _no_qk_norm})

    def test_llama_3_2_1b(self):
        """The real llama3 RoPE (base 500000, factor 32). Loaded without
        the head (the tied head would be a second copy of the 1 GB
        embedding table); the head is applied here."""
        def original_bug(net):
            _no_rescaling(net)
            _set_theta(10000.0)(net)
        self._check("unsloth/Llama-3.2-1B",
                    {"no llama3 rescaling": _no_rescaling,
                     "llama3 factor 8 instead of 32": _other_factor,
                     "RoPE base 10000, no rescaling (the original bug)": original_bug},
                    with_head=False)


if __name__ == "__main__":
    unittest.main()
