#!/usr/bin/env python
"""
    Tests for the checkpoint-loading foundation shared by the language
    model loaders (knowledgematrix/models/_hf.py) and for the pieces of
    neural_net.py it relies on:

    - `rope_inverse_frequencies`: the RoPE frequency tables, including the
      "linear" and "llama3" rescalings, against transformers for the
      published configs of every supported checkpoint,
    - `rope_settings`: reading the RoPE of a config written by
      transformers 4 or normalized by transformers 5,
    - `Checkpoint`: locating the text decoder and the LM head in the key
      layouts the supported checkpoints actually use,
    - `empty_network` / `WeightLoader`: building without storage, shape
      checks, completeness, casting, folding transforms, copy semantics,
      placement on a device pipeline, and the dry run,
    - `NN.residual(same_shape=True)` and `NN.shard_layout`.

    The model families have their own files (llama_family.py, gemma.py,
    qwen3_5.py); this one only needs a hand-built network, plus
    transformers for the frequency-table comparisons.
"""
import glob
import json
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from knowledgematrix.neural_net import (
    NN,
    RopeAttention,
    SUPPORTED_ROPE_TYPES,
    rope_inverse_frequencies,
)
from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.models._hf import (
    Checkpoint,
    WeightLoader,
    empty_network,
    open_checkpoint,
    rope_settings,
)

try:
    import transformers
    from transformers import AutoConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    transformers.logging.set_verbosity_error()
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import safetensors.torch
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

torch.set_default_dtype(torch.float64)

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "hf_configs")

LLAMA3 = {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}


def fixture_config(path):
    """The published config.json at `path`, normalized by transformers."""
    folder = tempfile.mkdtemp()
    try:
        shutil.copy(path, os.path.join(folder, "config.json"))
        return AutoConfig.from_pretrained(folder)
    finally:
        shutil.rmtree(folder, ignore_errors=True)


class Block(NN):
    """
        A small pre-norm residual network with a token embedding: enough
        structure (embedding, norms, attention, residuals, head) to
        exercise the loader without any transformers model.
    """
    def __init__(self, vocab=11, width=8, blocks=2, same_shape=True):
        super().__init__(input_shape=(1, 1, width))
        self.embedding(vocab, width)
        for _ in range(blocks):
            start = self.get_num_layers()
            self.rmsnorm(width)
            self.ropeattention(d_model=width, num_heads=2, head_dim=4)
            end = self.get_num_layers()
            self.identity()
            self.residual(start, end, same_shape=same_shape)
        self.rmsnorm(width)
        self.linear(width, vocab, bias=False)


class DictCheckpoint(Checkpoint):
    """A checkpoint backed by a plain dict of tensors."""
    def __init__(self, tensors, in_memory=False):
        super().__init__(config=SimpleNamespace(), in_memory=in_memory)
        self.tensors = tensors

    def keys(self):
        return self.tensors.keys()

    def __contains__(self, key):
        return key in self.tensors

    def tensor(self, key):
        return self.tensors[key].detach()


def reference_and_checkpoint(dtype=torch.float64, in_memory=False):
    """A randomly initialized Block and a checkpoint holding its weights."""
    torch.manual_seed(0)
    reference = Block()
    with torch.no_grad():
        for p in reference.parameters():
            if p.dim() == 1:        # the norm weights are initialized to exactly 1
                p.add_(0.1 * torch.randn_like(p))
    reference = reference.to(dtype)
    tensors = {name: p.detach().clone() for name, p in reference.named_parameters()}
    return reference, DictCheckpoint(tensors, in_memory=in_memory)


def fill(loader, checkpoint, skip=()):
    """Assign every parameter of a Block from `checkpoint` (keys = parameter names)."""
    for key in checkpoint.keys():
        if key in skip:
            continue
        _, index, attr = key.split(".", 2)      # "layers.<i>.<attr>"
        loader.put(int(index), attr, key)
    return loader


class TestRopeFrequencies(unittest.TestCase):
    def test_default_table(self):
        got = rope_inverse_frequencies(8, 10000.0)
        want = torch.tensor([1.0, 0.1, 0.01, 0.001])
        self.assertLess((got - want).abs().max().item(), 1e-15)

    def test_linear_divides_by_the_factor(self):
        plain = rope_inverse_frequencies(16, 1e6)
        scaled = rope_inverse_frequencies(16, 1e6, {"rope_type": "linear", "factor": 8.0})
        self.assertLess((scaled - plain / 8.0).abs().max().item(), 1e-18)

    def test_llama3_bands(self):
        # short wavelengths are kept, long ones slowed by the factor, and
        # the band in between interpolates monotonically between the two
        dim, theta = 128, 500000.0
        plain = rope_inverse_frequencies(dim, theta)
        scaled = rope_inverse_frequencies(dim, theta, LLAMA3)
        wavelen = 2 * torch.pi / plain
        high = wavelen < 8192 / 4.0
        low = wavelen > 8192 / 1.0
        mid = ~high & ~low
        self.assertTrue(high.any() and low.any() and mid.any())
        self.assertTrue(torch.equal(scaled[high], plain[high]))
        self.assertLess((scaled[low] - plain[low] / 8.0).abs().max().item(), 1e-18)
        ratio = scaled[mid] / plain[mid]
        self.assertTrue(((ratio > 1 / 8.0) & (ratio < 1.0)).all())
        self.assertTrue((ratio[1:] <= ratio[:-1]).all())

    def test_default_spellings_are_the_plain_table(self):
        plain = rope_inverse_frequencies(16, 1e4)
        for scaling in (None, {}, {"rope_type": "default"}, {"rope_type": None}, {"type": "default"}):
            self.assertTrue(torch.equal(rope_inverse_frequencies(16, 1e4, scaling), plain))

    def test_unsupported_types_are_refused(self):
        for rope_type in ("yarn", "dynamic", "longrope", "proportional"):
            with self.assertRaises(NotImplementedError):
                rope_inverse_frequencies(16, 1e4, {"rope_type": rope_type, "factor": 2.0})
            with self.assertRaises(NotImplementedError):
                RopeAttention(d_model=16, num_heads=4, rope_scaling={"rope_type": rope_type, "factor": 2.0})

    def test_incomplete_scaling_is_refused(self):
        with self.assertRaises(ValueError):
            rope_inverse_frequencies(16, 1e4, {"rope_type": "linear"})
        with self.assertRaises(ValueError):
            rope_inverse_frequencies(16, 1e4, {"rope_type": "llama3", "factor": 8.0})

    def test_attention_uses_the_rescaled_table(self):
        # same weights, same input: only the RoPE table differs, and the
        # difference has to show at positions the rescaling affects
        torch.manual_seed(0)
        plain = RopeAttention(d_model=16, num_heads=2, head_dim=8, rope_theta=5e5)
        scaled = RopeAttention(d_model=16, num_heads=2, head_dim=8, rope_theta=5e5,
                               rope_scaling=dict(LLAMA3, original_max_position_embeddings=16))
        scaled.load_state_dict(plain.state_dict())
        x = torch.randn(1, 1, 40, 16)
        self.assertGreater((plain(x) - scaled(x)).abs().max().item(), 1e-4)
        # and the frozen linearization follows the same table exactly
        self.assertEqual((scaled.frozen_forward(x, x) - scaled(x)).abs().max().item(), 0.0)

    @unittest.skipUnless(HAS_TRANSFORMERS, "requires the optional dependency transformers")
    def test_tables_match_transformers_for_every_supported_checkpoint(self):
        """
            For each published config, the float32 table equals the one
            transformers builds BIT FOR BIT, and the float64 table agrees
            with it to float32 rounding. Run on the layer types of the
            configs that have one RoPE per layer type (Gemma-3).
        """
        paths = sorted(glob.glob(os.path.join(FIXTURES, "*.json")))
        self.assertEqual(len(paths), 19)
        checked = 0
        for path in paths:
            config = fixture_config(path)
            text = getattr(config, "text_config", None) or config
            params = text.rope_parameters
            layer_types = [k for k, v in params.items() if isinstance(v, dict)] or [None]
            for layer_type in layer_types:
                with self.subTest(config=os.path.basename(path), layer_type=layer_type):
                    own = params[layer_type] if layer_type else params
                    settings = rope_settings(text, layer_type)
                    head_dim = getattr(text, "head_dim", None) or text.hidden_size // text.num_attention_heads
                    dim = int(head_dim * own.get("partial_rotary_factor", 1.0))
                    rope_type = own.get("rope_type", "default")
                    if rope_type == "default":
                        want = 1.0 / (own["rope_theta"] ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
                    else:
                        kwargs = {"layer_type": layer_type} if layer_type else {}
                        want, _ = ROPE_INIT_FUNCTIONS[rope_type](text, None, **kwargs)
                    self.assertEqual(settings["theta"], float(own["rope_theta"]))
                    got32 = rope_inverse_frequencies(dim, settings["theta"], settings["scaling"], dtype=torch.float32)
                    self.assertTrue(torch.equal(got32, want.to(torch.float32)))
                    got64 = rope_inverse_frequencies(dim, settings["theta"], settings["scaling"], dtype=torch.float64)
                    rel = ((got64 - want.double()).abs() / want.double()).max().item()
                    self.assertLess(rel, 1e-6)
                    checked += 1
        self.assertGreaterEqual(checked, 19 + 5)   # 5 Gemma-3 configs have two tables


class TestRopeSettings(unittest.TestCase):
    def test_transformers5_flat(self):
        config = SimpleNamespace(rope_parameters={"rope_theta": 500000.0, **LLAMA3})
        got = rope_settings(config)
        self.assertEqual(got["theta"], 500000.0)
        self.assertEqual(got["scaling"], {k: (v if k == "rope_type" else float(v)) for k, v in LLAMA3.items()})
        self.assertEqual(got["partial_rotary_factor"], 1.0)

    def test_transformers5_default_type_has_no_scaling(self):
        config = SimpleNamespace(rope_parameters={"rope_theta": 1000000, "rope_type": "default"})
        self.assertEqual(rope_settings(config), {"theta": 1e6, "scaling": None, "partial_rotary_factor": 1.0})

    def test_transformers5_partial_rotary(self):
        config = SimpleNamespace(rope_parameters={
            "rope_theta": 10000000, "rope_type": "default", "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10], "mrope_interleaved": True,
        })
        got = rope_settings(config)
        self.assertEqual((got["theta"], got["partial_rotary_factor"], got["scaling"]), (1e7, 0.25, None))

    def test_transformers5_per_layer_type(self):
        config = SimpleNamespace(rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "linear", "factor": 8.0, "rope_theta": 1000000.0},
        })
        local = rope_settings(config, "sliding_attention")
        glob_ = rope_settings(config, "full_attention")
        self.assertEqual((local["theta"], local["scaling"]), (1e4, None))
        self.assertEqual((glob_["theta"], glob_["scaling"]), (1e6, {"rope_type": "linear", "factor": 8.0}))
        with self.assertRaises(ValueError):
            rope_settings(config)                 # ambiguous without a layer type
        with self.assertRaises(KeyError):
            rope_settings(config, "chunked_attention")

    def test_transformers4_attributes(self):
        config = SimpleNamespace(rope_theta=500000.0, rope_scaling=dict(LLAMA3))
        got = rope_settings(config)
        self.assertEqual(got["theta"], 500000.0)
        self.assertEqual(got["scaling"]["rope_type"], "llama3")
        self.assertEqual(got["scaling"]["factor"], 8.0)
        plain = rope_settings(SimpleNamespace(rope_theta=100000, rope_scaling=None))
        self.assertEqual((plain["theta"], plain["scaling"]), (1e5, None))

    def test_transformers4_legacy_type_key(self):
        config = SimpleNamespace(rope_theta=1e6, rope_scaling={"type": "linear", "factor": 4.0})
        self.assertEqual(rope_settings(config)["scaling"], {"rope_type": "linear", "factor": 4.0})

    def test_transformers4_gemma3_local_base(self):
        config = SimpleNamespace(rope_theta=1e6, rope_local_base_freq=1e4,
                                 rope_scaling={"rope_type": "linear", "factor": 8.0})
        local = rope_settings(config, "sliding_attention")
        glob_ = rope_settings(config, "full_attention")
        self.assertEqual((local["theta"], local["scaling"]), (1e4, None))
        self.assertEqual((glob_["theta"], glob_["scaling"]), (1e6, {"rope_type": "linear", "factor": 8.0}))

    def test_missing_base_is_an_error_not_a_default(self):
        # the bug this module exists to prevent: a config without the
        # expected attribute must never silently become theta = 10000
        with self.assertRaises(ValueError):
            rope_settings(SimpleNamespace())
        with self.assertRaises(ValueError):
            rope_settings(SimpleNamespace(rope_parameters={"rope_type": "default"}))

    def test_unsupported_type_is_refused(self):
        self.assertEqual(SUPPORTED_ROPE_TYPES, ("default", "linear", "llama3"))
        with self.assertRaises(NotImplementedError):
            rope_settings(SimpleNamespace(rope_parameters={"rope_theta": 1e4, "rope_type": "yarn", "factor": 2.0}))

    @unittest.skipUnless(HAS_TRANSFORMERS, "requires the optional dependency transformers")
    def test_published_bases(self):
        want = {
            "meta-llama__Llama-3.3-70B-Instruct": (5e5, "llama3", 8.0),
            "meta-llama__Llama-3.1-8B": (5e5, "llama3", 8.0),
            "meta-llama__Llama-3.2-1B": (5e5, "llama3", 32.0),
            "meta-llama__Llama-3.2-3B": (5e5, "llama3", 32.0),
            "HuggingFaceTB__SmolLM2-135M": (1e5, None, None),
            "HuggingFaceTB__SmolLM2-360M": (1e5, None, None),
            "HuggingFaceTB__SmolLM2-1.7B": (1.3e5, None, None),
            "Qwen__Qwen3-8B": (1e6, None, None),
            "Qwen__Qwen3-14B": (1e6, None, None),
            "Qwen__Qwen3.5-4B": (1e7, None, None),
            "Qwen__Qwen3.5-27B": (1e7, None, None),
            "google__gemma-2-2b": (1e4, None, None),
        }
        for name, (theta, rope_type, factor) in want.items():
            with self.subTest(config=name):
                config = fixture_config(os.path.join(FIXTURES, name + ".json"))
                got = rope_settings(getattr(config, "text_config", None) or config)
                self.assertEqual(got["theta"], theta)
                if rope_type is None:
                    self.assertIsNone(got["scaling"])
                else:
                    self.assertEqual((got["scaling"]["rope_type"], got["scaling"]["factor"]), (rope_type, factor))


class TestCheckpointLayout(unittest.TestCase):
    """Where the text decoder and the LM head are, per key layout."""

    @staticmethod
    def _keys(prefix, head=None, extra=()):
        t = torch.zeros(1)
        keys = {prefix + "embed_tokens.weight": t, prefix + "layers.0.mlp.up_proj.weight": t,
                prefix + "norm.weight": t}
        if head:
            keys[head] = t
        for key in extra:
            keys[key] = t
        return DictCheckpoint(keys)

    def test_causal_lm(self):
        ckpt = self._keys("model.", head="lm_head.weight")
        self.assertEqual(ckpt.text_prefix, "model.")
        self.assertEqual(ckpt.lm_head_key, "lm_head.weight")
        self.assertTrue(ckpt.has_text("norm.weight"))
        self.assertFalse(ckpt.has_text("layers.0.self_attn.q_norm.weight"))

    def test_tied_head_falls_back_to_the_embedding(self):
        ckpt = self._keys("model.")
        self.assertEqual(ckpt.lm_head_key, "model.embed_tokens.weight")

    def test_multimodal_wrapper(self):
        # transformers 5 layout (Qwen3.5, Gemma-3 4b/12b/27b), with a
        # vision tower and Qwen3.5's multi-token-prediction head alongside
        extra = ("model.visual.blocks.0.attn.qkv.weight", "model.visual.patch_embed.proj.weight",
                 "mtp.layers.0.mlp.up_proj.weight", "mtp.norm.weight")
        ckpt = self._keys("model.language_model.", head="lm_head.weight", extra=extra)
        self.assertEqual(ckpt.text_prefix, "model.language_model.")
        self.assertEqual(ckpt.lm_head_key, "lm_head.weight")
        tied = self._keys("model.language_model.", extra=extra)
        self.assertEqual(tied.lm_head_key, "model.language_model.embed_tokens.weight")

    def test_older_multimodal_layout(self):
        ckpt = self._keys("language_model.model.", head="language_model.lm_head.weight",
                          extra=("vision_tower.vision_model.embeddings.patch_embedding.weight",))
        self.assertEqual(ckpt.text_prefix, "language_model.model.")
        self.assertEqual(ckpt.lm_head_key, "language_model.lm_head.weight")

    def test_bare_text_model(self):
        ckpt = self._keys("")
        self.assertEqual(ckpt.text_prefix, "")
        self.assertEqual(ckpt.lm_head_key, "embed_tokens.weight")

    def test_no_text_decoder_is_an_error(self):
        with self.assertRaises(ValueError):
            DictCheckpoint({"encoder.weight": torch.zeros(1)}).text_prefix

    def test_two_text_decoders_are_an_error(self):
        t = torch.zeros(1)
        ckpt = DictCheckpoint({"a.embed_tokens.weight": t, "a.layers.0.w": t,
                               "b.embed_tokens.weight": t, "b.layers.0.w": t})
        with self.assertRaises(ValueError):
            ckpt.text_prefix

    def test_text_config_unwraps_a_multimodal_config(self):
        inner = SimpleNamespace(hidden_size=8)
        ckpt = DictCheckpoint({})
        ckpt.config = SimpleNamespace(text_config=inner)
        self.assertIs(ckpt.text_config, inner)
        ckpt.config = inner
        self.assertIs(ckpt.text_config, inner)


class TestEmptyNetwork(unittest.TestCase):
    def test_no_storage(self):
        net = empty_network(Block, vocab=50000, width=4096, blocks=4)
        params = list(net.parameters())
        self.assertGreater(sum(p.numel() for p in params), 4e8)
        self.assertTrue(all(p.is_meta for p in params))

    def test_wiring_is_the_same_as_a_real_build(self):
        real = Block()
        empty = empty_network(Block)
        self.assertEqual(empty.get_num_layers(), real.get_num_layers())
        self.assertEqual(empty.residuals_starts, real.residuals_starts)
        self.assertEqual(sorted(empty.residuals), sorted(real.residuals))
        self.assertEqual([tuple(p.shape) for p in empty.parameters()],
                         [tuple(p.shape) for p in real.parameters()])

    def test_same_shape_matches_the_probed_residual(self):
        torch.manual_seed(0)
        probed = Block(same_shape=False)
        fast = Block(same_shape=True)
        fast.load_state_dict(probed.state_dict())
        self.assertEqual(sorted(probed.residuals), sorted(fast.residuals))
        self.assertEqual(len(probed.residual_modules), 0)
        x = torch.randint(0, 11, (1, 1, 5))
        self.assertEqual((probed.forward(x) - fast.forward(x)).abs().max().item(), 0.0)


class TestWeightLoader(unittest.TestCase):
    def test_loads_the_same_function(self):
        reference, ckpt = reference_and_checkpoint()
        net = fill(WeightLoader(empty_network(Block), ckpt, dtype=torch.float64), ckpt).finish()
        self.assertFalse(any(p.is_meta for p in net.parameters()))
        x = torch.randint(0, 11, (1, 1, 6))
        self.assertEqual((net.forward(x) - reference.forward(x)).abs().max().item(), 0.0)
        # and the result is a working knowledge-matrix network
        mat = KnowledgeMatrixComputer(net, batch_size=16).forward(x)
        self.assertLess(torch.norm(net.forward(x).reshape(1, -1) - mat.sum(1)).item(), 1e-10)

    def test_unassigned_parameter_is_refused(self):
        _, ckpt = reference_and_checkpoint()
        loader = fill(WeightLoader(empty_network(Block), ckpt), ckpt, skip=("layers.2.q_proj.weight",))
        with self.assertRaises(RuntimeError) as ctx:
            loader.finish()
        self.assertIn("layers.2.q_proj.weight", str(ctx.exception))

    def test_shape_mismatch_is_refused(self):
        _, ckpt = reference_and_checkpoint()
        ckpt.tensors["layers.2.q_proj.weight"] = torch.zeros(8, 7)
        loader = WeightLoader(empty_network(Block), ckpt)
        with self.assertRaises(ValueError) as ctx:
            loader.put(2, "q_proj.weight", "layers.2.q_proj.weight")
        self.assertIn("(8, 7)", str(ctx.exception))

    def test_missing_key_and_missing_parameter_are_refused(self):
        _, ckpt = reference_and_checkpoint()
        loader = WeightLoader(empty_network(Block), ckpt)
        with self.assertRaises(KeyError):
            loader.put(2, "q_proj.weight", "layers.2.q_proj.wieght")
        with self.assertRaises(AttributeError):
            loader.put(2, "q_proj.bias", "layers.2.q_proj.weight")    # built without biases

    def test_dtype_defaults(self):
        # from disk: float32 unless told otherwise; in memory: as stored
        _, on_disk = reference_and_checkpoint(dtype=torch.bfloat16, in_memory=False)
        net = fill(WeightLoader(empty_network(Block), on_disk), on_disk).finish()
        self.assertEqual({p.dtype for p in net.parameters()}, {torch.float32})
        _, live = reference_and_checkpoint(dtype=torch.bfloat16, in_memory=True)
        net = fill(WeightLoader(empty_network(Block), live), live).finish()
        self.assertEqual({p.dtype for p in net.parameters()}, {torch.bfloat16})
        net = fill(WeightLoader(empty_network(Block), live, dtype=torch.float64), live).finish()
        self.assertEqual({p.dtype for p in net.parameters()}, {torch.float64})

    def test_copy_semantics(self):
        _, live = reference_and_checkpoint(in_memory=True)
        key = "layers.2.q_proj.weight"
        copied = fill(WeightLoader(empty_network(Block), live, copy=True), live).finish()
        shared = fill(WeightLoader(empty_network(Block), live, copy=False), live).finish()
        source = live.tensors[key]
        self.assertNotEqual(copied.layers[2].q_proj.weight.data_ptr(), source.data_ptr())
        self.assertEqual(shared.layers[2].q_proj.weight.data_ptr(), source.data_ptr())
        # a cast always makes a new tensor, whatever `copy` says
        cast = fill(WeightLoader(empty_network(Block), live, dtype=torch.float32, copy=False), live).finish()
        self.assertNotEqual(cast.layers[2].q_proj.weight.data_ptr(), source.data_ptr())

    def test_transform_is_applied(self):
        reference, ckpt = reference_and_checkpoint()
        loader = WeightLoader(empty_network(Block), ckpt, dtype=torch.float64)
        for key in ckpt.keys():
            _, index, attr = key.split(".", 2)
            transform = (lambda w: w + 1.0) if key == "layers.1.weight" else None
            loader.put(int(index), attr, key, transform=transform)
        net = loader.finish()
        gap = (net.layers[1].weight - (reference.layers[1].weight + 1.0)).abs().max().item()
        self.assertEqual(gap, 0.0)
        # the checkpoint tensor itself is left alone
        self.assertTrue(torch.equal(ckpt.tensors["layers.1.weight"], reference.layers[1].weight))

    def test_transform_runs_above_half_precision(self):
        # folding 1 + w must be done before rounding to bfloat16: the
        # folded weight is the correctly rounded value of 1 + w, not
        # 1 + (w rounded), which can differ in the last bit
        torch.manual_seed(0)
        w = (torch.randn(8) * 0.05).to(torch.float32)
        ckpt = DictCheckpoint({"layers.1.weight": w.to(torch.bfloat16)})
        loader = WeightLoader(empty_network(Block), ckpt, dtype=torch.bfloat16)
        seen = {}

        def fold(t):
            seen["dtype"] = t.dtype
            return t + 1.0
        loader.put(1, "weight", "layers.1.weight", transform=fold)
        self.assertEqual(seen["dtype"], torch.float32)
        got = loader.net.layers[1].weight
        self.assertEqual(got.dtype, torch.bfloat16)
        want = (w.to(torch.bfloat16).to(torch.float32) + 1.0).to(torch.bfloat16)
        self.assertTrue(torch.equal(got, want))

    def test_requires_grad_is_preserved(self):
        _, ckpt = reference_and_checkpoint()
        net = fill(WeightLoader(empty_network(Block), ckpt), ckpt).finish()
        self.assertTrue(all(p.requires_grad for p in net.parameters()))

    def test_pipeline_placement_matches_shard(self):
        reference, ckpt = reference_and_checkpoint()
        want_devices, want_cuts = Block().shard_layout(["cpu", "cpu"])
        net = fill(WeightLoader(empty_network(Block), ckpt, dtype=torch.float64,
                                device=["cpu", "cpu"]), ckpt).finish()
        self.assertEqual(net.layer_devices, want_devices)
        self.assertEqual(net.shard_boundaries, want_cuts)
        x = torch.randint(0, 11, (1, 1, 6))
        self.assertEqual((net.forward(x) - reference.forward(x)).abs().max().item(), 0.0)
        # explicit cut points are honoured
        cut = reference.shard_cut_points()[0]
        net = fill(WeightLoader(empty_network(Block), ckpt, dtype=torch.float64,
                                device=["cpu", "cpu"], boundaries=[cut]), ckpt).finish()
        self.assertEqual(net.shard_boundaries, [cut])

    def test_boundaries_need_a_pipeline(self):
        _, ckpt = reference_and_checkpoint()
        with self.assertRaises(ValueError):
            WeightLoader(empty_network(Block), ckpt, device="cpu", boundaries=[3])

    def test_dry_run_stores_nothing_and_still_checks(self):
        _, ckpt = reference_and_checkpoint()
        net = fill(WeightLoader(empty_network(Block), ckpt, device="meta"), ckpt).finish()
        self.assertTrue(all(p.is_meta for p in net.parameters()))
        # completeness is still enforced ...
        loader = fill(WeightLoader(empty_network(Block), ckpt, device="meta"), ckpt, skip=("layers.0.weight",))
        with self.assertRaises(RuntimeError):
            loader.finish()
        # ... and so are the shapes
        ckpt.tensors["layers.0.weight"] = torch.zeros(3, 3)
        with self.assertRaises(ValueError):
            WeightLoader(empty_network(Block), ckpt, device="meta").put(0, "weight", "layers.0.weight")


class TestShardLayout(unittest.TestCase):
    def test_layout_is_what_shard_does(self):
        net = Block(blocks=4)
        layer_devices, cuts = net.shard_layout(["cpu", "cpu", "cpu"])
        net.shard(["cpu", "cpu", "cpu"])
        self.assertEqual(net.layer_devices, layer_devices)
        self.assertEqual(net.shard_boundaries, cuts)
        self.assertEqual(len(layer_devices), net.get_num_layers())

    def test_layout_needs_no_storage(self):
        real = Block(blocks=4)
        empty = empty_network(Block, blocks=4)
        self.assertEqual(empty.shard_layout(["a", "b"]), real.shard_layout(["a", "b"]))

    def test_layout_rejects_a_cut_through_a_residual(self):
        net = Block(blocks=2)
        inside = min(net.residuals_starts) + 1          # between a start and its end
        self.assertNotIn(inside, net.shard_cut_points())
        with self.assertRaises(ValueError):
            net.shard_layout(["cpu", "cpu"], boundaries=[inside])
        with self.assertRaises(ValueError):
            net.shard_layout(["cpu"])


@unittest.skipUnless(HAS_SAFETENSORS and HAS_TRANSFORMERS, "requires safetensors and transformers")
class TestOpenCheckpoint(unittest.TestCase):
    """A directory on disk, read lazily: one file, and several with an index."""

    def setUp(self):
        self.folder = tempfile.mkdtemp()
        config = {"model_type": "llama", "hidden_size": 8, "intermediate_size": 16,
                  "num_hidden_layers": 1, "num_attention_heads": 2, "vocab_size": 11}
        with open(os.path.join(self.folder, "config.json"), "w") as f:
            json.dump(config, f)
        torch.manual_seed(0)
        self.tensors = {
            "model.embed_tokens.weight": torch.randn(11, 8, dtype=torch.float32),
            "model.layers.0.mlp.up_proj.weight": torch.randn(16, 8, dtype=torch.float32),
            "model.norm.weight": torch.randn(8, dtype=torch.float32),
            "lm_head.weight": torch.randn(11, 8, dtype=torch.float32),
        }

    def tearDown(self):
        shutil.rmtree(self.folder, ignore_errors=True)

    def _check(self, ckpt):
        self.assertFalse(ckpt.in_memory)
        self.assertEqual(ckpt.config.model_type, "llama")
        self.assertEqual(sorted(ckpt.keys()), sorted(self.tensors))
        self.assertEqual(ckpt.text_prefix, "model.")
        self.assertEqual(ckpt.lm_head_key, "lm_head.weight")
        for key, want in self.tensors.items():
            self.assertTrue(torch.equal(ckpt.tensor(key), want))
        ckpt.close()

    def test_single_file(self):
        safetensors.torch.save_file(self.tensors, os.path.join(self.folder, "model.safetensors"))
        self._check(open_checkpoint(self.folder))

    def test_sharded_files_with_an_index(self):
        names = sorted(self.tensors)
        parts = {"model-00001-of-00002.safetensors": names[:2], "model-00002-of-00002.safetensors": names[2:]}
        weight_map = {}
        for filename, keys in parts.items():
            safetensors.torch.save_file({k: self.tensors[k] for k in keys}, os.path.join(self.folder, filename))
            weight_map.update({k: filename for k in keys})
        with open(os.path.join(self.folder, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {}, "weight_map": weight_map}, f)
        self._check(open_checkpoint(self.folder))

    def test_a_listed_file_that_is_missing_is_an_error(self):
        with open(os.path.join(self.folder, "model.safetensors.index.json"), "w") as f:
            json.dump({"weight_map": {"model.norm.weight": "model-00001-of-00001.safetensors"}}, f)
        with self.assertRaises(FileNotFoundError):
            open_checkpoint(self.folder)

    def test_no_weights_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            open_checkpoint(self.folder)

    def test_a_module_without_config_is_refused(self):
        with self.assertRaises(ValueError):
            open_checkpoint(nn.Linear(2, 2))


if __name__ == "__main__":
    unittest.main()
