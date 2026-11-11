#!/usr/bin/env python
"""
    Tests for the model split over several devices (`NN.shard`):

    - `shard_cut_points` never cuts through a residual span, and the
      default boundaries fall on admissible points;
    - the forward pass of a sharded LlamaLike / Qwen3_5 equals the
      unsharded one exactly (same arithmetic, only the tensors move);
    - KnowledgeRowComputer rows of the sharded model equal the unsharded
      rows, in frozen and ratio mode, and keep the row-sum identity;
    - `unshard` restores the single-device network;
    - the FULL matrix of a sharded network (KnowledgeMatrixComputer.forward
      and .segment, GradientMatrixComputer) equals the unsharded one, the
      network is never moved, and the passes follow the device map;
    - the same for a network a checkpoint loader built already split
      (`from_huggingface(..., device=[...])`), with the bias column alone
      and the rows with the independent bias.

    The split is exercised over ["cpu", "cpu"] (the bookkeeping) and, when
    the machine has one, over ["cpu", "mps"] or ["cuda:0", "cuda:1"] (a real
    cross-device transfer inside the autograd graph). MPS has no float64,
    so the cross-device case runs in float32 with a looser tolerance.
"""
import collections
import unittest

import torch
from torch import nn

from knowledgematrix.neural_net import NN, SoftCap
from knowledgematrix.models.llama import LlamaLike
from knowledgematrix.models.gemma import GemmaLike
from knowledgematrix.models.gpt2 import GPT2
from knowledgematrix.models.gpt_neox import GPTNeoX
from knowledgematrix.models.qwen3_5 import Qwen3_5
from knowledgematrix.row_computer import KnowledgeRowComputer
from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.gradient_matrix import GradientMatrixComputer

LLAMA = dict(vocab_size=50, hidden_size=24, intermediate_size=48,
             num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
             head_dim=6)
QWEN = dict(vocab_size=50, hidden_size=24, intermediate_size=48,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, linear_num_value_heads=4, linear_num_key_heads=2,
            linear_key_head_dim=6, linear_value_head_dim=6)
# Gemma-2 with its logit soft-cap: the network ENDS in a SoftCap layer, which
# has no parameter (the cap is small so that it bites on the tiny logits of
# a randomly initialized model).
GEMMA2 = dict(vocab_size=50, hidden_size=24, intermediate_size=48,
              num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
              head_dim=6, attn_softcap=30.0, sliding_window=4, final_softcap=0.25)


def second_device():
    """A device different from the CPU, when the machine has one."""
    if torch.cuda.device_count() >= 2:
        return ["cuda:0", "cuda:1"]
    if torch.cuda.is_available():
        return ["cpu", "cuda:0"]
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return ["cpu", "mps"]
    return None


def build(cls, params, dtype, seed=0):
    torch.manual_seed(seed)
    # (other test modules set float64 as the default dtype; build in the
    # requested one explicitly so the MPS case stays float32)
    with torch.device("cpu"):
        net = cls(**params, include_lm_head=True, device="cpu").to(dtype)
    net.eval()
    return net


def rows_of(net, x, rows, mode):
    comp = KnowledgeRowComputer(net, mixer_mode=mode)
    return comp.forward(x, rows).detach().cpu()


def assert_close(test, got, want, tol, what):
    """
        tol == 0: bitwise equality (one physical device: only the
        bookkeeping differs). Otherwise equality up to `tol` relative to the
        scale of the entries -- two devices round float32 differently, and a
        ratio-mode matrix has entries in the thousands (a gap of 2.4e-4 at
        an entry of 1070 is two units in its last place).
    """
    got, want = got.detach().cpu(), want.detach().cpu()
    gap = (got - want).abs().max().item()
    if tol == 0:
        test.assertTrue(torch.equal(got, want), f"{what} differs: {gap}")
    else:
        scale = max(1.0, want.abs().max().item())
        test.assertLessEqual(gap, tol * scale, f"{what} differs: {gap} (scale {scale})")


def assert_placed(test, net, devices):
    """
        Every parameter and buffer of every layer, and of the residual
        projections applied at a layer, sits on the device `layer_devices`
        says. Layers without storage (Identity, SoftCap, activations) have
        nothing to check, so the last layer of the network may be one.
    """
    test.assertEqual(len(net.layer_devices), net.get_num_layers())
    test.assertEqual(net.layer_devices[0], devices[0])
    test.assertEqual(net.layer_devices[-1], devices[-1])
    test.assertEqual(set(net.layer_devices), set(devices))
    for i, layer in enumerate(net.layers):
        want = torch.device(net.layer_devices[i]).type
        modules = [layer] + [m for _, proj in net.residuals.get(i, []) for m in proj]
        for module in modules:
            for t in list(module.parameters()) + list(module.buffers()):
                test.assertEqual(t.device.type, want, f"layer {i} ({type(layer).__name__})")


class TestCutPoints(unittest.TestCase):
    def test_no_cut_crosses_a_residual(self):
        for cls, params in ((LlamaLike, LLAMA), (Qwen3_5, QWEN)):
            net = build(cls, params, torch.float64)
            cuts = net.shard_cut_points()
            self.assertTrue(cuts)
            for c in cuts:
                for end, pairs in net.residuals.items():
                    for start, _ in pairs:
                        self.assertFalse(start < c <= end,
                                         f"cut {c} crosses residual {start}->{end}")
            # block boundaries (the start of each block) are admissible
            stride = getattr(cls, "BLOCK_STRIDE", None)
            if stride:
                self.assertIn(1 + stride, cuts)

    def test_default_boundaries_are_admissible_and_balanced(self):
        net = build(LlamaLike, LLAMA, torch.float64)
        counts = [sum(p.numel() for p in l.parameters()) for l in net.layers]
        net.shard(["cpu", "cpu", "cpu"])
        devs = net.layer_devices
        self.assertEqual(len(devs), net.get_num_layers())
        cuts = net.shard_boundaries
        self.assertEqual(len(cuts), 2)
        for c in cuts:
            self.assertIn(c, net.shard_cut_points())
        # each stage holds a non-trivial share of the parameters
        edges = [0] + cuts + [len(devs)]
        shares = [sum(counts[edges[k]:edges[k + 1]]) for k in range(3)]
        self.assertTrue(all(s > 0 for s in shares))

    def test_bad_boundaries_rejected(self):
        net = build(LlamaLike, LLAMA, torch.float64)
        bad = next(i for i in range(2, net.get_num_layers())
                   if i not in net.shard_cut_points())
        with self.assertRaises(ValueError):
            net.shard(["cpu", "cpu"], boundaries=[bad])
        with self.assertRaises(ValueError):
            net.shard(["cpu", "cpu"], boundaries=[])


class TestShardedEqualsUnsharded(unittest.TestCase):
    def _check(self, cls, params, devices, dtype, tol):
        torch.manual_seed(1)
        x = torch.randint(0, params["vocab_size"], (1, 1, 7))
        ref = build(cls, params, dtype)
        out_ref = ref(x).detach().clone()
        V = params["vocab_size"]
        rows = [6 * V + 3, 6 * V + 11, 2 * V + 0]
        ref_rows = {m: rows_of(ref, x, rows, m) for m in ("frozen", "ratio")}

        net = build(cls, params, dtype).shard(devices)
        self.assertEqual(net.device, devices[0])
        # every layer really is on its stage's device (the first stage on
        # devices[0], the last on devices[-1]); the last layer itself may
        # have no parameter at all
        assert_placed(self, net, devices)

        out = net(x.to(devices[0])).detach().cpu()
        self.assertTrue(torch.allclose(out, out_ref, atol=tol, rtol=0),
                        f"forward differs: {(out - out_ref).abs().max()}")
        for m in ("frozen", "ratio"):
            got = rows_of(net, x.to(devices[0]), rows, m)
            self.assertTrue(torch.allclose(got, ref_rows[m], atol=tol, rtol=0),
                            f"{m} rows differ: {(got - ref_rows[m]).abs().max()}")
            # row-sum identity against the sharded model's own forward
            want = out.reshape(-1)[rows]
            self.assertTrue(torch.allclose(got.sum(1), want, atol=tol, rtol=0))

        net.unshard("cpu")
        self.assertIsNone(net.layer_devices)
        self.assertIsNone(net.shard_boundaries)
        self.assertTrue(torch.allclose(net(x).detach(), out_ref, atol=tol, rtol=0))

    def test_llama_cpu_cpu(self):
        self._check(LlamaLike, LLAMA, ["cpu", "cpu"], torch.float64, 1e-12)

    def test_qwen_cpu_cpu_cpu(self):
        self._check(Qwen3_5, QWEN, ["cpu", "cpu", "cpu"], torch.float64, 1e-12)

    def test_llama_two_real_devices(self):
        devices = second_device()
        if devices is None:
            self.skipTest("one device only")
        dtype = torch.float32 if "mps" in devices else torch.float64
        tol = 2e-4 if dtype == torch.float32 else 1e-9
        self._check(LlamaLike, LLAMA, devices, dtype, tol)

    def test_qwen_two_real_devices(self):
        devices = second_device()
        if devices is None:
            self.skipTest("one device only")
        dtype = torch.float32 if "mps" in devices else torch.float64
        tol = 2e-4 if dtype == torch.float32 else 1e-9
        self._check(Qwen3_5, QWEN, devices, dtype, tol)

    def test_gemma2_ends_in_a_parameter_free_layer(self):
        net = build(GemmaLike, GEMMA2, torch.float64)
        self.assertIsInstance(net.layers[-1], SoftCap)
        self.assertEqual(len(list(net.layers[-1].parameters())), 0)
        # the cap does change the logits of this network
        x = torch.randint(0, GEMMA2["vocab_size"], (1, 1, 7))
        uncapped = build(GemmaLike, dict(GEMMA2, final_softcap=None), torch.float64)
        self.assertGreater((net(x) - uncapped(x)).abs().max().item(), 1e-3)

    def test_gemma2_with_cap_cpu_cpu(self):
        self._check(GemmaLike, GEMMA2, ["cpu", "cpu"], torch.float64, 1e-12)

    def test_gemma2_with_cap_cpu_cpu_cpu(self):
        self._check(GemmaLike, GEMMA2, ["cpu", "cpu", "cpu"], torch.float64, 1e-12)

    def test_gemma2_with_cap_two_real_devices(self):
        devices = second_device()
        if devices is None:
            self.skipTest("one device only")
        dtype = torch.float32 if "mps" in devices else torch.float64
        tol = 2e-4 if dtype == torch.float32 else 1e-9
        self._check(GemmaLike, GEMMA2, devices, dtype, tol)


### The full matrix of a sharded network ###

class Spy:
    """
        Watches a split network during a computation: counts how often each
        layer's device was requested (`to_layer_device`), and forbids moving
        the network (`.to`), which would silently collapse the split. Over
        ["cpu", "cpu"] this is the only way to see that a pass follows the
        device map -- nothing fails there if it does not.
    """
    def __init__(self, net):
        self.requests = collections.Counter()
        inner = net.to_layer_device

        def to_layer_device(x, i):
            self.requests[i] += 1
            return inner(x, i)

        def to(*args, **kwargs):
            raise AssertionError("a network split over devices must not be moved with .to()")

        net.to_layer_device = to_layer_device
        net.to = to


def build_lm(cls, params, dtype, seed=0):
    """A GPT-2 / GPT-NeoX style network (their constructors differ from
    the Llama-style ones `build` serves)."""
    torch.manual_seed(seed)
    with torch.device("cpu"):
        net = cls(**params, device="cpu").to(dtype)
    net.eval()
    return net


GPT = dict(vocab_size=50, d_model=16, d_ff=32, num_heads=4, num_layers=2, max_len=32)


class SmallResNet(NN):
    """conv / BatchNorm / ReLU with a projected residual, max-pooling and a
    linear head: piecewise linear, so GradientMatrixComputer applies."""
    def __init__(self):
        super().__init__((3, 8, 8))
        self.conv(3, 6, kernel_size=3, padding=1)
        self.batchnorm(6)
        self.relu()
        self.conv(6, 8, kernel_size=3, padding=1, stride=2)
        self.batchnorm(8)
        for layer in self.layers:
            layer.eval()                 # keep the shape probes of residual() off the statistics
        self.residual(0, 5)              # 1x1 conv + BatchNorm projection
        for module in self.residual_modules:
            module.eval()
        self.relu()
        self.maxpool(kernel_size=2, stride=2)
        self.flatten()
        self.linear(8 * 2 * 2, 16)
        self.relu()
        self.linear(16, 5)


def build_cnn(dtype, seed=0):
    torch.manual_seed(seed)
    with torch.device("cpu"):
        net = SmallResNet()
        with torch.no_grad():
            for module in net.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.running_mean.copy_(torch.randn_like(module.running_mean) * 0.3)
                    module.running_var.copy_(torch.rand_like(module.running_var) + 0.5)
                    module.weight.copy_(torch.rand_like(module.weight) + 0.5)
                    module.bias.copy_(torch.randn_like(module.bias) * 0.3)
        net = net.to(dtype)
    net.eval()
    return net


class TestShardedFullMatrix(unittest.TestCase):
    """KnowledgeMatrixComputer.forward / .segment on a network split with
    NN.shard: same matrix as the unsplit network, network left in place."""

    BATCH = 16

    def _language(self, builder, devices, dtype, tol, boundaries=None, modes=None, device=None):
        torch.manual_seed(1)
        x = torch.randint(0, 50, (1, 1, 5))
        # (the row sums are a float identity, never a bitwise one)
        row_tol = max(tol, 1e-10 if dtype == torch.float64 else 2e-4)
        modes = modes or (dict(mixer_mode="ratio"), dict(mixer_mode="frozen"),
                          dict(attention_mode="frozen_pattern"))
        for kw in modes:
            with self.subTest(devices=devices, **kw):
                ref = builder(dtype)
                ref_comp = KnowledgeMatrixComputer(ref, batch_size=self.BATCH, **kw)
                want = ref_comp.forward(x)
                want_weff = ref_comp.forward(x, extract_weff=True)
                out_ref = ref(x).detach().reshape(-1)

                net = builder(dtype).shard(devices, boundaries)
                assert_placed(self, net, devices)
                spy = Spy(net)
                comp = KnowledgeMatrixComputer(net, batch_size=self.BATCH, device=device, **kw)
                home = torch.device(device if device is not None else devices[0]).type
                got = comp.forward(x.to(devices[0]))
                self.assertEqual(got.device.type, home)          # the computer's device
                self.assertEqual(got.shape, want.shape)
                assert_close(self, got, want, tol, "matrix")
                # rows sum to the outputs of the split network's own forward pass
                out = comp.current_output.detach().reshape(-1)
                self.assertEqual(out.device.type, torch.device(devices[-1]).type)
                assert_close(self, got.sum(1), out, row_tol, "row sums")
                assert_close(self, out, out_ref, tol, "output")

                # the passes followed the device map: every layer's device was
                # requested by the reference pass, by each probe batch and by
                # the bias pass
                start = net._get_start_layer()
                batches = -(-want_weff.shape[1] // self.BATCH)
                for i in range(start, net.get_num_layers()):
                    self.assertGreaterEqual(spy.requests[i], batches + 2, f"layer {i}")
                # ... and the saved states stayed with their layer
                for i, saved in enumerate(net.pre_acts):
                    if saved is not None:
                        self.assertEqual(saved.device.type, torch.device(net.layer_devices[i]).type)
                assert_placed(self, net, devices)

                got_weff = comp.forward(x.to(devices[0]), extract_weff=True)
                self.assertEqual(got_weff.device.type, home)
                assert_close(self, got_weff, want_weff, tol, "W_eff")

                # segments: across the cut, and on either side of it
                n, cut = net.get_num_layers(), net.shard_boundaries[0]
                compared = 0
                for a, b, final in ((start, n, False), (start, cut, False), (cut, n, False), (cut, n, True)):
                    try:
                        want_seg = ref_comp.segment(x, a, b, final_position_only=final)
                    except ValueError:
                        continue    # not a valid pair of cuts for this network
                    got_seg = comp.segment(x.to(devices[0]), a, b, final_position_only=final)
                    self.assertEqual(got_seg.device.type, home)
                    assert_close(self, got_seg, want_seg, tol, f"segment ({a}, {b}, final={final})")
                    compared += 1
                self.assertEqual(compared, 4)     # none of them was skipped
                assert_placed(self, net, devices)

    # -- the bookkeeping, on one physical device (bitwise equal) --

    def test_llama_cpu_cpu(self):
        self._language(lambda dt: build(LlamaLike, LLAMA, dt), ["cpu", "cpu"], torch.float64, 0)

    def test_llama_three_stages(self):
        self._language(lambda dt: build(LlamaLike, LLAMA, dt), ["cpu", "cpu", "cpu"], torch.float64, 0)

    def test_qwen_cpu_cpu(self):
        self._language(lambda dt: build(Qwen3_5, QWEN, dt), ["cpu", "cpu"], torch.float64, 0)

    def test_gemma2_with_cap_cpu_cpu(self):
        self._language(lambda dt: build(GemmaLike, GEMMA2, dt), ["cpu", "cpu"], torch.float64, 0)

    def test_gpt2_cpu_cpu(self):
        # LayerNorm, MultiHeadAttention, chained residuals, two start layers
        self._language(lambda dt: build_lm(GPT2, dict(GPT, dropout=0.0), dt), ["cpu", "cpu"], torch.float64, 0)

    def test_gpt_neox_cpu_cpu(self):
        # parallel residual blocks
        self._language(lambda dt: build_lm(GPTNeoX, GPT, dt), ["cpu", "cpu"], torch.float64, 0)

    # -- two physical devices (a real transfer; float32 on MPS) --

    def _real(self, builder, reverse=False, **kw):
        devices = second_device()
        if devices is None:
            self.skipTest("one device only")
        if reverse:
            devices = devices[::-1]
        dtype = torch.float32 if "mps" in devices else torch.float64
        tol = 2e-4 if dtype == torch.float32 else 1e-9
        self._language(builder, devices, dtype, tol, **kw)

    def test_llama_two_real_devices(self):
        self._real(lambda dt: build(LlamaLike, LLAMA, dt))

    def test_llama_two_real_devices_reversed(self):
        # the probe batch starts on the accelerator and ends on the CPU
        self._real(lambda dt: build(LlamaLike, LLAMA, dt), reverse=True)

    def test_llama_two_real_devices_matrix_elsewhere(self):
        # the matrix is asked for on the LAST stage's device
        devices = second_device()
        if devices is None:
            self.skipTest("one device only")
        self._real(lambda dt: build(LlamaLike, LLAMA, dt), device=devices[-1])

    def test_qwen_two_real_devices(self):
        self._real(lambda dt: build(Qwen3_5, QWEN, dt))

    def test_gemma2_with_cap_two_real_devices(self):
        self._real(lambda dt: build(GemmaLike, GEMMA2, dt))

    def test_gpt2_two_real_devices(self):
        self._real(lambda dt: build_lm(GPT2, dict(GPT, dropout=0.0), dt))

    def test_gpt_neox_two_real_devices(self):
        self._real(lambda dt: build_lm(GPTNeoX, GPT, dt))


class TestShardedConvNet(unittest.TestCase):
    """A split convolutional network (projected residual, BatchNorm,
    max-pooling): KnowledgeMatrixComputer and GradientMatrixComputer."""

    def _check(self, devices, dtype, tol):
        torch.manual_seed(2)
        x = torch.rand(3, 8, 8, dtype=dtype)
        ref = build_cnn(dtype)
        want = KnowledgeMatrixComputer(ref, batch_size=50).forward(x)
        want_grad = {b: GradientMatrixComputer(ref, backend=b, batch_size=2).forward(x)
                     for b in ("func", "autograd")}
        out_ref = ref(x).detach().reshape(-1)
        row_tol = max(tol, 1e-10 if dtype == torch.float64 else 2e-4)

        def close(got, target, what, tolerance=tol):
            assert_close(self, got, target, tolerance, what)

        for cut in ref.shard_cut_points():
            with self.subTest(devices=devices, cut=cut):
                net = build_cnn(dtype).shard(devices, [cut])
                assert_placed(self, net, devices)
                spy = Spy(net)
                got = KnowledgeMatrixComputer(net, batch_size=50).forward(x.to(devices[0]))
                self.assertEqual(got.device.type, torch.device(devices[0]).type)
                close(got, want, "matrix")
                close(got.sum(1), out_ref, "row sums", row_tol)
                for i in range(net.get_num_layers()):
                    self.assertGreaterEqual(spy.requests[i], -(-x.numel() // 50) + 2)
                for backend in ("func", "autograd"):
                    gm = GradientMatrixComputer(net, backend=backend, batch_size=2)
                    got = gm.forward(x.to(devices[0]))
                    self.assertEqual(got.device.type, torch.device(devices[0]).type)
                    close(got, want_grad[backend], f"gradient matrix ({backend})")
                    close(got.sum(1), out_ref, f"gradient row sums ({backend})", row_tol)
                    weff = gm.forward(x.to(devices[0]), extract_weff=True)
                    self.assertEqual(weff.device.type, torch.device(devices[0]).type)
                    close(weff, want_grad[backend][:, :-1] / x.reshape(1, -1), f"W_eff ({backend})",
                          max(tol, 1e-9 if dtype == torch.float64 else 2e-3))
                assert_placed(self, net, devices)

    def test_cpu_cpu(self):
        self._check(["cpu", "cpu"], torch.float64, 0)

    def test_two_real_devices(self):
        devices = second_device()
        if devices is None:
            self.skipTest("one device only")
        dtype = torch.float32 if "mps" in devices else torch.float64
        self._check(devices, dtype, 2e-4 if dtype == torch.float32 else 1e-9)

    def test_two_real_devices_reversed(self):
        devices = second_device()
        if devices is None:
            self.skipTest("one device only")
        dtype = torch.float32 if "mps" in devices else torch.float64
        self._check(devices[::-1], dtype, 2e-4 if dtype == torch.float32 else 1e-9)


### Networks that come out of a checkpoint loader already split ###

try:
    from transformers.models.llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM
    from transformers.models.gemma2 import Gemma2Config
    from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

HF = dict(vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
          num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64)


@unittest.skipUnless(HAS_TRANSFORMERS, "transformers is not installed")
class TestLoaderShardedNetworks(unittest.TestCase):
    """
        `from_huggingface(..., device=[...])` places every tensor on the
        device of its layer while loading (models/_hf.py), without going
        through NN.shard. The matrix computers must treat such a network
        like one split afterwards.
    """

    def _references(self, dtype):
        torch.manual_seed(0)
        with torch.device("cpu"):
            # biases on the attention and MLP projections: a bias column
            # that is not zero in frozen mode
            llama = LlamaForCausalLM(LlamaConfig(**HF, attention_bias=True, mlp_bias=True)).to(dtype).eval()
            # Gemma-2 soft-caps its logits: the network ends in a SoftCap
            gemma = Gemma2ForCausalLM(Gemma2Config(**HF, sliding_window=4, attn_logit_softcapping=50.0,
                                                   final_logit_softcapping=0.5,
                                                   query_pre_attn_scalar=8)).to(dtype).eval()
        return (("llama", LlamaLike, llama), ("gemma-2", GemmaLike, gemma))

    def _check(self, devices, dtype, tol):
        torch.manual_seed(1)
        x = torch.randint(0, HF["vocab_size"], (1, 1, 5))
        rows = [3, 2 * HF["vocab_size"] + 1, 4 * HF["vocab_size"] + 7]
        for name, cls, hf in self._references(dtype):
            for mode in ("frozen", "ratio"):
                with self.subTest(model=name, devices=devices, mixer_mode=mode):
                    whole = cls.from_huggingface(hf, dtype=dtype)
                    self.assertIsNone(whole.layer_devices)
                    comp = KnowledgeMatrixComputer(whole, batch_size=32, mixer_mode=mode)
                    want = comp.forward(x)
                    want_bias = comp.bias_column(x)
                    want_rows = KnowledgeRowComputer(whole, mixer_mode=mode).forward(x, rows, independent_bias=True)

                    split = cls.from_huggingface(hf, dtype=dtype, device=devices)
                    assert_placed(self, split, devices)
                    if name == "gemma-2":
                        self.assertIsInstance(split.layers[-1], SoftCap)
                    Spy(split)
                    home = torch.device(devices[0]).type
                    comp = KnowledgeMatrixComputer(split, batch_size=32, mixer_mode=mode)
                    got = comp.forward(x.to(devices[0]))
                    self.assertEqual(got.device.type, home)
                    assert_close(self, got, want, tol, "matrix")
                    out = comp.current_output.detach().reshape(-1)
                    assert_close(self, got.sum(1), out, max(tol, 1e-10 if dtype == torch.float64 else 2e-4), "row sums")
                    assert_close(self, comp.bias_column(x.to(devices[0])), want_bias, tol, "bias column")
                    row_comp = KnowledgeRowComputer(split, mixer_mode=mode)
                    got_rows = row_comp.forward(x.to(devices[0]), rows, independent_bias=True)
                    assert_close(self, got_rows, want_rows, tol, "rows with the independent bias")
                    assert_close(self, got_rows.sum(1), row_comp.row_outputs,
                                 max(tol, 1e-10 if dtype == torch.float64 else 2e-4), "row sums of the rows")
                    assert_placed(self, split, devices)

    def test_cpu_cpu(self):
        self._check(["cpu", "cpu"], torch.float64, 0)

    def test_two_real_devices(self):
        devices = second_device()
        if devices is None:
            self.skipTest("one device only")
        dtype = torch.float32 if "mps" in devices else torch.float64
        self._check(devices, dtype, 2e-4 if dtype == torch.float32 else 1e-9)


if __name__ == "__main__":
    unittest.main()
