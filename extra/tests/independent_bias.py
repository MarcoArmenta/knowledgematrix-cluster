#!/usr/bin/env python
"""
    Tests for the independently computed bias column:

    - `KnowledgeMatrixComputer.bias_column(x)`: the bias column alone, by
      one affine pass from the zero input (no input column is computed);
    - `KnowledgeRowComputer.forward(x, rows, independent_bias=True)`: the
      rows `[A | bias]` with that bias instead of the remainder
      `output - A.sum(1)`, and `row_outputs`, the outputs to compare with.

    Why it matters. With the remainder, every row sums to the output BY
    CONSTRUCTION, whatever A is: the row-sum check of the row computer
    cannot fail. With the independent bias, `A.sum(1) + bias == output` is
    an identity between three separately computed things, and
    `TestTheCheckCanFail` shows that it does fail when a layer's linear map
    is deliberately broken -- while the remainder variant keeps passing on
    the very same broken computation.
"""
import inspect
import unittest
from unittest import mock

import torch
from torch import nn

from knowledgematrix.neural_net import NN, RMSNorm, SwiGLU
from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.row_computer import KnowledgeRowComputer
from knowledgematrix.models.llama import LlamaLike
from knowledgematrix.models.gemma import GemmaLike
from knowledgematrix.models.gpt2 import GPT2
from knowledgematrix.models.qwen3_5 import Qwen3_5
from knowledgematrix.models.transformer import Transformer

torch.set_default_dtype(torch.float64)

V, T = 50, 6
TOL = 1e-10          # float64: the identities hold to ~1e-13


def randomize(net, seed, scale=0.3):
    """Seeded random parameters (norm weights and biases included, so that
    no bias is zero by initialization)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * scale)
    net.eval()
    return net


def tokens(seed=0, length=T):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, V, (1, 1, length), generator=g)


LLAMA = dict(vocab_size=V, hidden_size=24, intermediate_size=48, num_hidden_layers=3,
             num_attention_heads=4, num_key_value_heads=2, head_dim=6)


def llama_with_biases(dtype=torch.float64):
    with torch.device("cpu"):
        net = LlamaLike(**LLAMA, qkv_bias=True, o_bias=True, mlp_bias=True, qk_norm=True)
        return randomize(net, 1).to(dtype)


def llama_plain(dtype=torch.float64):
    """No bias anywhere: the bias column of the frozen linearization is 0."""
    with torch.device("cpu"):
        return randomize(LlamaLike(**LLAMA), 2).to(dtype)


def gemma_with_dead_coordinates():
    """Norm scales of exactly 0 (trained Gemma-3 checkpoints have them): in
    ratio mode the output of the mixer behind such a norm is a constant of
    the linearization, i.e. a genuine bias entry in a bias-free network."""
    net = randomize(GemmaLike(vocab_size=V, hidden_size=24, intermediate_size=48, num_hidden_layers=2,
                              num_attention_heads=4, num_key_value_heads=2, head_dim=6,
                              attn_softcap=30.0, sliding_window=4, qk_norm=True, final_softcap=2.0), 3)
    stride = GemmaLike.BLOCK_STRIDE
    with torch.no_grad():
        net.layers[1 + 4].weight[5] = 0.0                 # pre-feedforward norm, block 0
        net.layers[1 + stride].weight[7] = 0.0            # input norm, block 1
        net.layers[1 + stride + 4].weight[[5, 11]] = 0.0
    return net


def qwen_with_biases():
    return randomize(Qwen3_5(vocab_size=V, hidden_size=24, intermediate_size=48, num_hidden_layers=4,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                             linear_num_value_heads=4, linear_num_key_heads=2,
                             linear_key_head_dim=6, linear_value_head_dim=6, attention_bias=True), 4)


def gpt2():
    return randomize(GPT2(vocab_size=V, d_model=16, d_ff=32, num_heads=4, num_layers=2,
                          max_len=32, dropout=0.0), 5)


def transformer():
    return randomize(Transformer(vocab_size=V, d_model=16, d_ff=32, num_heads=4, max_len=32), 6)


LANGUAGE_MODELS = (
    ("llama with biases", llama_with_biases),
    ("gemma with dead coordinates", gemma_with_dead_coordinates),
    ("qwen3.5 with biases", qwen_with_biases),
    ("gpt2", gpt2),
    ("transformer", transformer),
)
ROWS = [0, V + 3, 2 * V + 17, (T - 1) * V + 42]


def conv_net():
    """conv / BatchNorm / ReLU, a projected residual, max-pooling, a head."""
    net = NN(input_shape=(3, 8, 8))
    net.conv(3, 6, kernel_size=3, padding=1); net.batchnorm(6); net.relu()
    net.conv(6, 8, kernel_size=3, padding=1, stride=2); net.batchnorm(8)
    for layer in net.layers:
        layer.eval()
    net.residual(0, 5)
    for module in net.residual_modules:
        module.eval()
    net.relu(); net.maxpool(kernel_size=2, stride=2); net.flatten()
    net.linear(32, 16); net.sigmoid(); net.linear(16, 5)
    randomize(net, 7)
    g = torch.Generator().manual_seed(8)
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.copy_(torch.randn(m.running_mean.shape, generator=g) * 0.3)
                m.running_var.copy_(torch.rand(m.running_var.shape, generator=g) + 0.5)
    return net


class CountPasses:
    """Counts the passes of a computation: reference passes (NN.forward),
    and per-layer steps of the column pass and of the affine pass."""
    def __init__(self):
        self.forward = self.linear = self.affine = 0

    def __enter__(self):
        counter = self
        real_forward, real_linear, real_affine = (
            NN.forward, KnowledgeMatrixComputer._linear_step, KnowledgeMatrixComputer._affine_step)

        def forward(net, *a, **k):
            counter.forward += 1
            return real_forward(net, *a, **k)

        def linear(comp, *a, **k):
            counter.linear += 1
            return real_linear(comp, *a, **k)

        def affine(comp, *a, **k):
            counter.affine += 1
            return real_affine(comp, *a, **k)

        self._patches = [mock.patch.object(NN, "forward", forward),
                         mock.patch.object(KnowledgeMatrixComputer, "_linear_step", linear),
                         mock.patch.object(KnowledgeMatrixComputer, "_affine_step", affine)]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()


### 1. KnowledgeMatrixComputer.bias_column ###

class TestBiasColumn(unittest.TestCase):

    def test_is_the_last_column_of_the_matrix(self):
        x = tokens()
        for name, build in LANGUAGE_MODELS:
            for kw in (dict(mixer_mode="ratio"), dict(mixer_mode="frozen"),
                       dict(attention_mode="frozen_pattern")):
                with self.subTest(network=name, **kw):
                    net = build()
                    comp = KnowledgeMatrixComputer(net, batch_size=32, **kw)
                    mat = comp.forward(x)
                    bias = comp.bias_column(x)
                    self.assertEqual(bias.shape, (mat.shape[0],))
                    # the same code and arithmetic: bitwise the same numbers
                    self.assertTrue(torch.equal(bias, mat[:, -1]))
                    # with the input columns it reproduces the output
                    out = net(x).detach().reshape(-1)
                    self.assertLess((mat[:, :-1].sum(1) + bias - out).abs().max().item(), TOL)
                    self.assertTrue(torch.equal(comp.current_output.reshape(-1), out))

    def test_the_cases_are_not_vacuous(self):
        # every network above has a bias column that is far from zero in at
        # least one mode -- otherwise "equal to the last column" says little
        x = tokens()
        for name, build in LANGUAGE_MODELS:
            with self.subTest(network=name):
                net = build()
                size = max(KnowledgeMatrixComputer(net, mixer_mode=m).bias_column(x).abs().max().item()
                           for m in ("ratio", "frozen"))
                self.assertGreater(size, 1e-3)

    def test_the_modes_give_different_bias_columns(self):
        # the bias pass goes through the same linearization as the columns:
        # frozen routing puts the value / output biases of the mixers in the
        # bias column, the ratio does not
        x = tokens()
        net = llama_with_biases()
        ratio = KnowledgeMatrixComputer(net, mixer_mode="ratio").bias_column(x)
        frozen = KnowledgeMatrixComputer(net, mixer_mode="frozen").bias_column(x)
        self.assertGreater((ratio - frozen).abs().max().item(), 1e-3)
        # attention_mode="frozen_pattern" only freezes MultiHeadAttention
        net = gpt2()
        pattern = KnowledgeMatrixComputer(net, attention_mode="frozen_pattern").bias_column(x)
        frozen = KnowledgeMatrixComputer(net, mixer_mode="frozen").bias_column(x)
        ratio = KnowledgeMatrixComputer(net, mixer_mode="ratio").bias_column(x)
        self.assertTrue(torch.equal(pattern, frozen))
        self.assertGreater((ratio - frozen).abs().max().item(), 1e-3)

    def test_convolutional_network(self):
        net = conv_net()
        x = torch.rand(3, 8, 8, generator=torch.Generator().manual_seed(9))
        comp = KnowledgeMatrixComputer(net, batch_size=50)
        mat = comp.forward(x)
        bias = comp.bias_column(x)
        self.assertTrue(torch.equal(bias, mat[:, -1]))
        self.assertGreater(bias.abs().max().item(), 1e-3)
        self.assertLess((mat[:, :-1].sum(1) + bias - net(x).reshape(-1)).abs().max().item(), TOL)

    def test_a_network_without_constant_term(self):
        # no bias, no norm, no residual: the matrix has no bias column, and
        # the bias column asked for on its own is the zero it stands for
        net = NN(input_shape=(1, 5, 5))
        net.flatten(); net.linear(25, 12, bias=False); net.relu(); net.linear(12, 4, bias=False)
        randomize(net, 10)
        x = torch.randn(1, 5, 5, generator=torch.Generator().manual_seed(11))
        comp = KnowledgeMatrixComputer(net, batch_size=8)
        self.assertEqual(comp.forward(x).shape, (4, 25))
        bias = comp.bias_column(x)
        self.assertEqual(bias.shape, (4,))
        self.assertEqual(bias.abs().max().item(), 0.0)

    def test_it_costs_one_reference_pass_and_one_affine_pass(self):
        x = tokens()
        net = llama_with_biases()
        comp = KnowledgeMatrixComputer(net, batch_size=8, mixer_mode="frozen")
        layers = net.get_num_layers() - net._get_start_layer()
        with CountPasses() as count:
            comp.bias_column(x)
        self.assertEqual(count.forward, 1)
        self.assertEqual(count.affine, layers)      # every layer once
        self.assertEqual(count.linear, 0)           # no input column
        # ... against the full matrix: one affine pass and one column pass
        # per batch of T * hidden / batch_size = 18
        with CountPasses() as count:
            comp.forward(x)
        self.assertEqual((count.forward, count.affine, count.linear), (1, layers, 18 * layers))

    def test_the_network_is_left_out_of_save_mode(self):
        net = llama_plain()
        KnowledgeMatrixComputer(net).bias_column(tokens())
        self.assertFalse(net.save)


### 2. KnowledgeRowComputer.forward(..., independent_bias=True) ###

def row_gap(comp, net, x, rows, **kw):
    """max | sum of the row - output at that row |, the row-sum check."""
    got = comp.forward(x, rows, **kw)
    return (got.sum(1) - comp.row_outputs).abs().max().item()


class TestIndependentBiasRows(unittest.TestCase):

    def test_the_default_is_still_the_remainder(self):
        self.assertIs(inspect.signature(KnowledgeRowComputer.forward).parameters["independent_bias"].default,
                      False)
        x = tokens()
        net = llama_with_biases()
        comp = KnowledgeRowComputer(net, mixer_mode="frozen")
        default = comp.forward(x, ROWS)
        explicit = comp.forward(x, ROWS, independent_bias=False)
        self.assertTrue(torch.equal(default, explicit))
        out = net(x).detach().reshape(-1)[ROWS]
        self.assertTrue(torch.equal(default[:, -1], out - default[:, :-1].sum(1)))

    def test_rows_and_bias_are_those_of_the_matrix(self):
        x = tokens()
        for name, build in LANGUAGE_MODELS:
            for mode in ("frozen", "ratio"):
                with self.subTest(network=name, mixer_mode=mode):
                    net = build()
                    mat = KnowledgeMatrixComputer(net, batch_size=32, mixer_mode=mode).forward(x)
                    comp = KnowledgeRowComputer(net, mixer_mode=mode)
                    got = comp.forward(x, ROWS, independent_bias=True)
                    self.assertEqual(got.shape, (len(ROWS), mat.shape[1]))
                    self.assertLess((got - mat[ROWS]).abs().max().item(), TOL)
                    # the bias entries come from the very pass that builds the
                    # bias column of the matrix
                    self.assertTrue(torch.equal(got[:, -1], mat[ROWS, -1]))
                    # the input columns are those of the default call
                    self.assertTrue(torch.equal(got[:, :-1], comp.forward(x, ROWS)[:, :-1]))

    def test_the_row_sum_identity_holds(self):
        x = tokens()
        for name, build in LANGUAGE_MODELS:
            for mode in ("frozen", "ratio"):
                with self.subTest(network=name, mixer_mode=mode):
                    net = build()
                    comp = KnowledgeRowComputer(net, mixer_mode=mode)
                    self.assertLess(row_gap(comp, net, x, ROWS, independent_bias=True), TOL)
                    # row_outputs are the outputs of the network at the rows
                    self.assertTrue(torch.equal(comp.row_outputs, net(x).detach().reshape(-1)[ROWS]))

    def test_row_outputs_in_every_call(self):
        x = tokens()
        net = llama_plain()
        want = net(x).detach().reshape(-1)[ROWS]
        comp = KnowledgeRowComputer(net)
        self.assertIsNone(comp.row_outputs)
        for kw in (dict(), dict(extract_weff=True), dict(independent_bias=True)):
            comp.row_outputs = None
            comp.forward(x, ROWS, **kw)
            self.assertTrue(torch.equal(comp.row_outputs, want))

    def test_extract_weff_ignores_the_option(self):
        x = tokens()
        net = llama_with_biases()
        comp = KnowledgeRowComputer(net)
        self.assertTrue(torch.equal(comp.forward(x, ROWS, extract_weff=True),
                                    comp.forward(x, ROWS, extract_weff=True, independent_bias=True)))

    def test_it_adds_one_affine_pass_and_nothing_else(self):
        x = tokens()
        net = llama_with_biases()
        layers = net.get_num_layers() - net._get_start_layer()
        comp = KnowledgeRowComputer(net, mixer_mode="frozen")
        with CountPasses() as count:
            comp.forward(x, ROWS)
        self.assertEqual((count.forward, count.affine, count.linear), (1, 0, 0))
        with CountPasses() as count:
            comp.forward(x, ROWS, independent_bias=True)
        self.assertEqual((count.forward, count.affine, count.linear), (1, layers, 0))


### 3. The check can fail ###

class ScaledRowPass(KnowledgeRowComputer):
    """A row computer whose linear pass is wrong by a factor (its own
    per-layer code, the one thing the row-sum check is there to verify)."""
    factor = 1.01

    def _linear_forward(self, e):
        return super()._linear_forward(e) * self.factor


class NoRowPass(ScaledRowPass):
    """... and one whose rows are plainly zero."""
    factor = 0.0


class StaleNorm(KnowledgeRowComputer):
    """Linearizes one RMSNorm at the wrong point: its saved scale is off by
    10 % for the row pass AND for the bias pass that follows."""
    def _linear_forward(self, e):
        model = self.model
        index = next(i for i, layer in enumerate(model.layers) if isinstance(layer, RMSNorm))
        model.layernorms[index] = model.layernorms[index] * 1.1
        return super()._linear_forward(e)


class TestTheCheckCanFail(unittest.TestCase):
    """
        Each case breaks the linear map of the row computation and runs the
        row-sum check twice on the same broken computation: with the
        remainder bias (passes, by construction) and with the independent
        bias (fails). `healthy` is the same check on the intact computation.
    """
    # The independent check is off by more than this on every break below
    # (measured: 1e-3 to 2.8, the smallest being a 1 % error on outputs of
    # size 0.1), against ~1e-16 on the intact computation.
    BROKEN = 1e-4

    def _both(self, comp, net, x):
        return (row_gap(comp, net, x, ROWS),                          # remainder
                row_gap(comp, net, x, ROWS, independent_bias=True))   # independent

    def test_a_wrong_row_pass(self):
        x = tokens()
        for name, build in (("llama with biases", llama_with_biases), ("llama", llama_plain),
                            ("gemma with dead coordinates", gemma_with_dead_coordinates), ("gpt2", gpt2)):
            for mode in ("frozen", "ratio"):
                with self.subTest(network=name, mixer_mode=mode):
                    net = build()
                    healthy = self._both(KnowledgeRowComputer(net, mixer_mode=mode), net, x)
                    self.assertLess(max(healthy), TOL)
                    remainder, independent = self._both(ScaledRowPass(net, mixer_mode=mode), net, x)
                    self.assertLess(remainder, TOL)                   # cannot fail
                    self.assertGreater(independent, self.BROKEN)      # does fail

    def test_rows_that_are_all_zero(self):
        # the extreme case: no attribution at all, and the remainder variant
        # still "passes" -- its bias entry is simply the whole output
        x = tokens()
        net = llama_with_biases()
        comp = NoRowPass(net, mixer_mode="frozen")
        rows = comp.forward(x, ROWS)
        self.assertEqual(rows[:, :-1].abs().max().item(), 0.0)
        remainder, independent = self._both(comp, net, x)
        self.assertEqual(remainder, 0.0)
        self.assertGreater(independent, self.BROKEN)

    def test_a_layer_whose_linear_map_is_wrong_in_both_passes(self):
        # the frozen map of one SwiGLU block is 5 % off, for the columns and
        # for the bias pass alike (its forward pass is untouched)
        x = tokens()
        for name, build in (("llama with biases", llama_with_biases), ("llama", llama_plain)):
            with self.subTest(network=name):
                net = build()
                block = next(layer for layer in net.layers if isinstance(layer, SwiGLU))
                intact = block.frozen_forward
                comp = KnowledgeRowComputer(net, mixer_mode="frozen")
                self.assertLess(max(self._both(comp, net, x)), TOL)
                block.frozen_forward = lambda x_, x0, affine=True: intact(x_, x0, affine=affine) * 1.05
                try:
                    remainder, independent = self._both(comp, net, x)
                finally:
                    del block.frozen_forward
                self.assertLess(remainder, TOL)
                self.assertGreater(independent, self.BROKEN)
                self.assertLess(max(self._both(comp, net, x)), TOL)   # intact again

    def test_a_layer_whose_linear_map_is_wrong_in_the_row_pass_only(self):
        # ... and the same block wrong for the columns only (affine=False)
        x = tokens()
        net = llama_with_biases()
        block = next(layer for layer in net.layers if isinstance(layer, SwiGLU))
        intact = block.frozen_forward
        block.frozen_forward = lambda x_, x0, affine=True: intact(x_, x0, affine=affine) * (1.0 if affine else 1.05)
        try:
            remainder, independent = self._both(KnowledgeRowComputer(net, mixer_mode="frozen"), net, x)
        finally:
            del block.frozen_forward
        self.assertLess(remainder, TOL)
        self.assertGreater(independent, self.BROKEN)

    def test_a_norm_linearized_at_the_wrong_point(self):
        x = tokens()
        for mode in ("frozen", "ratio"):
            with self.subTest(mixer_mode=mode):
                net = llama_with_biases()
                remainder, independent = self._both(StaleNorm(net, mixer_mode=mode), net, x)
                self.assertLess(remainder, TOL)
                self.assertGreater(independent, self.BROKEN)

    def test_the_full_matrix_check_fails_on_the_same_break(self):
        # KnowledgeMatrixComputer always computed its bias column this way,
        # which is why ITS row sums are a real check: same break, same verdict
        x = tokens()
        net = llama_with_biases()
        block = next(layer for layer in net.layers if isinstance(layer, SwiGLU))
        intact = block.frozen_forward
        block.frozen_forward = lambda x_, x0, affine=True: intact(x_, x0, affine=affine) * 1.05
        try:
            mat = KnowledgeMatrixComputer(net, batch_size=32, mixer_mode="frozen").forward(x)
            rows = KnowledgeRowComputer(net, mixer_mode="frozen").forward(x, ROWS, independent_bias=True)
        finally:
            del block.frozen_forward
        out = net(x).detach().reshape(-1)
        self.assertGreater((mat.sum(1) - out).abs().max().item(), self.BROKEN)
        self.assertLess((rows - mat[ROWS]).abs().max().item(), TOL)   # the same (wrong) rows


### 4. Networks split over devices ###

def second_device():
    if torch.cuda.device_count() >= 2:
        return ["cuda:0", "cuda:1"]
    if torch.cuda.is_available():
        return ["cpu", "cuda:0"]
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return ["cpu", "mps"]
    return None


class TestSharded(unittest.TestCase):

    def _check(self, devices, dtype, tol):
        x = tokens()
        for mode in ("frozen", "ratio"):
            with self.subTest(devices=devices, mixer_mode=mode):
                whole = llama_with_biases(dtype)
                want_bias = KnowledgeMatrixComputer(whole, mixer_mode=mode).bias_column(x)
                want_rows = KnowledgeRowComputer(whole, mixer_mode=mode).forward(x, ROWS, independent_bias=True)

                split = llama_with_biases(dtype).shard(devices)

                def moved(*a, **k):
                    raise AssertionError("a network split over devices must not be moved with .to()")
                split.to = moved

                home = torch.device(devices[0]).type
                bias = KnowledgeMatrixComputer(split, mixer_mode=mode).bias_column(x.to(devices[0]))
                self.assertEqual(bias.device.type, home)
                comp = KnowledgeRowComputer(split, mixer_mode=mode)
                rows = comp.forward(x.to(devices[0]), ROWS, independent_bias=True)
                self.assertEqual(rows.device.type, home)
                self.assertEqual(comp.row_outputs.device.type, home)
                if tol == 0:
                    self.assertTrue(torch.equal(bias.cpu(), want_bias))
                    self.assertTrue(torch.equal(rows.cpu(), want_rows))
                    gap_tol = TOL
                else:
                    scale = max(1.0, want_rows.abs().max().item())
                    self.assertLessEqual((bias.cpu() - want_bias).abs().max().item(), tol * scale)
                    self.assertLessEqual((rows.cpu() - want_rows).abs().max().item(), tol * scale)
                    gap_tol = tol * scale
                self.assertLess((rows.sum(1) - comp.row_outputs).abs().max().item(), gap_tol)
                if mode == "frozen":      # (in ratio mode this network has no bias: all zeros)
                    self.assertGreater(bias.abs().max().item(), 1e-3)
                # ... and the check still fails there when the row pass is wrong
                broken = ScaledRowPass(split, mixer_mode=mode)
                wrong = broken.forward(x.to(devices[0]), ROWS, independent_bias=True)
                self.assertGreater((wrong.sum(1) - broken.row_outputs).abs().max().item(), 1e-4)

    def test_cpu_cpu(self):
        self._check(["cpu", "cpu"], torch.float64, 0)

    def test_three_stages(self):
        self._check(["cpu", "cpu", "cpu"], torch.float64, 0)

    def test_two_real_devices(self):
        devices = second_device()
        if devices is None:
            self.skipTest("one device only")
        dtype = torch.float32 if "mps" in devices else torch.float64
        self._check(devices, dtype, 2e-4 if dtype == torch.float32 else 1e-9)


if __name__ == "__main__":
    unittest.main()
