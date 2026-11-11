#!/usr/bin/env python
"""
    Biases inside composite layers, and the bias column.

    A composite layer (SwiGLU, GeGLU, RopeAttention, GatedAttention,
    MultiHeadAttention, GatedDeltaNet) keeps its parameters on SUBMODULES
    (`up_proj.bias`, `v_proj.bias`, ...). The condition that decides whether
    the knowledge matrix gets a bias column used to look at `layer.bias`
    only, so such a layer standing alone in a network without any other
    bias, norm or residual got no bias column:

    - mixer_mode="frozen": the biases of the value / output projections are
      constants of the frozen map; without the bias pass the rows did not
      sum to the output;
    - mixer_mode="ratio": the layer is an activation, linearized by the
      elementwise post/pre ratio. The matrix is then diag(output) and needs
      no constant -- except at an input coordinate that is exactly 0, where
      the ratio does not exist and the output (non-zero: the layer mixes
      coordinates and has biases) is a constant of the linearization. That
      holds for a bias-free mixing layer just as well.

    Now `NN.has_bias_column()` is the one condition for
    KnowledgeMatrixComputer.forward and NN.get_matrix_shape: any bias
    parameter (also inside a composite layer or a residual projection), a
    norm, a GLU constant, a mixing layer, or a residual connection.

    Checked here, for every composite layer type, both modes, the layer
    alone in an otherwise bias-free network: rows sum to the output, the
    bias column is exactly what the linearization says it is, and
    get_matrix_shape agrees with the matrix.
"""
import unittest
import warnings

import torch

from knowledgematrix.neural_net import (
    NN, GLU, GatedAttention, GatedDeltaNet, GeGLU, MultiHeadAttention, RopeAttention, SwiGLU,
    LINEARIZABLE_LAYERS,
)
from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.row_computer import KnowledgeRowComputer

torch.set_default_dtype(torch.float64)

D, T = 8, 4
TOL = 1e-10


def randomize(net, seed=0, scale=0.5):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * scale)
    net.eval()
    return net


def add_composite(net, kind):
    """Append one composite layer of the given kind to `net`."""
    if kind == "swiglu":
        net.swiglu(D, 16, bias=True)
    elif kind == "geglu":
        net.geglu(D, 16, bias=True)
    elif kind == "ropeattention":
        net.ropeattention(D, 2, bias=True, o_bias=True)
    elif kind == "ropeattention, q/k/v biases only":       # Qwen2.5
        net.ropeattention(D, 4, num_kv_heads=2, bias=True, o_bias=False, qk_norm=True)
    elif kind == "ropeattention, output bias only":
        net.ropeattention(D, 2, bias=False, o_bias=True, softcap=20.0, sliding_window=2)
    elif kind == "gatedattention":
        net.gatedattention(D, 2, head_dim=4, partial_rotary_factor=0.5, bias=True)
    elif kind == "multiheadattention":
        net.multiheadattention(D, 2)
    elif kind == "multiheadattention, causal":
        net.multiheadattention(D, 4, num_kv_heads=2, mask=torch.tril(torch.ones(16, 16)))
    # --- without any bias parameter ---
    elif kind == "swiglu, bias-free":
        net.swiglu(D, 16)
    elif kind == "geglu, bias-free":
        net.geglu(D, 16)
    elif kind == "ropeattention, bias-free":
        net.ropeattention(D, 2)
    elif kind == "gatedattention, bias-free":
        net.gatedattention(D, 2, head_dim=4, partial_rotary_factor=0.5)
    elif kind == "gateddeltanet":
        net.gateddeltanet(D, 2, 2, head_k_dim=4, head_v_dim=4)
    else:
        raise ValueError(kind)


WITH_BIASES = (
    "swiglu", "geglu", "ropeattention", "ropeattention, q/k/v biases only",
    "ropeattention, output bias only", "gatedattention", "multiheadattention",
    "multiheadattention, causal",
)
BIAS_FREE = (
    "swiglu, bias-free", "geglu, bias-free", "ropeattention, bias-free",
    "gatedattention, bias-free", "gateddeltanet",
)
MODES = (dict(mixer_mode="frozen"), dict(mixer_mode="ratio"))


def alone(kind, seed=0, length=T):
    """The composite layer alone: a network with no other layer at all."""
    net = NN(input_shape=(1, length, D))
    add_composite(net, kind)
    return randomize(net, seed)


def sandwiched(kind, seed=0, length=T):
    """... and between two bias-free Linear layers."""
    net = NN(input_shape=(1, length, D))
    net.linear(D, D, bias=False)
    add_composite(net, kind)
    net.linear(D, 5, bias=False)
    return randomize(net, seed)


def inputs(length=T, zeros=False, seed=1):
    x = torch.randn(1, length, D, generator=torch.Generator().manual_seed(seed))
    if zeros:
        x[0, 0, 0] = 0.0
        x[0, length - 1, 3] = 0.0
        if length > 1:
            x[0, 1, :] = 0.0         # a whole position
    return x


def matrix(net, x, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return KnowledgeMatrixComputer(net, batch_size=16, **kw).forward(x)


class TestDetection(unittest.TestCase):

    def test_biases_inside_composite_layers_are_seen(self):
        for kind in WITH_BIASES:
            with self.subTest(layer=kind):
                net = alone(kind)
                self.assertFalse(hasattr(net.layers[0], "bias"))      # not on the layer itself
                self.assertTrue(net._has_bias())
                self.assertTrue(net.has_bias_column())

    def test_a_bias_free_mixer_has_no_bias_but_has_the_column(self):
        for kind in BIAS_FREE:
            with self.subTest(layer=kind):
                net = alone(kind)
                self.assertFalse(net._has_bias())
                self.assertTrue(net._has_mixer())
                self.assertTrue(net.has_bias_column())

    def test_every_linearizable_layer_type_is_covered_here(self):
        seen = {type(alone(kind).layers[0]) for kind in WITH_BIASES + BIAS_FREE}
        self.assertEqual(seen, set(LINEARIZABLE_LAYERS))

    def test_plain_bias_free_networks_still_have_no_bias_column(self):
        net = NN(input_shape=(1, 1, D))
        net.linear(D, 6, bias=False); net.relu(); net.linear(6, 3, bias=False)
        self.assertFalse(net._has_bias())
        self.assertFalse(net.has_bias_column())
        glu = NN(input_shape=(1, 1, D))
        glu.glu(D, 6, activation="silu", bias=False)
        self.assertFalse(glu.has_bias_column())
        self.assertEqual(matrix(randomize(net), inputs(1)).shape, (3, D))

    def test_direct_biases_and_projection_biases(self):
        net = NN(input_shape=(1, 1, D))
        net.linear(D, 6)
        self.assertTrue(net._has_bias())
        net = NN(input_shape=(1, 1, D))
        net.glu(D, 6, bias=True)
        self.assertTrue(net._has_bias())
        # the projection a residual between different shapes gets (1x1 conv
        # + BatchNorm): its bias is on no layer of the network
        net = NN(input_shape=(2, 6, 6))
        net.conv(2, 4, kernel_size=3, padding=1, bias=False); net.relu()
        net.residual(0, 2)
        net.identity()
        self.assertTrue(all(getattr(layer, "bias", None) is None for layer in net.layers))
        self.assertEqual(len(net.residual_modules), 2)
        self.assertTrue(net._has_bias())


class TestCompositeLayerAlone(unittest.TestCase):
    """The layer is the whole network: no other bias, norm or residual."""

    def _row_sums(self, net, x, **kw):
        out = net.forward(x).detach().reshape(-1)
        mat = matrix(net, x, **kw)
        self.assertEqual(mat.shape, (out.numel(), x.numel() + 1))
        self.assertEqual(mat.shape[1], net.get_input_size() + int(net.has_bias_column()))
        self.assertLess((mat.sum(1) - out).abs().max().item(), TOL)
        return mat, out

    def test_rows_sum_to_the_output(self):
        for kind in WITH_BIASES + BIAS_FREE:
            for kw in MODES:
                for zeros in (False, True):
                    with self.subTest(layer=kind, zeros_in_the_input=zeros, **kw):
                        self._row_sums(alone(kind), inputs(zeros=zeros), **kw)

    def test_default_initialization(self):
        # the configuration of the report: NN.swiglu(d_model, d_ff, bias=True)
        # as built, frozen mode
        for kind in WITH_BIASES:
            with self.subTest(layer=kind):
                torch.manual_seed(0)
                net = NN(input_shape=(1, T, D))
                add_composite(net, kind)
                net.eval()
                mat, _ = self._row_sums(net, inputs(), mixer_mode="frozen")
                self.assertGreater(mat[:, -1].abs().max().item(), 1e-3)

    def test_frozen_bias_column_is_the_frozen_map_at_zero(self):
        # frozen mode: the layer is the affine map x -> frozen_forward(x, x0);
        # its constant term is its value at x = 0, and the input columns sum
        # to what is left
        for kind in WITH_BIASES + BIAS_FREE:
            with self.subTest(layer=kind):
                net = alone(kind)
                x = inputs()
                mat, out = self._row_sums(net, x, mixer_mode="frozen")
                x0 = x.unsqueeze(0)
                with torch.no_grad():
                    constant = net.layers[0].frozen_forward(torch.zeros_like(x0), x0, affine=True).reshape(-1)
                self.assertLess((mat[:, -1] - constant).abs().max().item(), TOL)
                self.assertLess((mat[:, :-1].sum(1) - (out - constant)).abs().max().item(), TOL)
                if kind in WITH_BIASES:
                    self.assertGreater(constant.abs().max().item(), 1e-3)      # a real bias
                else:
                    self.assertEqual(mat[:, -1].abs().max().item(), 0.0)       # none to carry

    def test_ratio_matrix_is_the_diagonal_of_the_output(self):
        # ratio mode: the layer is an activation, coordinate i of the output
        # is attributed to coordinate i of the input; nothing is constant
        for kind in WITH_BIASES + BIAS_FREE:
            with self.subTest(layer=kind):
                net = alone(kind)
                x = inputs()
                mat, out = self._row_sums(net, x, mixer_mode="ratio")
                self.assertLess((mat[:, :-1] - torch.diag(out)).abs().max().item(), TOL)
                self.assertEqual(mat[:, -1].abs().max().item(), 0.0)

    def test_ratio_at_a_zero_input_coordinate(self):
        # ... except where the input is exactly 0: no ratio exists there, the
        # column is 0 and the output of that coordinate is a constant
        for kind in WITH_BIASES + BIAS_FREE:
            with self.subTest(layer=kind):
                net = alone(kind)
                x = inputs(zeros=True)
                dead = (x.reshape(-1) == 0)
                self.assertEqual(int(dead.sum()), 2 + D)
                mat, out = self._row_sums(net, x, mixer_mode="ratio")
                # the layer does output there (not at every dead coordinate: a
                # bias-free per-position block maps a zero position to zero)
                self.assertGreater(out[dead].abs().max().item(), 1e-6)
                self.assertEqual(mat[:, :-1][:, dead].abs().max().item(), 0.0)
                self.assertTrue(torch.equal(mat[dead, -1], out[dead]))
                self.assertEqual(mat[~dead, -1].abs().max().item(), 0.0)
                live = torch.diag(out)[~dead][:, ~dead]
                self.assertLess((mat[:, :-1][~dead][:, ~dead] - live).abs().max().item(), TOL)

    def test_frozen_pattern_mode(self):
        # attention_mode="frozen_pattern" freezes MultiHeadAttention only
        for kind in ("multiheadattention", "multiheadattention, causal"):
            with self.subTest(layer=kind):
                net = alone(kind)
                x = inputs()
                mat, _ = self._row_sums(net, x, attention_mode="frozen_pattern")
                self.assertTrue(torch.equal(mat, matrix(net, x, mixer_mode="frozen")))
        net = alone("swiglu")
        mat, _ = self._row_sums(net, inputs(), attention_mode="frozen_pattern")
        self.assertTrue(torch.equal(mat, matrix(net, inputs(), mixer_mode="ratio")))

    def test_bias_column_alone_and_row_computer(self):
        # the bias column on its own, and the rows by reverse mode with the
        # independently computed bias, are those of the matrix
        for kind in WITH_BIASES + BIAS_FREE:
            for mode in ("frozen", "ratio"):
                for zeros in (False, True):
                    with self.subTest(layer=kind, mixer_mode=mode, zeros_in_the_input=zeros):
                        net = alone(kind)
                        x = inputs(zeros=zeros)
                        mat = matrix(net, x, mixer_mode=mode)
                        bias = KnowledgeMatrixComputer(net, mixer_mode=mode).bias_column(x)
                        self.assertTrue(torch.equal(bias, mat[:, -1]))
                        rows = [0, 3, D + 2, T * D - 1]
                        comp = KnowledgeRowComputer(net, mixer_mode=mode)
                        got = comp.forward(x, rows, independent_bias=True)
                        self.assertLess((got - mat[rows]).abs().max().item(), TOL)
                        self.assertLess((got.sum(1) - comp.row_outputs).abs().max().item(), TOL)


class TestCompositeLayerInABiasFreeNetwork(unittest.TestCase):
    """Linear(bias=False) -> composite -> Linear(bias=False)."""

    def test_rows_sum_to_the_output(self):
        for kind in WITH_BIASES + BIAS_FREE:
            for kw in MODES:
                for zeros in (False, True):
                    with self.subTest(layer=kind, zeros_in_the_input=zeros, **kw):
                        net = sandwiched(kind)
                        x = inputs(zeros=zeros)
                        out = net.forward(x).detach().reshape(-1)
                        mat = matrix(net, x, **kw)
                        self.assertEqual(mat.shape, (out.numel(), x.numel() + 1))
                        self.assertLess((mat.sum(1) - out).abs().max().item(), TOL)

    def test_frozen_bias_column_goes_through_the_head(self):
        for kind in WITH_BIASES:
            with self.subTest(layer=kind):
                net = sandwiched(kind)
                x = inputs()
                mat = matrix(net, x, mixer_mode="frozen")
                with torch.no_grad():
                    x0 = net.layers[0](x.unsqueeze(0))
                    constant = net.layers[1].frozen_forward(torch.zeros_like(x0), x0, affine=True)
                    want = net.layers[2](constant).reshape(-1)
                self.assertGreater(want.abs().max().item(), 1e-3)
                self.assertLess((mat[:, -1] - want).abs().max().item(), TOL)

    def test_a_dead_unit_in_front_of_the_layer(self):
        # ratio mode, zeros that are not in the input: the first layer
        # outputs exactly 0 in one coordinate (a zero row of its weight)
        for kind in WITH_BIASES + BIAS_FREE:
            with self.subTest(layer=kind):
                net = sandwiched(kind)
                with torch.no_grad():
                    net.layers[0].weight[2] = 0.0
                x = inputs()
                out = net.forward(x).detach().reshape(-1)
                mat = matrix(net, x, mixer_mode="ratio")
                self.assertLess((mat.sum(1) - out).abs().max().item(), TOL)
                self.assertGreater(mat[:, -1].abs().max().item(), 1e-6)


class TestMatrixShape(unittest.TestCase):
    """get_matrix_shape() is the shape of the matrix forward() returns."""

    def _agree(self, net, x, **kw):
        mat = matrix(net, x, **kw)
        self.assertEqual(tuple(mat.shape), tuple(net.get_matrix_shape()))
        out = net.forward(x).detach().reshape(-1)
        self.assertLess((mat.sum(1) - out).abs().max().item(), TOL)
        return mat

    def test_composite_layers(self):
        # one position, so that the row count get_matrix_shape reads off the
        # last layer (out_features) is the size of the output
        for kind in WITH_BIASES + BIAS_FREE:
            for kw in MODES:
                with self.subTest(layer=kind, **kw):
                    mat = self._agree(sandwiched(kind, length=1), inputs(1), **kw)
                    self.assertEqual(mat.shape, (5, D + 1))

    def test_bias_free_network_with_a_residual(self):
        # forward() has always run the bias pass for a network with
        # residual connections; get_matrix_shape did not count them
        net = NN(input_shape=(1, 1, D))
        net.linear(D, D, bias=False); net.relu(); net.linear(D, D, bias=False)
        net.residual(0, 3)
        net.linear(D, 3, bias=False)
        randomize(net)
        self.assertFalse(net._has_bias())
        mat = self._agree(net, inputs(1))
        self.assertEqual(mat.shape, (3, D + 1))
        self.assertEqual(mat[:, -1].abs().max().item(), 0.0)

    def test_networks_without_and_with_constant_terms(self):
        def mlp(bias):
            net = NN(input_shape=(1, 1, D))
            net.linear(D, 6, bias=bias); net.relu(); net.linear(6, 3, bias=bias)
            return randomize(net)

        def normed(which):
            net = NN(input_shape=(1, 1, D))
            net.linear(D, 6, bias=False)
            if which == "layernorm":
                net.layernorm(6)
            else:
                net.rmsnorm(6)
            net.linear(6, 3, bias=False)
            return randomize(net)

        def gated(activation, bias):
            net = NN(input_shape=(1, 1, D))
            net.glu(D, 6, activation=activation, bias=bias)
            net.linear(6, 3, bias=False)
            return randomize(net)

        cases = (
            ("bias-free MLP", mlp(False), False),
            ("MLP", mlp(True), True),
            ("LayerNorm", normed("layernorm"), True),
            ("RMSNorm (no shift, no bias)", normed("rmsnorm"), False),
            ("GLU silu, bias-free", gated("silu", False), False),
            ("GLU sigmoid, bias-free", gated("sigmoid", False), True),
            ("GLU silu with biases", gated("silu", True), True),
        )
        for name, net, column in cases:
            with self.subTest(network=name):
                self.assertEqual(net.has_bias_column(), column)
                mat = self._agree(net, inputs(1))
                self.assertEqual(mat.shape, (3, D + int(column)))

    def test_convolutional_network(self):
        net = NN(input_shape=(2, 6, 6))
        net.conv(2, 4, kernel_size=3, padding=1, bias=False); net.batchnorm(4); net.relu()
        net.adaptiveavgpool((1, 1)); net.flatten(); net.linear(4, 3, bias=False)
        randomize(net)
        self.assertTrue(net.has_bias_column())
        x = torch.rand(2, 6, 6, generator=torch.Generator().manual_seed(2))
        self.assertEqual(self._agree(net, x).shape, (3, 73))


if __name__ == "__main__":
    unittest.main()
