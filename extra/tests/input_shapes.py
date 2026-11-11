#!/usr/bin/env python
"""
    Two capabilities that came in from the cluster branch and are easy to
    regress silently:

    1. ``input_shape`` may be a bare feature count or a 1-, 2- or 3-tuple.
       MNIST-1D nets are built with a plain int, and the matrix computer
       used to unpack three values unconditionally -- a TypeError deep in
       the constructor, far from the cause.
    2. ``freeze_features`` freezes the CONVOLUTIONAL layers only, leaving
       the classifier trainable: the transfer-learning setup.

    Both are checked against the invariant that matters -- the knowledge
    matrix rows still sum to the forward pass at every accepted shape.
"""
import unittest

import torch
from torch import nn

from knowledgematrix.neural_net import NN
from knowledgematrix.matrix_computer import KnowledgeMatrixComputer

DEVICE = "cpu"
torch.set_default_dtype(torch.float64)


def mlp(input_shape, out=5):
    net = NN(input_shape=input_shape, device=DEVICE)
    net.flatten()
    net.linear(net.get_input_size(), 12)
    net.relu()
    net.linear(12, out)
    return net


class TestInputShapes(unittest.TestCase):

    CASES = [
        (40, (1, 40, 1), 40),
        ((40,), (1, 40, 1), 40),
        ((3, 40), (3, 1, 40), 120),
        ((3, 8, 8), (3, 8, 8), 192),
    ]

    def test_shapes_normalize_to_chw(self):
        for shape, chw, size in self.CASES:
            with self.subTest(input_shape=shape):
                comp = KnowledgeMatrixComputer(mlp(shape))
                self.assertEqual((comp.in_c, comp.in_h, comp.in_w), chw)
                self.assertEqual(comp.input_size, size)

    def test_get_input_size_accepts_a_bare_int(self):
        self.assertEqual(NN(input_shape=40).get_input_size(), 40)
        self.assertEqual(NN(input_shape=(3, 8, 8)).get_input_size(), 192)

    def test_matrix_invariant_at_every_accepted_shape(self):
        for shape, chw, _ in self.CASES:
            with self.subTest(input_shape=shape):
                torch.manual_seed(0)
                net = mlp(shape)
                net.eval()
                x = torch.randn(*chw)
                forward_pass = net.forward(x)
                net.save = True
                mat = KnowledgeMatrixComputer(net, batch_size=16).forward(x)
                diff = torch.norm(forward_pass.reshape(1, -1) - mat.sum(1)).item()
                self.assertAlmostEqual(diff, 0, places=None, delta=1e-10)

    def test_four_dimensional_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            KnowledgeMatrixComputer(mlp((2, 3, 4, 5)))


class TestFlatInputs(unittest.TestCase):
    """
        A network of fully connected layers may be fed plain feature
        vectors, (features,) or (1, features), instead of a (C, H, W)
        tensor -- how MNIST-1D pipelines feed their MLPs. The knowledge
        matrix is the one obtained from the same vector presented as
        (1, 1, features).
    """

    @staticmethod
    def _net(input_shape, flatten, act="tanh", residual=False):
        net = NN(input_shape=input_shape, device=DEVICE)
        if flatten:
            net.flatten()
        net.linear(40, 16)
        getattr(net, act)()
        if residual:
            start = net.get_num_layers()
            net.linear(16, 16)
            getattr(net, act)()
            end = net.get_num_layers()
            net.identity()
            net.residual(start, end)
        net.linear(16, 7)
        net.eval()
        return net

    def test_forward_accepts_vectors_and_batches(self):
        torch.manual_seed(0)
        net = self._net((1, 40), flatten=True)
        v = torch.randn(40)
        as_image = net.forward(v.reshape(1, 1, 40))
        self.assertEqual(net.forward(v).shape, (1, 7))
        self.assertEqual(net.forward(v.reshape(1, 40)).shape, (1, 7))
        self.assertTrue(torch.equal(net.forward(v).reshape(-1), as_image.reshape(-1)))
        self.assertTrue(torch.equal(net.forward(v.reshape(1, 40)).reshape(-1), as_image.reshape(-1)))
        batch = torch.randn(5, 40)
        out = net.forward(batch)
        self.assertEqual(out.shape, (5, 7))
        for k in range(5):
            self.assertLess((out[k] - net.forward(batch[k]).reshape(-1)).abs().max().item(), 1e-12)

    def test_matrix_of_a_flat_input(self):
        for input_shape in ((1, 40), (40,), 40):
            for flatten in (True, False):
                for act in ("relu", "tanh", "gelu"):
                    for residual in (False, True):
                        with self.subTest(input_shape=input_shape, flatten=flatten, act=act, residual=residual):
                            torch.manual_seed(1)
                            net = self._net(input_shape, flatten, act, residual)
                            v = torch.randn(40)
                            f = net.forward(v.reshape(1, 40)).reshape(-1)
                            M_image = KnowledgeMatrixComputer(net, batch_size=16).forward(v.reshape(1, 1, 40))
                            for x in (v, v.reshape(1, 40)):
                                M = KnowledgeMatrixComputer(net, batch_size=16).forward(x)
                                self.assertEqual(M.shape, (7, 41))
                                self.assertLess((M.sum(1) - f).abs().max().item(), 1e-12)
                                self.assertLess((M - M_image).abs().max().item(), 1e-12)

    def test_weff_and_bias_column_of_a_flat_input(self):
        torch.manual_seed(2)
        net = self._net((1, 40), flatten=True)
        x = torch.randn(1, 40)
        computer = KnowledgeMatrixComputer(net, batch_size=16)
        M = computer.forward(x)
        W_eff = computer.forward(x, extract_weff=True)
        self.assertEqual(W_eff.shape, (7, 40))
        self.assertLess((W_eff * x.reshape(1, -1) - M[:, :-1]).abs().max().item(), 1e-12)
        self.assertLess((computer.bias_column(x) - M[:, -1]).abs().max().item(), 1e-12)

    def test_the_matrix_takes_one_input_at_a_time(self):
        net = self._net((1, 40), flatten=True)
        with self.assertRaises(ValueError):
            KnowledgeMatrixComputer(net).forward(torch.randn(3, 40))

    def test_a_batch_of_one_image_is_that_image(self):
        # (1, C, H, W) used to be read as C=1, H=C, W=H
        torch.manual_seed(3)
        net = NN(input_shape=(1, 40, 1), device=DEVICE)
        net.conv(1, 4, kernel_size=(5, 1), stride=(2, 1), padding=(1, 0))
        net.tanh()
        net.conv(4, 4, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        net.relu()
        net.flatten()
        net.linear(4 * 19, 6)
        net.eval()
        x = torch.randn(1, 40, 1)
        computer = KnowledgeMatrixComputer(net, batch_size=16)
        M = computer.forward(x)
        M_batched = computer.forward(x.unsqueeze(0))
        self.assertEqual(M.shape, (6, 41))
        self.assertTrue(torch.equal(M, M_batched))
        self.assertLess((M.sum(1) - net.forward(x).reshape(-1)).abs().max().item(), 1e-12)
        self.assertTrue(torch.equal(computer.forward(x.unsqueeze(0), extract_weff=True),
                                    computer.forward(x, extract_weff=True)))
        with self.assertRaises(ValueError):
            computer.forward(torch.randn(2, 1, 40, 1))

    def test_token_ids_without_a_batch_dimension_are_an_error(self):
        net = NN(input_shape=(1, 1, 8), device=DEVICE)
        net.embedding(11, 8)
        net.linear(8, 3)
        with self.assertRaises(ValueError):
            net.forward(torch.randint(0, 11, (5,)))
        self.assertEqual(net.forward(torch.randint(0, 11, (1, 1, 5))).shape, (1, 1, 5, 3))


class TestFreezeFeatures(unittest.TestCase):

    def _convnet(self):
        net = NN(input_shape=(3, 8, 8), device=DEVICE)
        net.conv(3, 4, kernel_size=(3, 3), padding=(1, 1))
        net.relu()
        net.flatten()
        net.linear(4 * 8 * 8, 5)
        return net

    def test_freezes_convolutions_only(self):
        net = self._convnet()
        net.freeze_features()
        convs = [l for l in net.layers if isinstance(l, nn.Conv2d)]
        lins = [l for l in net.layers if isinstance(l, nn.Linear)]
        self.assertTrue(convs and lins)
        self.assertTrue(all(not p.requires_grad
                            for l in convs for p in l.parameters()))
        self.assertTrue(all(p.requires_grad
                            for l in lins for p in l.parameters()))

    def test_freeze_all_still_freezes_everything(self):
        # freeze_features must not have displaced the blanket freeze()
        net = self._convnet()
        net.freeze()
        self.assertTrue(all(not p.requires_grad for p in net.parameters()))
        net.unfreeze()
        self.assertTrue(all(p.requires_grad for p in net.parameters()))

    def test_freezing_does_not_change_the_forward_pass(self):
        torch.manual_seed(0)
        net = self._convnet()
        net.eval()
        x = torch.randn(3, 8, 8)
        before = net.forward(x)
        net.freeze_features()
        after = net.forward(x)
        self.assertEqual((before - after).abs().max().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
