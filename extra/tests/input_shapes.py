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
