#!/usr/bin/env python
"""
    The modules a vision wrapper takes from its torchvision source model
    must be REGISTERED submodules of the wrapper.

    `NN.residual` registers the projection it creates (1x1 conv + BatchNorm)
    in `NN.residual_modules`. A wrapper that then swaps torchvision's own
    `downsample` modules into the residual wiring by index leaves them
    outside `parameters()` / `buffers()` / `modules()` / `state_dict()` /
    `.to()`, and keeps the discarded auto-projection registered in their
    place. The forward pass does not notice (it goes through the wiring),
    but

    - `.to(device)` / `.to(dtype)` leave the real projections behind,
    - `state_dict()` saves the discarded projections instead of the real,
    - an optimizer never sees the real projections,
    - GradientMatrixComputer, whose bias column is a JVP over
      `named_parameters()`, misses the effective bias of the downsample
      BatchNorms: its rows no longer sum to the output.

    ResNet-18 was fixed earlier; ResNet-152 had the same defect. The other
    wrappers (GoogLeNet, Inception-v3, DenseNet, VGG-11, AlexNet) append
    every source module to `NN.layers` and use no residual projection; they
    are checked here so that it stays that way.

    Source models are randomly initialized torchvision networks with
    perturbed BatchNorm statistics (at a fresh initialization the running
    mean and the BatchNorm bias are 0, and so is the effective bias the
    gradient computer would miss): no download is needed.
"""
import gc
import unittest

import torch
from torch import nn
import torchvision

from knowledgematrix.gradient_matrix import GradientMatrixComputer
from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.models.alexnet import AlexNet
from knowledgematrix.models.densenet import DenseNet
from knowledgematrix.models.googlenet import GoogLeNet
from knowledgematrix.models.inception import Inception
from knowledgematrix.models.resnet18 import ResNet18
from knowledgematrix.models.resnet152 import ResNet152
from knowledgematrix.models.vgg11 import VGG11

torch.set_default_dtype(torch.float64)


def perturb(source, seed=0):
    """Random BatchNorm statistics and affine parameters, eval mode."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in source.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.copy_(torch.randn(m.running_mean.shape, generator=g) * 0.1)
                m.running_var.copy_(torch.rand(m.running_var.shape, generator=g) + 0.5)
                m.weight.copy_(torch.rand(m.weight.shape, generator=g) + 0.5)
                m.bias.copy_(torch.randn(m.bias.shape, generator=g) * 0.1)
    source.eval()
    return source


def resnet_source(depth, seed=0, num_classes=1000):
    torch.manual_seed(seed)
    builder = {18: torchvision.models.resnet18, 152: torchvision.models.resnet152}[depth]
    return perturb(builder(weights=None, num_classes=num_classes), seed + 1)


def wrap_resnet(depth, source, num_classes=1000, input_shape=(3, 8, 8)):
    cls = {18: ResNet18, 152: ResNet152}[depth]
    net = cls(input_shape=input_shape, num_classes=num_classes, pretrained=True, pretrained_model=source)
    net.eval()
    return net


def wiring_modules(net):
    """Every module the forward pass of `net` goes through: its layers and
    the projections of its residual connections."""
    modules = list(net.layers)
    for pairs in net.residuals.values():
        for _, projection in pairs:
            modules.extend(m for m in projection if not isinstance(m, nn.Identity))
    return modules


def assert_registered(test, net, source, dropped=()):
    """
        The wrapper owns exactly what its forward pass uses, and that is
        the source model's storage (apart from the parts named in
        `dropped`, which the wrapper deliberately leaves out).
    """
    registered = {id(m) for m in net.modules()}
    used = wiring_modules(net)
    for module in used:
        test.assertIn(id(module), registered, f"{type(module).__name__} is used but not registered")
    # no discarded projection is left registered
    in_use = {id(m) for m in used}
    for module in net.residual_modules:
        test.assertIn(id(module), in_use, f"{type(module).__name__} is registered but not used")

    def kept(name):
        return not any(name.startswith(prefix) for prefix in dropped)

    parameters = {id(p) for p in net.parameters()}
    missing = [n for n, p in source.named_parameters() if kept(n) and id(p) not in parameters]
    test.assertEqual(missing, [], f"{len(missing)} source parameters are not parameters of the wrapper")
    buffers = {id(b) for b in net.buffers()}
    missing = [n for n, b in source.named_buffers() if kept(n) and id(b) not in buffers]
    test.assertEqual(missing, [], f"{len(missing)} source buffers are not buffers of the wrapper")
    # ... and nothing else: no parameter that is not the source's
    theirs = {id(p) for n, p in source.named_parameters() if kept(n)}
    extra = [n for n, p in net.named_parameters() if id(p) not in theirs]
    test.assertEqual(extra, [], f"{len(extra)} wrapper parameters do not come from the source")
    test.assertEqual(sum(p.numel() for p in net.parameters()),
                     sum(p.numel() for n, p in source.named_parameters() if kept(n)))


class TestResNetDownsamples(unittest.TestCase):
    """ResNet-152 (and ResNet-18, fixed earlier): the torchvision
    downsample modules in the residual wiring are registered."""

    DEPTHS = (152, 18)

    def tearDown(self):
        gc.collect()

    def test_source_storage_is_the_wrapper_storage(self):
        for depth in self.DEPTHS:
            with self.subTest(resnet=depth):
                source = resnet_source(depth)
                net = wrap_resnet(depth, source)
                # one projected residual (1x1 conv + BatchNorm) per stage that
                # changes the shape: all four stages of the Bottleneck network
                # (its first stage widens 64 -> 256), three of the BasicBlock one
                stages = {152: 4, 18: 3}[depth]
                self.assertEqual(sum(m is not None for m in
                                     (block.downsample for block in source.modules()
                                      if hasattr(block, "downsample"))), stages)
                projections = list(net.residual_modules)
                self.assertEqual(len(projections), 2 * stages)
                self.assertEqual(sum(isinstance(m, nn.Conv2d) for m in projections), stages)
                self.assertEqual(sum(isinstance(m, nn.BatchNorm2d) for m in projections), stages)
                assert_registered(self, net, source)
                self.assertEqual(len(net.state_dict()), len(source.state_dict()))

    def test_forward_is_the_source_forward(self):
        x = torch.rand(3, 16, 16, generator=torch.Generator().manual_seed(3))
        for depth in self.DEPTHS:
            with self.subTest(resnet=depth):
                source = resnet_source(depth)
                net = wrap_resnet(depth, source, input_shape=(3, 16, 16))
                with torch.no_grad():
                    want = source(x.unsqueeze(0))
                    got = net.forward(x)
                self.assertTrue(torch.equal(got, want))

    def test_state_dict_carries_the_projections(self):
        # the weights of one wrapper loaded into another reproduce its
        # function -- which they cannot if the real projections are not in
        # the state dict
        x = torch.rand(3, 16, 16, generator=torch.Generator().manual_seed(4))
        for depth in self.DEPTHS:
            with self.subTest(resnet=depth):
                first = wrap_resnet(depth, resnet_source(depth, seed=0), input_shape=(3, 16, 16))
                second = wrap_resnet(depth, resnet_source(depth, seed=10), input_shape=(3, 16, 16))
                with torch.no_grad():
                    want = first.forward(x)
                    self.assertGreater((second.forward(x) - want).abs().max().item(), 1e-3)
                    second.load_state_dict(first.state_dict())
                    self.assertTrue(torch.equal(second.forward(x), want))

    def test_cast_and_device_move_reach_the_projections(self):
        x = torch.rand(3, 16, 16, generator=torch.Generator().manual_seed(5))
        for depth in self.DEPTHS:
            with self.subTest(resnet=depth):
                net = wrap_resnet(depth, resnet_source(depth), input_shape=(3, 16, 16))
                with torch.no_grad():
                    want = net.forward(x)
                net.to(torch.float32)
                for module in wiring_modules(net):
                    for t in list(module.parameters()) + list(module.buffers()):
                        if t.is_floating_point():
                            self.assertEqual(t.dtype, torch.float32, type(module).__name__)
                with torch.no_grad():
                    got = net.forward(x.float())
                self.assertEqual(got.dtype, torch.float32)
                self.assertLess((got.double() - want).abs().max().item() / want.abs().max().item(), 1e-4)

                if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
                else:
                    continue
                net.to(device)
                for module in wiring_modules(net):
                    for t in list(module.parameters()) + list(module.buffers()):
                        self.assertEqual(t.device.type, device, type(module).__name__)
                with torch.no_grad():
                    moved = net.forward(x.float().to(device)).cpu()
                self.assertLess((moved.double() - want).abs().max().item() / want.abs().max().item(), 1e-3)

    def test_an_optimizer_sees_the_projections(self):
        source = resnet_source(152)
        net = wrap_resnet(152, source)
        trainable = {id(p) for p in net.parameters() if p.requires_grad}
        for name, p in source.named_parameters():
            if "downsample" in name:
                self.assertIn(id(p), trainable, name)
        net.freeze()
        self.assertFalse(any(p.requires_grad for p in source.parameters()))

    def test_gradient_matrix_agrees_with_the_knowledge_matrix(self):
        # GradientMatrixComputer builds its bias column from
        # named_parameters(): with the downsample BatchNorms outside it, the
        # column lacks their effective bias beta - gamma * mu / sigma
        x = torch.rand(3, 8, 8, generator=torch.Generator().manual_seed(6))
        for depth in self.DEPTHS:
            with self.subTest(resnet=depth):
                # (a 10-class head keeps the Jacobian small; the head is then
                # a fresh Linear layer, everything else is the source's)
                net = wrap_resnet(depth, resnet_source(depth), num_classes=10)
                with torch.no_grad():
                    out = net.forward(x).reshape(-1)
                km = KnowledgeMatrixComputer(net, batch_size=64).forward(x)
                self.assertEqual(km.shape, (10, x.numel() + 1))
                self.assertLess(((km.sum(1) - out).norm() / out.norm()).item(), 1e-9)
                for backend in ("func", "autograd"):
                    gm = GradientMatrixComputer(net, backend=backend, batch_size=5).forward(x)
                    self.assertEqual(gm.shape, km.shape)
                    self.assertLess(((gm.sum(1) - out).norm() / out.norm()).item(), 1e-9,
                                    f"{backend}: rows do not sum to the output")
                    scale = km.abs().max().item()
                    self.assertLess((gm - km).abs().max().item() / scale, 1e-9, backend)
                # the effective bias of the downsample BatchNorms is not zero
                # here, so the test can see it missing
                bn = [m for m in net.residual_modules if isinstance(m, nn.BatchNorm2d)]
                effective = max((m.bias - m.weight * m.running_mean / torch.sqrt(m.running_var + m.eps))
                                .abs().max().item() for m in bn)
                self.assertGreater(effective, 1e-2)

    def test_custom_build_registers_its_projections(self):
        # the non-pretrained build uses the projections NN.residual creates
        for cls in (ResNet152, ResNet18):
            with self.subTest(wrapper=cls.__name__):
                net = cls(input_shape=(3, 8, 8), num_classes=7)
                registered = {id(m) for m in net.modules()}
                used = wiring_modules(net)
                self.assertTrue(all(id(m) in registered for m in used))
                self.assertEqual({id(m) for m in net.residual_modules},
                                 {id(m) for m in used} - {id(m) for m in net.layers})


class TestOtherWrappers(unittest.TestCase):
    """The wrappers without residual projections: every source module they
    use is one of their layers."""

    def tearDown(self):
        gc.collect()

    def _check(self, net, source, dropped=()):
        self.assertEqual(len(net.residuals), 0)
        self.assertEqual(len(net.residual_modules), 0)
        assert_registered(self, net, source, dropped)

    def test_alexnet(self):
        torch.manual_seed(0)
        source = torchvision.models.alexnet(weights=None).eval()
        self._check(AlexNet(input_shape=(3, 64, 64), num_classes=1000, pretrained=True,
                            pretrained_model=source), source)

    def test_vgg11(self):
        torch.manual_seed(0)
        source = torchvision.models.vgg11(weights=None).eval()
        self._check(VGG11(input_shape=(3, 32, 32), num_classes=1000, pretrained=True,
                          pretrained_model=source), source)

    def test_vgg11_bn(self):
        torch.manual_seed(0)
        source = perturb(torchvision.models.vgg11_bn(weights=None))
        net = VGG11(input_shape=(3, 32, 32), num_classes=1000, pretrained=True, pretrained_model=source)
        self.assertEqual(sum(isinstance(layer, nn.BatchNorm2d) for layer in net.layers), 8)
        self._check(net, source)

    def test_densenet121(self):
        torch.manual_seed(0)
        source = perturb(torchvision.models.densenet121(weights=None))
        net = DenseNet(input_shape=(3, 32, 32), num_classes=1000, pretrained=True, pretrained_model=source)
        self.assertGreater(len(net.concat_skips), 0)
        self._check(net, source)

    def test_googlenet(self):
        torch.manual_seed(0)
        source = perturb(torchvision.models.googlenet(weights=None, aux_logits=True, init_weights=False))
        net = GoogLeNet(input_shape=(3, 32, 32), num_classes=1000, pretrained=True, pretrained_model=source)
        # the wrapper removes the auxiliary heads from the source itself
        self.assertIsNone(source.aux1)
        self.assertIsNone(source.aux2)
        self.assertGreater(len(net.branch_inputs), 0)
        self._check(net, source)

    def test_inception_v3(self):
        torch.manual_seed(0)
        source = perturb(torchvision.models.inception_v3(weights=None, aux_logits=True, init_weights=False))
        net = Inception(input_shape=(3, 75, 75), num_classes=1000, pretrained=True, pretrained_model=source)
        self.assertGreater(len(net.branch_inputs), 0)
        # the auxiliary classifier is deliberately not part of the wrapper
        self._check(net, source, dropped=("AuxLogits.",))


if __name__ == "__main__":
    unittest.main()
