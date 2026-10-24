#!/usr/bin/env python
"""
    Tests for the model split over several devices (`NN.shard`):

    - `shard_cut_points` never cuts through a residual span, and the
      default boundaries fall on admissible points;
    - the forward pass of a sharded LlamaLike / Qwen3_5 equals the
      unsharded one exactly (same arithmetic, only the tensors move);
    - KnowledgeRowComputer rows of the sharded model equal the unsharded
      rows, in frozen and ratio mode, and keep the row-sum identity;
    - `unshard` restores the single-device network.

    The split is exercised over ["cpu", "cpu"] (the bookkeeping) and, when
    the machine has one, over ["cpu", "mps"] or ["cuda:0", "cuda:1"] (a real
    cross-device transfer inside the autograd graph). MPS has no float64,
    so the cross-device case runs in float32 with a looser tolerance.
"""
import unittest

import torch

from knowledgematrix.models.llama import LlamaLike
from knowledgematrix.models.qwen3_5 import Qwen3_5
from knowledgematrix.row_computer import KnowledgeRowComputer

LLAMA = dict(vocab_size=50, hidden_size=24, intermediate_size=48,
             num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
             head_dim=6)
QWEN = dict(vocab_size=50, hidden_size=24, intermediate_size=48,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, linear_num_value_heads=4, linear_num_key_heads=2,
            linear_key_head_dim=6, linear_value_head_dim=6)


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
        self.assertEqual(set(net.layer_devices), set(devices))
        # the first stage really is on devices[0], the last on devices[-1]
        self.assertEqual(next(net.layers[0].parameters()).device.type,
                         torch.device(devices[0]).type)
        self.assertEqual(next(net.layers[-1].parameters()).device.type,
                         torch.device(devices[-1]).type)

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


if __name__ == "__main__":
    unittest.main()
