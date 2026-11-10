#!/usr/bin/env python
"""
    Tests for the CustomResNet builder used in GH200 benchmarks.

    Verifies the fundamental knowledge matrix invariant (mat.sum(1) ≈ model(x))
    across different depths, widths, input sizes, and num_classes.

    Note: Uses small base_width (16-32) for fast CPU testing. The actual benchmark
    configs (128, 256) use the same builder logic and are tested on GPU.
"""
import sys
import os
import unittest

import torch

# Ensure project root is on path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from extra.GH200.models.custom_resnet import CustomResNet

DEVICE = "cpu"
torch.set_default_dtype(torch.float64)
BATCH_SIZE = 32


class TestCustomResNet(unittest.TestCase):

    def _test_config(
        self,
        base_width: int,
        blocks_per_stage: list[int],
        input_shape: tuple[int],
        num_classes: int,
    ) -> None:
        """Test a single ResNet configuration for KM invariant."""
        model = CustomResNet(
            input_shape=input_shape,
            num_classes=num_classes,
            base_width=base_width,
            blocks_per_stage=blocks_per_stage,
            device=DEVICE,
        )
        model.eval()

        computer = KnowledgeMatrixComputer(model, batch_size=BATCH_SIZE, device=DEVICE)
        x = torch.randn(input_shape)
        forward_pass = model(x)
        mat = computer.forward(x)

        diff = torch.norm(forward_pass - mat.sum(1)).item()
        label = f"R{2 + sum(2*b for b in blocks_per_stage)}-w{base_width}"
        inp_str = "x".join(str(s) for s in input_shape)

        self.assertAlmostEqual(
            first=diff,
            second=0,
            places=None,
            msg=f"{label} input={inp_str} classes={num_classes}: diff={diff}",
            delta=0.1,
        )

    # --- Depth variations (all small width for speed) ---

    def test_r18_depth_32x32(self) -> None:
        """ResNet-18 depth: [2,2,2,2] blocks."""
        self._test_config(16, [2, 2, 2, 2], (3, 32, 32), 10)

    def test_r34_depth_32x32(self) -> None:
        """ResNet-34 depth: [3,4,6,3] blocks."""
        self._test_config(16, [3, 4, 6, 3], (3, 32, 32), 10)

    def test_r46_depth_32x32(self) -> None:
        """ResNet-46 depth: [4,6,8,4] blocks."""
        self._test_config(16, [4, 6, 8, 4], (3, 32, 32), 10)

    # --- Width variations ---

    def test_width_16(self) -> None:
        self._test_config(16, [2, 2, 2, 2], (3, 32, 32), 10)

    def test_width_32(self) -> None:
        self._test_config(32, [2, 2, 2, 2], (3, 32, 32), 10)

    # --- Input size variations ---

    def test_input_32x32_no_maxpool(self) -> None:
        """32x32 input: no maxpool in stem (H < 64)."""
        self._test_config(16, [2, 2, 2, 2], (3, 32, 32), 10)

    def test_input_64x64_with_maxpool(self) -> None:
        """64x64 input: maxpool in stem (H >= 64)."""
        self._test_config(16, [2, 2, 2, 2], (3, 64, 64), 10)

    # --- num_classes variations ---

    def test_classes_10(self) -> None:
        self._test_config(16, [2, 2, 2, 2], (3, 32, 32), 10)

    def test_classes_100(self) -> None:
        self._test_config(16, [2, 2, 2, 2], (3, 32, 32), 100)

    def test_classes_1000(self) -> None:
        self._test_config(16, [2, 2, 2, 2], (3, 32, 32), 1000)

    # --- Combined variations ---

    def test_deep_wide_large_input(self) -> None:
        """R34 depth, wider, 64x64 input, 100 classes."""
        self._test_config(32, [3, 4, 6, 3], (3, 64, 64), 100)


if __name__ == "__main__":
    unittest.main()
