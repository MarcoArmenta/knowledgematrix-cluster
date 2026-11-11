#!/usr/bin/env python
"""
    Tests for the save_to_disk=False parameter on DatasetComputer and ExperimentRunner.
"""
import os
import sys
import tempfile
import shutil
import unittest

import torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from knowledgematrix.neural_net import NN
from knowledgematrix.dataset_computer import DatasetComputer
from knowledgematrix.experiment_runner import ExperimentRunner

DEVICE = "cpu"
torch.set_default_dtype(torch.float64)


class SmallMLP(NN):

    def __init__(
            self,
            input_shape: tuple[int],
            num_classes: int,
            save: bool = False,
            device: str = "cpu"
    ) -> None:
        super().__init__(input_shape, save, device)
        self.flatten()
        self.linear(in_features=self.get_input_size(), out_features=32)
        self.relu()
        self.linear(in_features=32, out_features=num_classes)


class TestDatasetComputerSaveToDisk(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.input_shape = (1, 4, 4)
        self.num_classes = 3
        self.model = SmallMLP(self.input_shape, self.num_classes).to(DEVICE)
        self.model.eval()
        self.data = [torch.randn(self.input_shape) for _ in range(5)]

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_save_to_disk_false_no_files(self) -> None:
        """When save_to_disk=False, no .pt files should be created."""
        output_dir = os.path.join(self.tmp_dir, "matrices")
        computer = DatasetComputer(self.model, batch_size=16, device=DEVICE)
        computer.compute(self.data, output_dir, save_to_disk=False)

        # Directory should not be created when save_to_disk=False
        if os.path.exists(output_dir):
            pt_files = [f for f in os.listdir(output_dir) if f.endswith(".pt")]
            self.assertEqual(pt_files, [], "No .pt files should exist with save_to_disk=False")

    def test_save_to_disk_false_computation_runs(self) -> None:
        """Verify that computation actually happens (no crash) with save_to_disk=False."""
        output_dir = os.path.join(self.tmp_dir, "matrices")
        computer = DatasetComputer(self.model, batch_size=16, device=DEVICE)
        # Should not raise
        computer.compute(self.data, output_dir, save_to_disk=False)

    def test_save_to_disk_true_default(self) -> None:
        """Verify backward compatibility: default save_to_disk=True still saves."""
        output_dir = os.path.join(self.tmp_dir, "matrices")
        computer = DatasetComputer(self.model, batch_size=16, device=DEVICE)
        computer.compute(self.data, output_dir)

        for i in range(5):
            filepath = os.path.join(output_dir, f"sample_{i}.pt")
            self.assertTrue(os.path.exists(filepath), f"sample_{i}.pt not found")


class TestExperimentRunnerSaveToDisk(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.input_shape = (1, 4, 4)
        self.num_classes = 3
        self.model = SmallMLP(self.input_shape, self.num_classes).to(DEVICE)
        self.data = [torch.randn(self.input_shape) for _ in range(5)]

        # Save a dummy weight checkpoint
        weights_dir = os.path.join(self.tmp_dir, "weights")
        os.makedirs(weights_dir)
        torch.save(
            self.model.state_dict(),
            os.path.join(weights_dir, "epoch_0.pt")
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_run_save_to_disk_false_no_files(self) -> None:
        """When save_to_disk=False, no .pt or .tar.gz should be created in matrices/."""
        runner = ExperimentRunner(self.model, self.tmp_dir, batch_size=16, device=DEVICE)
        runner.run(self.data, save_to_disk=False)

        matrix_dir = os.path.join(self.tmp_dir, "matrices", "epoch_0")
        if os.path.exists(matrix_dir):
            pt_files = [f for f in os.listdir(matrix_dir) if f.endswith(".pt")]
            tar_files = [f for f in os.listdir(matrix_dir) if f.endswith(".tar.gz")]
            self.assertEqual(pt_files, [], "No .pt files should exist")
            self.assertEqual(tar_files, [], "No .tar.gz files should exist")

    def test_run_save_to_disk_false_computation_runs(self) -> None:
        """Verify that run() completes without error when save_to_disk=False."""
        runner = ExperimentRunner(self.model, self.tmp_dir, batch_size=16, device=DEVICE)
        # Should not raise
        runner.run(self.data, save_to_disk=False)

    def test_run_single_save_to_disk_false(self) -> None:
        """Test run_single() with save_to_disk=False."""
        runner = ExperimentRunner(self.model, self.tmp_dir, batch_size=16, device=DEVICE)
        weight_path = os.path.join(self.tmp_dir, "weights", "epoch_0.pt")
        runner.run_single(self.data, weight_path, save_to_disk=False)

        matrix_dir = os.path.join(self.tmp_dir, "matrices", "epoch_0")
        if os.path.exists(matrix_dir):
            tar_files = [f for f in os.listdir(matrix_dir) if f.endswith(".tar.gz")]
            self.assertEqual(tar_files, [], "No archive should exist with save_to_disk=False")


if __name__ == "__main__":
    unittest.main()
