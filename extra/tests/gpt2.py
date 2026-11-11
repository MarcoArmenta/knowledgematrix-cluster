#!/usr/bin/env python
import unittest
import torch
import random
import psutil
import gc
from time import time

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.models.gpt2 import GPT2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float64)

def get_memory_usage() -> float:
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


class TestGPT2ForwardFidelity(unittest.TestCase):
    """Test that pretrained GPT-2 forward pass matches HuggingFace."""

    def test_pretrained_forward_matches_hf(self) -> None:
        from transformers import GPT2LMHeadModel

        # Use float32 for this test since HF model is float32
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            with torch.no_grad():
                model = GPT2(pretrained=True)
                model.eval()

                hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
                hf_model.eval()

                # "Hello, my name is"
                x = torch.tensor([[[15496, 11, 616, 1438, 318]]])
                our_out = model(x)
                hf_out = hf_model(x.squeeze(0)).logits

                max_diff = (our_out.squeeze(0) - hf_out).abs().max().item()
                print(f"\nForward fidelity: max abs diff = {max_diff}")
                self.assertLess(max_diff, 1e-4,
                    f"Pretrained forward pass differs from HF by {max_diff}")
        finally:
            torch.set_default_dtype(prev_dtype)


class TestGPT2KMIdentity(unittest.TestCase):
    """Test knowledge matrix identity: KM.sum(1) == forward_pass."""

    def test_km_identity_from_scratch(self) -> None:
        """KM identity on a small from-scratch GPT-2."""
        for test_num in range(3):
            print(f"\n--- From-scratch KM test {test_num + 1}/3 ---")
            gc.collect()

            start_mem = get_memory_usage()
            start_time = time()

            vocab_size = random.randint(30, 60)
            d_heads = random.randint(3, 8)
            num_heads = random.randint(2, 4) * 2
            d_model = num_heads * d_heads
            d_ff = random.randint(32, 64)
            num_layers = random.randint(1, 3)
            seq_len = random.randint(3, 8)

            model = GPT2(
                vocab_size=vocab_size, d_model=d_model, d_ff=d_ff,
                num_heads=num_heads, num_layers=num_layers, max_len=32
            ).to(DEVICE)
            model.eval()

            x = torch.randint(0, vocab_size, (1, 1, seq_len))
            forward_pass = model(x)

            batch_size = random.randint(16, 32)
            mc = KnowledgeMatrixComputer(model, batch_size=batch_size)
            mat = mc.forward(x)

            diff = torch.norm(forward_pass.reshape(1, -1) - mat.sum(1)).item()
            rel_err = diff / torch.norm(forward_pass).item()

            end_time = time()
            end_mem = get_memory_usage()

            self.assertAlmostEqual(diff, 0, delta=0.1,
                msg=f"KM identity failed: diff={diff}, rel_err={rel_err}")

            print(f"  d_model={d_model}, heads={num_heads}, layers={num_layers}, "
                  f"seq={seq_len}, vocab={vocab_size}")
            print(f"  Diff: {diff:.2e}  Rel err: {rel_err:.2e}  "
                  f"Time: {end_time-start_time:.3f}s  Mem: {end_mem-start_mem:.1f}MB")

    def test_km_identity_pretrained_single_token(self) -> None:
        """KM identity on pretrained GPT-2 with a single token."""
        print("\n--- Pretrained KM test (single token) ---")
        gc.collect()

        start_time = time()
        model = GPT2(pretrained=True)
        model.eval()
        model.double()

        x = torch.tensor([[[15496]]])  # 'Hello'
        forward_pass = model(x)

        mc = KnowledgeMatrixComputer(model, batch_size=64)
        mat = mc.forward(x)

        diff = torch.norm(forward_pass.reshape(1, -1) - mat.sum(1)).item()
        rel_err = diff / torch.norm(forward_pass).item()

        end_time = time()

        # KM shape should be (50257, 769): 50257 logits, 768 embedding dims + 1 bias
        self.assertEqual(mat.shape[0], 50257)
        self.assertEqual(mat.shape[1], 769)

        self.assertAlmostEqual(diff, 0, delta=0.1,
            msg=f"Pretrained KM identity failed: diff={diff}")

        print(f"  KM shape: {mat.shape}")
        print(f"  Diff: {diff:.2e}  Rel err: {rel_err:.2e}  Time: {end_time-start_time:.1f}s")


class TestGPT2CausalMasking(unittest.TestCase):
    """Test that causal masking works correctly through the KM."""

    def test_causal_independence(self) -> None:
        """Changing a later token shouldn't affect earlier token logits."""
        print("\n--- Causal masking test ---")

        model = GPT2(
            vocab_size=100, d_model=64, d_ff=128,
            num_heads=4, num_layers=2, max_len=32
        )
        model.eval()

        # Two inputs that differ only at position 3
        x1 = torch.tensor([[[10, 20, 30, 40, 50]]])
        x2 = torch.tensor([[[10, 20, 30, 99, 50]]])

        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)

        # Logits at positions 0, 1, 2 should be identical
        for pos in range(3):
            diff = torch.norm(out1[0, 0, pos] - out2[0, 0, pos]).item()
            self.assertAlmostEqual(diff, 0, delta=1e-10,
                msg=f"Position {pos} differs by {diff} when only position 3 changed")
            print(f"  Position {pos}: diff = {diff:.2e} (should be ~0)")

        # Position 3 and 4 should differ
        diff_pos3 = torch.norm(out1[0, 0, 3] - out2[0, 0, 3]).item()
        self.assertGreater(diff_pos3, 0.01,
            msg=f"Position 3 should differ but diff is only {diff_pos3}")
        print(f"  Position 3: diff = {diff_pos3:.2e} (should be >0)")


if __name__ == "__main__":
    unittest.main()
