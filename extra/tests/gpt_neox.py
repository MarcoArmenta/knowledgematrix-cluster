#!/usr/bin/env python
import unittest
from functools import reduce
import torch

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer, compose
from knowledgematrix.models.gpt_neox import GPTNeoX

torch.set_default_dtype(torch.float64)


class TestFromScratch(unittest.TestCase):
    def test_km_identity_and_composition(self):
        for seed, T in [(0, 4), (1, 6)]:
            torch.manual_seed(seed)
            m = GPTNeoX(vocab_size=40, d_model=24, d_ff=48, num_heads=4,
                        num_layers=2, max_len=16)
            m.eval()
            x = torch.randint(0, 40, (1, 1, T))
            for mode in ("monolithic", "frozen_pattern"):
                mc = KnowledgeMatrixComputer(m, batch_size=16, attention_mode=mode)
                mat = mc.forward(x)
                diff = torch.norm(mc.current_output.reshape(-1) - mat.sum(1)).item()
                self.assertLess(diff, 1e-10, f"{mode} identity: {diff}")
                cuts = m.block_boundaries()
                segs = [mc.segment(x, a, b) for a, b in zip(cuts[:-1], cuts[1:])]
                prod = reduce(compose, reversed(segs))
                self.assertLess((prod - mat).abs().max().item(), 1e-8, mode)

    def test_boundaries(self):
        m = GPTNeoX(vocab_size=40, d_model=24, d_ff=48, num_heads=4,
                    num_layers=2, max_len=16)
        # [0]=emb; per block 6 layers from 1; final LN + head
        self.assertEqual(m.block_boundaries(), [1, 7, 13, 15])

    def test_causality(self):
        torch.manual_seed(2)
        m = GPTNeoX(vocab_size=40, d_model=24, d_ff=48, num_heads=4,
                    num_layers=2, max_len=16)
        m.eval()
        x1 = torch.tensor([[[1, 2, 3, 4]]])
        x2 = torch.tensor([[[1, 2, 3, 9]]])
        with torch.no_grad():
            o1, o2 = m(x1), m(x2)
        for pos in range(3):
            self.assertLess((o1[0, 0, pos] - o2[0, 0, pos]).abs().max().item(), 1e-10)


class TestPretrainedPythia70m(unittest.TestCase):
    def test_forward_matches_hf(self):
        from transformers import GPTNeoXForCausalLM
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            with torch.no_grad():
                m = GPTNeoX(pretrained=True, hf_name="EleutherAI/pythia-70m")
                m.eval()
                hf = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-70m")
                hf.eval()
                x = torch.tensor([[[3856, 11, 619, 1416, 310]]])
                ours = m(x).squeeze(0)
                theirs = hf(x.squeeze(0)).logits
                diff = (ours - theirs).abs().max().item()
                self.assertLess(diff, 1e-4, f"HF fidelity: {diff}")
        finally:
            torch.set_default_dtype(prev)

    def test_km_identity_pretrained_short(self):
        m = GPTNeoX(pretrained=True, hf_name="EleutherAI/pythia-70m")
        m.eval()
        m.double()
        x = torch.tensor([[[3856, 11]]])          # T=2 keeps this fast on CPU
        mc = KnowledgeMatrixComputer(m, batch_size=64, attention_mode="frozen_pattern")
        mat = mc.forward(x)
        diff = torch.norm(mc.current_output.reshape(-1) - mat.sum(1)).item()
        self.assertEqual(mat.shape[1], 2 * 512 + 1)
        self.assertLess(diff, 1e-10, f"pretrained identity: {diff}")


if __name__ == "__main__":
    unittest.main()
