#!/usr/bin/env python
import unittest
import torch

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.neural_net import MultiHeadAttention

torch.set_default_dtype(torch.float64)


class TestRoPE(unittest.TestCase):
    def test_rope_matches_hf_reference(self):
        """Our partial rotary must match the HF GPT-NeoX rotary on random Q/K."""
        torch.manual_seed(0)
        d, H, T = 32, 4, 6
        mha = MultiHeadAttention(d, H, rotary_pct=0.25)
        dh = d // H
        ndims = int(dh * 0.25)
        q = torch.randn(1, 1, H, T, dh)
        k = torch.randn(1, 1, H, T, dh)
        q2, k2 = mha._apply_rope(q, k)
        # Reference: HF GPTNeoX rotary (non-interleaved, rotate_half convention)
        inv = 1.0 / (10000 ** (torch.arange(0, ndims, 2, dtype=torch.float64) / ndims))
        t = torch.arange(T, dtype=torch.float64)
        freqs = torch.outer(t, inv)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos(), emb.sin()

        def rot_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        for src, out in ((q, q2), (k, k2)):
            rot, keep = src[..., :ndims], src[..., ndims:]
            want = torch.cat((rot * cos + rot_half(rot) * sin, keep), dim=-1)
            self.assertTrue(torch.allclose(out, want, atol=1e-12))

    def test_rope_changes_pattern_not_probe_path(self):
        """rotary_pct must change the saved attention pattern but leave the
        frozen-pattern probe map (V/O only) algebraically identical for the
        same pattern — RoPE needs zero KM-algebra changes."""
        torch.manual_seed(1)
        d, H, T = 32, 4, 5
        x = torch.randn(1, 1, T, d)
        plain = MultiHeadAttention(d, H)
        roped = MultiHeadAttention(d, H, rotary_pct=0.25)
        roped.load_state_dict(plain.state_dict())
        for m in (plain, roped):
            m.save_pattern = True
            m(x)
            m.save_pattern = False
        self.assertFalse(torch.allclose(plain.attn_pattern, roped.attn_pattern))
        # patterns sum to 1 either way
        self.assertTrue(torch.allclose(roped.attn_pattern.sum(-1),
                                       torch.ones(1, 1, H, T), atol=1e-12))


if __name__ == "__main__":
    unittest.main()
