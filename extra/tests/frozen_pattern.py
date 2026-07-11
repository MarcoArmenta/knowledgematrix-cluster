#!/usr/bin/env python
import unittest
import torch

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.models.gpt2 import GPT2
from knowledgematrix.neural_net import MultiHeadAttention

torch.set_default_dtype(torch.float64)


def small_gpt2(seed=0, num_layers=2, num_heads=4, d_model=24, d_ff=48, vocab=40):
    torch.manual_seed(seed)
    m = GPT2(vocab_size=vocab, d_model=d_model, d_ff=d_ff,
             num_heads=num_heads, num_layers=num_layers, max_len=16)
    m.eval()
    return m


class TestPatternSaving(unittest.TestCase):
    def test_patterns_and_stream_saved_on_save_pass(self):
        m = small_gpt2()
        x = torch.randint(0, 40, (1, 1, 4))
        m.save = True
        out = m(x)
        m.save = False
        T, H = 4, 4
        mha_idxs = [i for i, l in enumerate(m.layers) if isinstance(l, MultiHeadAttention)]
        self.assertEqual(len(mha_idxs), 2)
        for i in mha_idxs:
            p = m.layers[i].attn_pattern
            self.assertIsNotNone(p, f"no pattern saved at layer {i}")
            self.assertEqual(tuple(p.shape), (1, 1, H, T, T))
            # rows sum to 1 (softmax) and causal upper triangle is 0
            self.assertTrue(torch.allclose(p.sum(-1), torch.ones(1, 1, H, T), atol=1e-12))
            self.assertEqual(p[..., torch.triu(torch.ones(T, T), 1) == 1].abs().max().item(), 0.0)
        # stream: cut 2 equals the post-embedding value
        emb = m.layers[1](m.layers[0](x))
        if emb.dim() == 3:
            emb = emb.unsqueeze(0)
        self.assertTrue(torch.allclose(m.stream[2], emb, atol=1e-12))
        # stream saved at every processed index
        for i in range(2, m.get_num_layers()):
            self.assertIsNotNone(m.stream[i], f"stream missing at {i}")

    def test_no_pattern_saved_on_plain_forward(self):
        m = small_gpt2()
        x = torch.randint(0, 40, (1, 1, 4))
        m(x)  # save=False
        mha_idxs = [i for i, l in enumerate(m.layers) if isinstance(l, MultiHeadAttention)]
        for i in mha_idxs:
            self.assertIsNone(m.layers[i].attn_pattern)


if __name__ == "__main__":
    unittest.main()
