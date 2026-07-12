"""GPT-NeoX / Pythia adapter: parallel residual + partial rotary + LayerNorm + GELU."""
import torch
from torch import nn

from knowledgematrix.neural_net import NN, MultiHeadAttention


class GPTNeoX(NN):
    def __init__(self, vocab_size=50304, d_model=512, d_ff=2048, num_heads=8,
                 num_layers=6, max_len=2048, rotary_pct=0.25, save=False,
                 pretrained=False, hf_name="EleutherAI/pythia-70m",
                 revision=None, device="cpu"):
        super().__init__(input_shape=(1, 1, d_model), save=save, device=device)
        self.num_layers = num_layers
        causal = torch.tril(torch.ones(max_len, max_len))
        if pretrained:
            self._build_pretrained(hf_name, revision, causal)
        else:
            self._build_from_scratch(vocab_size, d_model, d_ff, num_heads,
                                     num_layers, rotary_pct, causal)

    def _add_block(self, d_model, d_ff, num_heads, rotary_pct, causal,
                   ln1=None, mha=None, ln2=None, fc=None, proj=None):
        start = self.get_num_layers()
        self.layers.append(ln1 if ln1 is not None else nn.LayerNorm(d_model))
        if mha is None:
            mha = MultiHeadAttention(d_model, num_heads, mask=causal, rotary_pct=rotary_pct)
        self.layers.append(mha)
        mid = self.get_num_layers()
        self.layers.append(ln2 if ln2 is not None else nn.LayerNorm(d_model))
        self.layers.append(fc if fc is not None else nn.Linear(d_model, d_ff))
        self.gelu()
        self.layers.append(proj if proj is not None else nn.Linear(d_ff, d_model))
        end = self.get_num_layers()
        self.parallel_blocks[end] = (start, mid)

    def _build_from_scratch(self, vocab_size, d_model, d_ff, num_heads,
                            num_layers, rotary_pct, causal):
        self.embedding(vocab_size, d_model)
        for _ in range(num_layers):
            self._add_block(d_model, d_ff, num_heads, rotary_pct, causal)
        self.layernorm(d_model)
        self.linear(d_model, vocab_size)

    def _build_pretrained(self, hf_name, revision, causal):
        from transformers import GPTNeoXForCausalLM
        hf = GPTNeoXForCausalLM.from_pretrained(hf_name, revision=revision)
        cfg = hf.config
        assert cfg.use_parallel_residual, "only the parallel-residual NeoX variant is supported"
        # transformers <5 exposes rotary_pct/rotary_emb_base on the config; v5 consolidates
        # them into the rope_parameters dict as partial_rotary_factor/rope_theta.
        rp = getattr(cfg, "rope_parameters", None)
        rp = rp if isinstance(rp, dict) else {}
        rotary_pct = getattr(cfg, "rotary_pct", None)
        if rotary_pct is None:
            rotary_pct = rp.get("partial_rotary_factor",
                                getattr(cfg, "partial_rotary_factor", 1.0))
        rope_base = getattr(cfg, "rotary_emb_base", None)
        if rope_base is None:
            rope_base = rp.get("rope_theta", getattr(cfg, "rope_theta", 10000))
        d, H = cfg.hidden_size, cfg.num_attention_heads
        dh = d // H
        self.num_layers = cfg.num_hidden_layers
        self.layers.append(hf.gpt_neox.embed_in)
        for block in hf.gpt_neox.layers:
            mha = MultiHeadAttention(d, H, mask=causal, rotary_pct=rotary_pct,
                                     rope_base=rope_base)
            # query_key_value packs per-head [q(dh) k(dh) v(dh)] along rows
            w = block.attention.query_key_value.weight.view(H, 3, dh, d)
            b = block.attention.query_key_value.bias.view(H, 3, dh)
            for j, name in enumerate(("Q", "K", "V")):
                lin = getattr(mha, name)
                lin.weight = nn.Parameter(w[:, j].reshape(d, d).clone())
                lin.bias = nn.Parameter(b[:, j].reshape(d).clone())
            mha.O.weight = nn.Parameter(block.attention.dense.weight.clone())
            mha.O.bias = nn.Parameter(block.attention.dense.bias.clone())
            self._add_block(d, cfg.intermediate_size, H, rotary_pct, causal,
                            ln1=block.input_layernorm, mha=mha,
                            ln2=block.post_attention_layernorm,
                            fc=block.mlp.dense_h_to_4h, proj=block.mlp.dense_4h_to_h)
        self.layers.append(hf.gpt_neox.final_layer_norm)
        self.layers.append(hf.embed_out)

    def block_boundaries(self) -> list:
        cuts = [1 + 6 * b for b in range(self.num_layers + 1)]
        cuts.append(self.get_num_layers())
        return cuts
