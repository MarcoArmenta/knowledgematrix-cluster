"""
    Implementation of GPT-2 for knowledge matrix computation.
"""
import torch
from torch import nn
from typing import Union

from knowledgematrix.neural_net import NN, MultiHeadAttention, LearnedPositionalEncoding


class GPT2(NN):
    """
        GPT-2 model (decoder-only transformer with pre-LayerNorm).

        Supports both training from scratch and loading pretrained weights from HuggingFace.
        When pretrained=True, weights are extracted from HuggingFace's GPT2LMHeadModel
        and converted into the NN framework (Conv1D -> Linear, c_attn split into Q/K/V).

        Architecture per block (pre-norm):
            LN -> MHA -> Dropout -> Residual -> LN -> FFN -> Dropout -> Residual

        Output is logits (no final softmax).

        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Dimension of the model.
            d_ff (int): Dimension of the feed-forward network.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer blocks.
            max_len (int): Maximum sequence length.
            dropout (float): Dropout probability.
            save (bool): Whether to save activations for knowledge matrix computation.
            pretrained (bool): Whether to load pretrained GPT-2 weights from HuggingFace.
            device (str): Device to run the network on.
    """
    def __init__(
            self,
            vocab_size: int = 50257,
            d_model: int = 768,
            d_ff: int = 3072,
            num_heads: int = 12,
            num_layers: int = 12,
            max_len: int = 1024,
            dropout: float = 0.1,
            save: bool = False,
            pretrained: bool = False,
            device: str = "cpu"
        ) -> None:
        super().__init__(input_shape=(1, 1, d_model), save=save, device=device)

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = d_model
        self.d_ff = d_ff
        self.max_len = max_len
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_head = d_model // num_heads

        causal_mask = torch.tril(torch.ones(max_len, max_len))

        if pretrained:
            self._build_pretrained(vocab_size, d_model, d_ff, num_heads, num_layers,
                                   max_len, dropout, causal_mask)
        else:
            self._build_from_scratch(vocab_size, d_model, d_ff, num_heads, num_layers,
                                     max_len, dropout, causal_mask)

    def _build_from_scratch(
            self,
            vocab_size: int,
            d_model: int,
            d_ff: int,
            num_heads: int,
            num_layers: int,
            max_len: int,
            dropout: float,
            causal_mask: torch.Tensor
        ) -> None:
        self.embedding(vocab_size, d_model)
        self.learned_positionalencoding(max_len, d_model)

        for _ in range(num_layers):
            # Attention sub-block
            start_skip = self.get_num_layers()
            self.layernorm(d_model)
            self.multiheadattention(d_model, num_heads, mask=causal_mask)
            self.dropout(dropout)
            end_skip = self.get_num_layers()
            self.residual(start_skip, end_skip)

            # Feed-forward sub-block
            start_skip = self.get_num_layers()
            self.layernorm(d_model)
            self.linear(in_features=d_model, out_features=d_ff)
            self.gelu(approximate="tanh")
            self.linear(in_features=d_ff, out_features=d_model)
            self.dropout(dropout)
            end_skip = self.get_num_layers()
            self.residual(start_skip, end_skip)

        # Final layer norm and output projection
        self.layernorm(d_model)
        self.linear(in_features=d_model, out_features=vocab_size)

    def _build_pretrained(
            self,
            vocab_size: int,
            d_model: int,
            d_ff: int,
            num_heads: int,
            num_layers: int,
            max_len: int,
            dropout: float,
            causal_mask: torch.Tensor
        ) -> None:
        from transformers import GPT2LMHeadModel

        hf_model = GPT2LMHeadModel.from_pretrained("gpt2")

        # Token embedding
        self.layers.append(hf_model.transformer.wte)

        # Learned positional encoding (wrapping HF's wpe)
        lpe = LearnedPositionalEncoding(max_len, d_model)
        lpe.pos_embedding = hf_model.transformer.wpe
        self.layers.append(lpe)

        for block in hf_model.transformer.h:
            # Attention sub-block: LN -> MHA -> Dropout -> Residual
            start_skip = self.get_num_layers()

            # LayerNorm 1
            self.layers.append(block.ln_1)

            # MultiHeadAttention with weights from HF's c_attn and c_proj
            mha = MultiHeadAttention(d_model, num_heads, mask=causal_mask)
            self._load_attention_weights(mha, block.attn)
            self.layers.append(mha)

            # Dropout
            self.dropout(dropout)

            end_skip = self.get_num_layers()
            self.residual(start_skip, end_skip)

            # Feed-forward sub-block: LN -> Linear -> GELU -> Linear -> Dropout -> Residual
            start_skip = self.get_num_layers()

            # LayerNorm 2
            self.layers.append(block.ln_2)

            # FFN first linear (Conv1D -> Linear)
            fc = nn.Linear(d_model, d_ff)
            fc.weight = nn.Parameter(block.mlp.c_fc.weight.T)
            fc.bias = nn.Parameter(block.mlp.c_fc.bias.clone())
            self.layers.append(fc)

            # GELU (tanh approximation to match HF)
            self.gelu(approximate="tanh")

            # FFN second linear (Conv1D -> Linear)
            proj = nn.Linear(d_ff, d_model)
            proj.weight = nn.Parameter(block.mlp.c_proj.weight.T)
            proj.bias = nn.Parameter(block.mlp.c_proj.bias.clone())
            self.layers.append(proj)

            # Dropout
            self.dropout(dropout)

            end_skip = self.get_num_layers()
            self.residual(start_skip, end_skip)

        # Final LayerNorm
        self.layers.append(hf_model.transformer.ln_f)

        # Output head (lm_head)
        self.layers.append(hf_model.lm_head)

    def block_boundaries(self, sub_blocks: bool = False) -> list:
        """Cut-point layer indices for segment KMs. Consecutive pairs are the
        segments: L transformer blocks, then the final LN + lm_head segment."""
        cuts = []
        for b in range(self.num_layers):
            cuts.append(2 + 8 * b)
            if sub_blocks:
                cuts.append(5 + 8 * b)
        cuts.append(2 + 8 * self.num_layers)
        cuts.append(self.get_num_layers())
        return cuts

    @staticmethod
    def _load_attention_weights(mha: MultiHeadAttention, hf_attn) -> None:
        """
            Load weights from HuggingFace GPT-2 attention into our MultiHeadAttention.

            HF uses Conv1D with weight shape (in_features, out_features) and computes x @ weight + bias.
            nn.Linear uses weight shape (out_features, in_features) and computes x @ weight.T + bias.
            HF's c_attn concatenates Q, K, V into one (d_model, 3*d_model) Conv1D.
        """
        d_model = mha.d_model

        # Split c_attn (concatenated Q/K/V) weights and biases
        # c_attn.weight shape: (d_model, 3*d_model) in Conv1D format
        c_attn_weight = hf_attn.c_attn.weight  # (d_model, 3*d_model)
        c_attn_bias = hf_attn.c_attn.bias      # (3*d_model,)

        # Split along the output dimension and transpose for nn.Linear
        mha.Q.weight = nn.Parameter(c_attn_weight[:, :d_model].T)
        mha.Q.bias = nn.Parameter(c_attn_bias[:d_model].clone())

        mha.K.weight = nn.Parameter(c_attn_weight[:, d_model:2*d_model].T)
        mha.K.bias = nn.Parameter(c_attn_bias[d_model:2*d_model].clone())

        mha.V.weight = nn.Parameter(c_attn_weight[:, 2*d_model:].T)
        mha.V.bias = nn.Parameter(c_attn_bias[2*d_model:].clone())

        # Output projection (c_proj)
        mha.O.weight = nn.Parameter(hf_attn.c_proj.weight.T)
        mha.O.bias = nn.Parameter(hf_attn.c_proj.bias.clone())
