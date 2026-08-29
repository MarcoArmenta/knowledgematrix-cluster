from typing import List, Union

import torch

from knowledgematrix.neural_net import NN


class LlamaLike(NN):
    """
        The standard pre-norm decoder used by Llama, Mistral, Qwen2.5 and
        the dense Qwen3 models:

            x = x + attn(rmsnorm(x))
            x = x + swiglu(rmsnorm(x))

        with grouped-query attention and rotary position embeddings. It is
        strictly simpler than Qwen3_5 -- no linear-attention recurrence, no
        attention output gate -- so every layer is either a plain Linear, an
        RMSNorm, or one of the two linearizable blocks, and the knowledge
        matrix machinery applies unchanged.

        Family differences are flags rather than subclasses:
        - ``qkv_bias``: Qwen2.5 has biases on q/k/v, Llama and Mistral do not,
        - ``qk_norm``: Qwen3 RMS-normalizes q and k per head, Llama does not,
        - ``sliding_window``: Mistral's local attention (None = full causal).

        Args:
            vocab_size (int): Size of the vocabulary.
            hidden_size (int): Residual stream width.
            intermediate_size (int): SwiGLU hidden width.
            num_hidden_layers (int): Number of decoder blocks.
            num_attention_heads (int): Query heads.
            num_key_value_heads (int): Key/value heads (GQA).
            head_dim (int): Per-head dimension.
            rope_theta (float): RoPE base frequency.
            rms_norm_eps (float): Epsilon of all RMS normalizations.
            qkv_bias (bool): Biases on the q/k/v projections.
            qk_norm (bool): Per-head RMSNorm of queries and keys.
            sliding_window (int | None): Local attention width.
            include_lm_head (bool): Include the LM head; if False the network
                ends at the final RMSNorm and outputs hidden states.
            save (bool): Whether to save activations and preactivations.
            device (str): Device to run on.
    """

    #: layers appended per decoder block (rmsnorm, attn, identity,
    #: rmsnorm, swiglu, identity) -- the stride used when copying weights
    BLOCK_STRIDE = 6

    def __init__(
            self,
            vocab_size: int = 32000,
            hidden_size: int = 512,
            intermediate_size: int = 1024,
            num_hidden_layers: int = 4,
            num_attention_heads: int = 8,
            num_key_value_heads: Union[int, None] = None,
            head_dim: Union[int, None] = None,
            rope_theta: float = 10000.0,
            rms_norm_eps: float = 1e-5,
            qkv_bias: bool = False,
            o_bias: bool = False,
            qk_norm: bool = False,
            sliding_window: Union[int, None] = None,
            include_lm_head: bool = True,
            save: bool = False,
            device: str = "cpu"
        ) -> None:
        super().__init__(input_shape=(1, 1, hidden_size), save=save, device=device)

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        if head_dim is None:
            head_dim = hidden_size // num_attention_heads

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.include_lm_head = include_lm_head

        self.embedding(vocab_size, hidden_size)

        for _ in range(num_hidden_layers):
            # x = x + attn(rmsnorm(x))
            start = self.get_num_layers()
            self.rmsnorm(hidden_size, eps=rms_norm_eps)
            self.ropeattention(
                d_model=hidden_size,
                num_heads=num_attention_heads,
                num_kv_heads=num_key_value_heads,
                head_dim=head_dim,
                rope_theta=rope_theta,
                qk_norm=qk_norm,
                sliding_window=sliding_window,
                rms_norm_eps=rms_norm_eps,
                bias=qkv_bias,
                o_bias=o_bias,
            )
            end = self.get_num_layers()
            self.identity()
            self.residual(start, end)

            # x = x + swiglu(rmsnorm(x))
            start = self.get_num_layers()
            self.rmsnorm(hidden_size, eps=rms_norm_eps)
            self.swiglu(hidden_size, intermediate_size)
            end = self.get_num_layers()
            self.identity()
            self.residual(start, end)

        self.rmsnorm(hidden_size, eps=rms_norm_eps)
        if include_lm_head:
            self.linear(in_features=hidden_size, out_features=vocab_size, bias=False)

    @classmethod
    def from_huggingface(
            cls,
            source,
            include_lm_head: bool = True,
            dtype: Union[torch.dtype, None] = None,
            device: str = "cpu",
            save: bool = False,
        ) -> "LlamaLike":
        """
            Build a LlamaLike NN from a Hugging Face causal-LM (Llama,
            Mistral, Qwen2.5, dense Qwen3) and copy its weights, so both
            compute the same function.

            Args:
                source: a model id / path, or an already-loaded HF model.
                include_lm_head (bool): Include the LM head.
                dtype: Cast the copied weights to this dtype.
                device (str): Device of the resulting network.
                save (bool): Whether to save activations.
        """
        if isinstance(source, str):
            from transformers import AutoModelForCausalLM
            hf = AutoModelForCausalLM.from_pretrained(source)
        else:
            hf = source

        config = hf.config
        text = hf.model if hasattr(hf, "model") else hf
        if hasattr(text, "language_model"):
            text = text.language_model
        lm_head = getattr(hf, "lm_head", None)

        first_attn = text.layers[0].self_attn
        qkv_bias = getattr(first_attn.q_proj, "bias", None) is not None
        o_bias = getattr(first_attn.o_proj, "bias", None) is not None
        qk_norm = hasattr(first_attn, "q_norm") and first_attn.q_norm is not None

        nn_model = cls(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=getattr(config, "num_key_value_heads",
                                        config.num_attention_heads),
            head_dim=getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads,
            rope_theta=float(getattr(config, "rope_theta", 10000.0)),
            rms_norm_eps=float(getattr(config, "rms_norm_eps", 1e-5)),
            qkv_bias=qkv_bias,
            o_bias=o_bias,
            qk_norm=qk_norm,
            sliding_window=getattr(config, "sliding_window", None),
            include_lm_head=include_lm_head,
            save=save,
            device=device,
        )

        def cast(t: torch.Tensor) -> torch.Tensor:
            t = t.detach().clone()
            return t.to(dtype) if dtype is not None else t

        with torch.no_grad():
            layers = nn_model.layers
            layers[0].weight.data = cast(text.embed_tokens.weight)

            i = 1
            for hf_layer in text.layers:
                # NOTE: Llama-family RMSNorm weights are used as-is; the
                # zero-centered convention (weight + 1) is Qwen3.5's and
                # Gemma's, NOT this one.
                layers[i].weight.data = cast(hf_layer.input_layernorm.weight)
                attn = layers[i + 1]
                src = hf_layer.self_attn
                attn.q_proj.weight.data = cast(src.q_proj.weight)
                attn.k_proj.weight.data = cast(src.k_proj.weight)
                attn.v_proj.weight.data = cast(src.v_proj.weight)
                attn.o_proj.weight.data = cast(src.o_proj.weight)
                if qkv_bias:
                    attn.q_proj.bias.data = cast(src.q_proj.bias)
                    attn.k_proj.bias.data = cast(src.k_proj.bias)
                    attn.v_proj.bias.data = cast(src.v_proj.bias)
                if o_bias:
                    attn.o_proj.bias.data = cast(src.o_proj.bias)
                if qk_norm:
                    attn.q_norm.weight.data = cast(src.q_norm.weight)
                    attn.k_norm.weight.data = cast(src.k_norm.weight)
                layers[i + 3].weight.data = cast(hf_layer.post_attention_layernorm.weight)
                mlp = layers[i + 4]
                mlp.gate_proj.weight.data = cast(hf_layer.mlp.gate_proj.weight)
                mlp.up_proj.weight.data = cast(hf_layer.mlp.up_proj.weight)
                mlp.down_proj.weight.data = cast(hf_layer.mlp.down_proj.weight)
                i += cls.BLOCK_STRIDE

            layers[i].weight.data = cast(text.norm.weight)
            if include_lm_head:
                if lm_head is not None:
                    layers[i + 1].weight.data = cast(lm_head.weight)
                else:
                    layers[i + 1].weight.data = cast(text.embed_tokens.weight)

        nn_model.to(device)
        return nn_model
