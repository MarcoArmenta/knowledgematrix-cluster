from typing import Union

import torch

from knowledgematrix.neural_net import NN


class GemmaLike(NN):
    """
        The Gemma-2 / Gemma-3 decoder, which differs from the Llama block in
        five ways that all matter numerically:

            x = x + post_attn_norm(attn(input_norm(x)))
            x = x + post_ff_norm(geglu(pre_ff_norm(x)))

        1. TWO norms per sublayer (Llama has one),
        2. RMSNorm weights are zero-centered: the norm scales by (1 + w),
        3. the MLP gate is GELU, not SiLU,
        4. attention logits are tanh soft-capped (Gemma-2) and scaled by
           query_pre_attn_scalar rather than sqrt(head_dim),
        5. the embedding output is multiplied by sqrt(hidden_size) -- folded
           into the embedding weights here, which is exactly equivalent.

        Args:
            vocab_size (int): Size of the vocabulary.
            hidden_size (int): Residual stream width.
            intermediate_size (int): GeGLU hidden width.
            num_hidden_layers (int): Number of decoder blocks.
            num_attention_heads (int): Query heads.
            num_key_value_heads (int): Key/value heads (GQA).
            head_dim (int): Per-head dimension.
            rope_theta (float): RoPE base frequency.
            rms_norm_eps (float): Epsilon of all RMS normalizations.
            attn_softcap (float | None): tanh soft-cap of attention logits.
            query_pre_attn_scalar (float | None): score divisor is its sqrt;
                defaults to head_dim.
            sliding_window (int | None): width of the local-attention layers.
            alternate_sliding (bool): Gemma-2 alternates local/global layers.
            include_lm_head (bool): Include the LM head.
            save (bool): Whether to save activations and preactivations.
            device (str): Device to run on.
    """

    #: rmsnorm, attn, rmsnorm, identity, rmsnorm, geglu, rmsnorm, identity
    BLOCK_STRIDE = 8

    def __init__(
            self,
            vocab_size: int = 256000,
            hidden_size: int = 512,
            intermediate_size: int = 1024,
            num_hidden_layers: int = 4,
            num_attention_heads: int = 8,
            num_key_value_heads: Union[int, None] = None,
            head_dim: Union[int, None] = None,
            rope_theta: float = 10000.0,
            rms_norm_eps: float = 1e-6,
            attn_softcap: Union[float, None] = None,
            query_pre_attn_scalar: Union[float, None] = None,
            sliding_window: Union[int, None] = None,
            alternate_sliding: bool = True,
            include_lm_head: bool = True,
            save: bool = False,
            device: str = "cpu"
        ) -> None:
        super().__init__(input_shape=(1, 1, hidden_size), save=save, device=device)

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        if head_dim is None:
            head_dim = hidden_size // num_attention_heads
        scale_src = query_pre_attn_scalar if query_pre_attn_scalar else head_dim

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.include_lm_head = include_lm_head
        self.embed_scale = float(hidden_size) ** 0.5

        self.embedding(vocab_size, hidden_size)

        for idx in range(num_hidden_layers):
            # Gemma-2 alternates: even layers local, odd layers global.
            window = (sliding_window
                      if sliding_window and (not alternate_sliding or idx % 2 == 0)
                      else None)

            start = self.get_num_layers()
            self.rmsnorm(hidden_size, eps=rms_norm_eps)
            self.ropeattention(
                d_model=hidden_size,
                num_heads=num_attention_heads,
                num_kv_heads=num_key_value_heads,
                head_dim=head_dim,
                rope_theta=rope_theta,
                softcap=attn_softcap,
                query_scale=float(scale_src) ** 0.5,
                sliding_window=window,
                rms_norm_eps=rms_norm_eps,
            )
            self.rmsnorm(hidden_size, eps=rms_norm_eps)   # post_attention
            end = self.get_num_layers()
            self.identity()
            self.residual(start, end)

            start = self.get_num_layers()
            self.rmsnorm(hidden_size, eps=rms_norm_eps)   # pre_feedforward
            self.geglu(hidden_size, intermediate_size)
            self.rmsnorm(hidden_size, eps=rms_norm_eps)   # post_feedforward
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
        ) -> "GemmaLike":
        """
            Build a GemmaLike NN from a Hugging Face Gemma-2/3 causal LM.

            NOTE: Gemma-2 also soft-caps the FINAL logits
            (final_logit_softcapping). That is applied after the LM head, so
            a network built with include_lm_head=True reproduces the
            pre-softcap logits. The alignment pipeline runs with
            include_lm_head=False and applies the head itself, where the cap
            has to be applied alongside it.
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
            rms_norm_eps=float(getattr(config, "rms_norm_eps", 1e-6)),
            attn_softcap=getattr(config, "attn_logit_softcapping", None),
            query_pre_attn_scalar=getattr(config, "query_pre_attn_scalar", None),
            sliding_window=getattr(config, "sliding_window", None),
            include_lm_head=include_lm_head,
            save=save,
            device=device,
        )

        def cast(t: torch.Tensor) -> torch.Tensor:
            t = t.detach().clone()
            return t.to(dtype) if dtype is not None else t

        def centered(t: torch.Tensor) -> torch.Tensor:
            # Gemma's RMSNorm scales by (1 + weight).
            return cast(t) + 1.0

        with torch.no_grad():
            layers = nn_model.layers
            # fold Gemma's sqrt(hidden_size) embedding scaling into the table
            layers[0].weight.data = cast(text.embed_tokens.weight) * nn_model.embed_scale

            i = 1
            for hf_layer in text.layers:
                src = hf_layer.self_attn
                layers[i].weight.data = centered(hf_layer.input_layernorm.weight)
                attn = layers[i + 1]
                attn.q_proj.weight.data = cast(src.q_proj.weight)
                attn.k_proj.weight.data = cast(src.k_proj.weight)
                attn.v_proj.weight.data = cast(src.v_proj.weight)
                attn.o_proj.weight.data = cast(src.o_proj.weight)
                layers[i + 2].weight.data = centered(hf_layer.post_attention_layernorm.weight)
                layers[i + 4].weight.data = centered(hf_layer.pre_feedforward_layernorm.weight)
                mlp = layers[i + 5]
                mlp.gate_proj.weight.data = cast(hf_layer.mlp.gate_proj.weight)
                mlp.up_proj.weight.data = cast(hf_layer.mlp.up_proj.weight)
                mlp.down_proj.weight.data = cast(hf_layer.mlp.down_proj.weight)
                layers[i + 6].weight.data = centered(hf_layer.post_feedforward_layernorm.weight)
                i += cls.BLOCK_STRIDE

            layers[i].weight.data = centered(text.norm.weight)
            if include_lm_head:
                src_w = lm_head.weight if lm_head is not None else text.embed_tokens.weight
                layers[i + 1].weight.data = cast(src_w)

        nn_model.to(device)
        return nn_model
