from typing import List, Optional, Sequence, Union

import torch

from knowledgematrix.neural_net import NN
from knowledgematrix.models._hf import (
    WeightLoader,
    empty_network,
    open_checkpoint,
    rope_settings,
)


class LlamaLike(NN):
    """
        The standard pre-norm decoder used by Llama, SmolLM2, Mistral,
        Qwen2.5 and the dense Qwen3 models:

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
        - ``rope_theta`` / ``rope_scaling``: Llama-3.x rotates with base
          500000 and the "llama3" frequency rescaling, Qwen3 with base 1e6,
          SmolLM2 with base 1e5 (135M, 360M) or 1.3e5 (1.7B),
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
            o_bias (bool): Bias on the attention output projection.
            qk_norm (bool): Per-head RMSNorm of queries and keys.
            sliding_window (int | None | list): Local attention width; one
                value for every block, or one entry per block (None = full
                causal attention in that block).
            include_lm_head (bool): Include the LM head; if False the network
                ends at the final RMSNorm and outputs hidden states.
            save (bool): Whether to save activations and preactivations.
            device (str): Device to run on.
            rope_scaling (dict | None): RoPE frequency rescaling, see
                RopeAttention (Llama-3.x: ``{"rope_type": "llama3", ...}``).
            mlp_bias (bool): Biases on the three SwiGLU projections.
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
            sliding_window: Union[int, None, Sequence[Union[int, None]]] = None,
            include_lm_head: bool = True,
            save: bool = False,
            device: str = "cpu",
            rope_scaling: Union[dict, None] = None,
            mlp_bias: bool = False,
        ) -> None:
        super().__init__(input_shape=(1, 1, hidden_size), save=save, device=device)

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        if head_dim is None:
            head_dim = hidden_size // num_attention_heads
        if sliding_window is None or isinstance(sliding_window, int):
            windows = [sliding_window] * num_hidden_layers
        else:
            windows = list(sliding_window)
            if len(windows) != num_hidden_layers:
                raise ValueError(
                    f"sliding_window has {len(windows)} entries but "
                    f"num_hidden_layers is {num_hidden_layers}."
                )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.include_lm_head = include_lm_head

        self.embedding(vocab_size, hidden_size)

        for window in windows:
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
                sliding_window=window,
                rms_norm_eps=rms_norm_eps,
                bias=qkv_bias,
                o_bias=o_bias,
                rope_scaling=rope_scaling,
            )
            end = self.get_num_layers()
            self.identity()
            self.residual(start, end, same_shape=True)

            # x = x + swiglu(rmsnorm(x))
            start = self.get_num_layers()
            self.rmsnorm(hidden_size, eps=rms_norm_eps)
            self.swiglu(hidden_size, intermediate_size, bias=mlp_bias)
            end = self.get_num_layers()
            self.identity()
            self.residual(start, end, same_shape=True)

        self.rmsnorm(hidden_size, eps=rms_norm_eps)
        if include_lm_head:
            self.linear(in_features=hidden_size, out_features=vocab_size, bias=False)

    @classmethod
    def from_huggingface(
            cls,
            source,
            include_lm_head: bool = True,
            dtype: Union[torch.dtype, None] = None,
            device: Union[str, Sequence[str]] = "cpu",
            save: bool = False,
            copy: bool = True,
            boundaries: Optional[List[int]] = None,
            **hub_kwargs,
        ) -> "LlamaLike":
        """
            Build a LlamaLike NN computing the same function as a Hugging
            Face causal LM of the Llama family: Llama-3.1 / 3.2 / 3.3,
            SmolLM2, Mistral, Qwen2.5 and the dense Qwen3 models.

            Args:
                source: one of
                    - a hub repository id or a local directory
                      ("meta-llama/Llama-3.3-70B-Instruct"): the weights
                      are streamed from the safetensors files one tensor
                      at a time, without instantiating the transformers
                      model, so host memory stays at one tensor;
                    - a transformers model already in memory.
                include_lm_head (bool): Include the LM head. Without it
                    the network ends at the final RMSNorm (hidden states).
                dtype: dtype of the network. Default: float32 for a
                    checkpoint read from disk, the model's own dtype for
                    a model in memory.
                device (str | list[str]): one device, or the stages of a
                    pipeline over several devices (["cuda:0", "cuda:1"])
                    for a model that does not fit one -- the network comes
                    out sharded, see NN.shard.
                save (bool): Whether to save activations.
                copy (bool): for a model in memory, copy its weights
                    (True) or share them (False, no extra memory).
                boundaries (list[int] | None): pipeline cut indices, see
                    NN.shard (default: balance the parameter count).
                **hub_kwargs: revision, local_files_only, token,
                    cache_dir -- for a hub repository id.
        """
        ckpt = open_checkpoint(source, **hub_kwargs)
        config = ckpt.text_config
        p = ckpt.text_prefix

        act = getattr(config, "hidden_act", "silu")
        if act != "silu":
            raise NotImplementedError(
                f"LlamaLike has a SiLU-gated MLP; this checkpoint uses {act!r}."
            )
        rope = rope_settings(config)
        if rope["partial_rotary_factor"] != 1.0:
            raise NotImplementedError("LlamaLike rotates the full head dimension.")

        num_layers = config.num_hidden_layers
        # Mistral applies its window in every block; Qwen2.5 / Qwen3 list
        # the blocks that use it in config.layer_types.
        window = getattr(config, "sliding_window", None)
        layer_types = getattr(config, "layer_types", None)
        if window is not None and layer_types is not None:
            windows = [window if t == "sliding_attention" else None for t in layer_types]
        else:
            windows = [window] * num_layers

        attn0 = "layers.0.self_attn."
        qkv_bias = ckpt.has_text(attn0 + "q_proj.bias")
        o_bias = ckpt.has_text(attn0 + "o_proj.bias")
        qk_norm = ckpt.has_text(attn0 + "q_norm.weight")
        mlp_bias = ckpt.has_text("layers.0.mlp.gate_proj.bias")

        net = empty_network(
            cls,
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=num_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=getattr(config, "num_key_value_heads", None)
            or config.num_attention_heads,
            head_dim=getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads,
            rope_theta=rope["theta"],
            rope_scaling=rope["scaling"],
            rms_norm_eps=float(getattr(config, "rms_norm_eps", 1e-5)),
            qkv_bias=qkv_bias,
            o_bias=o_bias,
            qk_norm=qk_norm,
            mlp_bias=mlp_bias,
            sliding_window=windows,
            include_lm_head=include_lm_head,
            save=save,
        )

        w = WeightLoader(net, ckpt, dtype=dtype, device=device,
                         boundaries=boundaries, copy=copy)
        w.put(0, "weight", p + "embed_tokens.weight")

        i = 1
        for n in range(num_layers):
            block = f"{p}layers.{n}."
            # NOTE: Llama-family RMSNorm weights are used as-is; the
            # zero-centered convention (weight + 1) is Qwen3.5's and
            # Gemma's, NOT this one.
            w.put(i, "weight", block + "input_layernorm.weight")
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                w.put(i + 1, f"{proj}.weight", block + f"self_attn.{proj}.weight")
            if qkv_bias:
                for proj in ("q_proj", "k_proj", "v_proj"):
                    w.put(i + 1, f"{proj}.bias", block + f"self_attn.{proj}.bias")
            if o_bias:
                w.put(i + 1, "o_proj.bias", block + "self_attn.o_proj.bias")
            if qk_norm:
                w.put(i + 1, "q_norm.weight", block + "self_attn.q_norm.weight")
                w.put(i + 1, "k_norm.weight", block + "self_attn.k_norm.weight")
            w.put(i + 3, "weight", block + "post_attention_layernorm.weight")
            for proj in ("gate_proj", "up_proj", "down_proj"):
                w.put(i + 4, f"{proj}.weight", block + f"mlp.{proj}.weight")
                if mlp_bias:
                    w.put(i + 4, f"{proj}.bias", block + f"mlp.{proj}.bias")
            i += cls.BLOCK_STRIDE

        w.put(i, "weight", p + "norm.weight")
        if include_lm_head:
            # tied checkpoints store no head: the embedding table is the head
            w.put(i + 1, "weight", ckpt.lm_head_key)
        return w.finish()
