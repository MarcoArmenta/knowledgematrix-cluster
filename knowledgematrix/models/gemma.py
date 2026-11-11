from typing import List, Optional, Sequence, Union

import torch

from knowledgematrix.neural_net import NN
from knowledgematrix.models._hf import (
    WeightLoader,
    empty_network,
    open_checkpoint,
    rope_settings,
)

#: the two kinds of decoder block (the names transformers uses in
#: `config.layer_types`)
SLIDING = "sliding_attention"
FULL = "full_attention"

#: `config.hidden_activation` -> GeGLU(approximate=...)
_GELU_VARIANTS = {"gelu_pytorch_tanh": "tanh", "gelu": "none"}


class GemmaLike(NN):
    """
        The Gemma-2 / Gemma-3 decoder. Its block differs from the Llama
        block in ways that all matter numerically:

            x = x + post_attn_norm(attn(input_norm(x)))
            x = x + post_ff_norm(geglu(pre_ff_norm(x)))

        1. TWO norms per sublayer (Llama has one),
        2. RMSNorm weights are zero-centered: the norm scales by (1 + w),
        3. the MLP gate is GELU (tanh approximation), not SiLU,
        4. attention scores are divided by sqrt(query_pre_attn_scalar),
           which is NOT sqrt(head_dim) in general (Gemma-2 27b: 144 with
           head_dim 128; Gemma-3 27b: 168 with head_dim 128),
        5. the embedding output is multiplied by sqrt(hidden_size) -- folded
           into the embedding weights here, which is exactly equivalent,
        6. local (sliding window) and global (full causal) blocks are
           mixed: a local block lets a position attend to itself and the
           sliding_window - 1 positions before it.

        The two generations then differ from each other:

        ===================  ==========================  ==========================
        .                    Gemma-2                     Gemma-3
        ===================  ==========================  ==========================
        block pattern        local, global, local, ...   5 local, 1 global, ...
        attention logits     tanh soft-capped (50)       not capped
        queries and keys     as projected                RMS-normalized per head
        RoPE base            1e4 everywhere              1e4 local / 1e6 global,
                                                         global frequencies / 8 in
                                                         the 4b, 12b and 27b
        final logits         tanh soft-capped (30)       not capped
        ===================  ==========================  ==========================

        All of it is expressed by constructor options, so one class covers
        both; `from_huggingface` reads them from the checkpoint's config.

        Args:
            vocab_size (int): Size of the vocabulary.
            hidden_size (int): Residual stream width.
            intermediate_size (int): GeGLU hidden width.
            num_hidden_layers (int): Number of decoder blocks.
            num_attention_heads (int): Query heads.
            num_key_value_heads (int): Key/value heads (GQA).
            head_dim (int): Per-head dimension.
            rope_theta (float): RoPE base frequency (of the global blocks
                when `rope_local_theta` is given, of all blocks otherwise).
            rms_norm_eps (float): Epsilon of all RMS normalizations.
            attn_softcap (float | None): tanh soft-cap of attention logits.
            query_pre_attn_scalar (float | None): score divisor is its sqrt;
                defaults to head_dim.
            sliding_window (int | None): width of the local-attention layers.
            alternate_sliding (bool): Gemma-2 alternates local/global layers
                (ignored when `layer_types` is given).
            include_lm_head (bool): Include the LM head.
            save (bool): Whether to save activations and preactivations.
            device (str): Device to run on.
            layer_types (Sequence[str] | None): the kind of every block,
                "sliding_attention" or "full_attention" (Gemma-3: five
                local blocks then one global, repeating). None derives the
                pattern from `sliding_window` and `alternate_sliding`.
            qk_norm (bool): RMS-normalize queries and keys per head
                (Gemma-3).
            rope_scaling (dict | None): RoPE frequency rescaling going with
                `rope_theta`, see RopeAttention (Gemma-3 4b/12b/27b:
                ``{"rope_type": "linear", "factor": 8.0}``).
            rope_local_theta (float | None): RoPE base of the local blocks
                when they rotate differently from the global ones
                (Gemma-3: 1e4 against rope_theta = 1e6). None: all blocks
                share `rope_theta` and `rope_scaling`.
            rope_local_scaling (dict | None): frequency rescaling of the
                local blocks; only used together with `rope_local_theta`.
            final_softcap (float | None): tanh soft-cap of the logits
                (Gemma-2: 30), appended as a SoftCap layer after the LM
                head, so the network outputs -- and the rows of its
                knowledge matrix sum to -- the capped logits. None: the
                network ends at the LM head. Has no effect without the
                LM head.
            attention_bias (bool): Biases on the four attention projections.
            hidden_activation (str): the GeGLU gate, "gelu_pytorch_tanh"
                (every published Gemma) or "gelu" (exact erf).
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
            device: str = "cpu",
            layer_types: Union[Sequence[str], None] = None,
            qk_norm: bool = False,
            rope_scaling: Union[dict, None] = None,
            rope_local_theta: Union[float, None] = None,
            rope_local_scaling: Union[dict, None] = None,
            final_softcap: Union[float, None] = None,
            attention_bias: bool = False,
            hidden_activation: str = "gelu_pytorch_tanh",
        ) -> None:
        super().__init__(input_shape=(1, 1, hidden_size), save=save, device=device)

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        if head_dim is None:
            head_dim = hidden_size // num_attention_heads
        scale_src = query_pre_attn_scalar if query_pre_attn_scalar else head_dim
        if hidden_activation not in _GELU_VARIANTS:
            raise NotImplementedError(
                f"GemmaLike has a GELU-gated MLP ({', '.join(_GELU_VARIANTS)}); "
                f"got hidden_activation={hidden_activation!r}."
            )
        if rope_local_theta is None and rope_local_scaling is not None:
            raise ValueError("rope_local_scaling needs rope_local_theta.")

        if layer_types is None:
            # Gemma-2 alternates: even layers local, odd layers global.
            layer_types = [
                SLIDING if sliding_window and (not alternate_sliding or idx % 2 == 0) else FULL
                for idx in range(num_hidden_layers)
            ]
        else:
            layer_types = list(layer_types)
            if len(layer_types) != num_hidden_layers:
                raise ValueError(
                    f"layer_types has {len(layer_types)} entries but "
                    f"num_hidden_layers is {num_hidden_layers}."
                )
            unknown = sorted(set(layer_types) - {SLIDING, FULL})
            if unknown:
                raise NotImplementedError(
                    f"unsupported layer types {unknown}; GemmaLike has "
                    f"{SLIDING!r} and {FULL!r} blocks."
                )
            if SLIDING in layer_types and not sliding_window:
                raise ValueError("layer_types has sliding_attention blocks but sliding_window is not set.")

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.include_lm_head = include_lm_head
        self.embed_scale = float(hidden_size) ** 0.5
        self.layer_types = layer_types
        self.final_softcap = float(final_softcap) if (final_softcap and include_lm_head) else None

        self.embedding(vocab_size, hidden_size)

        for layer_type in layer_types:
            local = layer_type == SLIDING
            if local and rope_local_theta is not None:
                theta, scaling = rope_local_theta, rope_local_scaling
            else:
                theta, scaling = rope_theta, rope_scaling

            start = self.get_num_layers()
            self.rmsnorm(hidden_size, eps=rms_norm_eps)
            self.ropeattention(
                d_model=hidden_size,
                num_heads=num_attention_heads,
                num_kv_heads=num_key_value_heads,
                head_dim=head_dim,
                rope_theta=theta,
                qk_norm=qk_norm,
                softcap=attn_softcap,
                query_scale=float(scale_src) ** 0.5,
                sliding_window=sliding_window if local else None,
                rms_norm_eps=rms_norm_eps,
                bias=attention_bias,
                o_bias=attention_bias,
                rope_scaling=scaling,
            )
            self.rmsnorm(hidden_size, eps=rms_norm_eps)   # post_attention
            end = self.get_num_layers()
            self.identity()
            self.residual(start, end, same_shape=True)

            start = self.get_num_layers()
            self.rmsnorm(hidden_size, eps=rms_norm_eps)   # pre_feedforward
            self.geglu(hidden_size, intermediate_size,
                       approximate=_GELU_VARIANTS[hidden_activation])
            self.rmsnorm(hidden_size, eps=rms_norm_eps)   # post_feedforward
            end = self.get_num_layers()
            self.identity()
            self.residual(start, end, same_shape=True)

        self.rmsnorm(hidden_size, eps=rms_norm_eps)
        if include_lm_head:
            self.linear(in_features=hidden_size, out_features=vocab_size, bias=False)
            if self.final_softcap is not None:
                self.softcap(self.final_softcap)

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
            *,
            final_softcap: bool = True,
            embed_scale: Union[float, None] = None,
            **hub_kwargs,
        ) -> "GemmaLike":
        """
            Build a GemmaLike NN computing the same function as a Hugging
            Face Gemma-2 or Gemma-3 causal LM: google/gemma-2-{2b,9b,27b},
            google/gemma-3-{270m,1b,4b,12b,27b} and the checkpoints of the
            same architecture (-it / -pt variants, fine-tunes).

            The reference is the transformers implementation with
            `attn_implementation="eager"`. (With its default, "sdpa",
            transformers silently skips the attention soft-capping of
            Gemma-2 -- the `softcap` argument is not passed on to
            torch's scaled_dot_product_attention -- so that mode is not
            the Gemma-2 architecture.)

            The Gemma-3 4b/12b/27b checkpoints are multimodal
            (Gemma3ForConditionalGeneration): the network built here is
            their text decoder, i.e. the function they compute on a
            text-only input. The vision tower is not loaded.

            Args:
                source: one of
                    - a hub repository id or a local directory
                      ("google/gemma-3-27b-it"): the weights are streamed
                      from the safetensors files one tensor at a time,
                      without instantiating the transformers model;
                    - a transformers model already in memory.
                include_lm_head (bool): Include the LM head. Without it
                    the network ends at the final RMSNorm (hidden states).
                dtype: dtype of the network. Default: float32 for a
                    checkpoint read from disk, the model's own dtype for
                    a model in memory.
                device (str | list[str]): one device, or the stages of a
                    pipeline over several devices (the network comes out
                    sharded, see NN.shard); "meta" is a dry run that
                    checks every shape without storing anything.
                save (bool): Whether to save activations.
                copy (bool): for a model in memory, copy its weights
                    (True) or share them (False). Sharing applies to the
                    projection matrices and the LM head; the embedding
                    table (rescaled by sqrt(hidden_size)) and the norm
                    weights (1 + w) are new tensors either way.
                boundaries (list[int] | None): pipeline cut indices, see
                    NN.shard (default: balance the parameter count).
                final_softcap (bool): Gemma-2 soft-caps its logits,
                    logits -> c * tanh(logits / c) with c =
                    config.final_logit_softcapping = 30. True (default)
                    appends that as a SoftCap layer after the LM head, so
                    the network reproduces the reference logits and the
                    rows of its knowledge matrix sum to them. False stops
                    at the LM head: the uncapped logits. Checkpoints
                    without a cap (Gemma-3) and `include_lm_head=False`
                    are unaffected.
                embed_scale (float | None): the factor the embedding
                    output is multiplied by (folded into the table). None
                    (default) is the factor of the reference, which is
                    sqrt(hidden_size) ROUNDED TO THE DTYPE THE MODEL IS
                    LOADED IN: transformers keeps it in a tensor created
                    under that dtype (and the original Gemma code does
                    the same on purpose), so for gemma-3-4b it is 50.596
                    in float32 and 50.5 in bfloat16 -- 0.2% apart, which
                    is visible in the logits. For a model in memory the
                    factor is read from the model itself; for a
                    checkpoint on disk it is sqrt(hidden_size) rounded to
                    `dtype`, i.e. what `from_pretrained(..., dtype=dtype)`
                    multiplies by. Pass a number to pin it, e.g. the
                    bfloat16 value for a float32 network that is to be
                    compared with a reference loaded in bfloat16 (the
                    transformers default). The value used is stored in
                    `net.embed_scale`.
                **hub_kwargs: revision, local_files_only, token,
                    cache_dir -- for a hub repository id.
            Raises:
                NotImplementedError: when the config asks for something
                    this network does not reproduce (bidirectional
                    attention, an unsupported RoPE type or activation, a
                    wrapper other than Gemma-3's, ...). Nothing is
                    approximated silently.
        """
        ckpt = open_checkpoint(source, **hub_kwargs)
        config = ckpt.text_config
        arch = cls._read_architecture(ckpt)

        num_layers = arch["num_hidden_layers"]
        p = ckpt.text_prefix

        cap = arch.pop("final_softcap")
        net = empty_network(
            cls,
            include_lm_head=include_lm_head,
            final_softcap=cap if final_softcap else None,
            save=save,
            **arch,
        )

        w = WeightLoader(net, ckpt, dtype=dtype, device=device,
                         boundaries=boundaries, copy=copy)
        used = set()

        def put(layer: int, attr: str, key: str, transform=None) -> None:
            w.put(layer, attr, key, transform=transform)
            used.add(key)

        def centered(t: torch.Tensor) -> torch.Tensor:
            # Gemma's RMSNorm scales by (1 + weight)
            return 1.0 + t

        # The reference multiplies the embedding output by a tensor holding
        # sqrt(hidden_size), cast to the dtype of the table at every call.
        # Folding the factor into the table reproduces that product exactly
        # when both operands are rounded the way the reference rounds them
        # (see `embed_scale` above and `_reference_embed_scale`).
        embed_key = p + "embed_tokens.weight"
        table_dtype = w.dtype if w.dtype is not None else ckpt.tensor(embed_key).dtype
        if embed_scale is None:
            embed_scale = cls._reference_embed_scale(source, p, arch["hidden_size"], table_dtype)
        embed_scale = float(embed_scale)

        def scaled(t: torch.Tensor) -> torch.Tensor:
            # both operands as the reference holds them (in the dtype of
            # the table), multiplied in the working precision
            factor = torch.tensor(embed_scale, dtype=torch.float64).to(table_dtype)
            return t.to(table_dtype).to(t.dtype) * factor.to(device=t.device, dtype=t.dtype)

        put(0, "weight", embed_key, transform=scaled)
        net.embed_scale = embed_scale

        i = 1
        for n in range(num_layers):
            block = f"{p}layers.{n}."
            put(i, "weight", block + "input_layernorm.weight", transform=centered)
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                put(i + 1, f"{proj}.weight", block + f"self_attn.{proj}.weight")
                if arch["attention_bias"]:
                    put(i + 1, f"{proj}.bias", block + f"self_attn.{proj}.bias")
            if arch["qk_norm"]:
                put(i + 1, "q_norm.weight", block + "self_attn.q_norm.weight", transform=centered)
                put(i + 1, "k_norm.weight", block + "self_attn.k_norm.weight", transform=centered)
            put(i + 2, "weight", block + "post_attention_layernorm.weight", transform=centered)
            put(i + 4, "weight", block + "pre_feedforward_layernorm.weight", transform=centered)
            for proj in ("gate_proj", "up_proj", "down_proj"):
                put(i + 5, f"{proj}.weight", block + f"mlp.{proj}.weight")
            put(i + 6, "weight", block + "post_feedforward_layernorm.weight", transform=centered)
            i += cls.BLOCK_STRIDE

        put(i, "weight", p + "norm.weight", transform=centered)
        if include_lm_head:
            head_key = ckpt.lm_head_key
            if head_key == embed_key and not getattr(config, "tie_word_embeddings", True):
                raise ValueError(
                    "the checkpoint stores no LM head and its config does not tie the "
                    "head to the embedding table; pass include_lm_head=False."
                )
            # tied checkpoints store no head: the (unscaled) embedding
            # table is the head
            w.put(i + 1, "weight", head_key)

        # Every tensor of the text decoder has to be accounted for: one
        # that the network has no place for (a bias, an extra norm) means
        # the checkpoint is not the architecture built above.
        unused = sorted(k for k in ckpt.keys() if k.startswith(p) and k not in used)
        if unused:
            shown = ", ".join(unused[:8]) + (", ..." if len(unused) > 8 else "")
            raise NotImplementedError(
                f"{len(unused)} tensor(s) of the text decoder are not part of "
                f"{cls.__name__}: {shown}."
            )
        return w.finish()

    @staticmethod
    def _reference_embed_scale(source, prefix: str, hidden_size: int, dtype: torch.dtype) -> float:
        """
            The factor the reference implementation multiplies its
            embeddings by, for a network of the given dtype.

            transformers keeps sqrt(hidden_size) in a non-persistent buffer
            created under the dtype the model is instantiated in, and
            casts it to the dtype of the embedding table at every call
            (the original Gemma code builds the factor in the dtype of the
            hidden states, so a bfloat16 model scales by a bfloat16
            number -- transformers#29402). Hence:

            - a model in memory: its own buffer, cast the way its forward
              pass casts it. This is the function THAT model computes,
              whatever its history (a multimodal Gemma-3 built directly
              from its config carries a bfloat16-rounded factor even after
              .double(), because the sub-config says bfloat16);
            - a checkpoint on disk: sqrt(hidden_size) rounded to `dtype`,
              which is the buffer of `from_pretrained(..., dtype=dtype)`.
        """
        if isinstance(source, torch.nn.Module):
            try:
                module = source.get_submodule(prefix + "embed_tokens")
            except AttributeError:
                module = None
            buffer = getattr(module, "embed_scale", None)
            if isinstance(buffer, torch.Tensor) and not buffer.is_meta:
                own = buffer.detach().to(module.weight.dtype)
                return float(own.to(torch.float64).to(dtype))
        return float(torch.tensor(hidden_size ** 0.5, dtype=torch.float64).to(dtype))

    @staticmethod
    def _read_architecture(ckpt) -> dict:
        """
            The constructor arguments describing a Gemma-2 / Gemma-3
            checkpoint, read from its transformers config (plus the key
            "final_softcap": the cap the reference applies to its logits,
            or None). Raises on anything GemmaLike does not reproduce.
        """
        config = ckpt.text_config
        multimodal = config is not ckpt.config
        model_type = getattr(config, "model_type", None)
        if model_type not in ("gemma2", "gemma3_text"):
            raise NotImplementedError(
                f"GemmaLike reproduces the Gemma-2 and Gemma-3 text decoders "
                f"(model_type 'gemma2' / 'gemma3_text'); got {model_type!r}."
            )
        if multimodal and getattr(ckpt.config, "model_type", None) != "gemma3":
            # e.g. PaliGemma 2 wraps a Gemma-2 decoder but attends
            # bidirectionally over its prefix: not this function.
            raise NotImplementedError(
                f"the text decoder is wrapped by a "
                f"{getattr(ckpt.config, 'model_type', None)!r} model; only the Gemma-3 "
                f"multimodal wrapper ('gemma3') is known to run it as a plain causal LM."
            )
        gemma3 = model_type == "gemma3_text"

        if getattr(config, "use_bidirectional_attention", False):
            raise NotImplementedError("use_bidirectional_attention is not supported (causal attention only).")
        if not getattr(config, "is_causal", True):
            raise NotImplementedError("config.is_causal=False is not supported (causal attention only).")

        activation = getattr(config, "hidden_activation", None)
        if activation not in _GELU_VARIANTS:
            raise NotImplementedError(
                f"GemmaLike has a GELU-gated MLP ({', '.join(_GELU_VARIANTS)}); "
                f"this checkpoint uses hidden_activation={activation!r}."
            )

        num_layers = config.num_hidden_layers
        layer_types = getattr(config, "layer_types", None)
        if layer_types is None:
            # older transformers versions did not store the pattern (the
            # published Gemma-3 configs carry `sliding_window_pattern`
            # instead): Gemma-2 alternates starting with a local block,
            # Gemma-3 has one global block every `sliding_window_pattern`
            # (6).
            if gemma3:
                pattern = (getattr(config, "sliding_window_pattern", None)
                           or getattr(config, "_sliding_window_pattern", None) or 6)
            else:
                pattern = 2
            layer_types = [SLIDING if (n + 1) % pattern else FULL for n in range(num_layers)]
        layer_types = list(layer_types)
        if len(layer_types) < num_layers:
            raise ValueError(
                f"config.layer_types has {len(layer_types)} entries for "
                f"{num_layers} layers."
            )
        layer_types = layer_types[:num_layers]   # the reference zips the two
        unknown = sorted(set(layer_types) - {SLIDING, FULL})
        if unknown:
            raise NotImplementedError(f"unsupported layer types {unknown}.")

        sliding_window = getattr(config, "sliding_window", None)
        if SLIDING in layer_types and not sliding_window:
            raise ValueError("the config has sliding_attention layers but no sliding_window.")

        # RoPE: one setting (Gemma-2) or one per layer type (Gemma-3)
        rope = {t: rope_settings(config, t) for t in sorted(set(layer_types))}
        for layer_type, settings in rope.items():
            if settings["partial_rotary_factor"] != 1.0:
                raise NotImplementedError(
                    f"GemmaLike rotates the full head dimension; {layer_type} "
                    f"has partial_rotary_factor={settings['partial_rotary_factor']}."
                )
        global_rope = rope.get(FULL) or rope[SLIDING]
        local_rope = rope.get(SLIDING)
        if local_rope is None or local_rope == global_rope:
            local_theta, local_scaling = None, None
        else:
            local_theta, local_scaling = local_rope["theta"], local_rope["scaling"]

        attn_softcap = getattr(config, "attn_logit_softcapping", None)
        if gemma3 and attn_softcap is not None:
            raise NotImplementedError(
                "attn_logit_softcapping is set on a Gemma-3 config, but transformers' "
                "Gemma3Attention does not apply it: the intended function is ambiguous."
            )
        final_softcap = getattr(config, "final_logit_softcapping", None)
        if multimodal and final_softcap is not None:
            raise NotImplementedError(
                "final_logit_softcapping is set on the text config of a multimodal "
                "Gemma-3, but Gemma3ForConditionalGeneration does not apply it: the "
                "intended function is ambiguous."
            )

        attn0 = "layers.0.self_attn."
        attention_bias = ckpt.has_text(attn0 + "q_proj.bias")
        if attention_bias != bool(getattr(config, "attention_bias", attention_bias)):
            raise ValueError(
                f"config.attention_bias={config.attention_bias} but the checkpoint "
                f"{'has' if attention_bias else 'has no'} attention biases."
            )
        qk_norm = ckpt.has_text(attn0 + "q_norm.weight")
        if qk_norm != gemma3:
            raise ValueError(
                f"a {model_type} checkpoint {'with' if qk_norm else 'without'} "
                f"query/key norms: not the architecture its config names."
            )

        return dict(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=num_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=getattr(config, "num_key_value_heads", None)
            or config.num_attention_heads,
            head_dim=getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads,
            rope_theta=global_rope["theta"],
            rope_scaling=global_rope["scaling"],
            rope_local_theta=local_theta,
            rope_local_scaling=local_scaling,
            rms_norm_eps=float(config.rms_norm_eps),
            attn_softcap=attn_softcap,
            query_pre_attn_scalar=config.query_pre_attn_scalar,
            sliding_window=sliding_window,
            layer_types=layer_types,
            qk_norm=qk_norm,
            attention_bias=attention_bias,
            hidden_activation=activation,
            final_softcap=final_softcap,
        )
