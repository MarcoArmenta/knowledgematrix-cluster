from typing import List, Optional, Sequence, Union

import torch

from knowledgematrix.neural_net import NN
from knowledgematrix.models._hf import (
    WeightLoader,
    empty_network,
    open_checkpoint,
    rope_settings,
)


#: The transformers `model_type`s whose reference implementation LlamaLike
#: reproduces, each checked line by line against transformers 5.16.1 and by
#: the parity tests of extra/tests/llama_family.py:
#:
#:   "llama"      LlamaForCausalLM     -- Llama-2 / 3.1 / 3.2 / 3.3, SmolLM2
#:   "mistral"    MistralForCausalLM   -- Mistral-7B, Mistral-Nemo
#:   "ministral"  MinistralForCausalLM -- Ministral-8B (what transformers'
#:                Auto classes make of a "mistral" config.json that has
#:                `layer_types`)
#:   "qwen2"      Qwen2ForCausalLM     -- Qwen2 / Qwen2.5
#:   "qwen3"      Qwen3ForCausalLM     -- the dense Qwen3 models
#:
#: Every other type is refused. Several of them store their weights under
#: exactly the same keys and would load without a complaint into a network
#: that computes something else (measured on tiny random models, max logit
#: gap against the reference for logits of scale ~3: DiffLlama 4.1,
#: Granite 3.2, Doge 4.8, Gemma-2 0.6 for logits of scale 0.85), because
#: what distinguishes them is not in the tensor names: differential
#: attention, residual / embedding / logit multipliers, blocks without
#: RoPE (SmolLM3), another norm convention.
SUPPORTED_MODEL_TYPES = ("llama", "mistral", "ministral", "qwen2", "qwen3")

_LAYER_TYPES = ("full_attention", "sliding_attention")


def _layer_windows(config, num_layers: int) -> List[Optional[int]]:
    """
        The attention window of every block of a Qwen2 / Qwen3 / Ministral
        config (None = full causal attention), exactly as their forward
        pass picks the mask: block i uses the sliding-window mask iff
        ``config.layer_types[i] == "sliding_attention"``, and the window is
        ``config.sliding_window``.
    """
    window = getattr(config, "sliding_window", None)
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        # A config of a transformers version that predates `layer_types`.
        # Without a sliding window every block attends fully, in every
        # version (all published Qwen2.5 checkpoints); with one, those
        # versions applied it in some attention backends only, so there
        # is no single reference function to reproduce.
        if getattr(config, "use_sliding_window", False) and window is not None:
            raise NotImplementedError(
                "this config asks for sliding-window attention but has no "
                "layer_types (it comes from an older transformers version, "
                "where the window depended on the attention backend); "
                "upgrade transformers to load it."
            )
        return [None] * num_layers
    layer_types = list(layer_types)
    if len(layer_types) < num_layers:
        raise ValueError(
            f"config.layer_types has {len(layer_types)} entries for "
            f"{num_layers} layers."
        )
    layer_types = layer_types[:num_layers]
    unknown = sorted(set(layer_types) - set(_LAYER_TYPES))
    if unknown:
        raise NotImplementedError(
            f"unsupported attention layer type(s) {unknown} (supported: "
            f"{', '.join(_LAYER_TYPES)})."
        )
    if "sliding_attention" in layer_types:
        if window is None:
            raise ValueError(
                "config.layer_types asks for sliding-window attention but "
                "config.sliding_window is not set (use_sliding_window=False "
                "resets it); transformers cannot evaluate this model either."
            )
        if int(window) < 1:
            raise ValueError(f"config.sliding_window must be positive, got {window}.")
    return [int(window) if t == "sliding_attention" else None for t in layer_types]


def llama_architecture(config) -> dict:
    """
        The LlamaLike constructor arguments under which the network is the
        function transformers computes for `config` (a LlamaConfig,
        MistralConfig, MinistralConfig, Qwen2Config or Qwen3Config, as
        normalized by the installed transformers version).

        Everything the reference forward pass reads from the config is
        either reproduced or refused: an unsupported `model_type`,
        bidirectional attention (`is_causal=False`), an activation other
        than SiLU, a RoPE variant other than default / linear / llama3, a
        partial rotation, or an unknown attention layer type raise instead
        of producing a network that silently computes another function.

        What differs between the families (and is therefore read per
        family, exactly like the reference classes do):

        ===========  ===============  =========  ========  ====================
        model_type   q/k/v/o bias     MLP bias   q/k norm  sliding window
        ===========  ===============  =========  ========  ====================
        llama        attention_bias   mlp_bias   no        never
        mistral      none             none       no        every block
        ministral    none             none       no        per layer_types
        qwen2        q/k/v yes, o no  none       no        per layer_types
        qwen3        attention_bias   none       yes       per layer_types
        ===========  ===============  =========  ========  ====================

        Note in particular that LlamaForCausalLM ignores a stray
        `sliding_window` entry and MistralForCausalLM ignores `layer_types`
        (it logs a warning and windows every block).

        Returns:
            dict: keyword arguments for LlamaLike(...), without
            include_lm_head / save / device.
    """
    model_type = getattr(config, "model_type", None)
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise NotImplementedError(
            f"LlamaLike reproduces the transformers model types "
            f"{', '.join(SUPPORTED_MODEL_TYPES)}; this checkpoint is of type "
            f"{model_type!r}. Architectures that merely store their weights "
            f"under the same names are refused rather than loaded into a "
            f"network computing a different function."
        )
    if not getattr(config, "is_causal", True):
        raise NotImplementedError(
            "config.is_causal=False makes the reference model attend in both "
            "directions; LlamaLike is a causal decoder."
        )
    act = getattr(config, "hidden_act", None)
    if act not in ("silu", "swish"):   # two names of the same function
        raise NotImplementedError(
            f"LlamaLike has a SiLU-gated MLP; this checkpoint uses {act!r}."
        )
    rope = rope_settings(config)
    if rope["partial_rotary_factor"] != 1.0:
        raise NotImplementedError(
            f"LlamaLike rotates the full head dimension; this config has "
            f"partial_rotary_factor={rope['partial_rotary_factor']}."
        )
    eps = getattr(config, "rms_norm_eps", None)
    if eps is None:
        raise ValueError(f"{type(config).__name__} defines no rms_norm_eps.")

    num_layers = int(config.num_hidden_layers)
    heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", None) or heads)
    # the per-head dimension is NOT hidden_size / heads in general
    # (Qwen3-0.6B: 1024 / 16 = 64 but head_dim = 128; Mistral-Nemo:
    # 5120 / 32 = 160 but head_dim = 128)
    head_dim = int(getattr(config, "head_dim", None) or config.hidden_size // heads)

    windows: List[Optional[int]] = [None] * num_layers
    qkv_bias = o_bias = mlp_bias = qk_norm = False
    if model_type == "llama":
        qkv_bias = o_bias = bool(getattr(config, "attention_bias", False))
        mlp_bias = bool(getattr(config, "mlp_bias", False))
    elif model_type == "mistral":
        window = getattr(config, "sliding_window", None)
        if window is not None:
            if int(window) < 1:
                raise ValueError(f"config.sliding_window must be positive, got {window}.")
            windows = [int(window)] * num_layers
    elif model_type == "ministral":
        windows = _layer_windows(config, num_layers)
    elif model_type == "qwen2":
        qkv_bias = True
        windows = _layer_windows(config, num_layers)
    elif model_type == "qwen3":
        qkv_bias = o_bias = bool(getattr(config, "attention_bias", False))
        qk_norm = True
        windows = _layer_windows(config, num_layers)

    return dict(
        vocab_size=int(config.vocab_size),
        hidden_size=int(config.hidden_size),
        intermediate_size=int(config.intermediate_size),
        num_hidden_layers=num_layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        rope_theta=rope["theta"],
        rope_scaling=rope["scaling"],
        rms_norm_eps=float(eps),
        qkv_bias=qkv_bias,
        o_bias=o_bias,
        qk_norm=qk_norm,
        mlp_bias=mlp_bias,
        sliding_window=windows,
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
        - ``qkv_bias`` / ``o_bias`` / ``mlp_bias``: Qwen2.5 has biases on
          q/k/v only; a Llama config may switch them on everywhere
          (``attention_bias``, ``mlp_bias``); the published Llama, SmolLM2,
          Mistral and Qwen3 checkpoints have none,
        - ``qk_norm``: Qwen3 RMS-normalizes q and k per head, Llama does not,
        - ``rope_theta`` / ``rope_scaling``: Llama-3.x rotates with base
          500000 and the "llama3" frequency rescaling, Qwen3 with base 1e6,
          SmolLM2 with base 1e5 (135M, 360M) or 1.3e5 (1.7B),
        - ``sliding_window``: Mistral's local attention (None = full causal).

        `from_huggingface` reads all of them from a checkpoint
        (see `llama_architecture`).

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
            SmolLM2, Mistral, Qwen2.5 and the dense Qwen3 models
            (SUPPORTED_MODEL_TYPES; anything else is refused).

            The architecture is derived from the config alone, the way the
            reference classes do it (`llama_architecture`), and the
            checkpoint then has to match it exactly: a tensor the
            architecture needs and the checkpoint lacks, a tensor of
            another shape, and a decoder tensor the architecture has no
            use for all raise.

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
                    Checkpoints with tied embeddings (SmolLM2, Llama-3.2)
                    store no head: the embedding table is loaded a second
                    time as the head (an independent copy, so such a
                    network has vocab_size * hidden_size parameters more
                    than the transformers model counts).
                dtype: dtype of the network. Default: float32 for a
                    checkpoint read from disk, the model's own dtype for
                    a model in memory.
                device (str | list[str]): one device, or the stages of a
                    pipeline over several devices (["cuda:0", "cuda:1"])
                    for a model that does not fit one -- the network comes
                    out sharded, see NN.shard. "meta" is a dry run that
                    checks every tensor's presence and shape and stores
                    nothing.
                save (bool): Whether to save activations.
                copy (bool): for a model in memory, copy its weights
                    (True) or share them (False, no extra memory).
                boundaries (list[int] | None): pipeline cut indices, see
                    NN.shard (default: balance the parameter count).
                **hub_kwargs: revision, local_files_only, token,
                    cache_dir -- for a hub repository id.
        """
        ckpt = open_checkpoint(source, **hub_kwargs)
        try:
            return cls._from_checkpoint(ckpt, include_lm_head, dtype, device,
                                        save, copy, boundaries)
        except BaseException:
            ckpt.close()   # do not leave the weight files open
            raise

    @classmethod
    def _from_checkpoint(cls, ckpt, include_lm_head, dtype, device, save,
                         copy, boundaries) -> "LlamaLike":
        config = ckpt.text_config
        arch = llama_architecture(config)
        p = ckpt.text_prefix
        num_layers = arch["num_hidden_layers"]

        head_key = None
        if include_lm_head:
            head_key = ckpt.lm_head_key
            tied = bool(getattr(config, "tie_word_embeddings", False)
                        or getattr(ckpt.config, "tie_word_embeddings", False))
            if head_key == p + "embed_tokens.weight" and not tied:
                # transformers would leave such a head randomly initialized
                raise KeyError(
                    "the checkpoint stores no lm_head.weight and its config "
                    "does not tie the head to the embedding "
                    "(tie_word_embeddings=False): there is no LM head to "
                    "load. Pass include_lm_head=False for the hidden states."
                )

        net = empty_network(cls, include_lm_head=include_lm_head, save=save, **arch)
        w = WeightLoader(net, ckpt, dtype=dtype, device=device,
                         boundaries=boundaries, copy=copy)
        used = set()

        def put(layer: int, attr: str, key: str) -> None:
            w.put(layer, attr, key)
            used.add(key)

        put(0, "weight", p + "embed_tokens.weight")

        i = 1
        for n in range(num_layers):
            block = f"{p}layers.{n}."
            # NOTE: Llama-family RMSNorm weights are used as-is; the
            # zero-centered convention (weight + 1) is Qwen3.5's and
            # Gemma's, NOT this one.
            put(i, "weight", block + "input_layernorm.weight")
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                put(i + 1, f"{proj}.weight", block + f"self_attn.{proj}.weight")
            if arch["qkv_bias"]:
                for proj in ("q_proj", "k_proj", "v_proj"):
                    put(i + 1, f"{proj}.bias", block + f"self_attn.{proj}.bias")
            if arch["o_bias"]:
                put(i + 1, "o_proj.bias", block + "self_attn.o_proj.bias")
            if arch["qk_norm"]:
                put(i + 1, "q_norm.weight", block + "self_attn.q_norm.weight")
                put(i + 1, "k_norm.weight", block + "self_attn.k_norm.weight")
            put(i + 3, "weight", block + "post_attention_layernorm.weight")
            for proj in ("gate_proj", "up_proj", "down_proj"):
                put(i + 4, f"{proj}.weight", block + f"mlp.{proj}.weight")
                if arch["mlp_bias"]:
                    put(i + 4, f"{proj}.bias", block + f"mlp.{proj}.bias")
            i += cls.BLOCK_STRIDE

        put(i, "weight", p + "norm.weight")
        if include_lm_head:
            # tied checkpoints store no head: the embedding table is the head
            put(i + 1, "weight", head_key)

        # Every tensor of the decoder must have found its place. A leftover
        # is a parameter of the reference model that this network does not
        # have -- a bias the config did not announce, a layer beyond
        # num_hidden_layers, the extra weights of a look-alike architecture
        # -- and WeightLoader.finish only checks the opposite direction.
        # (The RoPE table `inv_freq`, saved as a buffer by old transformers
        # versions, is recomputed and carries no information; the LM head
        # is legitimately unused with include_lm_head=False.)
        leftover = sorted(
            k for k in ckpt.keys()
            if k.startswith(p) and k not in used
            and not k.endswith(("inv_freq", "lm_head.weight"))
        )
        if leftover:
            shown = ", ".join(leftover[:8]) + (", ..." if len(leftover) > 8 else "")
            raise RuntimeError(
                f"{len(leftover)} tensor(s) of the checkpoint's decoder have no "
                f"counterpart in the {config.model_type!r} architecture described "
                f"by its config, so the network would not compute the "
                f"checkpoint's function: {shown}."
            )
        return w.finish()
