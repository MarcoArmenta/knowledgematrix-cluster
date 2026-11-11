"""
    Implementation of Qwen3.5 (text model) as an NN, so that its knowledge
    matrices can be computed.

    Qwen3.5 (like Qwen3-Next) is a hybrid decoder: within every group of
    full_attention_interval (default 4) layers, the first ones use linear
    attention (GatedDeltaNet) and the last one uses full attention
    (GatedAttention). Each decoder layer is pre-norm:

        x = x + mixer(rmsnorm(x))        # GatedDeltaNet or GatedAttention
        x = x + swiglu(rmsnorm(x))

    followed by a final RMSNorm and the language-model head.

    The reference checkpoints use a zero-centered RMSNorm (scaling by
    1 + weight) for the layer norms, the final norm and the q/k norms of
    the attention layers; from_huggingface folds those weights into this
    library's plain RMSNorm (weight' = 1 + weight), which is exactly
    equivalent. The gated norm inside GatedDeltaNet (linear_attn.norm) is
    NOT zero-centered and is copied as is.

    Scope. The published checkpoints (Qwen/Qwen3.5-0.8B, -2B, -4B, -9B,
    -27B and their -Base variants) are multimodal; this class is their
    TEXT decoder, evaluated on text-only input. The vision tower
    (`model.visual.*`) and the multi-token-prediction head (`mtp.*`) of a
    checkpoint are ignored. For text-only input the three position axes
    of the multimodal rotary embedding coincide, so the interleaved
    M-RoPE table reduces to the plain 1-D table GatedAttention builds.
    The MoE checkpoints (Qwen3.5-35B-A3B, ...) and the quantized ones
    (-FP8, -GPTQ-Int4) are a different network and are refused.

    Loading the pretrained weights requires the optional dependencies
    `transformers` (for the config) and `safetensors` / `huggingface_hub`
    (for the files), used only inside from_huggingface.

    Example (Qwen3.5-4B):
        from knowledgematrix.models.qwen3_5 import Qwen3_5
        from knowledgematrix.matrix_computer import KnowledgeMatrixComputer

        model = Qwen3_5.from_huggingface("Qwen/Qwen3.5-4B")
        computer = KnowledgeMatrixComputer(model, batch_size=32)
        x = tokenizer("Hello", return_tensors="pt").input_ids.unsqueeze(0)
        mat = computer.forward(x)

    Note on size: the knowledge matrix has one row per output coordinate
    (seq_len * vocab_size with the LM head) and one column per embedded
    input coordinate (seq_len * hidden_size). For a 4B model this is very
    large; pass include_lm_head=False to stop at the final hidden states
    (seq_len * hidden_size rows), and push the (linear) LM head through
    afterwards for the logit rows you need.
"""

from typing import List, Optional, Sequence, Union

import torch

from knowledgematrix.neural_net import NN
from knowledgematrix.models._hf import (
    WeightLoader,
    empty_network,
    open_checkpoint,
    rope_settings,
)


LAYER_TYPES = ("linear_attention", "full_attention")


def _centered(w: torch.Tensor) -> torch.Tensor:
    """Zero-centered RMSNorm weight of the checkpoint -> plain weight."""
    return 1.0 + w


class Qwen3_5(NN):
    """
        The Qwen3.5 text model as an NN.

        Args:
            vocab_size (int): The size of the vocabulary.
            hidden_size (int): The dimension of the model.
            intermediate_size (int): The dimension of the SwiGLU MLP.
            num_hidden_layers (int): The number of decoder layers.
            num_attention_heads (int): Query heads of the full attention layers.
            num_key_value_heads (int): KV heads of the full attention layers.
            head_dim (int): Head dimension of the full attention layers
                (independent of hidden_size: 256 in every published model).
            linear_num_value_heads (int): Value heads of the DeltaNet layers.
            linear_num_key_heads (int): Key heads of the DeltaNet layers.
            linear_key_head_dim (int): Key head dim of the DeltaNet layers.
            linear_value_head_dim (int): Value head dim of the DeltaNet layers.
            linear_conv_kernel_dim (int): Kernel size of the DeltaNet conv.
            layer_types (list[str]): "linear_attention" / "full_attention"
                per layer. Default: every 4th layer is full attention.
            rope_theta (float): RoPE base frequency of the attention layers.
                NOTE: the default (10000) is the default of the transformers
                config class; the PUBLISHED checkpoints all use 1e7, which
                from_huggingface reads from their config.
            partial_rotary_factor (float): Fraction of each head rotated.
            rms_norm_eps (float): Epsilon of all RMS normalizations.
            include_lm_head (bool): Whether to include the LM head. If
                False the network ends at the final RMSNorm and outputs
                hidden states instead of logits.
            save (bool): Whether to save activations and preactivations.
            device (str): The device to run the network on.
            attention_bias (bool): Biases on the q/k/v/o projections of the
                full attention layers (False in every published model).
    """

    #: layers appended per decoder block (rmsnorm, mixer, identity,
    #: rmsnorm, swiglu, identity) -- the stride used when placing weights
    BLOCK_STRIDE = 6

    #: A checkpoint with tied embeddings stores no LM head. Read from disk
    #: onto a single device, the head of this network then reuses the
    #: embedding tensor instead of holding a second (vocab, hidden) table
    #: (see from_huggingface); llm_footprint reads this flag when it sizes
    #: a checkpoint.
    SHARES_TIED_HEAD = True

    def __init__(
            self,
            vocab_size: int = 248320,
            hidden_size: int = 2048,
            intermediate_size: int = 4096,
            num_hidden_layers: int = 8,
            num_attention_heads: int = 16,
            num_key_value_heads: int = 4,
            head_dim: int = 256,
            linear_num_value_heads: int = 32,
            linear_num_key_heads: int = 16,
            linear_key_head_dim: int = 128,
            linear_value_head_dim: int = 128,
            linear_conv_kernel_dim: int = 4,
            layer_types: Union[List[str], None] = None,
            rope_theta: float = 10000.0,
            partial_rotary_factor: float = 0.25,
            rms_norm_eps: float = 1e-6,
            include_lm_head: bool = True,
            save: bool = False,
            device: str = "cpu",
            attention_bias: bool = False
        ) -> None:
        super().__init__(input_shape=(1, 1, hidden_size), save=save, device=device)

        if layer_types is None:
            layer_types = [
                "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
                for i in range(num_hidden_layers)
            ]
        if len(layer_types) != num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(layer_types)} entries but num_hidden_layers is {num_hidden_layers}."
            )
        for lt in layer_types:
            if lt not in LAYER_TYPES:
                raise ValueError(f"Unknown layer type: {lt}.")

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.layer_types = list(layer_types)
        self.include_lm_head = include_lm_head

        self.embedding(vocab_size, hidden_size)

        for layer_type in layer_types:
            # Token mixer, pre-norm: x = x + mixer(rmsnorm(x)).
            # The identity layers mark the addition points so that no layer
            # index is both the end of one residual and the start of the
            # next (see NN.identity). Every block maps hidden_size ->
            # hidden_size, so the residuals are registered without the
            # shape probes (same_shape=True): building costs no forward
            # pass, and works on a network without storage (empty_network).
            start = self.get_num_layers()          # index of input_layernorm
            self.rmsnorm(hidden_size, eps=rms_norm_eps)
            if layer_type == "linear_attention":
                self.gateddeltanet(
                    d_model=hidden_size,
                    num_v_heads=linear_num_value_heads,
                    num_k_heads=linear_num_key_heads,
                    head_k_dim=linear_key_head_dim,
                    head_v_dim=linear_value_head_dim,
                    conv_kernel_size=linear_conv_kernel_dim,
                    rms_norm_eps=rms_norm_eps,
                )
            else:
                self.gatedattention(
                    d_model=hidden_size,
                    num_heads=num_attention_heads,
                    num_kv_heads=num_key_value_heads,
                    head_dim=head_dim,
                    rope_theta=rope_theta,
                    partial_rotary_factor=partial_rotary_factor,
                    rms_norm_eps=rms_norm_eps,
                    bias=attention_bias,
                )
            end = self.get_num_layers()            # index of the identity below
            self.identity()
            self.residual(start, end, same_shape=True)

            # Feed-forward, pre-norm: x = x + swiglu(rmsnorm(x))
            start = self.get_num_layers()          # index of post_attention_layernorm
            self.rmsnorm(hidden_size, eps=rms_norm_eps)
            self.swiglu(hidden_size, intermediate_size)
            end = self.get_num_layers()            # index of the identity below
            self.identity()
            self.residual(start, end, same_shape=True)

        self.rmsnorm(hidden_size, eps=rms_norm_eps)  # final norm
        if include_lm_head:
            self.linear(in_features=hidden_size, out_features=vocab_size, bias=False)

    @staticmethod
    def _refuse_unsupported(ckpt) -> None:
        """
            Raise on every config value this class does not reproduce,
            instead of building a network that computes something else.

            The reference implementation (transformers.models.qwen3_5)
            itself hard-codes several choices that the published configs
            still spell out -- the attention output gate is always on,
            there are no MLP-only layers, the delta rule always runs in
            float32. A checkpoint that says otherwise is not the network
            that was verified, whatever transformers would do with it.
        """
        top, config = ckpt.config, ckpt.text_config

        model_type = getattr(config, "model_type", None)
        if model_type != "qwen3_5_text":
            raise NotImplementedError(
                f"Qwen3_5 reproduces the dense Qwen3.5 text decoder (model_type "
                f"'qwen3_5_text'); this checkpoint is {model_type!r}"
                + (" -- the MoE variants (Qwen3.5-35B-A3B, ...) route every "
                   "token through expert MLPs, which this class does not have."
                   if "moe" in str(model_type) else ".")
            )
        if getattr(top, "quantization_config", None):
            raise NotImplementedError(
                "this checkpoint is quantized (quantization_config is set: the "
                "-FP8 / -GPTQ-Int4 repositories); its tensors are not the "
                "weights of the network. Load the unquantized checkpoint."
            )

        def refuse(name, value, expected, why):
            raise NotImplementedError(
                f"Qwen3_5 does not reproduce {name}={value!r} (supported: "
                f"{expected!r}): {why}"
            )

        act = getattr(config, "hidden_act", "silu")
        if act != "silu":
            refuse("hidden_act", act, "silu",
                   "the MLP gate and the DeltaNet convolution are SiLU-activated.")
        if not getattr(config, "attn_output_gate", True):
            refuse("attn_output_gate", config.attn_output_gate, True,
                   "GatedAttention always applies the sigmoid output gate.")
        mlp_only = getattr(config, "mlp_only_layers", None)
        if mlp_only:
            refuse("mlp_only_layers", mlp_only, [],
                   "every block has a token mixer.")
        ssm_dtype = getattr(config, "mamba_ssm_dtype", "float32")
        if ssm_dtype != "float32":
            refuse("mamba_ssm_dtype", ssm_dtype, "float32",
                   "the delta rule recurrence runs in float32 (float64 in a "
                   "float64 network).")
        layer_types = getattr(config, "layer_types", None)
        if not layer_types or len(layer_types) != config.num_hidden_layers:
            raise ValueError(
                f"config.layer_types must list the {config.num_hidden_layers} "
                f"layers, got {layer_types!r}."
            )
        unknown = sorted(set(layer_types) - set(LAYER_TYPES))
        if unknown:
            refuse("layer_types", unknown, list(LAYER_TYPES),
                   "only linear (GatedDeltaNet) and full (GatedAttention) "
                   "blocks exist.")
        if config.linear_num_value_heads % config.linear_num_key_heads != 0:
            refuse("linear_num_value_heads", config.linear_num_value_heads,
                   f"a multiple of linear_num_key_heads={config.linear_num_key_heads}",
                   "each key head is shared by a whole number of value heads.")

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
        ) -> "Qwen3_5":
        """
            Build a Qwen3_5 NN computing the same function as the text
            decoder of a Hugging Face Qwen3.5 checkpoint (0.8B, 2B, 4B,
            9B, 27B and their -Base variants) on text-only input.

            Args:
                source: one of
                    - a hub repository id or a local directory
                      ("Qwen/Qwen3.5-27B"): the weights are streamed from
                      the safetensors files one tensor at a time, without
                      instantiating the transformers model (nor its vision
                      tower), so host memory stays at one tensor;
                    - a transformers model already in memory:
                      Qwen3_5ForCausalLM, the multimodal
                      Qwen3_5ForConditionalGeneration (its text stack is
                      used), or a bare Qwen3_5TextModel (which has no LM
                      head: pass include_lm_head=False unless the config
                      ties the head to the embedding).
                include_lm_head (bool): Include the LM head. Without it
                    the network ends at the final RMSNorm (hidden states).
                    The 0.8B / 2B / 4B tie the head to the embedding and
                    store no head: read from disk, the head then shares
                    the embedding's tensor (when both are on one device).
                dtype: dtype of the network. Default: float32 for a
                    checkpoint read from disk (the files are bfloat16),
                    the model's own dtype for a model in memory.
                device (str | list[str]): one device, or the stages of a
                    pipeline over several devices (["cuda:0", "cuda:1"])
                    for a model that does not fit one -- the network comes
                    out sharded, see NN.shard. "meta" is a dry run: every
                    tensor is located and shape-checked, nothing is kept.
                save (bool): Whether to save activations.
                copy (bool): for a model in memory, copy its weights
                    (True) or share them (False, no extra memory; the
                    zero-centered norm weights are folded and therefore
                    always new tensors).
                boundaries (list[int] | None): pipeline cut indices, see
                    NN.shard (default: balance the parameter count).
                **hub_kwargs: revision, local_files_only, token,
                    cache_dir -- for a hub repository id.
            Raises:
                NotImplementedError: for a config value the class does not
                    reproduce (MoE or quantized checkpoint, another
                    activation, attn_output_gate=False, mlp_only_layers,
                    a RoPE rescaling, ...).
                KeyError / ValueError: for a missing or mis-shaped tensor,
                    or a decoder tensor the network has no place for.
        """
        ckpt = open_checkpoint(source, **hub_kwargs)
        config = ckpt.text_config
        cls._refuse_unsupported(ckpt)
        p = ckpt.text_prefix

        # Only the base and the rotated fraction of the multimodal RoPE
        # matter for text: with equal temporal / height / width positions
        # the (interleaved) mrope_section layout is the plain 1-D table.
        rope = rope_settings(config)
        if rope["scaling"] is not None:
            raise NotImplementedError(
                f"GatedAttention builds the plain RoPE table; this checkpoint "
                f"rescales its frequencies ({rope['scaling']})."
            )

        layer_types = list(config.layer_types)
        num_layers = config.num_hidden_layers
        attention_bias = bool(getattr(config, "attention_bias", False))

        net = empty_network(
            cls,
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=num_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads,
            linear_num_value_heads=config.linear_num_value_heads,
            linear_num_key_heads=config.linear_num_key_heads,
            linear_key_head_dim=config.linear_key_head_dim,
            linear_value_head_dim=config.linear_value_head_dim,
            linear_conv_kernel_dim=config.linear_conv_kernel_dim,
            layer_types=layer_types,
            rope_theta=rope["theta"],
            partial_rotary_factor=rope["partial_rotary_factor"],
            rms_norm_eps=float(config.rms_norm_eps),
            include_lm_head=include_lm_head,
            save=save,
            attention_bias=attention_bias,
        )

        w = WeightLoader(net, ckpt, dtype=dtype, device=device,
                         boundaries=boundaries, copy=copy)
        used = set()

        def put(layer: int, attr: str, key: str, transform=None) -> None:
            w.put(layer, attr, key, transform=transform)
            used.add(key)

        put(0, "weight", p + "embed_tokens.weight")

        i = 1  # first layer after the embedding
        for n, layer_type in enumerate(layer_types):
            block = f"{p}layers.{n}."
            put(i, "weight", block + "input_layernorm.weight", _centered)
            if layer_type == "linear_attention":
                src = block + "linear_attn."
                put(i + 1, "conv1d.weight", src + "conv1d.weight")
                put(i + 1, "dt_bias", src + "dt_bias")
                put(i + 1, "A_log", src + "A_log")
                # the gated norm scales by its weight, NOT by 1 + weight
                put(i + 1, "norm.weight", src + "norm.weight")
                for proj in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
                    put(i + 1, f"{proj}.weight", src + f"{proj}.weight")
            else:
                src = block + "self_attn."
                for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    put(i + 1, f"{proj}.weight", src + f"{proj}.weight")
                    if attention_bias:
                        put(i + 1, f"{proj}.bias", src + f"{proj}.bias")
                put(i + 1, "q_norm.weight", src + "q_norm.weight", _centered)
                put(i + 1, "k_norm.weight", src + "k_norm.weight", _centered)
            # layer i + 2 is the identity boundary of the mixer residual
            put(i + 3, "weight", block + "post_attention_layernorm.weight", _centered)
            for proj in ("gate_proj", "up_proj", "down_proj"):
                put(i + 4, f"{proj}.weight", block + f"mlp.{proj}.weight")
            i += cls.BLOCK_STRIDE

        put(i, "weight", p + "norm.weight", _centered)
        if include_lm_head:
            head = ckpt.lm_head_key
            if not head.endswith("lm_head.weight") and not getattr(ckpt.config, "tie_word_embeddings", False):
                # No stored head and no tying: transformers would leave the
                # head randomly initialized; the embedding table is not it.
                raise ValueError(
                    "the checkpoint stores no lm_head.weight and its config does "
                    "not tie the head to the embedding (tie_word_embeddings is "
                    "false), so it defines no LM head. Pass include_lm_head=False "
                    "to end at the final hidden states."
                )
            embed = p + "embed_tokens.weight"
            one_device = w.layer_devices is None or w.layer_devices[0] == w.layer_devices[i + 1]
            if head == embed and one_device and w.devices != ["meta"]:
                # Tied checkpoints store no head: the embedding table is
                # the head. Reuse the tensor that was just placed instead
                # of reading and storing the (V, D) table a second time
                # (2.5 GB in float32 for the 4B).
                w.put_tensor(i + 1, "weight", net.layers[0].weight.detach(),
                             _borrowed=False, _label=head)
            else:
                put(i + 1, "weight", head)

        # A tensor of the decoder that found no place means the checkpoint
        # has something this network does not (a bias, another norm, ...).
        # Tensors outside the decoder (vision tower, mtp head) are ignored.
        decoder = (p + "embed_tokens.", p + "layers.", p + "norm.")
        leftover = sorted(k for k in ckpt.keys() if k.startswith(decoder) and k not in used)
        if leftover:
            shown = ", ".join(leftover[:8]) + (", ..." if len(leftover) > 8 else "")
            raise ValueError(
                f"{len(leftover)} tensor(s) of the text decoder have no "
                f"counterpart in Qwen3_5: {shown}."
            )
        return w.finish()
