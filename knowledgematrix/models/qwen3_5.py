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
    1 + weight); from_huggingface folds those weights into this library's
    plain RMSNorm (weight' = 1 + weight), which is exactly equivalent.

    Loading the pretrained weights requires the optional dependency
    `transformers` (pip install transformers), used only inside
    from_huggingface.

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

from typing import List, Union

import torch

from knowledgematrix.neural_net import NN


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
            head_dim (int): Head dimension of the full attention layers.
            linear_num_value_heads (int): Value heads of the DeltaNet layers.
            linear_num_key_heads (int): Key heads of the DeltaNet layers.
            linear_key_head_dim (int): Key head dim of the DeltaNet layers.
            linear_value_head_dim (int): Value head dim of the DeltaNet layers.
            linear_conv_kernel_dim (int): Kernel size of the DeltaNet conv.
            layer_types (list[str]): "linear_attention" / "full_attention"
                per layer. Default: every 4th layer is full attention.
            rope_theta (float): RoPE base frequency of the attention layers.
            partial_rotary_factor (float): Fraction of each head rotated.
            rms_norm_eps (float): Epsilon of all RMS normalizations.
            include_lm_head (bool): Whether to include the LM head. If
                False the network ends at the final RMSNorm and outputs
                hidden states instead of logits.
            save (bool): Whether to save activations and preactivations.
            device (str): The device to run the network on.
    """

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
            device: str = "cpu"
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
            if lt not in ("linear_attention", "full_attention"):
                raise ValueError(f"Unknown layer type: {lt}.")

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.layer_types = list(layer_types)
        self.include_lm_head = include_lm_head

        self.embedding(vocab_size, hidden_size)

        for layer_type in layer_types:
            # Token mixer, pre-norm: x = x + mixer(rmsnorm(x)).
            # The identity layers mark the addition points so that no layer
            # index is both the end of one residual and the start of the
            # next (see NN.identity).
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
                )
            end = self.get_num_layers()            # index of the identity below
            self.identity()
            self.residual(start, end)

            # Feed-forward, pre-norm: x = x + swiglu(rmsnorm(x))
            start = self.get_num_layers()          # index of post_attention_layernorm
            self.rmsnorm(hidden_size, eps=rms_norm_eps)
            self.swiglu(hidden_size, intermediate_size)
            end = self.get_num_layers()            # index of the identity below
            self.identity()
            self.residual(start, end)

        self.rmsnorm(hidden_size, eps=rms_norm_eps)  # final norm
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
        ) -> "Qwen3_5":
        """
            Build a Qwen3_5 NN from a Hugging Face transformers model and
            copy its weights, so both compute the same network function.

            Args:
                source: One of
                    - a hub id or local path (e.g. "Qwen/Qwen3.5-4B"),
                    - a transformers Qwen3_5ForCausalLM (or the multimodal
                      Qwen3_5ForConditionalGeneration, whose text stack is
                      used; the vision tower is out of scope),
                    - a transformers Qwen3_5TextModel (no LM head: pass
                      include_lm_head=False or tied embeddings are used).
                include_lm_head: Whether the NN ends with the LM head.
                dtype: Optional dtype to convert the weights to (e.g.
                    torch.float64 for exactness tests).
                device: The device of the resulting NN.
                save: The save flag of the resulting NN.
        """
        if isinstance(source, str):
            try:
                from transformers import AutoConfig, AutoModelForCausalLM
            except ImportError as e:
                raise ImportError(
                    "Loading pretrained Qwen3.5 weights requires the optional "
                    "dependency `transformers` (pip install transformers)."
                ) from e
            config = AutoConfig.from_pretrained(source)
            if hasattr(config, "vision_config") or config.model_type == "qwen3_5":
                from transformers import AutoModelForImageTextToText
                source = AutoModelForImageTextToText.from_pretrained(source)
            else:
                source = AutoModelForCausalLM.from_pretrained(source)

        # Resolve the text decoder stack and the LM head.
        lm_head = getattr(source, "lm_head", None)
        text = source
        if hasattr(text, "model"):
            text = text.model                       # ForCausalLM / ForConditionalGeneration
        if hasattr(text, "language_model"):
            text = text.language_model              # multimodal wrapper
        if not hasattr(text, "embed_tokens") or not hasattr(text, "layers"):
            raise ValueError(f"Could not find a Qwen3.5 text decoder in {type(source).__name__}.")

        config = text.config
        rope = getattr(config, "rope_parameters", None) or {}

        nn_model = cls(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
            linear_num_value_heads=config.linear_num_value_heads,
            linear_num_key_heads=config.linear_num_key_heads,
            linear_key_head_dim=config.linear_key_head_dim,
            linear_value_head_dim=config.linear_value_head_dim,
            linear_conv_kernel_dim=config.linear_conv_kernel_dim,
            layer_types=list(config.layer_types),
            rope_theta=float(rope.get("rope_theta", 10000.0)),
            partial_rotary_factor=float(rope.get("partial_rotary_factor", 0.25)),
            rms_norm_eps=config.rms_norm_eps,
            include_lm_head=include_lm_head,
            save=save,
            device=device,
        )

        def cast(t: torch.Tensor) -> torch.Tensor:
            t = t.detach().clone()
            return t.to(dtype) if dtype is not None else t

        def centered(t: torch.Tensor) -> torch.Tensor:
            # Zero-centered RMSNorm of the checkpoint -> plain RMSNorm weight.
            return cast(t) + 1.0

        with torch.no_grad():
            layers = nn_model.layers
            layers[0].weight.data = cast(text.embed_tokens.weight)

            i = 1  # first layer after the embedding
            for hf_layer in text.layers:
                block_type = hf_layer.block_type
                layers[i].weight.data = centered(hf_layer.input_layernorm.weight)
                mixer = layers[i + 1]
                if block_type == "linear_attention":
                    src = hf_layer.linear_attn
                    mixer.conv1d.weight.data = cast(src.conv1d.weight)
                    mixer.dt_bias.data = cast(src.dt_bias)
                    mixer.A_log.data = cast(src.A_log)
                    mixer.norm.weight.data = cast(src.norm.weight)
                    mixer.out_proj.weight.data = cast(src.out_proj.weight)
                    mixer.in_proj_qkv.weight.data = cast(src.in_proj_qkv.weight)
                    mixer.in_proj_z.weight.data = cast(src.in_proj_z.weight)
                    mixer.in_proj_b.weight.data = cast(src.in_proj_b.weight)
                    mixer.in_proj_a.weight.data = cast(src.in_proj_a.weight)
                else:
                    src = hf_layer.self_attn
                    mixer.q_proj.weight.data = cast(src.q_proj.weight)
                    mixer.k_proj.weight.data = cast(src.k_proj.weight)
                    mixer.v_proj.weight.data = cast(src.v_proj.weight)
                    mixer.o_proj.weight.data = cast(src.o_proj.weight)
                    if getattr(src.q_proj, "bias", None) is not None:
                        mixer.q_proj.bias.data = cast(src.q_proj.bias)
                        mixer.k_proj.bias.data = cast(src.k_proj.bias)
                        mixer.v_proj.bias.data = cast(src.v_proj.bias)
                        mixer.o_proj.bias.data = cast(src.o_proj.bias)
                    mixer.q_norm.weight.data = centered(src.q_norm.weight)
                    mixer.k_norm.weight.data = centered(src.k_norm.weight)
                # layers[i + 2] is the identity boundary of the mixer residual
                layers[i + 3].weight.data = centered(hf_layer.post_attention_layernorm.weight)
                mlp = layers[i + 4]
                mlp.gate_proj.weight.data = cast(hf_layer.mlp.gate_proj.weight)
                mlp.up_proj.weight.data = cast(hf_layer.mlp.up_proj.weight)
                mlp.down_proj.weight.data = cast(hf_layer.mlp.down_proj.weight)
                i += 6  # rmsnorm, mixer, identity, rmsnorm, swiglu, identity

            layers[i].weight.data = centered(text.norm.weight)
            if include_lm_head:
                if lm_head is not None:
                    layers[i + 1].weight.data = cast(lm_head.weight)
                else:  # tied embeddings
                    layers[i + 1].weight.data = cast(text.embed_tokens.weight)

        nn_model.to(device)
        return nn_model
