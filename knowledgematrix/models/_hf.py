"""
    Shared plumbing of the Hugging Face loaders (models/llama.py,
    models/gemma.py, models/qwen3_5.py).

    A loader has to do three things, and each one has a failure mode that
    stays invisible on a small model and is fatal on a large one:

    1. BUILD the network. `empty_network` builds it without storage (its
       parameters live on torch's "meta" device: shapes, no memory), so a
       70B-parameter model costs nothing to construct instead of 280 GB of
       float32 that is about to be overwritten.

    2. READ the checkpoint. `open_checkpoint` gives uniform access to the
       tensors by their state-dict key, whether the checkpoint is a
       transformers model already in memory or safetensors files on disk.
       From disk the tensors are read ONE AT A TIME, so the transformers
       model is never instantiated and peak host memory is one tensor.

    3. PLACE the tensors. `WeightLoader.put` casts a tensor and sends it
       straight to the device of the layer that owns it -- one device, or
       a pipeline over several (NN.shard_layout) for a model that does not
       fit one GPU. Every shape is checked against the network, and
       `finish` refuses a network with a parameter nobody assigned, so a
       loader cannot silently leave random weights behind.

    Writing a loader:

        ckpt = open_checkpoint(source)             # model | repo id | path
        cfg = ckpt.text_config
        net = empty_network(cls, hidden_size=cfg.hidden_size, ...)
        w = WeightLoader(net, ckpt, dtype=dtype, device=device, copy=copy)
        p = ckpt.text_prefix                       # "model." etc.
        w.put(0, "weight", p + "embed_tokens.weight")
        w.put(2, "q_proj.weight", p + "layers.0.self_attn.q_proj.weight")
        ...
        return w.finish()

    Only `transformers` (for the config) and `safetensors` /
    `huggingface_hub` (for files on disk or on the hub) are needed, and
    only when a loader is actually called.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

import torch
from torch import nn

from knowledgematrix.neural_net import NN, SUPPORTED_ROPE_TYPES


### Building without storage ###

def empty_network(cls, **kwargs) -> NN:
    """
        Build `cls(**kwargs)` with every parameter on the "meta" device:
        the modules, their shapes and the residual wiring exist, the
        storage does not. The result cannot be evaluated until a
        `WeightLoader` has assigned every parameter.

        The constructor must not evaluate the network (shape probes cannot
        run without data): residual connections have to be registered with
        `residual(start, end, same_shape=True)`.
    """
    with torch.device("meta"):
        return cls(**kwargs)


### Reading a checkpoint ###

class Checkpoint:
    """
        Read access to the tensors of a causal language model checkpoint,
        by state-dict key.

        Attributes:
            config: the transformers config (as normalized by the installed
                transformers version).
            in_memory (bool): True when the tensors belong to a live
                transformers model (they are returned by reference).
    """

    def __init__(self, config, in_memory: bool) -> None:
        self.config = config
        self.in_memory = in_memory
        self._text_prefix: Optional[str] = None

    # -- storage interface, implemented by the two subclasses below --

    def keys(self) -> Iterable[str]:
        raise NotImplementedError

    def __contains__(self, key: str) -> bool:
        raise NotImplementedError

    def tensor(self, key: str) -> torch.Tensor:
        """The tensor stored under `key`: detached, in its stored dtype."""
        raise NotImplementedError

    def close(self) -> None:
        pass

    # -- where the text decoder lives --

    @property
    def text_config(self):
        """The config of the text decoder (multimodal checkpoints wrap it)."""
        return getattr(self.config, "text_config", None) or self.config

    @property
    def text_prefix(self) -> str:
        """
            The key prefix of the text decoder, such that
            `text_prefix + "embed_tokens.weight"` and
            `text_prefix + "layers.0. ..."` are keys: "model." for a
            causal LM, "model.language_model." for a multimodal wrapper
            (or "language_model.model." in checkpoints saved by older
            transformers versions), "" for a bare text model.
        """
        if self._text_prefix is None:
            suffix = "embed_tokens.weight"
            keys = list(self.keys())
            candidates = []
            for key in keys:
                if key.endswith(suffix):
                    prefix = key[: -len(suffix)]
                    if any(k.startswith(prefix + "layers.0.") for k in keys):
                        candidates.append(prefix)
            if len(candidates) != 1:
                raise ValueError(
                    f"could not locate the text decoder in the checkpoint: "
                    f"{len(candidates)} candidate prefixes {candidates} "
                    f"(looked for '<prefix>embed_tokens.weight' next to "
                    f"'<prefix>layers.0.*')."
                )
            self._text_prefix = candidates[0]
        return self._text_prefix

    @property
    def lm_head_key(self) -> str:
        """
            The key of the LM head matrix (V, D). Checkpoints with tied
            embeddings do not store one: the embedding table is the head.
        """
        prefix = self.text_prefix
        # "model.language_model." -> the head sits next to "model.";
        # "language_model.model." -> next to "language_model.".
        roots = {""}
        parts = prefix.split(".")[:-1]
        for n in range(len(parts) + 1):
            roots.add(".".join(parts[:n]) + ("." if n else ""))
        found = [r + "lm_head.weight" for r in sorted(roots) if (r + "lm_head.weight") in self]
        if len(found) > 1:
            raise ValueError(f"ambiguous LM head: {found}.")
        if found:
            return found[0]
        return prefix + "embed_tokens.weight"

    def has_text(self, key: str) -> bool:
        """Whether `text_prefix + key` is stored."""
        return (self.text_prefix + key) in self


class _ModelCheckpoint(Checkpoint):
    """The tensors of a transformers model held in memory."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__(config=model.config, in_memory=True)
        # references, not copies; includes the non-persistent buffers'
        # absence and the tied lm_head.weight's presence
        self._state: Dict[str, torch.Tensor] = dict(model.state_dict())

    def keys(self) -> Iterable[str]:
        return self._state.keys()

    def __contains__(self, key: str) -> bool:
        return key in self._state

    def tensor(self, key: str) -> torch.Tensor:
        return self._state[key].detach()


class _SafetensorsCheckpoint(Checkpoint):
    """The tensors of a checkpoint stored as safetensors files, read lazily."""

    def __init__(self, config, files: Sequence[str], key_to_file: Optional[Dict[str, str]] = None) -> None:
        super().__init__(config=config, in_memory=False)
        from safetensors import safe_open
        self._open = safe_open
        self._handles: Dict[str, object] = {}
        if key_to_file is None:
            key_to_file = {}
            for path in files:
                for key in self._handle(path).keys():
                    key_to_file[key] = path
        self._key_to_file = key_to_file

    def _handle(self, path: str):
        if path not in self._handles:
            self._handles[path] = self._open(path, framework="pt", device="cpu")
        return self._handles[path]

    def keys(self) -> Iterable[str]:
        return self._key_to_file.keys()

    def __contains__(self, key: str) -> bool:
        return key in self._key_to_file

    def tensor(self, key: str) -> torch.Tensor:
        return self._handle(self._key_to_file[key]).get_tensor(key)

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.__exit__(None, None, None)
            except Exception:
                pass
        self._handles = {}


def _require(module: str, purpose: str):
    try:
        return __import__(module)
    except ImportError as e:
        raise ImportError(
            f"{purpose} requires the optional dependency `{module}` "
            f"(pip install knowledgematrix[hf])."
        ) from e


def _locate(source: str, filename: str, hub_kwargs: dict) -> Optional[str]:
    """Path of `filename` in a local directory or a hub repository, or None."""
    if os.path.isdir(source):
        path = os.path.join(source, filename)
        return path if os.path.isfile(path) else None
    hub = _require("huggingface_hub", "Loading a checkpoint from the Hugging Face hub")
    try:
        from huggingface_hub.errors import EntryNotFoundError
    except ImportError:  # older huggingface_hub
        from huggingface_hub.utils import EntryNotFoundError
    try:
        return hub.hf_hub_download(source, filename, **hub_kwargs)
    except EntryNotFoundError:
        return None


def open_checkpoint(
        source,
        revision: Optional[str] = None,
        local_files_only: bool = False,
        token: Union[str, bool, None] = None,
        cache_dir: Optional[str] = None,
    ) -> Checkpoint:
    """
        Open a causal-LM checkpoint for reading.

        Args:
            source: a transformers model (its tensors are used in place),
                a local directory written by `save_pretrained`, or a hub
                repository id ("meta-llama/Llama-3.3-70B-Instruct").
            revision, local_files_only, token, cache_dir: forwarded to
                the hub when `source` is a repository id. Gated
                repositories (Llama, Gemma) need a token with access --
                `huggingface-cli login` once, or pass it here.
        Returns:
            Checkpoint
    """
    if isinstance(source, nn.Module):
        if not hasattr(source, "config"):
            raise ValueError(f"{type(source).__name__} has no .config; expected a transformers model.")
        return _ModelCheckpoint(source)

    source = os.fspath(source)
    transformers = _require("transformers", "Loading a pretrained language model")
    _require("safetensors", "Reading a checkpoint from disk")
    hub_kwargs = {}
    if not os.path.isdir(source):
        hub_kwargs = dict(revision=revision, local_files_only=local_files_only,
                          token=token, cache_dir=cache_dir)
    config = transformers.AutoConfig.from_pretrained(source, **hub_kwargs)

    index = _locate(source, "model.safetensors.index.json", hub_kwargs)
    if index is not None:
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
        paths: Dict[str, str] = {}
        for name in sorted(set(weight_map.values())):
            path = _locate(source, name, hub_kwargs)
            if path is None:
                raise FileNotFoundError(f"{source}: weight file {name} listed in the index is missing.")
            paths[name] = path
        key_to_file = {key: paths[name] for key, name in weight_map.items()}
        return _SafetensorsCheckpoint(config, list(paths.values()), key_to_file)

    single = _locate(source, "model.safetensors", hub_kwargs)
    if single is None:
        raise FileNotFoundError(
            f"{source}: no safetensors weights found (model.safetensors or "
            f"model.safetensors.index.json). For a checkpoint in another "
            f"format, load it with transformers and pass the model instead."
        )
    return _SafetensorsCheckpoint(config, [single])


### Reading the architecture ###

def rope_settings(config, layer_type: Optional[str] = None) -> dict:
    """
        The rotary position embedding of a text decoder, read from its
        transformers config in a version-independent way.

        transformers 5 keeps everything in `config.rope_parameters` --
        either one dict, or one dict per layer type (Gemma-3:
        "sliding_attention" / "full_attention"); transformers 4 used the
        attributes `rope_theta`, `rope_scaling`, `partial_rotary_factor`
        and Gemma-3's `rope_local_base_freq`. Reading `config.rope_theta`
        on transformers 5 finds nothing, and a default of 10000 is then
        the wrong model for every checkpoint that uses another base
        (Llama-3: 500000, Qwen3: 1e6, SmolLM2: 1e5, Qwen3.5: 1e7).

        Args:
            config: the text config.
            layer_type (str | None): for configs with per-layer-type RoPE.
        Returns:
            dict with
              "theta" (float): the base,
              "scaling" (dict | None): frequency rescaling, in the format
                  RopeAttention(rope_scaling=...) takes (None = plain),
              "partial_rotary_factor" (float).
        Raises:
            NotImplementedError: for a RoPE type the library does not
                reproduce (yarn, dynamic, longrope, proportional).
    """
    params = getattr(config, "rope_parameters", None)
    if params:
        params = dict(params)
        per_layer = [k for k, v in params.items() if isinstance(v, dict)]
        if per_layer:
            if layer_type is None:
                raise ValueError(
                    f"this config has one RoPE per layer type ({per_layer}); "
                    f"pass layer_type."
                )
            if layer_type not in params:
                raise KeyError(f"no RoPE parameters for layer type {layer_type!r} (have {per_layer}).")
            params = dict(params[layer_type])
    else:
        # transformers 4 attributes
        scaling = getattr(config, "rope_scaling", None)
        theta = getattr(config, "rope_theta", None)
        if layer_type == "sliding_attention" and getattr(config, "rope_local_base_freq", None) is not None:
            theta, scaling = config.rope_local_base_freq, None
        if theta is None:
            raise ValueError(
                f"{type(config).__name__} defines neither rope_parameters nor "
                f"rope_theta; cannot determine the RoPE base."
            )
        params = dict(scaling or {})
        params["rope_theta"] = theta
        if getattr(config, "partial_rotary_factor", None) is not None:
            params.setdefault("partial_rotary_factor", config.partial_rotary_factor)

    if "rope_theta" not in params:
        raise ValueError(f"RoPE parameters without rope_theta: {params}.")
    rope_type = params.get("rope_type", params.get("type", "default")) or "default"
    if rope_type not in SUPPORTED_ROPE_TYPES:
        raise NotImplementedError(
            f"RoPE type {rope_type!r} is not supported (supported: "
            f"{', '.join(SUPPORTED_ROPE_TYPES)})."
        )
    scaling = None
    if rope_type != "default":
        scaling = {"rope_type": rope_type, "factor": float(params["factor"])}
        if rope_type == "llama3":
            for key in ("low_freq_factor", "high_freq_factor", "original_max_position_embeddings"):
                scaling[key] = float(params[key])
    return {
        "theta": float(params["rope_theta"]),
        "scaling": scaling,
        "partial_rotary_factor": float(params.get("partial_rotary_factor", 1.0)),
    }


### Placing the tensors ###

_HALF = (torch.float16, torch.bfloat16)


class WeightLoader:
    """
        Assigns checkpoint tensors to the parameters of a network built
        with `empty_network`.

        Args:
            net (NN): the network (parameters on the "meta" device).
            checkpoint (Checkpoint): where the tensors come from.
            dtype (torch.dtype | None): dtype of the network. None keeps
                the dtype of an in-memory model, and means float32 for a
                checkpoint read from disk (those are usually stored in
                bfloat16, which is a storage format: on CPU it is slow,
                and the knowledge matrix row-sum identity only holds to
                ~1e-2 in it). Pass torch.bfloat16 explicitly for the
                models that only fit in half precision.
            device (str | Sequence[str]): one device, or the stages of a
                pipeline over several -- then the network comes out
                sharded exactly as NN.shard(devices, boundaries) would
                leave it, without ever being assembled on one device.
            boundaries (List[int] | None): cut indices of the pipeline
                (see NN.shard); default balances the parameter count.
            copy (bool): for an in-memory model, copy the tensors (True,
                the network is independent of the model) or share them
                (False: no extra memory; the network then aliases the
                model's parameters wherever no cast or move is needed).
    """

    def __init__(
            self,
            net: NN,
            checkpoint: Checkpoint,
            dtype: Optional[torch.dtype] = None,
            device: Union[str, Sequence[str]] = "cpu",
            boundaries: Optional[List[int]] = None,
            copy: bool = True,
        ) -> None:
        self.net = net
        self.checkpoint = checkpoint
        if dtype is None and not checkpoint.in_memory:
            dtype = torch.float32
        self.dtype = dtype
        self.copy = copy
        if isinstance(device, (str, torch.device)):
            devices = [str(device)]
        else:
            devices = [str(d) for d in device]
        if len(devices) == 0:
            raise ValueError("need at least one device.")
        if len(devices) == 1:
            if boundaries:
                raise ValueError("boundaries were given for a single device.")
            self.layer_devices = None
            self.boundaries = None
        else:
            self.layer_devices, self.boundaries = net.shard_layout(devices, boundaries)
        self.devices = devices

    def _device(self, layer: int) -> str:
        return self.devices[0] if self.layer_devices is None else self.layer_devices[layer]

    def _owner(self, layer: int, attr: str):
        module = self.net.layers[layer]
        *path, name = attr.split(".")
        for part in path:
            module = getattr(module, part)
        if name not in module._parameters or module._parameters[name] is None:
            raise AttributeError(
                f"layer {layer} ({type(self.net.layers[layer]).__name__}) has no parameter {attr!r}."
            )
        return module, name

    def put(
            self,
            layer: int,
            attr: str,
            key: str,
            transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        ) -> None:
        """
            Assign the checkpoint tensor `key` to the parameter `attr`
            ("weight", "q_proj.weight", ...) of `net.layers[layer]`.

            Args:
                transform: applied to the tensor before it is stored, for
                    the conventions that are folded into the weights
                    (Gemma's and Qwen3.5's zero-centered RMSNorm,
                    w -> 1 + w; Gemma's sqrt(d) embedding scale). It is
                    evaluated in at least float32 and the result is then
                    cast, so a half-precision network gets the correctly
                    rounded folded weight.
        """
        if key not in self.checkpoint:
            raise KeyError(f"the checkpoint has no tensor {key!r} (wanted for layer {layer}, {attr}).")
        self.put_tensor(layer, attr, self.checkpoint.tensor(key), transform=transform,
                        _borrowed=self.checkpoint.in_memory, _label=key)

    def put_tensor(
            self,
            layer: int,
            attr: str,
            tensor: torch.Tensor,
            transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            _borrowed: bool = True,
            _label: str = "<tensor>",
        ) -> None:
        """`put` for a tensor the caller already holds."""
        module, name = self._owner(layer, attr)
        target = module._parameters[name]
        if tuple(tensor.shape) != tuple(target.shape):
            raise ValueError(
                f"shape mismatch for layer {layer} {attr}: the network expects "
                f"{tuple(target.shape)}, {_label} is {tuple(tensor.shape)}."
            )
        source = tensor.detach()
        dtype = self.dtype if self.dtype is not None else source.dtype
        device = self._device(layer)
        out = source
        if transform is not None:
            work = dtype
            if work in _HALF:
                work = torch.float32
            if source.dtype == torch.float64:
                work = torch.float64
            out = transform(out.to(device=device, dtype=work))
        out = out.to(device=device, dtype=dtype)
        if self.copy and _borrowed and out.data_ptr() == source.data_ptr():
            out = out.clone()
        module._parameters[name] = nn.Parameter(out, requires_grad=target.requires_grad)

    def finish(self) -> NN:
        """
            Check that every parameter received a tensor, record where the
            network lives, and return it.
        """
        missing = [name for name, p in self.net.named_parameters() if p.is_meta]
        missing += [name for name, b in self.net.named_buffers() if b.is_meta]
        if missing:
            shown = ", ".join(missing[:8]) + (", ..." if len(missing) > 8 else "")
            raise RuntimeError(
                f"{len(missing)} parameter(s) of {type(self.net).__name__} were "
                f"not assigned by the loader: {shown}."
            )
        self.net.layer_devices = self.layer_devices
        self.net.shard_boundaries = self.boundaries
        self.net.device = self.devices[0]
        if not self.checkpoint.in_memory:
            self.checkpoint.close()
        return self.net
