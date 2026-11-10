"""
    One entry point for the pretrained language models whose knowledge
    matrices the library can compute.

        from knowledgematrix.models.hub import load_llm
        from knowledgematrix.row_computer import KnowledgeRowComputer

        net = load_llm("SmolLM2-135M")                     # or a hub id / path
        x = tokenizer("The capital of France is", return_tensors="pt").input_ids
        rows = KnowledgeRowComputer(net).forward(x.unsqueeze(0), rows=[...])

    `load_llm` reads the checkpoint's config, picks the network class of
    its architecture family and streams the weights into it:

        family     config.model_type                   class
        ---------  ----------------------------------  ---------
        llama      llama, mistral, qwen2, qwen3        LlamaLike
        gemma      gemma2, gemma3, gemma3_text         GemmaLike
        qwen3_5    qwen3_5, qwen3_5_text               Qwen3_5

    so any checkpoint of a supported architecture loads -- an instruct
    variant, a fine-tune, an ungated mirror -- not only the ones named in
    `SUPPORTED_LLMS`. That table lists the checkpoints the test suite
    builds from their published configs.

    `llm_footprint` answers "what does it take to hold this model" from
    the config alone (no weights are downloaded), including how a model
    too large for one GPU splits over several.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import torch
from torch import nn

from knowledgematrix.neural_net import NN


@dataclass(frozen=True)
class LLM:
    """A supported checkpoint: its hub repository and architecture family."""
    repo: str
    family: str
    #: the repository requires accepting a license on the hub (pass a
    #: token, or use an ungated mirror of the same weights)
    gated: bool = False


#: The checkpoints the library is tested against (their published
#: config.json files are in extra/tests/fixtures/hf_configs).
SUPPORTED_LLMS: Dict[str, LLM] = {
    llm.repo: llm for llm in (
        LLM("Qwen/Qwen3.5-4B", "qwen3_5"),
        LLM("Qwen/Qwen3.5-27B", "qwen3_5"),
        LLM("google/gemma-2-2b", "gemma", gated=True),
        LLM("google/gemma-2-9b", "gemma", gated=True),
        LLM("google/gemma-2-27b", "gemma", gated=True),
        LLM("google/gemma-3-270m", "gemma", gated=True),
        LLM("google/gemma-3-1b-it", "gemma", gated=True),
        LLM("google/gemma-3-4b-it", "gemma", gated=True),
        LLM("google/gemma-3-12b-it", "gemma", gated=True),
        LLM("google/gemma-3-27b-it", "gemma", gated=True),
        LLM("Qwen/Qwen3-8B", "llama"),
        LLM("Qwen/Qwen3-14B", "llama"),
        LLM("meta-llama/Llama-3.3-70B-Instruct", "llama", gated=True),
        LLM("meta-llama/Llama-3.2-1B", "llama", gated=True),
        LLM("meta-llama/Llama-3.2-3B", "llama", gated=True),
        LLM("meta-llama/Llama-3.1-8B", "llama", gated=True),
        LLM("HuggingFaceTB/SmolLM2-135M", "llama"),
        LLM("HuggingFaceTB/SmolLM2-360M", "llama"),
        LLM("HuggingFaceTB/SmolLM2-1.7B", "llama"),
    )
}

#: transformers `config.model_type` -> architecture family
FAMILIES: Dict[str, str] = {
    "llama": "llama",
    "mistral": "llama",
    "qwen2": "llama",
    "qwen3": "llama",
    "gemma2": "gemma",
    "gemma3": "gemma",
    "gemma3_text": "gemma",
    "qwen3_5": "qwen3_5",
    "qwen3_5_text": "qwen3_5",
}

#: model types that are multimodal wrappers around a text decoder
_WRAPPERS = ("gemma3", "qwen3_5")


def family_class(family: str):
    """The network class of an architecture family ("llama", "gemma", "qwen3_5")."""
    if family == "llama":
        from knowledgematrix.models.llama import LlamaLike
        return LlamaLike
    if family == "gemma":
        from knowledgematrix.models.gemma import GemmaLike
        return GemmaLike
    if family == "qwen3_5":
        from knowledgematrix.models.qwen3_5 import Qwen3_5
        return Qwen3_5
    raise ValueError(f"unknown family {family!r}.")


def llm_class(model_type: str):
    """The network class for a transformers `config.model_type`."""
    if model_type not in FAMILIES:
        raise ValueError(
            f"model_type {model_type!r} is not supported. Supported: "
            f"{', '.join(sorted(FAMILIES))}."
        )
    return family_class(FAMILIES[model_type])


def resolve_llm(name: str) -> str:
    """
        The hub repository for a supported checkpoint given by its full id
        or its bare name, case-insensitively: "Llama-3.3-70B-Instruct",
        "llama-3.3-70b-instruct" and "meta-llama/Llama-3.3-70B-Instruct"
        all give "meta-llama/Llama-3.3-70B-Instruct". Anything else -- a
        local directory, a repository that is not in the table -- is
        returned unchanged.
    """
    if os.path.isdir(name):
        return name
    lowered = name.lower()
    for repo in SUPPORTED_LLMS:
        if lowered == repo.lower() or lowered == repo.split("/", 1)[1].lower():
            return repo
    return name


def _model_type(source, hub_kwargs: dict) -> str:
    if isinstance(source, nn.Module):
        return source.config.model_type
    try:
        from transformers import AutoConfig
    except ImportError as e:
        raise ImportError(
            "Loading a pretrained language model requires the optional "
            "dependency `transformers` (pip install knowledgematrix[hf])."
        ) from e
    kwargs = {} if os.path.isdir(source) else hub_kwargs
    return AutoConfig.from_pretrained(source, **kwargs).model_type


def load_llm(
        source,
        include_lm_head: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Union[str, Sequence[str]] = "cpu",
        save: bool = False,
        copy: bool = True,
        boundaries: Optional[List[int]] = None,
        **hub_kwargs,
    ) -> NN:
    """
        Build the knowledge-matrix network of a pretrained language model.

        Args:
            source: the name of a supported checkpoint ("SmolLM2-135M",
                "Llama-3.3-70B-Instruct", see `SUPPORTED_LLMS`), any hub
                repository id or local directory holding a checkpoint of
                a supported architecture, or a transformers model already
                in memory.
            include_lm_head (bool): end the network with the LM head
                (output: logits, T * vocab rows) or at the final norm
                (output: hidden states, T * hidden rows). The head is a
                linear map, so for large models it is cheaper to stop at
                the hidden states and push the few rows you need through
                it afterwards.
            dtype: dtype of the network. Default float32 when reading a
                checkpoint from disk; pass torch.bfloat16 for the models
                that only fit in half precision (the row-sum identity
                then holds to ~1e-2 instead of ~1e-6).
            device (str | list[str]): one device, or the stages of a
                pipeline over several (["cuda:0", "cuda:1", ...]) for a
                model that does not fit one; each tensor is read from
                disk and sent straight to its stage.
            save (bool): the `save` flag of the network.
            copy (bool): for a model in memory, copy its weights (True)
                or share them (False).
            boundaries (list[int] | None): pipeline cut indices (see
                NN.shard); default balances the parameter count.
            **hub_kwargs: revision, local_files_only, token, cache_dir.
        Returns:
            NN: a LlamaLike, GemmaLike or Qwen3_5.
    """
    if isinstance(source, (str, os.PathLike)):
        source = resolve_llm(os.fspath(source))
    cls = llm_class(_model_type(source, hub_kwargs))
    return cls.from_huggingface(
        source,
        include_lm_head=include_lm_head,
        dtype=dtype,
        device=device,
        save=save,
        copy=copy,
        boundaries=boundaries,
        **hub_kwargs,
    )


def llm_footprint(
        source,
        dtype: torch.dtype = torch.float32,
        devices: Optional[Sequence[str]] = None,
        include_lm_head: bool = True,
        boundaries: Optional[List[int]] = None,
        **hub_kwargs,
    ) -> dict:
    """
        The size of the knowledge-matrix network of a checkpoint, from its
        config alone: no weights are downloaded and no memory is used
        (both the reference model and ours are built on torch's "meta"
        device). Use it to size a job before launching it.

        Args:
            source: as in `load_llm` (a name, a repository id, a path, or
                a transformers config object).
            dtype: the dtype the network would be loaded in.
            devices (list[str] | None): a pipeline of devices; the report
                then includes the weights each stage would hold under the
                default (or the given) cut points.
            include_lm_head (bool): as in `load_llm`.
            boundaries (list[int] | None): as in `load_llm`.
        Returns:
            dict with "class" (str), "num_layers" (int), "parameters"
            (int), "bytes" (int), "gib" (float), and, when `devices` is
            given, "boundaries" (list[int]) and "per_device" (a list of
            {"device", "layers", "parameters", "gib"}).

        The figures are WEIGHTS only. A knowledge-matrix row costs about
        one forward and one backward pass on top (activations saved for
        every layer), which grows with the sequence length.
    """
    try:
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as e:
        raise ImportError(
            "llm_footprint requires the optional dependency `transformers` "
            "(pip install knowledgematrix[hf])."
        ) from e

    # the base class was renamed between transformers 4 and 5
    config_class = getattr(transformers, "PreTrainedConfig", None) or transformers.PretrainedConfig
    if isinstance(source, config_class):
        config = source
    else:
        source = resolve_llm(os.fspath(source))
        kwargs = {} if os.path.isdir(source) else hub_kwargs
        config = AutoConfig.from_pretrained(source, **kwargs)
    cls = llm_class(config.model_type)

    if config.model_type in _WRAPPERS:
        from transformers import AutoModelForImageTextToText as auto
    else:
        auto = AutoModelForCausalLM
    with torch.device("meta"):
        reference = auto.from_config(config)
    net = cls.from_huggingface(reference, include_lm_head=include_lm_head, device="meta")

    itemsize = torch.empty((), dtype=dtype).element_size()
    counts = [sum(p.numel() for p in layer.parameters()) for layer in net.layers]
    total = sum(counts)
    report = {
        "class": cls.__name__,
        "num_layers": net.get_num_layers(),
        "parameters": total,
        "bytes": total * itemsize,
        "gib": total * itemsize / 2 ** 30,
    }
    if devices is not None and len(devices) > 1:
        layer_devices, cuts = net.shard_layout(list(devices), boundaries)
        edges = [0] + cuts + [len(counts)]
        report["boundaries"] = cuts
        report["per_device"] = [
            {
                "device": str(devices[k]),
                "layers": edges[k + 1] - edges[k],
                "parameters": sum(counts[edges[k]:edges[k + 1]]),
                "gib": sum(counts[edges[k]:edges[k + 1]]) * itemsize / 2 ** 30,
            }
            for k in range(len(devices))
        ]
    return report
