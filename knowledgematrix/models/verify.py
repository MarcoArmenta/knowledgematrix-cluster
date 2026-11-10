"""
    Check a real checkpoint end to end: does the knowledge-matrix network
    built by `load_llm` compute the same function as the transformers
    model, and do its knowledge-matrix rows sum to its outputs?

        python -m knowledgematrix.models.verify HuggingFaceTB/SmolLM2-135M
        python -m knowledgematrix.models.verify meta-llama/Llama-3.3-70B-Instruct \\
            --dtype bfloat16 --device cuda:0,cuda:1,cuda:2 --tokens 256

    or, from Python, `verify_llm(...)`, which returns the measurements.

    The two models are never in memory together: the reference logits are
    computed and kept (T x vocab numbers), the transformers model is
    freed, and only then is the knowledge-matrix network loaded. Peak
    memory is therefore one copy of the weights, which is what makes the
    check possible for the checkpoints that barely fit.

    The test suite verifies every supported ARCHITECTURE against
    transformers with scaled-down random models; this tool is for the
    actual WEIGHTS, on the machine that can hold them (the gated and the
    large checkpoints cannot be part of an offline test suite).
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from typing import List, Optional, Sequence, Union

import torch

from knowledgematrix.models.hub import _WRAPPERS, load_llm, resolve_llm
from knowledgematrix.row_computer import KnowledgeRowComputer

DEFAULT_PROMPT = (
    "The knowledge matrix of a neural network at an input is a matrix whose "
    "rows sum to the outputs of the network at that input. Each entry "
    "attributes a part of one output coordinate to one input coordinate, and "
    "the last column collects what the biases and normalizations contribute. "
    "For a language model the inputs are the coordinates of the token "
    "embeddings and the outputs are the logits of the next token at every "
    "position of the sequence."
)

#: default pass threshold on the logit gap relative to the logit scale
TOLERANCES = {
    torch.float64: 1e-5,     # transformers keeps float32 internals (RoPE, norms)
    torch.float32: 1e-3,
    torch.bfloat16: 1e-1,    # half precision is a storage format: loose by nature
    torch.float16: 1e-1,
}

DTYPES = {
    "float64": torch.float64,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _token_ids(source: str, prompt: Optional[str], tokens: Optional[int], hub_kwargs: dict) -> torch.Tensor:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(source, **hub_kwargs)
    text = prompt if prompt is not None else DEFAULT_PROMPT
    ids = tokenizer(text, return_tensors="pt").input_ids
    if tokens is not None:
        while ids.shape[1] < tokens:           # repeat the text to reach the length
            text = text + " " + (prompt if prompt is not None else DEFAULT_PROMPT)
            ids = tokenizer(text, return_tensors="pt").input_ids
        ids = ids[:, :tokens]
    return ids


def _reference_logits(source, ids, dtype, device, hub_kwargs, reference_kwargs) -> torch.Tensor:
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(source, **hub_kwargs)
    if config.model_type in _WRAPPERS:
        from transformers import AutoModelForImageTextToText as auto
    else:
        auto = AutoModelForCausalLM
    kwargs = dict(hub_kwargs)
    kwargs.update(reference_kwargs or {})
    try:
        model = auto.from_pretrained(source, dtype=dtype, **kwargs)
    except TypeError:                           # transformers 4: the kwarg was torch_dtype
        model = auto.from_pretrained(source, torch_dtype=dtype, **kwargs)
    model.eval()
    if "device_map" not in kwargs:
        model.to(device)
    with torch.no_grad():
        logits = model(input_ids=ids.to(model.device)).logits
    logits = logits.detach().to("cpu", torch.float64)
    del model
    _free()
    return logits


def verify_llm(
        source: str,
        prompt: Optional[str] = None,
        tokens: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        device: Union[str, Sequence[str]] = "cpu",
        rows: int = 4,
        mixer_mode: str = "frozen",
        reference_kwargs: Optional[dict] = None,
        **hub_kwargs,
    ) -> dict:
    """
        Compare the network built by `load_llm` with the transformers
        model on one prompt, and check the knowledge-matrix row sums.

        Args:
            source: a supported name, a hub repository id or a local path.
            prompt (str | None): the text (default: a short paragraph).
            tokens (int | None): cut or repeat the text to exactly this
                many tokens. Use a few hundred to exercise what only
                shows at larger positions: RoPE frequency rescaling,
                sliding windows, the chunked linear-attention recurrence.
            dtype: dtype of both models.
            device (str | list[str]): where the knowledge-matrix network
                goes (a list = a pipeline over several devices).
            rows (int): number of knowledge-matrix rows to extract (the
                most likely next tokens at the last position); 0 skips
                the knowledge-matrix check.
            mixer_mode (str): "frozen" or "ratio" (see KnowledgeRowComputer).
            reference_kwargs (dict | None): extra arguments for the
                transformers `from_pretrained` of the reference, e.g.
                {"device_map": "auto"} to spread it over several GPUs.
            **hub_kwargs: revision, local_files_only, token, cache_dir.
        Returns:
            dict of measurements:
              "tokens": sequence length T,
              "logit_scale": max |reference logit|,
              "max_abs_gap", "relative_gap" (= max_abs_gap / logit_scale),
              "worst_position": the position of the largest gap,
              "argmax_agreement": fraction of positions with the same
                  top token,
              "row_sum_gap": max |sum of a knowledge-matrix row - its
                  logit| (None when rows == 0), "row_sum_relative",
              "seconds_reference", "seconds_load", "seconds_rows",
              "passed": relative_gap and row_sum_relative below
                  TOLERANCES[dtype].
    """
    source = resolve_llm(os.fspath(source))
    devices = [device] if isinstance(device, (str, torch.device)) else list(device)
    ids = _token_ids(source, prompt, tokens, hub_kwargs)
    T = ids.shape[1]

    start = time.time()
    reference = _reference_logits(source, ids, dtype, str(devices[0]), hub_kwargs, reference_kwargs)
    seconds_reference = time.time() - start

    start = time.time()
    net = load_llm(source, dtype=dtype, device=device, **hub_kwargs)
    seconds_load = time.time() - start
    x = ids.unsqueeze(0)                         # (1, 1, T)
    with torch.no_grad():
        ours = net.forward(x.to(net.layer_device(0)))
    ours = ours.detach().to("cpu", torch.float64).reshape(reference.shape)

    gap = (ours - reference).abs()
    scale = reference.abs().max().item()
    per_position = gap.reshape(T, -1).max(dim=1).values
    report = {
        "source": source,
        "class": type(net).__name__,
        "dtype": str(dtype).replace("torch.", ""),
        "tokens": T,
        "logit_scale": scale,
        "max_abs_gap": gap.max().item(),
        "relative_gap": gap.max().item() / scale,
        "worst_position": int(per_position.argmax().item()),
        "argmax_agreement": (ours.argmax(-1) == reference.argmax(-1)).double().mean().item(),
        "row_sum_gap": None,
        "row_sum_relative": None,
        "seconds_reference": seconds_reference,
        "seconds_load": seconds_load,
        "seconds_rows": None,
    }

    if rows > 0:
        vocab = ours.shape[-1]
        top = ours.reshape(T, vocab)[-1].topk(rows).indices.tolist()
        indices: List[int] = [(T - 1) * vocab + v for v in top]
        start = time.time()
        A = KnowledgeRowComputer(net, mixer_mode=mixer_mode).forward(x, indices)
        report["seconds_rows"] = time.time() - start
        sums = A.sum(dim=1).detach().to("cpu", torch.float64)
        want = ours.reshape(-1)[indices]
        report["row_sum_gap"] = (sums - want).abs().max().item()
        report["row_sum_relative"] = report["row_sum_gap"] / scale
        report["rows"] = indices

    tolerance = TOLERANCES.get(dtype, 1e-3)
    report["tolerance"] = tolerance
    report["passed"] = bool(
        report["relative_gap"] < tolerance
        and (report["row_sum_relative"] is None or report["row_sum_relative"] < tolerance)
    )
    del net
    _free()
    return report


def format_report(report: dict) -> str:
    lines = [
        f"{report['source']}  ->  {report['class']}  ({report['dtype']}, T = {report['tokens']})",
        f"  logits vs transformers : max |gap| {report['max_abs_gap']:.3e}  "
        f"= {report['relative_gap']:.2e} of the logit scale ({report['logit_scale']:.1f}), "
        f"worst at position {report['worst_position']}",
        f"  same top token         : {100 * report['argmax_agreement']:.1f}% of positions",
    ]
    if report["row_sum_gap"] is not None:
        lines.append(
            f"  knowledge-matrix rows  : max |row sum - logit| {report['row_sum_gap']:.3e}  "
            f"= {report['row_sum_relative']:.2e} of the logit scale ({len(report['rows'])} rows)"
        )
    timing = f"  time                   : reference {report['seconds_reference']:.1f}s, load {report['seconds_load']:.1f}s"
    if report["seconds_rows"] is not None:
        timing += f", rows {report['seconds_rows']:.1f}s"
    lines.append(timing)
    lines.append(f"  {'PASS' if report['passed'] else 'FAIL'} (tolerance {report['tolerance']:.0e} relative)")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m knowledgematrix.models.verify",
        description="Compare the knowledge-matrix network of a checkpoint with "
                    "the transformers model, one model in memory at a time.",
    )
    parser.add_argument("source", nargs="+", help="supported name, hub repository id or local path")
    parser.add_argument("--dtype", default="float32", choices=sorted(DTYPES))
    parser.add_argument("--device", default="cpu",
                        help="device, or a comma-separated pipeline (cuda:0,cuda:1)")
    parser.add_argument("--tokens", type=int, default=None, help="sequence length to test")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--rows", type=int, default=4, help="knowledge-matrix rows to check (0: none)")
    parser.add_argument("--mixer-mode", default="frozen", choices=("frozen", "ratio"))
    parser.add_argument("--reference-device-map", default=None,
                        help='device_map of the transformers reference, e.g. "auto"')
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--revision", default=None)
    args = parser.parse_args(argv)

    devices = [d for d in args.device.split(",") if d]
    hub_kwargs = {}
    if args.local_files_only:
        hub_kwargs["local_files_only"] = True
    if args.revision:
        hub_kwargs["revision"] = args.revision
    reference_kwargs = {"device_map": args.reference_device_map} if args.reference_device_map else None

    ok = True
    for source in args.source:
        report = verify_llm(
            source, prompt=args.prompt, tokens=args.tokens, dtype=DTYPES[args.dtype],
            device=devices[0] if len(devices) == 1 else devices, rows=args.rows,
            mixer_mode=args.mixer_mode, reference_kwargs=reference_kwargs, **hub_kwargs,
        )
        print(format_report(report), flush=True)
        ok = ok and report["passed"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
