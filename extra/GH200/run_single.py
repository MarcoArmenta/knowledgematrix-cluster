#!/usr/bin/env python3
"""
    Subprocess entry point for a single benchmark configuration.

    Handles RMM-before-torch import ordering: when --allocator=rmm, RMM must
    be initialized BEFORE any torch import. Since knowledgematrix modules import
    torch at module level, all knowledgematrix imports are deferred until after
    the RMM/torch setup block.
"""
from __future__ import annotations
import argparse
import json
import os
import statistics
import sys
import time


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments. No torch imports here."""
    parser = argparse.ArgumentParser(description="Single benchmark configuration runner")
    parser.add_argument("--base-width", type=int, required=True)
    parser.add_argument("--blocks-per-stage", type=str, required=True,
                        help="JSON list, e.g. '[2,2,2,2]'")
    parser.add_argument("--input-size", type=str, required=True,
                        help="Comma-separated, e.g. '3,64,64'")
    parser.add_argument("--column-batch-size", type=int, required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--allocator", choices=["default", "rmm"], required=True)
    parser.add_argument("--results-path", type=str, required=True)
    parser.add_argument("--warmup-samples", type=int, default=2)
    parser.add_argument("--enable-profiler", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    blocks_per_stage = json.loads(args.blocks_per_stage)
    input_size = tuple(int(x) for x in args.input_size.split(","))

    # --- RMM must be initialized BEFORE any torch import ---
    if args.allocator == "rmm":
        import rmm
        from rmm.allocators.torch import rmm_torch_allocator
        rmm.reinitialize(pool_allocator=True, managed_memory=True)
        import torch
        torch.cuda.memory.change_current_allocator(rmm_torch_allocator)
    else:
        import torch

    # NOW safe to import knowledgematrix (which imports torch internally)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
    from extra.GH200.utils import append_result, now_iso, get_peak_rss_gb

    # Lazy import to avoid circular issues with NN base
    from extra.GH200.models.custom_resnet import CustomResNet

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build result template
    depth_label = 2 + sum(2 * b for b in blocks_per_stage)
    model_label = f"R{depth_label}-w{args.base_width}"
    result = {
        "timestamp": now_iso(),
        "model_label": model_label,
        "base_width": args.base_width,
        "blocks_per_stage": blocks_per_stage,
        "num_classes": args.num_classes,
        "input_size": list(input_size),
        "column_batch_size": args.column_batch_size,
        "num_samples": args.num_samples,
        "allocator": args.allocator,
        "status": "ok",
        "median_matrix_ms": None,
        "mean_matrix_ms": None,
        "p95_matrix_ms": None,
        "total_time_s": None,
        "columns_per_sec": None,
        "matrices_per_sec": None,
        "peak_mem_allocated_gb": None,
        "peak_mem_reserved_gb": None,
        "peak_rss_gb": None,
        "model_params": None,
        "matrix_shape": None,
        "sanity_check_diff": None,
        "samples_completed": 0,
        "oom_at_sample": None,
    }

    try:
        # Build model
        model = CustomResNet(
            input_shape=input_size,
            num_classes=args.num_classes,
            base_width=args.base_width,
            blocks_per_stage=blocks_per_stage,
            device=device,
        )
        model.eval()  # NN.eval() returns None — standalone call only
        model.to(device)
        result["model_params"] = sum(p.numel() for p in model.parameters())

        # Create computer
        computer = KnowledgeMatrixComputer(
            model,
            batch_size=args.column_batch_size,
            device=device,
        )

        # Reset memory stats (only works with default allocator)
        if args.allocator != "rmm" and device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # Warmup
        for _ in range(args.warmup_samples):
            x = torch.randn(input_size, device=device, dtype=torch.float32)
            try:
                mat = computer.forward(x)
                del mat
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                result["status"] = "oom"
                result["oom_at_sample"] = -1  # OOM during warmup
                _finalize(result, args, torch)
                return

        # Timed loop
        times_ms: list[float] = []
        total_start = time.monotonic()

        for i in range(args.num_samples):
            x = torch.randn(input_size, device=device, dtype=torch.float32)
            try:
                if device == "cuda":
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    mat = computer.forward(x)
                    end_event.record()
                    torch.cuda.synchronize()
                    elapsed_ms = start_event.elapsed_time(end_event)
                else:
                    t0 = time.monotonic()
                    mat = computer.forward(x)
                    elapsed_ms = (time.monotonic() - t0) * 1000

                times_ms.append(elapsed_ms)

                # Sanity check on first sample
                if i == 0:
                    model.save = True
                    out = model.forward(x)
                    model.save = False
                    diff = torch.norm(out - mat.sum(1)).item()
                    result["sanity_check_diff"] = diff
                    result["matrix_shape"] = list(mat.shape)

                del mat
                result["samples_completed"] = i + 1

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                result["status"] = "oom"
                result["oom_at_sample"] = i
                break

        total_elapsed = time.monotonic() - total_start
        result["total_time_s"] = round(total_elapsed, 3)

        # Compute timing stats
        if times_ms:
            sorted_times = sorted(times_ms)
            result["median_matrix_ms"] = round(statistics.median(sorted_times), 3)
            result["mean_matrix_ms"] = round(statistics.mean(sorted_times), 3)
            p95_idx = int(len(sorted_times) * 0.95)
            result["p95_matrix_ms"] = round(sorted_times[min(p95_idx, len(sorted_times) - 1)], 3)

            total_columns = input_size[0] * input_size[1] * input_size[2]
            median_s = result["median_matrix_ms"] / 1000
            if median_s > 0:
                result["columns_per_sec"] = round(total_columns / median_s, 1)
                result["matrices_per_sec"] = round(1 / median_s, 3)

        # Memory stats
        if device == "cuda" and args.allocator != "rmm":
            result["peak_mem_allocated_gb"] = round(
                torch.cuda.max_memory_allocated() / (1024 ** 3), 3
            )
            result["peak_mem_reserved_gb"] = round(
                torch.cuda.max_memory_reserved() / (1024 ** 3), 3
            )
        result["peak_rss_gb"] = get_peak_rss_gb()

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        result["status"] = "oom"
        result["oom_at_sample"] = -1  # OOM during model creation
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    _finalize(result, args, torch)

    # Optional profiler trace
    if args.enable_profiler and result["status"] == "ok" and device == "cuda":
        _run_profiler(computer, input_size, device, torch, project_root, model_label, args)


def _finalize(result: dict, args: argparse.Namespace, torch_module) -> None:
    """Write result to JSONL file and stdout."""
    from extra.GH200.utils import append_result
    append_result(result, args.results_path)
    print(json.dumps(result))


def _run_profiler(
    computer, input_size: tuple, device: str, torch_module,
    project_root: str, model_label: str, args: argparse.Namespace
) -> None:
    """Run one extra sample under torch.profiler and export Chrome trace."""
    traces_dir = os.path.join(project_root, "extra", "GH200", "results", "traces")
    os.makedirs(traces_dir, exist_ok=True)

    x = torch_module.randn(input_size, device=device, dtype=torch_module.float32)
    trace_name = (
        f"{model_label}_{'x'.join(str(s) for s in input_size)}"
        f"_cb{args.column_batch_size}_c{args.num_classes}_{args.allocator}"
    )
    trace_path = os.path.join(traces_dir, f"{trace_name}.json")

    with torch_module.profiler.profile(
        activities=[
            torch_module.profiler.ProfilerActivity.CPU,
            torch_module.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        mat = computer.forward(x)
        del mat

    prof.export_chrome_trace(trace_path)


if __name__ == "__main__":
    main()
