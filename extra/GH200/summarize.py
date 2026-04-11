#!/usr/bin/env python3
"""
    Parse benchmark results (JSONL) and generate a markdown summary report.
"""
import argparse
import os
import sys
from collections import defaultdict

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from extra.GH200.utils import load_results
from extra.GH200.configs import COLUMN_BATCH_SIZES


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize benchmark results")
    parser.add_argument("--results-path", type=str,
                        default="extra/GH200/results/results.jsonl")
    parser.add_argument("--output", type=str,
                        default="extra/GH200/results/summary.md")
    args = parser.parse_args()

    results_path = os.path.join(project_root, args.results_path)
    output_path = os.path.join(project_root, args.output)
    results = load_results(results_path)

    if not results:
        print("No results found.")
        return

    lines: list[str] = []
    lines.append("# GH200 Knowledge Matrix Benchmark Summary\n")
    lines.append(f"Total results: {len(results)}\n")

    ok_results = [r for r in results if r.get("status") == "ok"]
    oom_results = [r for r in results if r.get("status") == "oom"]
    error_results = [r for r in results if r.get("status") not in ("ok", "oom")]

    lines.append(f"- OK: {len(ok_results)}")
    lines.append(f"- OOM: {len(oom_results)}")
    lines.append(f"- Error/Timeout: {len(error_results)}\n")

    # --- Throughput tables ---
    lines.append("## Throughput (matrices/sec) by Model and Column Batch Size\n")
    _throughput_tables(lines, ok_results)

    # --- Memory tables ---
    lines.append("## Peak Memory (GB) by Model and Column Batch Size\n")
    _memory_tables(lines, ok_results)

    # --- HBM vs UM comparison ---
    lines.append("## HBM vs Unified Memory Comparison\n")
    _allocator_comparison(lines, ok_results)

    # --- Sustained workload ---
    lines.append("## Sustained Workload: Metrics vs Number of Samples\n")
    _sustained_workload(lines, ok_results)

    # --- OOM boundary ---
    lines.append("## OOM Boundary Map\n")
    _oom_boundary(lines, results)

    # --- num_classes scaling ---
    lines.append("## num_classes Scaling\n")
    _num_classes_scaling(lines, ok_results)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Summary written to {output_path}")


def _throughput_tables(lines: list[str], results: list[dict]) -> None:
    """Group by (model, input, num_classes, allocator, num_samples=smallest) and show matrices/sec."""
    grouped = defaultdict(dict)
    for r in results:
        key = (r["model_label"], tuple(r["input_size"]), r["num_classes"],
               r["allocator"], r["num_samples"])
        grouped[key][r["column_batch_size"]] = r.get("matrices_per_sec")

    if not grouped:
        lines.append("No throughput data available.\n")
        return

    batch_cols = COLUMN_BATCH_SIZES
    lines.append(f"| Config | " + " | ".join(str(b) for b in batch_cols) + " |")
    lines.append(f"|--------|" + "|".join("--------" for _ in batch_cols) + "|")

    for key in sorted(grouped.keys()):
        model, inp, nc, alloc, ns = key
        inp_str = "x".join(str(x) for x in inp)
        label = f"{model} {inp_str} c{nc} {alloc} n{ns}"
        vals = []
        for b in batch_cols:
            v = grouped[key].get(b)
            vals.append(f"{v:.1f}" if v is not None else "-")
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    lines.append("")


def _memory_tables(lines: list[str], results: list[dict]) -> None:
    """Peak memory per config."""
    grouped = defaultdict(dict)
    for r in results:
        key = (r["model_label"], tuple(r["input_size"]), r["num_classes"],
               r["allocator"], r["num_samples"])
        mem = r.get("peak_mem_allocated_gb") or r.get("peak_rss_gb")
        grouped[key][r["column_batch_size"]] = mem

    if not grouped:
        lines.append("No memory data available.\n")
        return

    batch_cols = COLUMN_BATCH_SIZES
    lines.append(f"| Config | " + " | ".join(str(b) for b in batch_cols) + " |")
    lines.append(f"|--------|" + "|".join("--------" for _ in batch_cols) + "|")

    for key in sorted(grouped.keys()):
        model, inp, nc, alloc, ns = key
        inp_str = "x".join(str(x) for x in inp)
        label = f"{model} {inp_str} c{nc} {alloc} n{ns}"
        vals = []
        for b in batch_cols:
            v = grouped[key].get(b)
            vals.append(f"{v:.2f}" if v is not None else "-")
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    lines.append("")


def _allocator_comparison(lines: list[str], results: list[dict]) -> None:
    """Side-by-side HBM vs UM for matching configs."""
    by_key = {}
    for r in results:
        key = (r["model_label"], tuple(r["input_size"]), r["num_classes"],
               r["column_batch_size"], r["num_samples"])
        by_key.setdefault(key, {})[r["allocator"]] = r

    lines.append("| Config | HBM median ms | UM median ms | HBM peak GB | UM peak GB |")
    lines.append("|--------|---------------|--------------|-------------|------------|")

    for key in sorted(by_key.keys()):
        if len(by_key[key]) < 2:
            continue
        model, inp, nc, cb, ns = key
        inp_str = "x".join(str(x) for x in inp)
        label = f"{model} {inp_str} c{nc} cb{cb} n{ns}"
        hbm = by_key[key].get("default", {})
        um = by_key[key].get("rmm", {})
        hbm_ms = hbm.get("median_matrix_ms")
        um_ms = um.get("median_matrix_ms")
        hbm_mem = hbm.get("peak_mem_allocated_gb") or hbm.get("peak_rss_gb")
        um_mem = um.get("peak_rss_gb")
        lines.append(
            f"| {label} | "
            f"{f'{hbm_ms:.1f}' if hbm_ms else '-'} | "
            f"{f'{um_ms:.1f}' if um_ms else '-'} | "
            f"{f'{hbm_mem:.2f}' if hbm_mem else '-'} | "
            f"{f'{um_mem:.2f}' if um_mem else '-'} |"
        )

    lines.append("")


def _sustained_workload(lines: list[str], results: list[dict]) -> None:
    """Show how metrics change across num_samples for same config."""
    grouped = defaultdict(dict)
    for r in results:
        key = (r["model_label"], tuple(r["input_size"]), r["num_classes"],
               r["column_batch_size"], r["allocator"])
        grouped[key][r["num_samples"]] = r.get("median_matrix_ms")

    if not grouped:
        lines.append("No sustained workload data available.\n")
        return

    sample_counts = sorted({r["num_samples"] for r in results})
    lines.append(f"| Config | " + " | ".join(f"n={s}" for s in sample_counts) + " |")
    lines.append(f"|--------|" + "|".join("--------" for _ in sample_counts) + "|")

    for key in sorted(grouped.keys()):
        model, inp, nc, cb, alloc = key
        inp_str = "x".join(str(x) for x in inp)
        label = f"{model} {inp_str} c{nc} cb{cb} {alloc}"
        vals = []
        for s in sample_counts:
            v = grouped[key].get(s)
            vals.append(f"{v:.1f}" if v is not None else "-")
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    lines.append("")


def _oom_boundary(lines: list[str], results: list[dict]) -> None:
    """For each (model, input, num_classes), find the batch_size where OOM occurs."""
    oom_at = defaultdict(lambda: float("inf"))
    for r in results:
        if r.get("status") == "oom":
            key = (r["model_label"], tuple(r["input_size"]), r["num_classes"], r["allocator"])
            oom_at[key] = min(oom_at[key], r["column_batch_size"])

    if not oom_at:
        lines.append("No OOM events recorded.\n")
        return

    lines.append("| Model | Input | Classes | Allocator | First OOM at batch_size |")
    lines.append("|-------|-------|---------|-----------|------------------------|")
    for key in sorted(oom_at.keys()):
        model, inp, nc, alloc = key
        inp_str = "x".join(str(x) for x in inp)
        lines.append(f"| {model} | {inp_str} | {nc} | {alloc} | {oom_at[key]} |")

    lines.append("")


def _num_classes_scaling(lines: list[str], results: list[dict]) -> None:
    """Show how num_classes affects median ms for same (model, input, batch, allocator)."""
    grouped = defaultdict(dict)
    for r in results:
        key = (r["model_label"], tuple(r["input_size"]), r["column_batch_size"],
               r["allocator"], r["num_samples"])
        grouped[key][r["num_classes"]] = r.get("median_matrix_ms")

    if not grouped:
        lines.append("No num_classes data available.\n")
        return

    class_counts = sorted({r["num_classes"] for r in results})
    lines.append(f"| Config | " + " | ".join(f"c={c}" for c in class_counts) + " |")
    lines.append(f"|--------|" + "|".join("--------" for _ in class_counts) + "|")

    for key in sorted(grouped.keys()):
        model, inp, cb, alloc, ns = key
        inp_str = "x".join(str(x) for x in inp)
        label = f"{model} {inp_str} cb{cb} {alloc} n{ns}"
        vals = []
        for c in class_counts:
            v = grouped[key].get(c)
            vals.append(f"{v:.1f}" if v is not None else "-")
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    lines.append("")


if __name__ == "__main__":
    main()
