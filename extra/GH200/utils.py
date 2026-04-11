"""
    Benchmark utilities: JSONL logging, memory helpers, subprocess runner.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from datetime import datetime, timezone


def append_result(result: dict, results_path: str) -> None:
    """Append a single result dict as a JSONL line. Creates parent dirs if needed."""
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "a") as f:
        f.write(json.dumps(result) + "\n")


def load_results(results_path: str) -> list[dict]:
    """Read all JSONL lines from a results file."""
    if not os.path.exists(results_path):
        return []
    results = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def config_key(config: dict) -> tuple:
    """Return a hashable key for resume deduplication."""
    return (
        config["model_label"],
        tuple(config["input_size"]),
        config["column_batch_size"],
        config["num_samples"],
        config["num_classes"],
        config["allocator"],
    )


def result_to_config_key(result: dict) -> tuple:
    """Extract config key from a JSONL result dict."""
    return (
        result["model_label"],
        tuple(result["input_size"]),
        result["column_batch_size"],
        result["num_samples"],
        result["num_classes"],
        result["allocator"],
    )


def get_peak_rss_gb() -> float | None:
    """Read peak RSS (VmHWM) from /proc/self/status. Returns None on non-Linux."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except FileNotFoundError:
        return None
    return None


def now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def run_benchmark_subprocess(
    config: dict,
    results_path: str,
    timeout: int,
    project_root: str,
    enable_profiler: bool = False,
) -> dict:
    """
    Dispatch a single benchmark config as a subprocess.

    Returns the result dict parsed from the subprocess stdout (JSONL),
    or a synthetic error/timeout result if the subprocess fails.
    """
    cmd = [
        sys.executable,
        os.path.join(project_root, "extra", "GH200", "run_single.py"),
        "--base-width", str(config["base_width"]),
        "--blocks-per-stage", json.dumps(config["blocks_per_stage"]),
        "--input-size", ",".join(str(x) for x in config["input_size"]),
        "--column-batch-size", str(config["column_batch_size"]),
        "--num-samples", str(config["num_samples"]),
        "--num-classes", str(config["num_classes"]),
        "--allocator", config["allocator"],
        "--results-path", results_path,
    ]
    if enable_profiler:
        cmd.append("--enable-profiler")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=project_root,
        )

        # Parse last JSONL line from stdout
        for line in reversed(proc.stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)

        # No parseable output — return error with stderr
        return {
            **config,
            "status": "error",
            "error": f"No JSONL output. returncode={proc.returncode}",
            "stderr": proc.stderr[-2000:] if proc.stderr else "",
            "timestamp": now_iso(),
        }

    except subprocess.TimeoutExpired:
        result = {
            **config,
            "status": "timeout",
            "error": f"Subprocess exceeded {timeout}s timeout",
            "timestamp": now_iso(),
        }
        append_result(result, results_path)
        return result


def print_header() -> None:
    """Print table header for progress display."""
    print(f"{'Model':<12} {'Input':<12} {'Classes':>7} {'ColBatch':>8} "
          f"{'Samples':>7} {'Status':<8} {'Median ms':>10} {'Peak GB':>8}")
    print("-" * 82)


def print_result_row(result: dict) -> None:
    """Print a single result row."""
    input_str = "x".join(str(x) for x in result.get("input_size", []))
    median = result.get("median_matrix_ms")
    peak = result.get("peak_mem_allocated_gb") or result.get("peak_rss_gb")
    print(f"{result.get('model_label', '?'):<12} {input_str:<12} "
          f"{result.get('num_classes', '?'):>7} {result.get('column_batch_size', '?'):>8} "
          f"{result.get('num_samples', '?'):>7} {result.get('status', '?'):<8} "
          f"{median if median is not None else '-':>10} "
          f"{f'{peak:.2f}' if peak is not None else '-':>8}")
