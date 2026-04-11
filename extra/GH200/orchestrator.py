#!/usr/bin/env python3
"""
    Benchmark orchestrator: enumerates all configurations and dispatches each
    as an independent subprocess via run_single.py.

    Configurations are sorted smallest-to-largest so that most runs complete
    before hitting OOM on larger configs. Resume is supported via JSONL dedup.
"""
from __future__ import annotations
import argparse
import os
import sys

# Ensure project root is on path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from extra.GH200.configs import (
    MODEL_LADDER,
    INPUT_SIZES,
    COLUMN_BATCH_SIZES,
    NUM_SAMPLES,
    NUM_CLASSES_LIST,
)
from extra.GH200.utils import (
    config_key,
    load_results,
    print_header,
    print_result_row,
    result_to_config_key,
    run_benchmark_subprocess,
)


def generate_configs(allocator: str) -> list[dict]:
    """
    Generate all benchmark configurations for a given allocator.

    Sorted smallest-to-largest: model → input → num_classes → batch_size → samples.
    """
    configs = []
    for model_cfg in MODEL_LADDER:
        for input_size in INPUT_SIZES:
            for num_classes in NUM_CLASSES_LIST:
                for col_batch in COLUMN_BATCH_SIZES:
                    for num_samples in NUM_SAMPLES:
                        configs.append({
                            "model_label": model_cfg.label,
                            "base_width": model_cfg.base_width,
                            "blocks_per_stage": model_cfg.blocks_per_stage,
                            "input_size": list(input_size),
                            "num_classes": num_classes,
                            "column_batch_size": col_batch,
                            "num_samples": num_samples,
                            "allocator": allocator,
                        })
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description="GH200 Knowledge Matrix Benchmark Orchestrator")
    parser.add_argument("--allocator", choices=["default", "rmm"], required=True)
    parser.add_argument("--results-path", type=str,
                        default="extra/GH200/results/results.jsonl")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print all configs without running")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip already-completed configs (default: True)")
    parser.add_argument("--no-resume", action="store_false", dest="resume",
                        help="Re-run all configs from scratch")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Per-config subprocess timeout in seconds (default: 3600)")
    parser.add_argument("--enable-profiler", action="store_true",
                        help="Run torch.profiler trace for each config")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_path = os.path.join(project_root, args.results_path)

    configs = generate_configs(args.allocator)
    total = len(configs)
    print(f"Allocator: {args.allocator}")
    print(f"Total configurations: {total}")

    if args.dry_run:
        print(f"\n--- DRY RUN (no subprocesses will be launched) ---\n")
        print_header()
        for cfg in configs:
            print(f"{cfg['model_label']:<12} "
                  f"{'x'.join(str(x) for x in cfg['input_size']):<12} "
                  f"{cfg['num_classes']:>7} {cfg['column_batch_size']:>8} "
                  f"{cfg['num_samples']:>7} {'pending':<8}")
        print(f"\nTotal: {total} configurations")
        return

    # Load completed results for resume
    completed_keys = set()
    if args.resume:
        existing = load_results(results_path)
        for r in existing:
            try:
                completed_keys.add(result_to_config_key(r))
            except KeyError:
                continue
        if completed_keys:
            print(f"Resuming: {len(completed_keys)} configs already completed")

    print()
    print_header()

    done = 0
    skipped = 0
    for i, cfg in enumerate(configs):
        key = config_key(cfg)
        if key in completed_keys:
            skipped += 1
            continue

        result = run_benchmark_subprocess(
            config=cfg,
            results_path=results_path,
            timeout=args.timeout,
            project_root=project_root,
            enable_profiler=args.enable_profiler,
        )
        done += 1
        print_result_row(result)

    print(f"\nDone: {done} run, {skipped} skipped, {total} total")


if __name__ == "__main__":
    main()
