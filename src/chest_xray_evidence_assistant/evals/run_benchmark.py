"""Command-line entry point for the deterministic offline benchmark matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from .datasets import BenchmarkDataset, load_benchmark
from .matrix import run_matrix, write_matrix_bundle

DEFAULT_DATASET = Path("data/benchmark/cxr-agent-bench-v0.jsonl")
DEFAULT_OUTPUT = Path("artifacts/evaluation/latest")


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(arguments)


def _load_dataset(path: Path) -> BenchmarkDataset:
    dataset_path = path.resolve()
    if dataset_path.suffix != ".jsonl" or not dataset_path.is_file():
        raise ValueError("benchmark dataset path is invalid")
    benchmark = load_benchmark(dataset_path.parent / "manifest.json")
    tracked = (benchmark.root / benchmark.manifest.dataset_path).resolve()
    if tracked != dataset_path:
        raise ValueError("benchmark dataset is not tracked by its manifest")
    return benchmark


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    try:
        benchmark = _load_dataset(args.dataset)
    except (OSError, ValueError):
        print(json.dumps({"error_code": "benchmark_input_invalid", "status": "failed"}))
        return 1

    try:
        bundle = run_matrix(benchmark)
    except Exception:
        print(json.dumps({"error_code": "benchmark_evaluation_failed", "status": "failed"}))
        return 1

    try:
        summary_path = write_matrix_bundle(bundle, args.output)
        summary_sha256 = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    except (OSError, ValueError):
        print(json.dumps({"error_code": "benchmark_output_failed", "status": "failed"}))
        return 1

    passed = bundle.summary.exit_gate.passed
    print(
        json.dumps(
            {
                "exit_gate_passed": passed,
                "output": str(summary_path),
                "status": "succeeded" if passed else "failed",
                "summary_sha256": summary_sha256,
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
