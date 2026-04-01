"""Evaluation helpers for repo-level benchmark workflows."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .refactor_benchmark import BenchmarkReport


def run_refactor_benchmark(output_dir: str | Path = "benchmarks/results/latest") -> "BenchmarkReport":
    """Lazily import the benchmark runner so ``python -m`` stays warning-free."""

    from .refactor_benchmark import run_refactor_benchmark as _run_refactor_benchmark

    return _run_refactor_benchmark(output_dir)


__all__ = ["run_refactor_benchmark"]
