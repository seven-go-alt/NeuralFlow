from __future__ import annotations

import argparse

from benchmarks.config import BenchmarkConfig


def parse_args(args: list[str] | None = None) -> BenchmarkConfig:
    """Parse CLI arguments into BenchmarkConfig."""
    parser = argparse.ArgumentParser(description="NeuralFlow Benchmark Suite")
    parser.add_argument(
        "--suite",
        choices=["retrieval", "end_to_end", "memory", "all"],
        default="all",
        help="Benchmark suite to run",
    )
    parser.add_argument("--samples", type=int, default=30, help="Number of samples per benchmark")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations")
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent tasks")
    parser.add_argument("--live-url", type=str, help="URL for live API testing")
    parser.add_argument("--output", type=str, default="benchmark_results", help="Output directory")
    parser.add_argument(
        "--fixtures", type=str, default="benchmarks/data/fixtures.jsonl", help="Path to fixtures"
    )
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=0.15,
        help="Regression detection threshold (fraction)",
    )

    parsed = parser.parse_args(args)
    return BenchmarkConfig(
        suite=parsed.suite,
        num_samples=parsed.samples,
        num_warmup=parsed.warmup,
        max_concurrency=parsed.concurrency,
        live_url=parsed.live_url,
        output_dir=parsed.output,
        fixtures_path=parsed.fixtures,
        regression_threshold=parsed.regression_threshold,
    )
