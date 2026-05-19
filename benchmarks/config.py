from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BenchmarkConfig:
    """Configuration for benchmark suite execution."""

    suite: str = "all"  # retrieval | end_to_end | memory | all
    top_k: int = 5
    num_warmup: int = 3
    num_samples: int = 30
    max_concurrency: int = 5
    live_url: str | None = None
    output_dir: str = "benchmark_results"
    regression_threshold: float = 0.15
    fixtures_path: str = "benchmarks/data/fixtures.jsonl"
