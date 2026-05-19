from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from benchmarks.config import BenchmarkConfig
from benchmarks.end_to_end.benchmark import end_to_end_benchmark
from benchmarks.memory.benchmark import memory_benchmark
from benchmarks.models import BenchmarkResult
from benchmarks.reporting.html_reporter import html_report
from benchmarks.reporting.json_reporter import to_json_report
from benchmarks.retrieval.benchmark import retrieve_benchmark


async def run_all(
    config: BenchmarkConfig, queries: list[str] | None = None
) -> list[BenchmarkResult]:
    """Run all benchmark suites and return combined results."""
    results: list[BenchmarkResult] = []

    if config.suite in ("all", "retrieval"):
        r = await retrieve_benchmark(queries=queries, config=config)
        results.extend(r)

    if config.suite in ("all", "end_to_end"):
        e = await end_to_end_benchmark(queries=queries, config=config)
        results.extend(e)

    if config.suite in ("all", "memory"):
        m = await memory_benchmark(config=config)
        results.extend(m)

    return results


def save_reports(
    results: list[BenchmarkResult],
    config: BenchmarkConfig,
    metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Save benchmark results as JSON and HTML reports."""
    os.makedirs(config.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    meta = {
        "timestamp": timestamp,
        "suite": config.suite,
        "num_samples": config.num_samples,
        "num_warmup": config.num_warmup,
        **(metadata or {}),
    }

    json_path = os.path.join(config.output_dir, f"benchmark_{timestamp}.json")
    with open(json_path, "w") as f:
        f.write(to_json_report(results, metadata=meta))

    html_path = os.path.join(config.output_dir, f"benchmark_{timestamp}.html")
    with open(html_path, "w") as f:
        f.write(html_report(results))

    return {"json": json_path, "html": html_path}
