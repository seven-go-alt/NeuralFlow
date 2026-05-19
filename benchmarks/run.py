#!/usr/bin/env python3
"""NeuralFlow Benchmark Suite — CLI entry point."""

from __future__ import annotations

import asyncio
import json

from benchmarks.cli import parse_args
from benchmarks.runner import run_all, save_reports


async def main() -> None:
    config = parse_args()
    print(f"Running benchmark suite: {config.suite}")
    print(f"  Samples: {config.num_samples}, Warmup: {config.num_warmup}")

    queries = _load_queries(config.fixtures_path) if config.fixtures_path else None

    results = await run_all(config, queries=queries)

    paths = save_reports(results, config)
    print("\nResults saved:")
    print(f"  JSON: {paths['json']}")
    print(f"  HTML: {paths['html']}")

    _print_summary(results)


def _load_queries(path: str) -> list[str]:
    try:
        queries = []
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                queries.append(data.get("query", ""))
        return [q for q in queries if q]
    except FileNotFoundError:
        print(f"Warning: fixtures file not found at {path}")
        return []


def _print_summary(results: list) -> None:
    print(f"\n{'─' * 50}")
    print(f"Total benchmarks: {len(results)}")
    for r in results:
        print(f"  {r.name}: mean={r.latency.mean_ms:.1f}ms, qps={r.throughput_qps:.1f}")


if __name__ == "__main__":
    asyncio.run(main())
