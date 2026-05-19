from __future__ import annotations

import statistics
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from benchmarks.models import LatencyStats

T = TypeVar("T")


async def measure_latency_samples(
    fn: Callable[[], Awaitable[T]],
    num_samples: int = 30,
    num_warmup: int = 3,
) -> tuple[list[float], list[T]]:
    """Measure latency of an async function over multiple samples."""
    for _ in range(num_warmup):
        await fn()

    latencies: list[float] = []
    results: list[T] = []
    for _ in range(num_samples):
        start = time.perf_counter()
        result = await fn()
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)
        results.append(result)

    return latencies, results


def compute_stats(latencies: list[float]) -> LatencyStats:
    """Compute statistical summary from latency samples."""
    if not latencies:
        return LatencyStats()

    sorted_lat = sorted(latencies)
    n = len(sorted_lat)

    return LatencyStats(
        min_ms=sorted_lat[0],
        max_ms=sorted_lat[-1],
        mean_ms=statistics.mean(sorted_lat),
        median_ms=sorted_lat[n // 2],
        p50_ms=sorted_lat[int(n * 0.50)],
        p90_ms=sorted_lat[int(n * 0.90)],
        p95_ms=sorted_lat[int(n * 0.95)],
        p99_ms=sorted_lat[int(n * 0.99)],
        stddev_ms=statistics.stdev(sorted_lat) if n > 1 else 0.0,
        samples=n,
    )


def compute_stats_from(latencies: list[float], stats_cls: type = LatencyStats) -> LatencyStats:
    """Alias for compute_stats."""
    return compute_stats(latencies)
