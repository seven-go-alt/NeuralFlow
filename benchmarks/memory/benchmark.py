from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable

from benchmarks.config import BenchmarkConfig
from benchmarks.models import BenchmarkResult
from benchmarks.utils.timing import compute_stats, measure_latency_samples

CacheFn = Callable[[str], Awaitable[bool]]


async def _stub_cache_check(key: str) -> bool:
    """Simulate cache hit/miss."""
    await asyncio.sleep(random.uniform(0.001, 0.005))
    return random.random() < 0.6


async def memory_benchmark(
    cache_fn: CacheFn | None = None,
    config: BenchmarkConfig | None = None,
) -> list[BenchmarkResult]:
    """Benchmark memory and caching layer performance."""
    cfg = config or BenchmarkConfig()
    fn = cache_fn or _stub_cache_check

    keys = [f"query_{i}" for i in range(100)]

    async def _call() -> bool:
        return await fn(random.choice(keys))

    latencies, hits = await measure_latency_samples(
        _call,
        num_samples=cfg.num_samples,
        num_warmup=cfg.num_warmup,
    )

    hit_rate = sum(1 for h in hits if h) / len(hits) if hits else 0.0
    stats = compute_stats(latencies)

    return [
        BenchmarkResult(
            name="memory_cache",
            suite="memory",
            latency=stats,
            throughput_qps=1000.0 / stats.mean_ms if stats.mean_ms > 0 else 0.0,
            accuracy=hit_rate,
            raw_samples=latencies,
            metadata={"hit_rate": hit_rate, "cache_size": 100},
        )
    ]
