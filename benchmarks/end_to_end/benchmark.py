from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable

from benchmarks.config import BenchmarkConfig
from benchmarks.models import BenchmarkResult
from benchmarks.utils.timing import compute_stats, measure_latency_samples

ChatFn = Callable[[str], Awaitable[str]]


async def _stub_chat(message: str) -> str:
    """Simulate an LLM chat response."""
    latency = random.uniform(0.2, 2.0)
    await asyncio.sleep(latency)
    return f"Simulated response to: {message[:50]}"


async def end_to_end_benchmark(
    chat_fn: ChatFn | None = None,
    queries: list[str] | None = None,
    config: BenchmarkConfig | None = None,
) -> list[BenchmarkResult]:
    """Benchmark end-to-end RAG pipeline latency."""
    cfg = config or BenchmarkConfig()
    fn = chat_fn or _stub_chat
    qs = queries or ["What is RAG?", "How does retrieval work?", "Explain chunking"]

    results: list[BenchmarkResult] = []
    for query in qs:

        async def _call(q: str = query) -> str:
            return await fn(q)

        latencies, _responses = await measure_latency_samples(
            _call,
            num_samples=cfg.num_samples,
            num_warmup=cfg.num_warmup,
        )

        stats = compute_stats(latencies)
        results.append(
            BenchmarkResult(
                name=f"e2e_{query[:20]}",
                suite="end_to_end",
                latency=stats,
                throughput_qps=1000.0 / stats.mean_ms if stats.mean_ms > 0 else 0.0,
                raw_samples=latencies,
            )
        )

    return results
