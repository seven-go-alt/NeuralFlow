from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable

from app.retrieval.schemas import RetrievalResult
from benchmarks.config import BenchmarkConfig
from benchmarks.models import BenchmarkResult
from benchmarks.utils.metrics import mrr, recall_at_k
from benchmarks.utils.timing import compute_stats, measure_latency_samples

RetrieveFn = Callable[[str, int], Awaitable[list[RetrievalResult]]]


async def _stub_retrieve(query: str, top_k: int) -> list[RetrievalResult]:
    """Default stub that simulates retrieval latency with random results."""
    latency = random.uniform(0.005, 0.05)
    await asyncio.sleep(latency)
    return [
        RetrievalResult(
            chunk_id=f"c_{i}",
            document_id=f"d_{hash(query) % 100}",
            content=f"stub result {i} for {query}",
            score=round(random.uniform(0.5, 0.95), 2),
        )
        for i in range(top_k)
    ]


async def retrieve_benchmark(
    retrieve_fn: RetrieveFn | None = None,
    queries: list[str] | None = None,
    config: BenchmarkConfig | None = None,
) -> list[BenchmarkResult]:
    """Benchmark retrieval latency and quality metrics."""
    cfg = config or BenchmarkConfig()
    fn = retrieve_fn or _stub_retrieve
    qs = queries or ["sample query", "another test", "benchmark query"]

    results: list[BenchmarkResult] = []

    for query in qs:

        async def _call(q: str = query) -> list:
            return await fn(q, cfg.top_k)

        latencies, responses = await measure_latency_samples(
            _call,
            num_samples=cfg.num_samples,
            num_warmup=cfg.num_warmup,
        )

        relevant = [r.document_id for r in responses[-1]] if responses else []
        top_k_results = responses[-1][: cfg.top_k] if responses else []
        r_at_k = recall_at_k(top_k_results, relevant, k=cfg.top_k)
        mrr_score = mrr(top_k_results, relevant, k=cfg.top_k)

        stats = compute_stats(latencies)
        results.append(
            BenchmarkResult(
                name=f"retrieval_{query[:20]}",
                suite="retrieval",
                latency=stats,
                throughput_qps=1000.0 / stats.mean_ms if stats.mean_ms > 0 else 0.0,
                recall_at_k=r_at_k,
                mrr=mrr_score,
                raw_samples=latencies,
            )
        )

    return results
