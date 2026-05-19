from __future__ import annotations

import json
from typing import Any

from benchmarks.models import BenchmarkResult


def to_json_report(
    results: list[BenchmarkResult],
    metadata: dict[str, Any] | None = None,
) -> str:
    """Serialize benchmark results to JSON string."""
    report: dict[str, Any] = {
        "metadata": metadata or {},
        "results": [_result_to_dict(r) for r in results],
    }
    return json.dumps(report, indent=2, ensure_ascii=False)


def _result_to_dict(r: BenchmarkResult) -> dict[str, Any]:
    return {
        "name": r.name,
        "suite": r.suite,
        "latency": {
            "min_ms": r.latency.min_ms,
            "max_ms": r.latency.max_ms,
            "mean_ms": r.latency.mean_ms,
            "median_ms": r.latency.median_ms,
            "p50_ms": r.latency.p50_ms,
            "p90_ms": r.latency.p90_ms,
            "p95_ms": r.latency.p95_ms,
            "p99_ms": r.latency.p99_ms,
            "stddev_ms": r.latency.stddev_ms,
            "samples": r.latency.samples,
        },
        "throughput_qps": r.throughput_qps,
        "recall_at_k": r.recall_at_k,
        "mrr": r.mrr,
        "accuracy": r.accuracy,
        "error_rate": r.error_rate,
        "metadata": r.metadata,
    }
