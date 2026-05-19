from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class BenchmarkCase:
    """A single benchmark query case."""

    query: str
    expected_topics: list[str] = field(default_factory=list)
    expected_document_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LatencyStats:
    """Statistical summary of latency samples."""

    min_ms: float = 0.0
    max_ms: float = 0.0
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    stddev_ms: float = 0.0
    samples: int = 0


@dataclass(slots=True)
class BenchmarkResult:
    """Full result of a benchmark run."""

    name: str
    suite: str
    latency: LatencyStats
    throughput_qps: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    accuracy: float = 0.0
    error_rate: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_samples: list[float] = field(default_factory=list)
