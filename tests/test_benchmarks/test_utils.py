from __future__ import annotations

from benchmarks.config import BenchmarkConfig
from benchmarks.models import BenchmarkResult, LatencyStats
from benchmarks.utils.metrics import mrr, recall_at_k


def _make_result(doc_id: str, chunk_id: str = "c1", content: str = "test", score: float = 0.9):
    from app.retrieval.schemas import RetrievalResult

    return RetrievalResult(chunk_id=chunk_id, document_id=doc_id, content=content, score=score)


def test_recall_at_k_all_relevant() -> None:
    results = [_make_result("d1"), _make_result("d2")]
    assert recall_at_k(results, ["d1", "d2"]) == 1.0


def test_recall_at_k_partial() -> None:
    results = [_make_result("d1"), _make_result("d2")]
    assert recall_at_k(results, ["d1", "d3"]) == 0.5


def test_recall_at_k_no_match() -> None:
    results = [_make_result("d1")]
    assert recall_at_k(results, ["d2"]) == 0.0


def test_mrr_first_rank() -> None:
    results = [_make_result("d1"), _make_result("d2")]
    assert mrr(results, ["d1"]) == 1.0


def test_mrr_second_rank() -> None:
    results = [_make_result("d1"), _make_result("d2")]
    assert mrr(results, ["d2"]) == 0.5


def test_mrr_no_match() -> None:
    results = [_make_result("d1")]
    assert mrr(results, ["d2"]) == 0.0


def test_benchmark_config_defaults() -> None:
    cfg = BenchmarkConfig()
    assert cfg.suite == "all"
    assert cfg.num_samples == 30
    assert cfg.num_warmup == 3


def test_benchmark_result_defaults() -> None:
    r = BenchmarkResult(name="test", suite="retrieval", latency=LatencyStats())
    assert r.name == "test"
    assert r.suite == "retrieval"
    assert r.latency.samples == 0
    assert r.throughput_qps == 0.0
