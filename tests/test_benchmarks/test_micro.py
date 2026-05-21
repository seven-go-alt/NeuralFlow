"""Microbenchmarks for core retrieval and scoring functions."""

from app.retrieval.reranker import _sigmoid, heuristic_rerank
from app.retrieval.schemas import RetrievalResult


def _chunks(n: int, base_score: float = 0.5) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id=f"c{i}",
            document_id="d1",
            content=f"sample chunk content with query term {i}",
            score=base_score,
            metadata={},
            source={"title": f"doc_{i}", "page_number": i if i % 2 == 0 else None},
        )
        for i in range(n)
    ]


def test_sigmoid_benchmark(benchmark) -> None:
    """Benchmark sigmoid function throughput."""
    result = benchmark(_sigmoid, 0.5)
    assert 0.5 < result < 0.7


def test_sigmoid_large_positive(benchmark) -> None:
    """Benchmark sigmoid with large positive input."""
    result = benchmark(_sigmoid, 100.0)
    assert result > 0.999


def test_sigmoid_large_negative(benchmark) -> None:
    """Benchmark sigmoid with large negative input."""
    result = benchmark(_sigmoid, -100.0)
    assert result < 0.001


def test_heuristic_rerank_10_chunks(benchmark) -> None:
    """Benchmark heuristic rerank with 10 chunks."""
    chunks = _chunks(10)
    result = benchmark(heuristic_rerank, chunks, "query term")
    assert len(result) == 10


def test_heuristic_rerank_100_chunks(benchmark) -> None:
    """Benchmark heuristic rerank with 100 chunks."""
    chunks = _chunks(100)
    result = benchmark(heuristic_rerank, chunks, "query term with more words")
    assert len(result) == 100


def test_heuristic_rerank_empty(benchmark) -> None:
    """Benchmark heuristic rerank with empty input."""
    result = benchmark(heuristic_rerank, [], "query")
    assert result == []


def test_heuristic_rerank_custom_weights(benchmark) -> None:
    """Benchmark with non-default weights."""
    chunks = _chunks(50)
    result = benchmark(
        heuristic_rerank,
        chunks,
        "custom weight query",
        vector_weight=0.4,
        keyword_weight=0.4,
        metadata_weight=0.2,
    )
    assert len(result) == 50
