from __future__ import annotations

from app.retrieval.schemas import RetrievalResult


def _keyword_overlap(result: RetrievalResult, query_terms: list[str]) -> float:
    if not query_terms:
        return 0.0
    lower = result.content.lower()
    matched = sum(1 for t in query_terms if t.lower() in lower)
    return matched / len(query_terms)


def heuristic_rerank(
    results: list[RetrievalResult],
    query: str,
    vector_weight: float = 0.5,
    keyword_weight: float = 0.3,
    metadata_weight: float = 0.2,
) -> list[RetrievalResult]:
    """Heuristic reranker combining vector score, keyword overlap, and metadata signal.

    Args:
        results: Deduplicated retrieval results with existing scores.
        query: Original user query (used for keyword overlap).
        vector_weight: Weight for existing vector similarity score.
        keyword_weight: Weight for keyword overlap ratio.
        metadata_weight: Weight for metadata signal (title match, page number, etc.).

    Returns:
        Re-ranked results (sorted by combined score descending).
    """
    if not results:
        return results

    query_terms = [t.lower() for t in query.split() if len(t) > 1]

    scored: list[tuple[float, int, RetrievalResult]] = []
    for idx, r in enumerate(results):
        overlap = _keyword_overlap(r, query_terms)
        meta_bonus = 0.0
        title = r.source.get("title") or ""
        if title and any(t in title.lower() for t in query_terms):
            meta_bonus += 0.2
        if r.source.get("page_number") is not None:
            meta_bonus += 0.05
        combined = vector_weight * r.score + keyword_weight * overlap + metadata_weight * meta_bonus
        scored.append((combined, idx, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, _, r in scored]
