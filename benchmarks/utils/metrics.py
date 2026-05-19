from __future__ import annotations

from app.retrieval.schemas import RetrievalResult


def recall_at_k(
    retrieved: list[RetrievalResult],
    relevant: list[str],
    k: int | None = None,
) -> float:
    """Compute recall@k: fraction of relevant documents in top-k results."""
    if not relevant:
        return 0.0
    k = k or len(retrieved)
    retrieved_ids = {r.document_id for r in retrieved[:k]}
    hits = sum(1 for doc_id in relevant if doc_id in retrieved_ids)
    return hits / len(relevant)


def mrr(
    retrieved: list[RetrievalResult],
    relevant: list[str],
    k: int | None = None,
) -> float:
    """Compute Mean Reciprocal Rank: 1/rank of first relevant document."""
    if not relevant or not retrieved:
        return 0.0
    k = k or len(retrieved)
    relevant_set = set(relevant)
    for rank, r in enumerate(retrieved[:k], start=1):
        if r.document_id in relevant_set:
            return 1.0 / rank
    return 0.0
