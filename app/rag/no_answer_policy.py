from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.retrieval.schemas import RetrievalResult


@dataclass(slots=True)
class NoAnswerDecision:
    should_answer: bool
    reason: str = ""
    confidence: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


def evaluate_retrieval_confidence(results: list[RetrievalResult]) -> float:
    """Average score of retrieval results, 0 if empty."""
    if not results:
        return 0.0
    return sum(r.score for r in results) / len(results)


def has_query_context_overlap(query: str, results: list[RetrievalResult]) -> float:
    """Fraction of query terms appearing in at least one retrieved chunk."""
    if not query.strip() or not results:
        return 0.0
    terms = {t.lower() for t in query.split() if len(t) > 1}
    if not terms:
        return 0.0
    matched = sum(1 for t in terms if any(t in r.content.lower() for r in results))
    return matched / len(terms)


def decide_no_answer(
    query: str,
    results: list[RetrievalResult],
    *,
    min_confidence: float = 0.3,
    min_overlap: float = 0.2,
    empty_result_policy: str = "refuse",
) -> NoAnswerDecision:
    """Determine whether the system should answer based on retrieval quality.

    Args:
        query: User query.
        results: Retrieved chunks.
        min_confidence: Minimum average retrieval confidence.
        min_overlap: Minimum query-context term overlap ratio.
        empty_result_policy: "refuse" (default) declines when no results
                             are retrieved; "fallback" allows answering anyway.

    Returns:
        NoAnswerDecision with should_answer, reason, and confidence details.
    """
    if not results:
        if empty_result_policy == "fallback":
            return NoAnswerDecision(
                should_answer=True,
                reason="No results retrieved but fallback policy allows answering",
                confidence=0.0,
                details={"empty_result_policy": empty_result_policy},
            )
        return NoAnswerDecision(
            should_answer=False,
            reason="No relevant documents retrieved",
            confidence=0.0,
            details={"empty_result_policy": empty_result_policy},
        )

    avg_confidence = evaluate_retrieval_confidence(results)
    overlap = has_query_context_overlap(query, results)

    fails_confidence = avg_confidence < min_confidence
    fails_overlap = overlap < min_overlap

    if fails_confidence and fails_overlap:
        return NoAnswerDecision(
            should_answer=False,
            reason=f"Low retrieval confidence ({avg_confidence:.2f}) and low term overlap ({overlap:.2f})",
            confidence=avg_confidence,
            details={
                "avg_confidence": avg_confidence,
                "term_overlap": overlap,
                "min_confidence": min_confidence,
                "min_overlap": min_overlap,
            },
        )
    if fails_confidence:
        return NoAnswerDecision(
            should_answer=False,
            reason=f"Low retrieval confidence ({avg_confidence:.2f})",
            confidence=avg_confidence,
            details={
                "avg_confidence": avg_confidence,
                "term_overlap": overlap,
                "min_confidence": min_confidence,
                "min_overlap": min_overlap,
            },
        )
    if fails_overlap:
        return NoAnswerDecision(
            should_answer=False,
            reason=f"Low term overlap ({overlap:.2f}) between query and retrieved context",
            confidence=avg_confidence,
            details={
                "avg_confidence": avg_confidence,
                "term_overlap": overlap,
                "min_confidence": min_confidence,
                "min_overlap": min_overlap,
            },
        )

    return NoAnswerDecision(
        should_answer=True,
        reason="Retrieval quality meets thresholds",
        confidence=avg_confidence,
        details={
            "avg_confidence": avg_confidence,
            "term_overlap": overlap,
            "min_confidence": min_confidence,
            "min_overlap": min_overlap,
        },
    )
