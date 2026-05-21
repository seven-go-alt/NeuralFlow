from __future__ import annotations

import logging
import math
from typing import Any

from app.retrieval.schemas import RetrievalResult

logger = logging.getLogger(__name__)

_MODEL: Any = None  # singleton cross-encoder model


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-max(min(x, 100), -100)))
    except OverflowError:
        return 1.0 if x > 0 else 0.0


class CrossEncoderReranker:
    """Cross-encoder reranker with lazy-loaded model and heuristic fallback."""

    def __init__(self, model_name: str | None = None, enabled: bool = True) -> None:
        self._model_name = model_name
        self._enabled = enabled

    def rerank(
        self,
        query: str,
        chunks: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        if not chunks:
            return chunks

        from app.config import get_settings

        s = get_settings()
        if top_k is None:
            top_k = s.reranker_top_k
        model_name = self._model_name or s.reranker_model
        enabled = self._enabled and s.cross_encoder_enabled

        if not enabled:
            hw = s.reranker_heuristic_weights
            return heuristic_rerank(
                chunks,
                query,
                vector_weight=hw.get("vector_weight", 0.5),
                keyword_weight=hw.get("keyword_weight", 0.3),
                metadata_weight=hw.get("metadata_weight", 0.2),
            )

        model = _load_model(model_name)
        if model is None:
            logger.warning("cross-encoder model unavailable, falling back to heuristic rerank")
            hw = s.reranker_heuristic_weights
            return heuristic_rerank(
                chunks,
                query,
                vector_weight=hw.get("vector_weight", 0.5),
                keyword_weight=hw.get("keyword_weight", 0.3),
                metadata_weight=hw.get("metadata_weight", 0.2),
            )

        pairs = [(query, c.content) for c in chunks]
        try:
            scores: list[float] = model.predict(pairs).tolist()
        except Exception as exc:
            logger.warning("cross-encoder inference failed: %s", exc)
            hw = s.reranker_heuristic_weights
            return heuristic_rerank(
                chunks,
                query,
                vector_weight=hw.get("vector_weight", 0.5),
                keyword_weight=hw.get("keyword_weight", 0.3),
                metadata_weight=hw.get("metadata_weight", 0.2),
            )

        normalized = [_sigmoid(s) for s in scores]
        indexed = list(enumerate(chunks))
        indexed.sort(key=lambda x: normalized[x[0]], reverse=True)
        reranked = [chunk for _, chunk in indexed[:top_k]]
        for chunk, score in zip(reranked, sorted(normalized, reverse=True)[:top_k], strict=False):
            chunk.score = score
        return reranked


def _load_model(model_name: str) -> Any:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        from sentence_transformers import CrossEncoder

        _MODEL = CrossEncoder(model_name)
        logger.info("cross-encoder model loaded: %s", model_name)
        return _MODEL
    except Exception as exc:
        logger.warning("failed to load cross-encoder model '%s': %s", model_name, exc)
        return None


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
