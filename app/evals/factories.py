from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.core.llm import LLMClient, TokenUsage
from app.db.session import SessionLocal
from app.documents.repository import DocumentRepository
from app.evals.judge_cache import JudgeCache
from app.evals.runner import AnswerEvalFn, AnswerFn, RetrieveFn
from app.rag.answer_evaluator import AnswerEvalResult, evaluate_answer
from app.retrieval.schemas import RetrievalFilters, RetrievalRequest
from app.retrieval.service import RetrievalService


def make_live_retrieve_fn(tenant_id: str = "public") -> RetrieveFn:
    """Factory: build a retrieve function wired to the real RetrievalService."""

    def retrieve(query: str, top_k: int) -> list[dict[str, Any]]:
        db = SessionLocal()
        try:
            service = RetrievalService(document_repo=DocumentRepository(db))
            request = RetrievalRequest(
                query=query,
                top_k=top_k,
                score_threshold=0.0,
                filters=RetrievalFilters(),
            )
            import asyncio

            response = asyncio.run(service.search(tenant_id, request))
        finally:
            db.close()
        return [
            {
                "document_id": r.metadata.get("canonical_doc_id", r.document_id),
                "content": r.content,
                "score": r.score,
                "metadata": r.metadata,
            }
            for r in response.results
        ]

    return retrieve


def make_live_answer_fn(model: str | None = None) -> AnswerFn:
    """Factory: build an answer function that uses the real LLM for RAG generation."""
    settings = get_settings()
    llm = LLMClient(model=model or settings.litellm_model)

    def answer(query: str, context: str) -> tuple[str | None, TokenUsage]:
        if not context.strip():
            return "Unable to answer: no relevant context found.", {}
        prompt = (
            "You are a helpful enterprise knowledge assistant. "
            "Answer the user's question based solely on the provided context. "
            "If the context does not contain enough information, say so clearly.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )
        import asyncio

        content, usage = asyncio.run(llm.generate_with_usage(prompt))
        return content, usage

    return answer


def make_live_answer_eval_fn(
    model: str | None = None,
    judge_cache: JudgeCache | None = None,
) -> AnswerEvalFn:
    """Factory: build a Judge function with caching.

    Uses evaluate_answer (which now returns TokenUsage) and caches
    results keyed by (question, answer, model) hash.
    """
    settings = get_settings()
    judge_model = model or settings.litellm_model
    llm = LLMClient(model=judge_model)
    cache = judge_cache or JudgeCache()

    def answer_eval(
        query: str, answer: str, context_chunks: list[str]
    ) -> tuple[AnswerEvalResult | None, TokenUsage]:
        # check cache first
        cached = cache.get(query, answer, judge_model)
        if cached is not None:
            return (
                AnswerEvalResult(
                    relevance=cached["relevance"],
                    faithfulness=cached["faithfulness"],
                    completeness=cached["completeness"],
                    overall=(
                        cached["relevance"]
                        + cached["faithfulness"]
                        + cached["completeness"]
                    )
                    / 3.0,
                    reason=cached.get("reason", "cached"),
                    details=cached,
                ),
                {},
            )

        # not cached — call LLM (evaluate_answer returns (result, usage) from Task 4)
        import asyncio

        eval_result, eval_usage = asyncio.run(
            evaluate_answer(query, answer, context_chunks, llm)
        )

        # cache the result for future runs
        if eval_result.reason != "empty answer" and "error" not in eval_result.reason:
            cache.set(
                query,
                answer,
                judge_model,
                {
                    "relevance": eval_result.relevance,
                    "faithfulness": eval_result.faithfulness,
                    "completeness": eval_result.completeness,
                    "reason": eval_result.reason,
                },
            )
        return eval_result, eval_usage

    return answer_eval
