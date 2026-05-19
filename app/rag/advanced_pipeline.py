from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from app.core.llm import LLMClient
from app.observability.trace_manager import TraceManager
from app.rag.context_builder import RAGContextBuilder
from app.rag.corrective_loop import CorrectiveRetriever
from app.rag.query_transformer import (
    expand_multi_query,
    hyde_transform,
    merge_deduplicated,
)
from app.rag.retrieval_grader import GradeResult, grade_retrieval
from app.retrieval.schemas import RetrievalResult

RetrieveFn = Callable[[str, int], Awaitable[list[RetrievalResult]]]


@dataclass(slots=True)
class AdvancedRAGResult:
    query: str
    original_query: str
    final_query: str
    results: list[RetrievalResult]
    context: str
    citations: list[dict[str, Any]]
    grade: GradeResult | None = None
    corrections: int = 0
    transform_history: list[dict[str, Any]] = field(default_factory=list)
    should_answer: bool = True


class AdvancedRAGPipeline:
    """Orchestrates query transformation, retrieval, and quality grading."""

    def __init__(
        self,
        retrieve_fn: RetrieveFn,
        llm: LLMClient,
        context_builder: RAGContextBuilder | None = None,
    ) -> None:
        self._retrieve_fn = retrieve_fn
        self._llm = llm
        self._context_builder = context_builder or RAGContextBuilder()
        self._corrective = CorrectiveRetriever(retrieve_fn, llm)
        self.trace: TraceManager | None = None

    async def execute(
        self,
        query: str,
        top_k: int = 5,
        use_multi_query: bool = False,
        use_hyde: bool = False,
        max_corrections: int = 2,
    ) -> AdvancedRAGResult:
        """Execute the full advanced RAG pipeline with tracing."""
        self.trace = TraceManager("advanced_rag")
        transform_history: list[dict[str, Any]] = []
        search_query = query
        results: list[RetrievalResult] = []
        grade: GradeResult | None = None
        corrections = 0

        if use_hyde:
            with self.trace.span("hyde_transform"):
                search_query = await hyde_transform(query, self._llm)
                transform_history.append(
                    {
                        "strategy": "hyde",
                        "original": query,
                        "transformed": search_query,
                    }
                )

        if use_multi_query:
            with self.trace.span("multi_query_transform"):
                transform_result = await expand_multi_query(query, self._llm)
                transform_history.append(
                    {
                        "strategy": "multi_query",
                        "variants": transform_result.variants,
                    }
                )

                all_results: list[list[RetrievalResult]] = []
                for vq in transform_result.variants:
                    vr = await self._retrieve_fn(vq, top_k)
                    all_results.append(vr)
                results = merge_deduplicated(all_results)

                grade = await self._grade_or_none(query, results)
                if grade is not None and not grade.sufficient:
                    with self.trace.span("corrective_retry"):
                        corrective = await self._corrective.retrieve(search_query, top_k)
                        results = corrective.results if corrective.results else results
                        grade = corrective.grade
                        corrections = corrective.corrections
                        transform_history.extend(corrective.histories)
        else:
            with self.trace.span("retrieve"):
                corrective = await self._corrective.retrieve(search_query, top_k)
                results = corrective.results
                grade = corrective.grade
                corrections = corrective.corrections
                transform_history.extend(corrective.histories)

        with self.trace.span("grade"):
            if grade is None:
                grade = await self._grade_or_none(query, results)

        with self.trace.span("context_build"):
            rag_build = self._context_builder.build(query, results)
            should_answer = grade.sufficient if grade is not None else bool(results)

        self.trace.close()
        return AdvancedRAGResult(
            query=query,
            original_query=query,
            final_query=search_query,
            results=results,
            context=rag_build.context,
            citations=rag_build.citations,
            grade=grade,
            corrections=corrections,
            transform_history=transform_history,
            should_answer=should_answer,
        )

    async def _grade_or_none(
        self, query: str, results: list[RetrievalResult]
    ) -> GradeResult | None:
        if not results:
            return GradeResult(score=0.0, sufficient=False, reason="No results")
        return await grade_retrieval(query, results, llm=self._llm)
