from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from app.core.llm import LLMClient
from app.rag.query_transformer import rewrite_query
from app.rag.retrieval_grader import GradeResult, grade_retrieval
from app.retrieval.schemas import RetrievalResult

RetrieveFn = Callable[[str, int], Awaitable[list[RetrievalResult]]]


@dataclass(slots=True)
class CorrectiveResult:
    query: str
    results: list[RetrievalResult]
    grade: GradeResult
    corrections: int = 0
    histories: list[dict[str, Any]] = field(default_factory=list)


class CorrectiveRetriever:
    """Self-RAG retriever that retries with query transformation on low quality."""

    def __init__(
        self,
        retrieve_fn: RetrieveFn,
        llm: LLMClient,
        max_corrections: int = 2,
    ) -> None:
        self._retrieve_fn = retrieve_fn
        self._llm = llm
        self._max_corrections = max_corrections

    async def retrieve(self, query: str, top_k: int = 5) -> CorrectiveResult:
        """Attempt retrieval with up to max_corrections retries via query rewriting."""
        current_query = query
        histories: list[dict[str, Any]] = []

        for attempt in range(self._max_corrections + 1):
            results = await self._retrieve_fn(current_query, top_k)
            grade = await grade_retrieval(current_query, results, llm=self._llm)

            histories.append(
                {
                    "attempt": attempt,
                    "query": current_query,
                    "result_count": len(results),
                    "grade_score": grade.score,
                    "grade_sufficient": grade.sufficient,
                }
            )

            if grade.sufficient:
                return CorrectiveResult(
                    query=current_query,
                    results=results,
                    grade=grade,
                    corrections=attempt,
                    histories=histories,
                )

            if attempt < self._max_corrections:
                current_query = await rewrite_query(current_query, self._llm)

        # All attempts exhausted
        return CorrectiveResult(
            query=current_query,
            results=[],
            grade=grade,
            corrections=self._max_corrections,
            histories=histories,
        )
