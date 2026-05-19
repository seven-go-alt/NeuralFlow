from __future__ import annotations

from app.rag.corrective_loop import CorrectiveRetriever
from app.retrieval.schemas import RetrievalResult


class FakeLLM:
    def __init__(self, rewrite_result: str | None = None) -> None:
        self.rewrite_result = rewrite_result or "rewritten query"
        self.call_count = 0

    async def generate(self, prompt: str) -> str:
        self.call_count += 1
        return self.rewrite_result


class TestCorrectiveRetriever:
    async def test_successful_first_attempt(self) -> None:
        async def retrieve_fn(query: str, top_k: int) -> list[RetrievalResult]:
            return [
                RetrievalResult(
                    chunk_id="c1",
                    document_id="d1",
                    content="relevant content " + query,
                    score=0.9,
                )
            ]

        llm = FakeLLM()
        retriever = CorrectiveRetriever(retrieve_fn=retrieve_fn, llm=llm, max_corrections=2)
        result = await retriever.retrieve("test query")

        assert len(result.results) > 0
        assert result.corrections == 0
        assert result.grade.sufficient is True

    async def test_corrective_retry_on_low_quality(self) -> None:
        attempt_count = 0

        async def retrieve_fn(query: str, top_k: int) -> list[RetrievalResult]:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                return [
                    RetrievalResult(
                        chunk_id="c1", document_id="d1", content="irrelevant", score=0.05
                    )
                ]
            return [
                RetrievalResult(
                    chunk_id="c1",
                    document_id="d1",
                    content="improved query matching content",
                    score=0.9,
                )
            ]

        llm = FakeLLM("improved query")
        retriever = CorrectiveRetriever(retrieve_fn=retrieve_fn, llm=llm, max_corrections=1)
        result = await retriever.retrieve("test query")

        assert result.corrections == 1
        assert len(result.results) > 0

    async def test_all_attempts_exhausted(self) -> None:
        async def retrieve_fn(query: str, top_k: int) -> list[RetrievalResult]:
            return [
                RetrievalResult(chunk_id="c1", document_id="d1", content="irrelevant", score=0.01)
            ]

        llm = FakeLLM("still bad query")
        retriever = CorrectiveRetriever(retrieve_fn=retrieve_fn, llm=llm, max_corrections=1)
        result = await retriever.retrieve("test")

        # All attempts exhausted, returns empty results
        assert result.corrections == 1
        assert len(result.results) == 0
        assert result.grade.sufficient is False

    async def test_history_traces_attempts(self) -> None:
        async def retrieve_fn(query: str, top_k: int) -> list[RetrievalResult]:
            return [RetrievalResult(chunk_id="c1", document_id="d1", content="bad", score=0.01)]

        llm = FakeLLM("rewritten")
        retriever = CorrectiveRetriever(retrieve_fn=retrieve_fn, llm=llm, max_corrections=1)
        result = await retriever.retrieve("test")

        assert len(result.histories) >= 1
        assert all("attempt" in h for h in result.histories)
        assert all("query" in h for h in result.histories)
        assert all("grade_score" in h for h in result.histories)

    async def test_empty_retrieve_fn(self) -> None:
        async def retrieve_fn(query: str, top_k: int) -> list[RetrievalResult]:
            return []

        llm = FakeLLM()
        retriever = CorrectiveRetriever(retrieve_fn=retrieve_fn, llm=llm, max_corrections=1)
        result = await retriever.retrieve("test")

        assert len(result.results) == 0
        assert result.grade.sufficient is False
