from __future__ import annotations

from app.rag.advanced_pipeline import AdvancedRAGPipeline
from app.retrieval.schemas import RetrievalResult


class FakeLLM:
    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self.responses = responses or {}
        self.call_count = 0

    async def generate(self, prompt: str) -> str:
        self.call_count += 1
        for key, response in self.responses.items():
            if key in prompt:
                return response
        return "default response"


class TestAdvancedRAGPipeline:
    async def test_basic_execution(self) -> None:
        async def retrieve_fn(query: str, top_k: int) -> list[RetrievalResult]:
            return [
                RetrievalResult(
                    chunk_id="c1",
                    document_id="d1",
                    content="test query relevant content",
                    score=0.9,
                )
            ]

        llm = FakeLLM()
        pipeline = AdvancedRAGPipeline(retrieve_fn=retrieve_fn, llm=llm)
        result = await pipeline.execute("test query")

        assert result.query == "test query"
        assert len(result.results) > 0
        assert len(result.context) > 0
        assert result.should_answer is True

    async def test_empty_results(self) -> None:
        async def retrieve_fn(query: str, top_k: int) -> list[RetrievalResult]:
            return []

        llm = FakeLLM()
        pipeline = AdvancedRAGPipeline(retrieve_fn=retrieve_fn, llm=llm)
        result = await pipeline.execute("test")

        assert len(result.results) == 0
        assert result.should_answer is False

    async def test_hyde_transformation(self) -> None:
        async def retrieve_fn(query: str, top_k: int) -> list[RetrievalResult]:
            return [
                RetrievalResult(
                    chunk_id="c1",
                    document_id="d1",
                    content=f"年假政策 result for {query}",
                    score=0.9,
                )
            ]

        llm = FakeLLM({"假设性": "公司年假政策规定每年5天带薪年假。"})
        pipeline = AdvancedRAGPipeline(retrieve_fn=retrieve_fn, llm=llm)
        result = await pipeline.execute("年假", use_hyde=True)

        assert result.final_query != "年假"
        assert any("hyde" in h["strategy"] for h in result.transform_history)

    async def test_multi_query_transformation(self) -> None:
        async def retrieve_fn(query: str, top_k: int) -> list[RetrievalResult]:
            return [
                RetrievalResult(
                    chunk_id=f"c_{query[:4]}",
                    document_id="d1",
                    content=f"年假内容 {query}",
                    score=0.9,
                )
            ]

        llm = FakeLLM({"改写": "1. 年假天数\n2. 年假申请"})
        pipeline = AdvancedRAGPipeline(retrieve_fn=retrieve_fn, llm=llm)
        result = await pipeline.execute("年假", use_multi_query=True)

        assert len(result.results) > 0
        assert any("multi_query" in h["strategy"] for h in result.transform_history)

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

        llm = FakeLLM({"改写成": "improved query"})
        pipeline = AdvancedRAGPipeline(retrieve_fn=retrieve_fn, llm=llm)
        result = await pipeline.execute("test query", max_corrections=1)

        assert len(result.results) > 0
        assert result.should_answer is True

    async def test_preserves_original_query(self) -> None:
        async def retrieve_fn(query: str, top_k: int) -> list[RetrievalResult]:
            return [
                RetrievalResult(
                    chunk_id="c1", document_id="d1", content=f"data for {query}", score=0.9
                )
            ]

        llm = FakeLLM()
        pipeline = AdvancedRAGPipeline(retrieve_fn=retrieve_fn, llm=llm)
        result = await pipeline.execute("original query")

        assert result.original_query == "original query"
        assert result.query == "original query"
