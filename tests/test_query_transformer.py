from __future__ import annotations

from app.rag.query_transformer import (
    TransformResult,
    _parse_variants,
    expand_multi_query,
    hyde_transform,
    merge_deduplicated,
    rewrite_query,
)
from app.retrieval.schemas import RetrievalResult


class FakeLLM:
    """Synchronous stub that records prompts and returns canned responses."""

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self.responses = responses or {}
        self.last_prompt: str | None = None

    async def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        for key, response in self.responses.items():
            if key in prompt:
                return response
        return "default response"


class TestRewriteQuery:
    async def test_basic_rewrite(self) -> None:
        llm = FakeLLM({"改写成": "公司年假政策的具体规定是什么？"})
        result = await rewrite_query("年假", llm)
        assert result == "公司年假政策的具体规定是什么？"

    async def test_fallback_to_original_when_empty(self) -> None:
        llm = FakeLLM({"改写成": ""})
        result = await rewrite_query("年假", llm)
        assert result == "年假"  # should fall back to original

    async def test_strips_quotes(self) -> None:
        llm = FakeLLM({"改写成": '"年假政策规定"'})
        result = await rewrite_query("年假", llm)
        assert result == "年假政策规定"


class TestExpandMultiQuery:
    async def test_generates_variants(self) -> None:
        llm = FakeLLM(
            {
                "改写": "1. 年假天数规定\n2. 年假申请流程\n3. 年假工资计算",
            }
        )
        result = await expand_multi_query("年假", llm, n=3)
        assert isinstance(result, TransformResult)
        assert result.strategy == "multi_query"
        assert len(result.variants) >= 2

    async def test_empty_variant_falls_back(self) -> None:
        llm = FakeLLM({"改写": ""})
        result = await expand_multi_query("年假", llm)
        assert len(result.variants) >= 1

    async def test_original_query_preserved(self) -> None:
        llm = FakeLLM({"改写": "variant"})
        result = await expand_multi_query("年假", llm)
        assert result.original_query == "年假"


class TestHydeTransform:
    async def test_generates_hypothesis(self) -> None:
        llm = FakeLLM({"假设性": "根据公司规定，年假为5天。"})
        result = await hyde_transform("年假多少天", llm)
        assert len(result) > len("年假多少天")

    async def test_fallback_when_hypothesis_too_short(self) -> None:
        llm = FakeLLM({"假设性": "短"})
        result = await hyde_transform("年假多少天", llm)
        assert result == "年假多少天"


class TestParseVariants:
    def test_numbered_lines(self) -> None:
        text = "1. 第一变体\n2. 第二变体\n3. 第三变体"
        result = _parse_variants(text)
        assert result == ["第一变体", "第二变体", "第三变体"]

    def test_short_variants_filtered(self) -> None:
        text = "1. a\n2. bc\n3. defg"
        result = _parse_variants(text)
        assert result == ["defg"]

    def test_empty_input(self) -> None:
        assert _parse_variants("") == []

    def test_max_five_variants(self) -> None:
        text = "\n".join(f"{i}. variant_{i}" for i in range(10))
        result = _parse_variants(text)
        assert len(result) <= 5


class TestMergeDeduplicated:
    def test_merges_unique_results(self) -> None:
        r1 = RetrievalResult(chunk_id="c1", document_id="d1", content="a", score=0.9)
        r2 = RetrievalResult(chunk_id="c2", document_id="d1", content="b", score=0.8)
        merged = merge_deduplicated([[r1], [r2]])
        assert len(merged) == 2

    def test_deduplicates_by_doc_chunk_id(self) -> None:
        r1 = RetrievalResult(chunk_id="c1", document_id="d1", content="a", score=0.9)
        r2 = RetrievalResult(chunk_id="c1", document_id="d1", content="a", score=0.9)
        merged = merge_deduplicated([[r1], [r2]])
        assert len(merged) == 1

    def test_order_preserved(self) -> None:
        r1 = RetrievalResult(chunk_id="c1", document_id="d1", content="first", score=0.9)
        r2 = RetrievalResult(chunk_id="c2", document_id="d1", content="second", score=0.8)
        r3 = RetrievalResult(chunk_id="c1", document_id="d1", content="dup", score=0.7)
        merged = merge_deduplicated([[r1], [r2], [r3]])
        assert [r.content for r in merged] == ["first", "second"]

    def test_empty_input(self) -> None:
        assert merge_deduplicated([]) == []
