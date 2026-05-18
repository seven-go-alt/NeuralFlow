from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.retrieval.hybrid_service import HybridRetrievalService
from app.retrieval.keyword_store import KeywordResult, KeywordStore, tokenize
from app.retrieval.reranker import heuristic_rerank
from app.retrieval.schemas import RetrievalRequest, RetrievalResult

# --- tokenize ---


class TestTokenize:
    def test_english_words(self) -> None:
        assert tokenize("hello world") == ["hello", "world"]

    def test_cjk_chars(self) -> None:
        tokens = tokenize("公司年假政策")
        assert tokens == ["公", "司", "年", "假", "政", "策"]

    def test_mixed(self) -> None:
        tokens = tokenize("年假 policy 2024")
        assert "年" in tokens
        assert "假" in tokens
        assert "policy" in tokens
        assert "2024" in tokens

    def test_empty(self) -> None:
        assert tokenize("") == []


# --- KeywordStore ---


class TestKeywordStore:
    def test_search_hit(self) -> None:
        store = KeywordStore()
        store.index(
            [
                {
                    "chunk_id": "c1",
                    "document_id": "d1",
                    "content": "年假政策 15 天",
                    "tenant_id": "t1",
                },
                {
                    "chunk_id": "c2",
                    "document_id": "d2",
                    "content": "考勤制度 迟到 规定",
                    "tenant_id": "t1",
                },
            ]
        )
        results = store.search("年假", top_k=5, tenant_id="t1")
        assert len(results) == 1
        assert results[0].chunk_id == "c1"
        assert results[0].score > 0

    def test_search_miss(self) -> None:
        store = KeywordStore()
        store.index(
            [
                {"chunk_id": "c1", "document_id": "d1", "content": "年假政策", "tenant_id": "t1"},
            ]
        )
        results = store.search("考勤", top_k=5, tenant_id="t1")
        assert len(results) == 0

    def test_search_empty_query(self) -> None:
        store = KeywordStore()
        store.index([{"chunk_id": "c1", "document_id": "d1", "content": "hello"}])
        results = store.search("", top_k=5)
        assert len(results) == 0

    def test_tenant_filter(self) -> None:
        store = KeywordStore()
        store.index(
            [
                {"chunk_id": "c1", "document_id": "d1", "content": "年假", "tenant_id": "t1"},
                {"chunk_id": "c2", "document_id": "d2", "content": "年假", "tenant_id": "t2"},
            ]
        )
        results = store.search("年假", top_k=5, tenant_id="t1")
        assert len(results) == 1
        assert results[0].document_id == "d1"

    def test_no_index(self) -> None:
        store = KeywordStore()
        results = store.search("hello", top_k=5)
        assert len(results) == 0


# --- KeywordResult ---


def test_keyword_result_defaults() -> None:
    r = KeywordResult(chunk_id="c1", document_id="d1", content="text", score=0.5)
    assert r.matched_terms == []
    assert r.metadata == {}
    assert r.source == {}


# --- reranker ---


def _make_result(
    chunk_id: str,
    doc_id: str,
    content: str,
    score: float,
    title: str | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=doc_id,
        content=content,
        score=score,
        source={"title": title, "filename": None, "page_number": None},
    )


class TestHeuristicRerank:
    def test_empty(self) -> None:
        assert heuristic_rerank([], "query") == []

    def test_preserves_order_when_no_keyword_match(self) -> None:
        results = [
            _make_result("c1", "d1", "some content", 0.9),
            _make_result("c2", "d2", "other content", 0.5),
        ]
        reranked = heuristic_rerank(results, "unrelated query")
        assert len(reranked) == 2
        assert reranked[0].chunk_id == "c1"

    def test_boosts_with_keyword_match(self) -> None:
        results = [
            _make_result("c1", "d1", "python code", 0.3),
            _make_result("c2", "d2", "javascript code", 0.8),
        ]
        reranked = heuristic_rerank(results, "python")
        assert reranked[0].chunk_id == "c1"


# --- HybridRetrievalService ---


@pytest.fixture
def stub_vector_store() -> AsyncMock:
    store = AsyncMock()
    store.query = AsyncMock(
        return_value={
            "documents": [["vector content"]],
            "metadatas": [[{"chunk_id": "v1", "document_id": "d1", "tenant_id": "t1"}]],
            "ids": [["v1"]],
            "distances": [[0.1]],
        }
    )
    return store


@pytest.fixture
def stub_kw_store() -> KeywordStore:
    store = KeywordStore()
    store.index(
        [
            {
                "chunk_id": "k1",
                "document_id": "d2",
                "content": "keyword content",
                "tenant_id": "t1",
            },
        ]
    )
    return store


class TestHybridRetrievalService:
    @pytest.mark.asyncio
    async def test_vector_mode(
        self, stub_vector_store: AsyncMock, stub_kw_store: KeywordStore
    ) -> None:
        service = HybridRetrievalService(stub_vector_store, stub_kw_store)
        request = RetrievalRequest(query="test", top_k=5)
        response = await service.search("t1", request, mode="vector")
        assert len(response.results) == 1
        assert response.results[0].chunk_id == "v1"

    @pytest.mark.asyncio
    async def test_keyword_mode(
        self, stub_vector_store: AsyncMock, stub_kw_store: KeywordStore
    ) -> None:
        service = HybridRetrievalService(stub_vector_store, stub_kw_store)
        request = RetrievalRequest(query="keyword", top_k=5)
        response = await service.search("t1", request, mode="keyword")
        # keyword store has "keyword content" which matches "keyword"
        assert len(response.results) == 1
        assert response.results[0].document_id == "d2"

    @pytest.mark.asyncio
    async def test_hybrid_mode(
        self, stub_vector_store: AsyncMock, stub_kw_store: KeywordStore
    ) -> None:
        service = HybridRetrievalService(stub_vector_store, stub_kw_store)
        request = RetrievalRequest(query="keyword", top_k=10)
        response = await service.search("t1", request, mode="hybrid")
        # Both vector and keyword results should be present
        doc_ids = {r.document_id for r in response.results}
        assert "d1" in doc_ids
        assert "d2" in doc_ids

    @pytest.mark.asyncio
    async def test_hybrid_dedup(
        self, stub_vector_store: AsyncMock, stub_kw_store: KeywordStore
    ) -> None:
        """When vector and keyword return the same chunk, deduplicate."""
        store = KeywordStore()
        store.index(
            [
                {
                    "chunk_id": "v1",
                    "document_id": "d1",
                    "content": "vector content",
                    "tenant_id": "t1",
                },
            ]
        )
        service = HybridRetrievalService(stub_vector_store, store)
        request = RetrievalRequest(query="vector", top_k=10)
        response = await service.search("t1", request, mode="hybrid")
        assert len(response.results) == 1

    @pytest.mark.asyncio
    async def test_score_threshold_applied(
        self, stub_vector_store: AsyncMock, stub_kw_store: KeywordStore
    ) -> None:
        """Results below score threshold are filtered."""
        service = HybridRetrievalService(stub_vector_store, stub_kw_store)
        request = RetrievalRequest(query="test", top_k=10, score_threshold=0.99)
        response = await service.search("t1", request, mode="vector")
        assert len(response.results) == 0

    def test_build_where(self) -> None:
        service = HybridRetrievalService(AsyncMock())
        where = service._build_where("t1", {})
        assert where == {"tenant_id": "t1"}
