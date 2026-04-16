from __future__ import annotations

import asyncio

import pytest

from app.memory.vector_retriever import VectorRetriever


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.set_calls: list[tuple[str, int, str]] = []

    def get(self, key: str) -> str | None:
        return self.store.get(key)

    def setex(self, key: str, ttl: int, value: str) -> bool:
        self.store[key] = value
        self.set_calls.append((key, ttl, value))
        return True


class SpyCollection:
    def __init__(self) -> None:
        self.query_calls: list[dict] = []
        self.get_calls: list[dict] = []
        self.query_result = {
            "documents": [["summary one"]],
            "metadatas": [[{"session_id": "s1", "type": "summary"}]],
            "distances": [[0.12]],
            "ids": [["doc-1"]],
        }
        self.documents = [
            {"id": "doc-1", "document": "redis cache summary", "metadata": {"session_id": "s1", "type": "summary"}},
            {"id": "doc-2", "document": "python traceback fix", "metadata": {"session_id": "s1", "type": "summary"}},
        ]

    def query(self, query_texts, n_results=3, where=None):
        self.query_calls.append({"query_texts": query_texts, "n_results": n_results, "where": where})
        return self.query_result

    def get(self, where=None, include=None):
        self.get_calls.append({"where": where, "include": include})
        matched = [
            item for item in self.documents if all(item["metadata"].get(k) == v for k, v in (where or {}).items())
        ]
        return {
            "documents": [item["document"] for item in matched],
            "metadatas": [item["metadata"] for item in matched],
            "ids": [item["id"] for item in matched],
        }


@pytest.mark.asyncio
async def test_vector_retriever_uses_metadata_filter_and_returns_structured_results() -> None:
    collection = SpyCollection()
    retriever = VectorRetriever(collection=collection, cache_client=FakeRedis(), cache_ttl_seconds=300)

    results = await retriever.search("summary", session_id="s1", memory_type="summary", top_k=2)

    assert collection.query_calls == [
        {"query_texts": ["summary"], "n_results": 2, "where": {"session_id": "s1", "type": "summary"}}
    ]
    assert results == [
        {
            "content": "summary one",
            "metadata": {"session_id": "s1", "type": "summary"},
            "score": pytest.approx(0.88),
            "source": "vector",
        }
    ]


@pytest.mark.asyncio
async def test_vector_retriever_uses_cache_on_repeated_queries() -> None:
    collection = SpyCollection()
    cache = FakeRedis()
    retriever = VectorRetriever(collection=collection, cache_client=cache, cache_ttl_seconds=300)

    first = await retriever.search("summary", session_id="s1", memory_type="summary", top_k=2)
    second = await retriever.search("summary", session_id="s1", memory_type="summary", top_k=2)

    assert first == second
    assert len(collection.query_calls) == 1
    assert len(cache.set_calls) == 1


@pytest.mark.asyncio
async def test_vector_retriever_falls_back_to_keyword_search_when_vector_query_fails() -> None:
    collection = SpyCollection()

    def raise_query(*args, **kwargs):
        raise RuntimeError("chroma unavailable")

    collection.query = raise_query  # type: ignore[method-assign]
    retriever = VectorRetriever(collection=collection, cache_client=FakeRedis(), cache_ttl_seconds=300)

    results = await retriever.search("python fix", session_id="s1", memory_type="summary", top_k=2)

    assert collection.get_calls == [
        {"where": {"session_id": "s1", "type": "summary"}, "include": ["documents", "metadatas"]}
    ]
    assert results[0]["content"] == "python traceback fix"
    assert results[0]["source"] == "keyword"
    assert results[0]["score"] > 0
