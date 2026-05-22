from __future__ import annotations

import pytest

from app.utils.vector_client import (
    InMemoryVectorClient,
    VectorStoreUnavailableError,
    get_vector_client,
)


def test_in_memory_query_filters_by_where_clause() -> None:
    client = InMemoryVectorClient()
    coll = client.get_or_create_collection("test")
    coll.add(
        documents=["leave policy", "salary info", "office hours"],
        metadatas=[
            {"doc_id": "d1", "dept": "hr"},
            {"doc_id": "d2", "dept": "finance"},
            {"doc_id": "d3", "dept": "hr"},
        ],
        ids=["c1", "c2", "c3"],
    )

    result = coll.query(query_texts=["leave"], n_results=5, where={"dept": "hr"})
    assert len(result["documents"][0]) == 1


def test_in_memory_query_filters_with_and_clause() -> None:
    client = InMemoryVectorClient()
    coll = client.get_or_create_collection("test")
    coll.add(
        documents=["doc1", "doc2"],
        metadatas=[
            {"tenant": "pub", "dept": "hr"},
            {"tenant": "pub", "dept": "eng"},
        ],
        ids=["c1", "c2"],
    )

    result = coll.query(
        query_texts=["doc"], n_results=5, where={"$and": [{"tenant": "pub"}, {"dept": "hr"}]}
    )
    assert len(result["documents"][0]) == 1
    assert result["ids"][0][0] == "c1"


def test_in_memory_get_with_where() -> None:
    client = InMemoryVectorClient()
    coll = client.get_or_create_collection("g")
    coll.add(
        documents=["hello"],
        metadatas=[{"lang": "en"}],
        ids=["c1"],
    )

    result = coll.get(where={"lang": "en"})
    assert result["documents"] == ["hello"]


def test_get_vector_client_returns_in_memory_when_chroma_unavailable(monkeypatch) -> None:
    """When ChromaDB is unreachable and allow_in_memory=True, return InMemoryVectorClient."""
    import chromadb

    def failing_client(*args, **kwargs):
        raise ConnectionError("Chroma unreachable")

    monkeypatch.setattr(chromadb, "HttpClient", failing_client)

    client = get_vector_client(allow_in_memory=True)
    assert isinstance(client, InMemoryVectorClient)


def test_get_vector_client_raises_when_chroma_unavailable_and_not_allowed(monkeypatch) -> None:
    import chromadb

    def failing_client(*args, **kwargs):
        raise ConnectionError("Chroma unreachable")

    monkeypatch.setattr(chromadb, "HttpClient", failing_client)

    with pytest.raises(VectorStoreUnavailableError, match="ChromaDB unavailable"):
        get_vector_client(allow_in_memory=False)
