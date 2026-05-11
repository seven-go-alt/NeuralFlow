from __future__ import annotations

import pytest

from app.retrieval.chroma_store import ChromaDocumentStore


class SpyEmbeddingService:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str]] = []

    async def embed_texts(self, texts: list[str], model: str) -> list[list[float]]:
        self.calls.append((texts, model))
        return [[0.1, 0.2, 0.3]]


class SpyCollection:
    def __init__(self) -> None:
        self.query_calls: list[dict] = []

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        return {
            "documents": [["Annual leave policy: 12 days"]],
            "metadatas": [[{"document_id": "doc_1", "chunk_id": "chk_1", "title": "Policy"}]],
            "ids": [["chk_1"]],
            "distances": [[0.01]],
        }


class SpyClient:
    def __init__(self, collection: SpyCollection) -> None:
        self.collection = collection

    def get_or_create_collection(self, name: str):
        return self.collection


@pytest.mark.asyncio
async def test_chroma_document_store_queries_with_explicit_query_embeddings(monkeypatch) -> None:
    collection = SpyCollection()
    embedding_service = SpyEmbeddingService()
    monkeypatch.setattr(
        "app.retrieval.chroma_store.get_vector_client",
        lambda allow_in_memory=False: SpyClient(collection),
    )

    store = ChromaDocumentStore(embedding_service=embedding_service)
    result = await store.query("annual leave", top_k=3, where={"tenant_id": "public"})

    assert embedding_service.calls == [(["annual leave"], "text-embedding-3-small")]
    assert collection.query_calls == [
        {
            "query_embeddings": [[0.1, 0.2, 0.3]],
            "n_results": 3,
            "where": {"tenant_id": "public"},
        }
    ]
    assert result["documents"][0][0] == "Annual leave policy: 12 days"
