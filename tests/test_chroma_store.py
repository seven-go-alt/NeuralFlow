from __future__ import annotations

import pytest

from app.config import Settings
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
        "app.retrieval.chroma_store.get_settings",
        lambda: Settings(embedding_model="test-embedding-model"),
    )
    monkeypatch.setattr(
        "app.retrieval.chroma_store.get_vector_client",
        lambda allow_in_memory=False: SpyClient(collection),
    )

    store = ChromaDocumentStore(embedding_service=embedding_service)
    result = await store.query("annual leave", top_k=3, where={"tenant_id": "public"})

    assert embedding_service.calls == [(["annual leave"], "test-embedding-model")]
    assert collection.query_calls == [
        {
            "query_embeddings": [[0.1, 0.2, 0.3]],
            "n_results": 3,
            "where": {"tenant_id": "public"},
        }
    ]
    assert result["documents"][0][0] == "Annual leave policy: 12 days"


def test_upsert_chunks_raises_on_missing_embedding() -> None:
    store = ChromaDocumentStore(allow_in_memory=True)
    chunks = [{"chunk_id": "c1", "content": "text", "metadata": {}, "embedding": None}]
    with pytest.raises(ValueError, match="All chunks must have embeddings"):
        store.upsert_chunks(chunks)


def test_upsert_chunks_empty_list_returns_early() -> None:
    store = ChromaDocumentStore(allow_in_memory=True)
    store.upsert_chunks([])  # should not raise


def test_upsert_chunks_calls_add_on_attribute_error(monkeypatch) -> None:
    class CollectionWithoutUpsert:
        def __init__(self) -> None:
            self.add_calls: list[dict] = []

        def add(self, ids, documents, metadatas, embeddings=None):
            self.add_calls.append(
                {
                    "ids": ids,
                    "documents": documents,
                    "metadatas": metadatas,
                    "embeddings": embeddings,
                }
            )

    class ClientWithoutUpsert:
        def __init__(self) -> None:
            self.collection = CollectionWithoutUpsert()

        def get_or_create_collection(self, name: str):
            return self.collection

    monkeypatch.setattr(
        "app.retrieval.chroma_store.get_vector_client",
        lambda allow_in_memory=False: ClientWithoutUpsert(),
    )

    store = ChromaDocumentStore()
    chunks = [
        {
            "chunk_id": "c1",
            "content": "text",
            "metadata": {"doc_id": "d1"},
            "embedding": [0.1, 0.2, 0.3],
        }
    ]
    store.upsert_chunks(chunks)
    assert len(store.client.collection.add_calls) == 1  # type: ignore[union-attr]


def test_delete_document_filters_by_tenant_and_document_id(monkeypatch) -> None:
    class CollectionWithDelete:
        def __init__(self) -> None:
            self.delete_calls: list[dict] = []

        def delete(self, where: dict):
            self.delete_calls.append(where)

    class ClientWithDelete:
        def __init__(self) -> None:
            self.collection = CollectionWithDelete()

        def get_or_create_collection(self, name: str):
            return self.collection

    monkeypatch.setattr(
        "app.retrieval.chroma_store.get_vector_client",
        lambda allow_in_memory=False: ClientWithDelete(),
    )

    store = ChromaDocumentStore()
    store.delete_document(tenant_id="public", document_id="doc_1")
    assert store.client.collection.delete_calls[0] == {  # type: ignore[union-attr]
        "$and": [{"tenant_id": "public"}, {"document_id": "doc_1"}]
    }
