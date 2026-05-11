from __future__ import annotations

import dataclasses
from typing import Any, Protocol, Sequence, cast

from app.embeddings.service import EmbeddingService
from app.utils.vector_client import get_vector_client


class EmbeddingServiceLike(Protocol):
    async def embed_texts(self, texts: list[str], model: str) -> list[list[float]]: ...


class ChromaDocumentStore:
    def __init__(
        self,
        collection_name: str = "document_knowledge",
        *,
        allow_in_memory: bool = False,
        embedding_service: EmbeddingServiceLike | None = None,
    ) -> None:
        self.client = get_vector_client(allow_in_memory=allow_in_memory)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedding_service = embedding_service or EmbeddingService()

    def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return
        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["content"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        embeddings = [chunk.get("embedding") for chunk in chunks]
        if any(embedding is None for embedding in embeddings):
            raise ValueError("All chunks must have embeddings before upsert")
        try:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,  # type: ignore[arg-type]
            )
        except AttributeError:
            self.collection.add(ids=ids, documents=documents, metadatas=metadatas)

    async def query(
        self, query_text: str, top_k: int, where: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        query_embedding = await self.embedding_service.embed_texts(
            [query_text], model="text-embedding-3-small"
        )
        query_vectors: Sequence[Sequence[float]] = query_embedding
        result = self.collection.query(
            query_embeddings=query_vectors,  # type: ignore[arg-type]
            n_results=top_k,
            where=where,
        )
        if isinstance(result, dict):
            return cast(dict[str, Any], result)
        return dataclasses.asdict(result)

    def delete_document(self, tenant_id: str, document_id: str) -> None:
        where = {"$and": [{"tenant_id": tenant_id}, {"document_id": document_id}]}
        if hasattr(self.collection, "delete"):
            self.collection.delete(where=where)  # type: ignore[arg-type]
