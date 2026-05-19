from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence
from typing import Any, Protocol, cast

from app.embeddings.service import EmbeddingService
from app.utils.retry import retry_sync
from app.utils.vector_client import get_vector_client

logger = logging.getLogger(__name__)


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

        def _do_upsert() -> None:
            try:
                self.collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                )
            except AttributeError:
                self.collection.add(ids=ids, documents=documents, metadatas=metadatas)

        try:
            retry_sync(_do_upsert, max_attempts=3, base_delay=0.5)
        except Exception:
            logger.warning("chroma upsert failed after retries", exc_info=True)

    async def query(
        self, query_text: str, top_k: int, where: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        query_embedding = await self.embedding_service.embed_texts(
            [query_text], model="text-embedding-3-small"
        )
        query_vectors: Sequence[Sequence[float]] = query_embedding
        result = self.collection.query(
            query_embeddings=query_vectors,
            n_results=top_k,
            where=where,
        )
        if isinstance(result, dict):
            return cast(dict[str, Any], result)
        return dataclasses.asdict(result)

    def delete_document(self, tenant_id: str, document_id: str) -> None:
        where = {"$and": [{"tenant_id": tenant_id}, {"document_id": document_id}]}
        if hasattr(self.collection, "delete"):
            self.collection.delete(where=where)
