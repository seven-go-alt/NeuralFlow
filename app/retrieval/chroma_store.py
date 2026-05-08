from __future__ import annotations

from typing import Any

from app.utils.vector_client import get_vector_client


class ChromaDocumentStore:
    def __init__(self, collection_name: str = "document_knowledge") -> None:
        self.client = get_vector_client()
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return
        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["content"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        embeddings = [chunk.get("embedding") for chunk in chunks]
        try:
            self.collection.upsert(
                ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
            ) # type: ignore[arg-type]
        except AttributeError:
            self.collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def query(
        self, query_text: str, top_k: int, where: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self.collection.query(query_texts=[query_text], n_results=top_k, where=where)

    def delete_document(self, tenant_id: str, document_id: str) -> None:
        where = {"$and": [{"tenant_id": tenant_id}, {"document_id": document_id}]}
        if hasattr(self.collection, "delete"):
            self.collection.delete(where=where)
