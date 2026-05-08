from __future__ import annotations

from typing import Any

from app.documents.repository import DocumentRepository
from app.embeddings.service import EmbeddingService
from app.retrieval.chroma_store import ChromaDocumentStore
from app.retrieval.schemas import RetrievalRequest, RetrievalResponse, RetrievalResult


class RetrievalService:
    def __init__(
        self, document_repo: DocumentRepository, embedding_service: EmbeddingService | None = None
    ) -> None:
        self.document_repo = document_repo
        self.embedding_service = embedding_service or EmbeddingService()
        self.store = ChromaDocumentStore()

    async def search(self, tenant_id: str, request: RetrievalRequest) -> RetrievalResponse:
        where = self._build_where(tenant_id=tenant_id, filters=request.filters.model_dump())
        response = self.store.query(query_text=request.query, top_k=request.top_k, where=where)
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        ids = (response.get("ids") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        results: list[RetrievalResult] = []
        for idx, content in enumerate(documents):
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            score = max(0.0, 1.0 - float(distances[idx] if idx < len(distances) else 0.0))
            if score < request.score_threshold:
                continue
            results.append(
                RetrievalResult(
                    chunk_id=metadata.get(
                        "chunk_id", ids[idx] if idx < len(ids) else f"chunk_{idx}"
                    ),
                    document_id=metadata.get("document_id", "unknown"),
                    content=content,
                    score=score,
                    metadata=metadata,
                    source={
                        "title": metadata.get("title"),
                        "filename": metadata.get("filename"),
                        "page_number": metadata.get("page_number"),
                    },
                )
            )
        return RetrievalResponse(query=request.query, results=self._dedupe(results))

    def _build_where(self, tenant_id: str, filters: dict[str, Any]) -> dict[str, Any]:
        clauses: list[dict[str, Any]] = [{"tenant_id": tenant_id}]
        document_ids = filters.get("document_ids") or []
        if len(document_ids) == 1:
            clauses.append({"document_id": document_ids[0]})
        file_types = filters.get("file_types") or []
        if len(file_types) == 1:
            clauses.append({"file_type": file_types[0]})
        return {"$and": clauses} if len(clauses) > 1 else clauses[0]

    def _dedupe(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        seen: set[str] = set()
        deduped: list[RetrievalResult] = []
        for item in results:
            key = f"{item.document_id}:{item.chunk_id}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped
