from __future__ import annotations

from typing import Any, Protocol

from app.retrieval.keyword_store import KeywordResult, KeywordStore
from app.retrieval.reranker import heuristic_rerank
from app.retrieval.schemas import RetrievalRequest, RetrievalResponse, RetrievalResult


class VectorStore(Protocol):
    async def query(
        self,
        query_text: str,
        top_k: int,
        where: dict[str, Any],
    ) -> dict[str, Any]: ...


class HybridRetrievalService:
    """Retrieval service supporting vector, keyword, and hybrid modes.

    Usage:
        store = ChromaDocumentStore()
        kw_store = KeywordStore()
        service = HybridRetrievalService(store, kw_store)

        # Vector mode (default, same as current RetrievalService)
        result = await service.search(tenant_id, request, mode="vector")

        # Keyword mode
        result = await service.search(tenant_id, request, mode="keyword")

        # Hybrid mode (merge + rerank)
        result = await service.search(tenant_id, request, mode="hybrid")
    """

    def __init__(
        self,
        vector_store: VectorStore,
        keyword_store: KeywordStore | None = None,
    ) -> None:
        self._vector_store: VectorStore = vector_store
        self._keyword_store = keyword_store or KeywordStore()

    async def search(
        self,
        tenant_id: str,
        request: RetrievalRequest,
        mode: str = "vector",
    ) -> RetrievalResponse:
        if mode == "keyword":
            return await self._keyword_search(tenant_id, request)
        if mode == "hybrid":
            return await self._hybrid_search(tenant_id, request)
        return await self._vector_search(tenant_id, request)

    async def _vector_search(
        self,
        tenant_id: str,
        request: RetrievalRequest,
    ) -> RetrievalResponse:
        where = self._build_where(tenant_id, request.filters.model_dump())
        response = await self._vector_store.query(
            query_text=request.query,
            top_k=request.top_k,
            where=where,
        )
        return self._parse_chroma_response(request.query, response, request.score_threshold)

    async def _keyword_search(
        self,
        tenant_id: str,
        request: RetrievalRequest,
    ) -> RetrievalResponse:
        kw_results = self._keyword_store.search(
            query=request.query,
            top_k=request.top_k,
            tenant_id=tenant_id,
        )
        results = self._kw_to_retrieval_results(kw_results, request.score_threshold)
        return RetrievalResponse(query=request.query, results=results)

    async def _hybrid_search(
        self,
        tenant_id: str,
        request: RetrievalRequest,
    ) -> RetrievalResponse:
        vector_response = await self._vector_search(tenant_id, request)
        kw_results = self._keyword_store.search(
            query=request.query,
            top_k=request.top_k,
            tenant_id=tenant_id,
        )
        kw_results_filtered = self._kw_to_retrieval_results(kw_results, 0.0)

        merged = self._merge_results(vector_response.results, kw_results_filtered)
        reranked = heuristic_rerank(merged, request.query)

        if request.score_threshold > 0:
            reranked = [r for r in reranked if r.score >= request.score_threshold]

        return RetrievalResponse(query=request.query, results=reranked)

    def _merge_results(
        self,
        vector_results: list[RetrievalResult],
        keyword_results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        seen: set[str] = set()
        merged: list[RetrievalResult] = []
        for r in vector_results + keyword_results:
            key = f"{r.document_id}:{r.chunk_id}"
            if key in seen:
                continue
            seen.add(key)
            merged.append(r)
        return merged

    def _kw_to_retrieval_results(
        self,
        kw_results: list[KeywordResult],
        score_threshold: float,
    ) -> list[RetrievalResult]:
        results: list[RetrievalResult] = []
        for kw in kw_results:
            if kw.score < score_threshold:
                continue
            results.append(
                RetrievalResult(
                    chunk_id=kw.chunk_id,
                    document_id=kw.document_id,
                    content=kw.content,
                    score=kw.score,
                    metadata=kw.metadata,
                    source=kw.source,
                )
            )
        return results

    def _build_where(self, tenant_id: str, filters: dict[str, Any]) -> dict[str, Any]:
        clauses: list[dict[str, Any]] = [{"tenant_id": tenant_id}]
        document_ids = filters.get("document_ids") or []
        if len(document_ids) == 1:
            clauses.append({"document_id": document_ids[0]})
        file_types = filters.get("file_types") or []
        if len(file_types) == 1:
            clauses.append({"file_type": file_types[0]})
        return {"$and": clauses} if len(clauses) > 1 else clauses[0]

    def _parse_chroma_response(
        self,
        query: str,
        response: dict[str, Any],
        score_threshold: float,
    ) -> RetrievalResponse:
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        ids = (response.get("ids") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        results: list[RetrievalResult] = []
        for idx, content in enumerate(documents):
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            distance = float(distances[idx] if idx < len(distances) else 0.0)
            score = 1.0 / (1.0 + max(0.0, distance))
            if score < score_threshold:
                continue
            results.append(
                RetrievalResult(
                    chunk_id=metadata.get(
                        "chunk_id",
                        ids[idx] if idx < len(ids) else f"chunk_{idx}",
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
        return RetrievalResponse(query=query, results=self._dedupe(results))

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
