from __future__ import annotations

from app.rag.context_builder import RAGContextBuilder
from app.retrieval.schemas import RetrievalRequest
from app.retrieval.service import RetrievalService


class RAGService:
    def __init__(self, retrieval_service: RetrievalService, context_builder: RAGContextBuilder | None = None) -> None:
        self.retrieval_service = retrieval_service
        self.context_builder = context_builder or RAGContextBuilder()

    async def build_context(self, tenant_id: str, query: str, *, top_k: int = 5, score_threshold: float = 0.0):
        retrieval = await self.retrieval_service.search(
            tenant_id=tenant_id,
            request=RetrievalRequest(query=query, top_k=top_k, score_threshold=score_threshold),
        )
        return self.context_builder.build(query=query, results=retrieval.results)
