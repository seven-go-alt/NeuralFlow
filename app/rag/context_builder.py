from __future__ import annotations

from app.core.token_budget import ContextSegment, TokenBudgetManager
from app.rag.schemas import RAGBuildResponse
from app.retrieval.schemas import RetrievalResult


class RAGContextBuilder:
    def __init__(self, token_budget_manager: TokenBudgetManager | None = None) -> None:
        self.token_budget_manager = token_budget_manager or TokenBudgetManager()

    def build(self, query: str, results: list[RetrievalResult]) -> RAGBuildResponse:
        unique_results: list[RetrievalResult] = []
        seen: set[str] = set()
        for item in results:
            key = item.content[:160]
            if key in seen:
                continue
            seen.add(key)
            unique_results.append(item)
        context_blocks = []
        citations = []
        segments: list[ContextSegment] = []
        for index, item in enumerate(unique_results, start=1):
            label = item.source.get("title") or item.source.get("filename") or item.document_id
            citation = f"[{index}] {label}"
            block = f"{citation}\n{item.content}"
            context_blocks.append(block)
            citations.append(
                {
                    "index": index,
                    "label": label,
                    "document_id": item.document_id,
                    "chunk_id": item.chunk_id,
                    "page_number": item.source.get("page_number"),
                }
            )
            segments.append(ContextSegment(name=f"retrieval_{index}", text=block, priority=2))
        trimmed = self.token_budget_manager.trim_context(segments)
        return RAGBuildResponse(
            query=query,
            context=trimmed.trimmed_text,
            citations=citations,
            used_chunks=unique_results,
            token_before_trim=trimmed.token_before_trim,
            token_after_trim=trimmed.token_after_trim,
        )
