from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.retrieval.schemas import RetrievalResult


class RAGBuildResponse(BaseModel):
    query: str
    context: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    used_chunks: list[RetrievalResult] = Field(default_factory=list)
    token_before_trim: int = 0
    token_after_trim: int = 0
