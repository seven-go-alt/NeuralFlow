from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RetrievalFilters(BaseModel):
    document_ids: list[str] = Field(default_factory=list)
    file_types: list[str] = Field(default_factory=list)
    content_types: list[str] = Field(default_factory=list)


class RetrievalRequest(BaseModel):
    query: str
    top_k: int = 5
    score_threshold: float = 0.0
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)


class RetrievalResult(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: dict[str, Any] = Field(default_factory=dict)


class RetrievalResponse(BaseModel):
    query: str
    results: list[RetrievalResult] = Field(default_factory=list)
