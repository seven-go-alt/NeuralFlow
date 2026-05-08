from __future__ import annotations

from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    texts: list[str]
    model: str


class EmbeddingResult(BaseModel):
    vectors: list[list[float]] = Field(default_factory=list)
    model: str
