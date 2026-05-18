from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class KeywordResult:
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    matched_terms: list[str] = field(default_factory=list)
    source: dict[str, Any] = field(default_factory=dict)


def tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens, keeping CJK characters as individual tokens."""
    tokens: list[str] = []
    for char in text:
        if "一" <= char <= "鿿" or "぀" <= char <= "ヿ":
            tokens.append(char)
            continue
    tokens.extend(re.findall(r"[a-zA-Z0-9_]+", text.lower()))
    return tokens


class KeywordStore:
    """In-memory keyword store backed by chunk metadata.

    Primarily designed for testing and evaluation. In production, this
    would be backed by a database full-text search index.
    """

    def __init__(self) -> None:
        self._chunks: list[dict[str, Any]] = []

    def index(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = list(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        tenant_id: str | None = None,
    ) -> list[KeywordResult]:
        query_terms = tokenize(query)
        if not query_terms:
            return []

        scored: list[tuple[float, int, dict[str, Any]]] = []

        for idx, chunk in enumerate(self._chunks):
            if tenant_id and chunk.get("tenant_id") != tenant_id:
                continue

            chunk_text = (chunk.get("content") or "").lower()
            chunk_tokens = Counter(tokenize(chunk_text))

            matched_terms: list[str] = []
            score = 0.0
            for term in query_terms:
                tf = chunk_tokens.get(term, 0)
                if tf > 0:
                    matched_terms.append(term)
                    score += math.log(1 + tf) * math.log(
                        1 + len(query_terms) / (query_terms.count(term))
                    )

            if score > 0:
                scored.append((score, idx, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[KeywordResult] = []
        for score, _, chunk in scored[:top_k]:
            chunk_text = chunk.get("content") or ""
            matched = [t for t in query_terms if t in chunk_text.lower()]
            results.append(
                KeywordResult(
                    chunk_id=chunk.get("chunk_id", ""),
                    document_id=chunk.get("document_id", ""),
                    content=chunk_text,
                    score=min(1.0, score / 10.0),
                    metadata=chunk.get("metadata", {}),
                    matched_terms=matched,
                    source={
                        "title": chunk.get("title"),
                        "filename": chunk.get("filename"),
                        "page_number": chunk.get("page_number"),
                    },
                )
            )
        return results
