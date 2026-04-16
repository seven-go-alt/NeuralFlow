from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class RetrievedDocument:
    content: str
    metadata: dict[str, Any]
    score: float
    source: str


class VectorRetriever:
    def __init__(
        self,
        collection: Any,
        cache_client: Any | None = None,
        cache_ttl_seconds: int = 300,
        tenant_id: str = "public",
    ) -> None:
        self.collection = collection
        self.cache_client = cache_client
        self.cache_ttl_seconds = cache_ttl_seconds
        self.tenant_id = tenant_id or "public"
        self.last_cache_hit = False

    async def search(
        self,
        query: str,
        session_id: str | None = None,
        memory_type: str = "summary",
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        where = self._build_where(session_id=session_id, memory_type=memory_type)
        cache_key = self._build_cache_key(query=query, where=where, top_k=top_k)

        cached = await self._cache_get(cache_key)
        if cached is not None:
            self.last_cache_hit = True
            logger.info("vector retrieval cache hit", extra={"cache_hit": True, "query": query})
            return cached

        self.last_cache_hit = False
        results = await self._vector_search(query=query, where=where, top_k=top_k)
        if not results:
            results = await self._keyword_fallback(query=query, where=where, top_k=top_k)

        await self._cache_set(cache_key, results)
        logger.info(
            "vector retrieval completed",
            extra={"cache_hit": False, "query": query, "result_count": len(results)},
        )
        return results

    def _build_where(self, session_id: str | None, memory_type: str) -> dict[str, Any]:
        where: dict[str, Any] = {"type": memory_type, "tenant_id": self.tenant_id}
        if session_id:
            where["session_id"] = session_id
        return where

    def _build_cache_key(self, query: str, where: dict[str, Any], top_k: int) -> str:
        payload = json.dumps({"query": query, "where": where, "top_k": top_k}, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"vsearch:{digest}"

    async def _cache_get(self, key: str) -> list[dict[str, Any]] | None:
        if self.cache_client is None:
            return None
        try:
            raw_value = await asyncio.to_thread(self.cache_client.get, key)
        except Exception:
            logger.warning("vector cache read failed", exc_info=True)
            return None
        if not raw_value:
            return None
        if isinstance(raw_value, bytes):
            raw_value = raw_value.decode("utf-8")
        return json.loads(raw_value)

    async def _cache_set(self, key: str, results: list[dict[str, Any]]) -> None:
        if self.cache_client is None:
            return
        payload = json.dumps(results, ensure_ascii=False)
        try:
            await asyncio.to_thread(self.cache_client.setex, key, self.cache_ttl_seconds, payload)
        except Exception:
            logger.warning("vector cache write failed", exc_info=True)

    async def _vector_search(self, query: str, where: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
        try:
            response = await asyncio.to_thread(
                self.collection.query,
                query_texts=[query],
                n_results=top_k,
                where=where,
            )
        except Exception:
            logger.warning("vector search failed, falling back to keyword search", exc_info=True)
            return []
        return self._normalize_vector_results(response)

    async def _keyword_fallback(self, query: str, where: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
        response = await asyncio.to_thread(
            self.collection.get,
            where=where,
            include=["documents", "metadatas"],
        )
        documents = response.get("documents", []) or []
        metadatas = response.get("metadatas", []) or []
        query_terms = [term for term in query.lower().split() if term]
        ranked: list[RetrievedDocument] = []
        for content, metadata in zip(documents, metadatas, strict=False):
            lowered = content.lower()
            overlap = sum(1 for term in query_terms if term in lowered)
            if overlap <= 0:
                continue
            score = overlap / max(len(query_terms), 1)
            ranked.append(
                RetrievedDocument(
                    content=content,
                    metadata=metadata,
                    score=score,
                    source="keyword",
                )
            )
        ranked.sort(key=lambda item: item.score, reverse=True)
        return [asdict(item) for item in ranked[:top_k]]

    def _normalize_vector_results(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        normalized: list[dict[str, Any]] = []
        for index, content in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = distances[index] if index < len(distances) else 0.0
            score = max(0.0, 1.0 - float(distance))
            normalized.append(
                asdict(
                    RetrievedDocument(
                        content=content,
                        metadata=metadata,
                        score=score,
                        source="vector",
                    )
                )
            )
        return normalized
