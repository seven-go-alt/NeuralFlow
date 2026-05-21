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
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> None:
        self.collection = collection
        self.cache_client = cache_client
        self.cache_ttl_seconds = cache_ttl_seconds
        self.tenant_id = tenant_id or "public"
        self.last_cache_hit = False
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b

    RRF_K = 60  # Reciprocal Rank Fusion constant

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

        # Hybrid Search: run vector and BM25 in parallel, fuse with RRF
        candidate_k = top_k * 2
        vector_results, bm25_results = await asyncio.gather(
            self._vector_search(query=query, where=where, top_k=candidate_k),
            self._bm25_search(query=query, where=where, top_k=candidate_k),
        )

        if vector_results and bm25_results:
            results = self._rrf_fuse(vector_results, bm25_results, top_k=top_k)
        elif vector_results:
            results = vector_results[:top_k]
        elif bm25_results:
            results = bm25_results[:top_k]
        else:
            results = []

        await self._cache_set(cache_key, results)
        logger.info(
            "hybrid retrieval completed",
            extra={
                "cache_hit": False,
                "query": query,
                "vector_count": len(vector_results),
                "bm25_count": len(bm25_results),
                "result_count": len(results),
            },
        )
        return results

    def _build_where(self, session_id: str | None, memory_type: str) -> dict[str, Any]:
        clauses: list[dict[str, Any]] = [{"type": memory_type}, {"tenant_id": self.tenant_id}]
        if session_id:
            clauses.append({"session_id": session_id})
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def _build_cache_key(self, query: str, where: dict[str, Any], top_k: int) -> str:
        payload = json.dumps(
            {"query": query, "where": where, "top_k": top_k}, sort_keys=True, ensure_ascii=False
        )
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

    async def _vector_search(
        self, query: str, where: dict[str, Any], top_k: int
    ) -> list[dict[str, Any]]:
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

    async def _bm25_search(
        self, query: str, where: dict[str, Any], top_k: int
    ) -> list[dict[str, Any]]:
        """BM25-style retrieval using TF-IDF scoring."""
        try:
            response = await asyncio.to_thread(
                self.collection.get,
                where=where,
                include=["documents", "metadatas"],
            )
        except Exception:
            logger.warning("BM25 search failed", exc_info=True)
            return []

        documents = response.get("documents", []) or []
        metadatas = response.get("metadatas", []) or []
        if not documents:
            return []

        query_terms = [t for t in query.lower().split() if t]
        if not query_terms:
            return []

        # Compute document frequency for each query term
        doc_count = len(documents)
        df: dict[str, int] = {}
        for term in query_terms:
            df[term] = sum(1 for doc in documents if term in doc.lower())

        # BM25 scoring (default: k1=1.5, b=0.75, avgdl approximation)
        avg_dl = max(1, sum(len(d) for d in documents) / doc_count)
        k1, b = self.bm25_k1, self.bm25_b
        ranked: list[RetrievedDocument] = []
        for content, metadata in zip(documents, metadatas, strict=False):
            lowered = content.lower()
            doc_len = len(content)
            score = 0.0
            for term in query_terms:
                tf = lowered.count(term)
                if tf == 0:
                    continue
                idf = max(0.0, (doc_count - df[term] + 0.5) / (df[term] + 0.5))
                import math

                idf = math.log(1 + idf)
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_dl))
                score += idf * tf_norm
            if score > 0:
                ranked.append(
                    RetrievedDocument(
                        content=content,
                        metadata=metadata,
                        score=score,
                        source="bm25",
                    )
                )

        ranked.sort(key=lambda item: item.score, reverse=True)
        return [asdict(item) for item in ranked[:top_k]]

    def _rrf_fuse(
        self,
        vector_results: list[dict[str, Any]],
        bm25_results: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Reciprocal Rank Fusion: merge two ranked lists."""
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, dict[str, Any]] = {}

        for rank, item in enumerate(vector_results):
            key = item["content"][:100]  # Use content prefix as dedup key
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.RRF_K + rank + 1)
            doc_map[key] = item

        for rank, item in enumerate(bm25_results):
            key = item["content"][:100]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.RRF_K + rank + 1)
            if key not in doc_map:
                doc_map[key] = item

        sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
        results = []
        for key in sorted_keys[:top_k]:
            item = dict(doc_map[key])
            item["score"] = round(rrf_scores[key], 6)
            item["source"] = "hybrid"
            results.append(item)
        return results

    def _normalize_vector_results(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        normalized: list[dict[str, Any]] = []
        for index, content in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = distances[index] if index < len(distances) else 0.0
            score = 1.0 / (1.0 + max(0.0, float(distance)))
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
