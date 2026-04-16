from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from app.config import get_settings
from app.memory.vector_retriever import VectorRetriever
from app.utils.redis_client import get_redis_client
from app.utils.vector_client import get_vector_client


class LongTermMemory:
    def __init__(self, client=None, collection_name: str | None = None, retriever: VectorRetriever | None = None) -> None:
        settings = get_settings()
        self.client = client or get_vector_client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name or settings.chroma_collection
        )
        self.retriever = retriever or VectorRetriever(
            collection=self.collection,
            cache_client=get_redis_client(),
            cache_ttl_seconds=settings.vector_search_cache_ttl_seconds,
        )
        self.default_top_k = settings.vector_search_default_top_k

    def save_summary(self, summary: str, metadata: dict) -> str:
        item_id = str(uuid4())
        payload_metadata = {
            **metadata,
            "type": metadata.get("type", "summary"),
            "created_at": metadata.get("created_at")
            or datetime.now(UTC).isoformat(timespec="seconds"),
        }
        self.collection.add(
            documents=[summary],
            metadatas=[payload_metadata],
            ids=[item_id],
        )
        return item_id

    async def search(self, query: str, top_k: int | None = None, session_id: str | None = None) -> list[str]:
        results = await self.retriever.search(
            query=query,
            session_id=session_id,
            memory_type="summary",
            top_k=top_k or self.default_top_k,
        )
        return [str(item["content"]) for item in results]

    async def search_documents(
        self,
        query: str,
        top_k: int | None = None,
        session_id: str | None = None,
    ) -> list[dict]:
        return await self.retriever.search(
            query=query,
            session_id=session_id,
            memory_type="summary",
            top_k=top_k or self.default_top_k,
        )
