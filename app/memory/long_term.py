from __future__ import annotations

from uuid import uuid4

from app.config import get_settings
from app.utils.vector_client import get_vector_client


class LongTermMemory:
    def __init__(self, client=None, collection_name: str | None = None) -> None:
        settings = get_settings()
        self.client = client or get_vector_client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name or settings.chroma_collection
        )

    def save_summary(self, summary: str, metadata: dict) -> str:
        item_id = str(uuid4())
        self.collection.add(
            documents=[summary],
            metadatas=[metadata],
            ids=[item_id],
        )
        return item_id

    async def search(self, query: str, top_k: int = 3) -> list[str]:
        results = self.collection.query(query_texts=[query], n_results=top_k)
        documents = results.get("documents", [[]])
        return documents[0] if documents else []
