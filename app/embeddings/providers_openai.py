from __future__ import annotations

import logging
import math

import httpx

from app.config import get_settings
from app.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        settings = get_settings()
        self.api_base = (
            settings.embedding_api_base or settings.llm_api_base or "https://api.openai.com/v1"
        ).rstrip("/")
        self.api_key = settings.embedding_api_key or settings.llm_api_key or settings.openai_api_key
        self.offline_fallback_enabled = settings.offline_fallback_enabled

    async def embed_texts(self, texts: list[str], model: str) -> list[list[float]]:
        if self.api_key:
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.post(
                        f"{self.api_base}/embeddings",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={"model": model, "input": texts},
                    )
                    response.raise_for_status()
                    payload = response.json()
                    return [item["embedding"] for item in payload.get("data", [])]
            except Exception:
                if not self.offline_fallback_enabled:
                    raise
                logger.warning(
                    "embedding provider failed at %s/embeddings; using deterministic offline fallback",
                    self.api_base,
                    exc_info=True,
                )
        return [self._fake_embedding(text) for text in texts]

    def _fake_embedding(self, text: str, dims: int = 32) -> list[float]:
        base = sum(ord(ch) for ch in text) or 1
        return [round(math.sin(base * (i + 1)) * 0.5 + 0.5, 6) for i in range(dims)]
