from __future__ import annotations

from app.config import get_settings
from app.embeddings.base import EmbeddingProvider as BaseEmbeddingProvider
from app.embeddings.cache import EmbeddingCache
from app.embeddings.providers_openai import OpenAICompatibleEmbeddingProvider


class EmbeddingService:
    def __init__(
        self,
        provider: BaseEmbeddingProvider | None = None,
        cache: EmbeddingCache | None = None,
    ) -> None:
        if provider is None:
            settings = get_settings()
            if settings.embedding_provider == "local":
                from app.embeddings.providers_local import LocalSentenceTransformerProvider

                provider = LocalSentenceTransformerProvider()
            else:
                provider = OpenAICompatibleEmbeddingProvider()
        self.provider = provider
        self.cache = cache or EmbeddingCache()

    async def embed_texts(self, texts: list[str], model: str) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []
        for index, text in enumerate(texts):
            cached = self.cache.get(model=model, text=text)
            if cached is not None:
                results[index] = cached
            else:
                missing_indices.append(index)
                missing_texts.append(text)
        if missing_texts:
            fresh = await self.provider.embed_texts(missing_texts, model=model)
            for index, vector in zip(missing_indices, fresh, strict=False):
                self.cache.set(model=model, text=texts[index], vector=vector)
                results[index] = vector
        return [item or [] for item in results]
