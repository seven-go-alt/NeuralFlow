from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from app.config import get_settings
from app.embeddings.base import EmbeddingProvider


class LocalSentenceTransformerProvider(EmbeddingProvider):
    """Embeds with a local sentence-transformers model.

    Always encodes with ``settings.embedding_model`` regardless of the
    ``model`` argument, so callers that hard-code OpenAI model names
    (e.g. intent router) still work against the single local model.
    """

    _model_cache: ClassVar[dict[str, Any]] = {}

    def __init__(self) -> None:
        self.model_name = get_settings().embedding_model

    async def embed_texts(self, texts: list[str], model: str) -> list[list[float]]:
        return await asyncio.to_thread(self._encode, texts)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - ml extra not installed
            raise RuntimeError(
                "sentence-transformers is required for EMBEDDING_PROVIDER=local "
                "(install the 'ml' extra)"
            ) from exc
        st = self._model_cache.get(self.model_name)
        if st is None:
            st = SentenceTransformer(self.model_name)
            self._model_cache[self.model_name] = st
        vectors = st.encode(list(texts), normalize_embeddings=True)
        return [list(map(float, vector)) for vector in vectors]
