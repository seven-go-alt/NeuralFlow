from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from app.config import Settings

# ---------------------------------------------------------------------------
# Stub for sentence-transformers — no real model downloads in tests
# ---------------------------------------------------------------------------


class StubSentenceTransformer:
    """Records construction arguments; returns fake embeddings."""

    last_model_name: str | None = None
    construction_count: int = 0

    def __init__(self, model_name: str) -> None:
        StubSentenceTransformer.last_model_name = model_name
        StubSentenceTransformer.construction_count += 1

    def encode(
        self, texts: list[str], *, normalize_embeddings: bool = True
    ) -> list[list[float]]:
        return [[0.1] * 4 for _ in texts]


# ---------------------------------------------------------------------------
# Autouse fixtures  (order: inject fake module -> reset stub -> clear cache)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _inject_fake_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a fake sentence_transformers module so ``from sentence_transformers
    import SentenceTransformer`` succeeds without the real package installed."""
    fake_mod = types.ModuleType("sentence_transformers")
    fake_mod.SentenceTransformer = StubSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)


@pytest.fixture(autouse=True)
def _reset_stub_tracking() -> None:
    """Reset stub class-level state between tests."""
    StubSentenceTransformer.last_model_name = None
    StubSentenceTransformer.construction_count = 0


@pytest.fixture(autouse=True)
def _clear_model_cache() -> None:
    """Clear the class-level model cache between tests."""
    from app.embeddings.providers_local import LocalSentenceTransformerProvider

    LocalSentenceTransformerProvider._model_cache.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_service_selects_local_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EmbeddingService selects LocalSentenceTransformerProvider
    when embedding_provider = "local"."""
    monkeypatch.setattr(
        "app.embeddings.service.get_settings",
        lambda: Settings(embedding_provider="local"),
    )

    from app.embeddings.providers_local import LocalSentenceTransformerProvider
    from app.embeddings.service import EmbeddingService

    service = EmbeddingService()
    assert isinstance(service.provider, LocalSentenceTransformerProvider)


@pytest.mark.asyncio
async def test_local_provider_encodes_with_configured_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local provider uses settings.embedding_model regardless of
    the ``model`` argument passed to embed_texts."""
    monkeypatch.setattr(
        "app.embeddings.providers_local.get_settings",
        lambda: Settings(embedding_model="all-MiniLM-L6-v2"),
    )

    from app.embeddings.providers_local import LocalSentenceTransformerProvider

    provider = LocalSentenceTransformerProvider()

    # --- First call: should construct SentenceTransformer ---
    result = await provider.embed_texts(
        ["hello", "world"], model="text-embedding-3-small"
    )

    assert StubSentenceTransformer.last_model_name == "all-MiniLM-L6-v2"
    assert StubSentenceTransformer.construction_count == 1
    assert isinstance(result, list)
    assert isinstance(result[0], list)
    assert isinstance(result[0][0], float)

    # --- Second call: should hit _model_cache ---
    result2 = await provider.embed_texts(["foo"], model="some-other-model")
    assert StubSentenceTransformer.construction_count == 1  # cached
    assert len(result2) == 1
    assert len(result2[0]) == 4


@pytest.mark.asyncio
async def test_chroma_query_uses_configured_embedding_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ChromaDocumentStore.query passes settings.embedding_model
    to the embedding service."""
    monkeypatch.setattr(
        "app.retrieval.chroma_store.get_settings",
        lambda: Settings(embedding_model="all-MiniLM-L6-v2"),
    )

    from app.retrieval.chroma_store import ChromaDocumentStore

    # --- Stubs ---
    class ModelRecordingEmbeddingService:
        def __init__(self) -> None:
            self.model: str | None = None

        async def embed_texts(
            self, texts: list[str], model: str
        ) -> list[list[float]]:
            self.model = model
            return [[0.1, 0.2, 0.3]]

    class SpyCollection:
        def query(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "documents": [["test"]],
                "metadatas": [[{"document_id": "doc_1", "chunk_id": "chk_1"}]],
                "ids": [["chk_1"]],
                "distances": [[0.01]],
            }

    class SpyClient:
        def __init__(self) -> None:
            self.collection = SpyCollection()

        def get_or_create_collection(self, name: str) -> SpyCollection:
            return self.collection

    monkeypatch.setattr(
        "app.retrieval.chroma_store.get_vector_client",
        lambda **kw: SpyClient(),
    )

    embedding_service = ModelRecordingEmbeddingService()
    store = ChromaDocumentStore(embedding_service=embedding_service)
    await store.query("test query", top_k=3)

    assert embedding_service.model == "all-MiniLM-L6-v2"
