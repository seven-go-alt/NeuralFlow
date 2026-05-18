from __future__ import annotations

import pytest

from app.embeddings.cache import EmbeddingCache
from app.embeddings.providers_openai import OpenAICompatibleEmbeddingProvider
from app.embeddings.service import EmbeddingService


class FakeCache:
    def __init__(self) -> None:
        self._store: dict[str, list[float]] = {}
        self.get_calls: list[tuple[str, str]] = []
        self.set_calls: list[tuple[str, str, list[float]]] = []

    def get(self, model: str, text: str) -> list[float] | None:
        self.get_calls.append((model, text))
        return self._store.get(f"{model}:{text}")

    def set(self, model: str, text: str, vector: list[float]) -> None:
        self.set_calls.append((model, text, vector))
        self._store[f"{model}:{text}"] = vector


class FakeProvider:
    def __init__(self, vectors: list[list[float]] | None = None) -> None:
        self.vectors = vectors or [[0.1, 0.2], [0.3, 0.4]]
        self.calls: list[tuple[list[str], str]] = []

    async def embed_texts(self, texts: list[str], model: str) -> list[list[float]]:
        self.calls.append((texts, model))
        return self.vectors[: len(texts)]


@pytest.mark.asyncio
async def test_embed_texts_all_cached() -> None:
    cache = FakeCache()
    cache._store["test-model:hello"] = [1.0, 2.0]
    cache._store["test-model:world"] = [3.0, 4.0]
    provider = FakeProvider()

    service = EmbeddingService(provider=provider, cache=cache)
    result = await service.embed_texts(["hello", "world"], model="test-model")

    assert result == [[1.0, 2.0], [3.0, 4.0]]
    assert provider.calls == []  # No provider call needed


@pytest.mark.asyncio
async def test_embed_texts_partial_cache_miss() -> None:
    cache = FakeCache()
    cache._store["test-model:hello"] = [1.0, 2.0]  # Populate without tracking
    provider = FakeProvider(vectors=[[3.0, 4.0]])

    service = EmbeddingService(provider=provider, cache=cache)
    result = await service.embed_texts(["hello", "world"], model="test-model")

    assert result == [[1.0, 2.0], [3.0, 4.0]]
    assert provider.calls == [(["world"], "test-model")]
    assert cache.set_calls == [("test-model", "world", [3.0, 4.0])]


@pytest.mark.asyncio
async def test_embed_texts_no_cache() -> None:
    cache = FakeCache()
    provider = FakeProvider(vectors=[[1.0], [2.0], [3.0]])

    service = EmbeddingService(provider=provider, cache=cache)
    result = await service.embed_texts(["a", "b", "c"], model="m")

    assert result == [[1.0], [2.0], [3.0]]
    assert provider.calls == [(["a", "b", "c"], "m")]
    assert len(cache.set_calls) == 3


@pytest.mark.asyncio
async def test_embed_texts_empty_list() -> None:
    service = EmbeddingService(provider=FakeProvider(), cache=FakeCache())
    result = await service.embed_texts([], model="test")
    assert result == []


@pytest.mark.asyncio
async def test_embed_texts_provider_returns_fewer_vectors() -> None:
    cache = FakeCache()
    provider = FakeProvider(vectors=[[1.0]])

    service = EmbeddingService(provider=provider, cache=cache)
    result = await service.embed_texts(["a", "b"], model="test")

    assert result == [[1.0], []]  # Missing vector becomes []


def test_cache_build_key() -> None:
    c = EmbeddingCache()
    k1 = c.build_key("m1", "hello")
    k2 = c.build_key("m1", "hello")
    k3 = c.build_key("m2", "hello")
    assert k1 == k2
    assert k1 != k3
    assert len(k1) == 64  # sha256 hexdigest


def test_cache_get_set() -> None:
    c = EmbeddingCache()
    assert c.get("m", "t") is None
    c.set("m", "t", [0.5, 0.6])
    assert c.get("m", "t") == [0.5, 0.6]


def test_cache_isolation() -> None:
    c = EmbeddingCache()
    c.set("m1", "t", [1.0])
    c.set("m2", "t", [2.0])
    c.set("m1", "other", [3.0])
    assert c.get("m1", "t") == [1.0]
    assert c.get("m2", "t") == [2.0]
    assert c.get("m1", "other") == [3.0]


@pytest.mark.asyncio
async def test_provider_fake_embedding_deterministic() -> None:
    p = OpenAICompatibleEmbeddingProvider()
    emb1 = p._fake_embedding("hello")
    emb2 = p._fake_embedding("hello")
    assert emb1 == emb2
    assert all(0 <= v <= 1 for v in emb1)


@pytest.mark.asyncio
async def test_provider_fake_embedding_different_inputs_differ() -> None:
    p = OpenAICompatibleEmbeddingProvider()
    emb_a = p._fake_embedding("apple")
    emb_b = p._fake_embedding("banana")
    assert emb_a != emb_b


@pytest.mark.asyncio
async def test_provider_fake_embedding_dimensions() -> None:
    p = OpenAICompatibleEmbeddingProvider()
    emb = p._fake_embedding("test", dims=8)
    assert len(emb) == 8


@pytest.mark.asyncio
async def test_provider_embed_texts_no_api_key_falls_back(monkeypatch) -> None:
    p = OpenAICompatibleEmbeddingProvider()
    p.api_key = None
    p.offline_fallback_enabled = True

    result = await p.embed_texts(["hello", "world"], model="test")
    assert len(result) == 2
    assert all(isinstance(v, float) for v in result[0])


@pytest.mark.asyncio
async def test_provider_embed_texts_success(monkeypatch) -> None:
    import httpx

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {
                "data": [
                    {"embedding": [0.1, 0.2]},
                    {"embedding": [0.3, 0.4]},
                ]
            }

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

        async def post(self, *args: object, **kwargs: object) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: FakeClient())

    p = OpenAICompatibleEmbeddingProvider()
    p.api_base = "https://fake.api/v1"
    p.api_key = "sk-fake"

    result = await p.embed_texts(["hello", "world"], model="test-model")
    assert result == [[0.1, 0.2], [0.3, 0.4]]
