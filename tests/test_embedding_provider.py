from __future__ import annotations

import httpx
import pytest

from app.embeddings.providers_openai import OpenAICompatibleEmbeddingProvider


class FakeResponse:
    def raise_for_status(self) -> None:
        raise httpx.HTTPStatusError(
            "404 Not Found",
            request=httpx.Request("POST", "https://example.test/v1/embeddings"),
            response=httpx.Response(
                404, request=httpx.Request("POST", "https://example.test/v1/embeddings")
            ),
        )


class FakeAsyncClient:
    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, *args, **kwargs):
        return FakeResponse()


@pytest.mark.asyncio
async def test_embedding_provider_falls_back_to_fake_vectors_when_remote_endpoint_fails(
    monkeypatch,
) -> None:
    monkeypatch.setattr("app.embeddings.providers_openai.httpx.AsyncClient", FakeAsyncClient)

    provider = OpenAICompatibleEmbeddingProvider()
    provider.api_base = "https://example.test/v1"
    provider.api_key = "test-key"
    provider.offline_fallback_enabled = True

    vectors = await provider.embed_texts(["hello world"], model="text-embedding-3-small")

    assert len(vectors) == 1
    assert len(vectors[0]) == 32
    assert vectors[0] == provider._fake_embedding("hello world")


@pytest.mark.asyncio
async def test_embedding_provider_raises_when_remote_endpoint_fails_and_fallback_disabled(
    monkeypatch,
) -> None:
    monkeypatch.setattr("app.embeddings.providers_openai.httpx.AsyncClient", FakeAsyncClient)

    provider = OpenAICompatibleEmbeddingProvider()
    provider.api_base = "https://example.test/v1"
    provider.api_key = "test-key"
    provider.offline_fallback_enabled = False

    with pytest.raises(httpx.HTTPStatusError):
        await provider.embed_texts(["hello world"], model="text-embedding-3-small")
