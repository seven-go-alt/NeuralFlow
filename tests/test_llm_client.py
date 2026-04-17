from __future__ import annotations

import pytest

from app.core.llm import LLMClient


class FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = type("Message", (), {"content": content})()


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [FakeChoice(content)]


@pytest.mark.asyncio
async def test_llm_client_falls_back_to_ollama_when_primary_provider_fails(monkeypatch) -> None:
    calls: list[str] = []

    async def fake_acompletion(*, model: str, messages: list[dict], stream: bool = False):
        calls.append(model)
        if model == "primary-model":
            raise RuntimeError("invalid openai key")
        return FakeResponse("ollama reply")

    monkeypatch.setattr("app.core.llm.acompletion", fake_acompletion)

    client = LLMClient(model="primary-model")
    client.fallback_model = "ollama/qwen2.5:7b"
    client.offline_fallback_enabled = True

    reply = await client.generate("你好")

    assert reply == "ollama reply"
    assert calls == ["primary-model", "ollama/qwen2.5:7b"]


@pytest.mark.asyncio
async def test_llm_client_returns_rule_based_summary_when_all_models_fail(monkeypatch) -> None:
    async def fake_acompletion(*, model: str, messages: list[dict], stream: bool = False):
        raise RuntimeError(f"{model} unavailable")

    monkeypatch.setattr("app.core.llm.acompletion", fake_acompletion)

    client = LLMClient(model="primary-model")
    client.fallback_model = "ollama/qwen2.5:7b"
    client.offline_fallback_enabled = True

    reply = await client.generate("用户问：帮我总结 Redis 和 Chroma 的隔离差异")

    assert "离线兜底摘要" in reply
    assert "Redis" in reply
    assert "Chroma" in reply