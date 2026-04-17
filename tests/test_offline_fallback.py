from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from app.core.intent_router import IntentDetectionResult, IntentPolicy
from app.main import app


class StubWorkingMemory:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.messages: list[tuple[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        self.messages.append((role, content))

    def get_messages(self):
        return [{"role": role, "content": content} for role, content in self.messages]


class StubContextBuilder:
    def __init__(self, session_id: str, working_mem=None, long_mem=None, token_budget_manager=None) -> None:
        self.session_id = session_id

    async def build_prompt(self, user_query: str, intent: str, **kwargs) -> str:
        return f"intent={intent}\nquery={user_query}"


class StubRouter:
    async def detect(self, text: str) -> IntentDetectionResult:
        return IntentDetectionResult(
            intents=["general"],
            primary_intent="general",
            used_fallback=False,
            policies={"general": IntentPolicy(memory_strategy="working_only", skill_whitelist=[])},
        )


class BrokenLLMClient:
    async def generate(self, prompt: str) -> str:
        raise RuntimeError("invalid openai key")

    async def stream_generate(self, prompt: str, include_thinking: bool = False):
        if False:
            yield {"event": "message", "data": "unused"}
        raise RuntimeError("invalid openai key")


def test_chat_endpoint_returns_non_500_rule_fallback_when_llm_fails(monkeypatch) -> None:
    monkeypatch.setattr("app.main.intent_router", StubRouter())
    monkeypatch.setattr("app.main.WorkingMemory", StubWorkingMemory)
    monkeypatch.setattr("app.main.ContextBuilder", StubContextBuilder)
    monkeypatch.setattr("app.main.llm_client", BrokenLLMClient())

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/chat", json={"session_id": "s-offline", "message": "帮我总结 Redis 和 Chroma 的隔离差异"})

    assert response.status_code != 500
    assert response.status_code == 200
    body = response.json()
    assert "离线兜底摘要" in body["reply"]
    assert "Redis" in body["reply"]
    assert "Chroma" in body["reply"]


@pytest.mark.asyncio
async def test_chat_stream_endpoint_returns_rule_fallback_event_when_llm_fails(monkeypatch) -> None:
    monkeypatch.setattr("app.main.intent_router", StubRouter())
    monkeypatch.setattr("app.main.WorkingMemory", StubWorkingMemory)
    monkeypatch.setattr("app.main.ContextBuilder", StubContextBuilder)
    monkeypatch.setattr("app.main.llm_client", BrokenLLMClient())

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream(
            "POST",
            "/chat/stream",
            json={"session_id": "s-offline-stream", "message": "帮我总结 Redis 和 Chroma 的隔离差异"},
        ) as response:
            assert response.status_code == 200
            body = [line async for line in response.aiter_lines() if line]

    assert "event: message" in body
    assert any("离线兜底摘要" in line for line in body)
    assert any("Redis" in line for line in body)
    assert any("Chroma" in line for line in body)
    assert "event: done" in body