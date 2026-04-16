from __future__ import annotations

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.streaming import StreamTaskRegistry
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
        return f"prompt::{intent}::{user_query}"


class StubRouter:
    async def detect(self, text: str) -> IntentDetectionResult:
        return IntentDetectionResult(
            intents=["general"],
            primary_intent="general",
            used_fallback=False,
            policies={"general": IntentPolicy(memory_strategy="working_only", skill_whitelist=[])},
        )


class StubStreamingLLM:
    async def generate(self, prompt: str) -> str:
        return "ok"

    async def stream_generate(self, prompt: str, include_thinking: bool = False):
        if include_thinking:
            yield {"event": "thinking", "data": "analysis"}
        yield {"event": "message", "data": "hello"}
        yield {"event": "message", "data": " world"}


@pytest.mark.asyncio
async def test_stream_task_registry_cancels_previous_task_for_same_session() -> None:
    registry = StreamTaskRegistry()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def first() -> None:
        started.set()
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    first_task = asyncio.create_task(first())
    await started.wait()
    registry.register("session-1", first_task)

    second_task = asyncio.create_task(asyncio.sleep(0))
    registry.register("session-1", second_task)
    await asyncio.sleep(0)

    assert first_task.cancelled()
    assert cancelled.is_set()
    registry.clear("session-1", second_task)


@pytest.mark.asyncio
async def test_chat_stream_endpoint_returns_sse_events(monkeypatch) -> None:
    llm = StubStreamingLLM()
    monkeypatch.setattr("app.main.intent_router", StubRouter())
    monkeypatch.setattr("app.main.llm_client", llm)
    monkeypatch.setattr("app.main.WorkingMemory", StubWorkingMemory)
    monkeypatch.setattr("app.main.ContextBuilder", StubContextBuilder)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream(
            "POST",
            "/chat/stream?include_thinking=true",
            json={"session_id": "s-stream", "message": "你好"},
        ) as response:
            assert response.status_code == 200
            body = [line async for line in response.aiter_lines() if line]

    assert "event: thinking" in body
    assert 'data: {"delta":"analysis"}' in body
    assert "event: message" in body
    assert 'data: {"delta":"hello"}' in body
    assert 'data: {"delta":" world"}' in body
    assert "event: done" in body
