from __future__ import annotations

from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.retrieval.schemas import RetrievalResponse, RetrievalResult


class StubWorkingMemory:
    def __init__(self, session_id: str, tenant_id: str | None = None) -> None:
        self.session_id = session_id
        self.tenant_id = tenant_id
        self.messages: list[tuple[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        self.messages.append((role, content))

    def get_messages(self):
        return [{"role": role, "content": content} for role, content in self.messages]


class StubContextBuilder:
    def __init__(
        self,
        session_id: str,
        working_mem=None,
        long_mem=None,
        token_budget_manager=None,
        tenant_id: str | None = None,
    ) -> None:
        self.session_id = session_id
        self.tenant_id = tenant_id

    async def build_prompt(self, user_query: str, intent: str, **kwargs) -> str:
        return f"prompt::{intent}::{user_query}"


class StubRouter:
    async def detect(self, text: str):
        from app.core.intent_router import IntentDetectionResult, IntentPolicy

        return IntentDetectionResult(
            intents=["general"],
            primary_intent="general",
            used_fallback=False,
            policies={"general": IntentPolicy(memory_strategy="working_only", skill_whitelist=[])},
        )


class StubLLM:
    async def generate(self, prompt: str) -> str:
        return "answer with citations"


class StubStreamingLLM:
    async def stream_generate(self, prompt: str, include_thinking: bool = False):
        yield {"event": "message", "data": "hello"}
        yield {"event": "message", "data": " world"}


class StubRetrievalService:
    def __init__(self, *args, **kwargs) -> None:
        pass

    async def search(self, tenant_id: str, request):
        return RetrievalResponse(
            query=request.query,
            results=[
                RetrievalResult(
                    chunk_id="chk_1",
                    document_id="doc_1",
                    content="employee handbook leave policy",
                    score=0.91,
                    metadata={"page_number": 3},
                    source={
                        "title": "Employee Handbook",
                        "filename": "handbook.pdf",
                        "page_number": 3,
                    },
                )
            ],
        )


class StubRAGContextBuilder:
    def build(self, query: str, results: list[RetrievalResult]):
        return SimpleNamespace(
            context="[1] Employee Handbook\nemployee handbook leave policy",
            citations=[
                {
                    "index": 1,
                    "label": "Employee Handbook",
                    "document_id": "doc_1",
                    "chunk_id": "chk_1",
                    "page_number": 3,
                }
            ],
        )


class DummyDB:
    def close(self):
        return None


@pytest.mark.asyncio
async def test_chat_returns_citations(monkeypatch) -> None:
    monkeypatch.setattr("app.main.intent_router", StubRouter())
    monkeypatch.setattr("app.main.llm_client", StubLLM())
    monkeypatch.setattr("app.main.WorkingMemory", StubWorkingMemory)
    monkeypatch.setattr("app.main.ContextBuilder", StubContextBuilder)
    monkeypatch.setattr("app.main.RetrievalService", StubRetrievalService)
    monkeypatch.setattr("app.db.session.SessionLocal", lambda: DummyDB())
    monkeypatch.setattr("app.rag.context_builder.RAGContextBuilder", StubRAGContextBuilder)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat", json={"session_id": "s1", "message": "请假制度是什么？", "use_retrieval": True}
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["reply"] == "answer with citations"
    assert payload["citations"][0]["document_id"] == "doc_1"
    assert payload["citations"][0]["page_number"] == 3


@pytest.mark.asyncio
async def test_chat_stream_emits_retrieval_and_chunk_events(monkeypatch) -> None:
    monkeypatch.setattr("app.main.intent_router", StubRouter())
    monkeypatch.setattr("app.main.llm_client", StubStreamingLLM())
    monkeypatch.setattr("app.main.WorkingMemory", StubWorkingMemory)
    monkeypatch.setattr("app.main.ContextBuilder", StubContextBuilder)
    monkeypatch.setattr("app.main.RetrievalService", StubRetrievalService)
    monkeypatch.setattr("app.db.session.SessionLocal", lambda: DummyDB())
    monkeypatch.setattr("app.rag.context_builder.RAGContextBuilder", StubRAGContextBuilder)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream(
            "POST",
            "/chat/stream",
            json={"session_id": "s2", "message": "请假制度是什么？", "use_retrieval": True},
        ) as response:
            assert response.status_code == 200
            body = [line async for line in response.aiter_lines() if line]

    assert "event: retrieval" in body
    assert any('"count":1' in line for line in body if line.startswith("data:"))
    assert "event: chunk" in body
    assert any(
        "employee handbook leave policy" in line for line in body if line.startswith("data:")
    )
