from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.utils.vector_client import VectorStoreUnavailableError


@pytest.mark.asyncio
async def test_document_upload_returns_503_when_ingestion_queue_unavailable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DOCUMENTS_STORAGE_DIR", str(tmp_path / "uploads"))

    class BrokenTaskApp:
        @staticmethod
        def send_task(*args, **kwargs):
            raise ConnectionError("broker down")

    monkeypatch.setattr("app.api.documents.celery_app", BrokenTaskApp)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("handbook.txt", b"leave policy\nsubmit request first", "text/plain")}
        data = {"title": "Employee Handbook"}
        response = await client.post("/api/documents/upload", files=files, data=data)

    assert response.status_code == 503
    assert response.json()["detail"] == "Document ingestion queue is unavailable"


@pytest.mark.asyncio
async def test_reindex_returns_503_when_ingestion_queue_unavailable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DOCUMENTS_STORAGE_DIR", str(tmp_path / "uploads"))

    class FailingTaskApp:
        @staticmethod
        def send_task(*args, **kwargs):
            raise ConnectionError("broker down")

    class CreateOnlyTaskApp:
        @staticmethod
        def send_task(*args, **kwargs):
            return {"queued": True}

    monkeypatch.setattr("app.api.documents.celery_app", CreateOnlyTaskApp)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("handbook.txt", b"leave policy\nsubmit request first", "text/plain")}
        data = {"title": "Employee Handbook"}
        upload_response = await client.post("/api/documents/upload", files=files, data=data)
        document_id = upload_response.json()["document_id"]

    monkeypatch.setattr("app.api.documents.celery_app", FailingTaskApp)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(f"/api/documents/{document_id}/reindex")

    assert response.status_code == 503
    assert response.json()["detail"] == "Document ingestion queue is unavailable"


@pytest.mark.asyncio
async def test_chat_reports_vector_store_unavailable_instead_of_silent_empty_retrieval(monkeypatch) -> None:
    class StubRouter:
        async def detect(self, text: str):
            from app.core.intent_router import IntentDetectionResult, IntentPolicy

            return IntentDetectionResult(
                intents=["general"],
                primary_intent="general",
                used_fallback=False,
                policies={"general": IntentPolicy(memory_strategy="working_only", skill_whitelist=[])},
            )

    class StubWorkingMemory:
        def __init__(self, session_id: str, tenant_id: str | None = None) -> None:
            self.messages: list[tuple[str, str]] = []

        def add_message(self, role: str, content: str) -> None:
            self.messages.append((role, content))

        def get_messages(self):
            return [{"role": role, "content": content} for role, content in self.messages]

    class StubContextBuilder:
        def __init__(self, session_id: str, working_mem=None, tenant_id: str | None = None, **kwargs) -> None:
            self.session_id = session_id

        async def build_prompt(self, user_query: str, intent: str, **kwargs) -> str:
            return f"prompt::{intent}::{user_query}"

    class StubLLM:
        async def generate(self, prompt: str) -> str:
            return "answer without kb"

    class FailingRetrievalService:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def search(self, tenant_id: str, request):
            raise VectorStoreUnavailableError("ChromaDB unavailable at localhost:8001")

    monkeypatch.setattr("app.main.intent_router", StubRouter())
    monkeypatch.setattr("app.main.WorkingMemory", StubWorkingMemory)
    monkeypatch.setattr("app.main.ContextBuilder", StubContextBuilder)
    monkeypatch.setattr("app.main.llm_client", StubLLM())
    monkeypatch.setattr("app.main.RetrievalService", FailingRetrievalService)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat",
            json={"session_id": "s-kb", "message": "请总结知识库文档", "use_retrieval": True},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "Knowledge base vector store is unavailable"
