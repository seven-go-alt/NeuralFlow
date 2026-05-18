from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.utils.vector_client import VectorStoreUnavailableError


@pytest.mark.asyncio
async def test_document_upload_falls_back_to_sync_when_celery_down(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DOCUMENTS_STORAGE_DIR", str(tmp_path / "uploads"))

    class BrokenTaskApp:
        @staticmethod
        def send_task(*args, **kwargs):
            raise ConnectionError("broker down")

    monkeypatch.setattr("app.api.documents.celery_app", BrokenTaskApp)

    # Mock the sync pipeline to avoid ChromaDB dependency
    class FakePipeline:
        async def run(self, **kwargs):
            return {"status": "ok"}

    monkeypatch.setattr("app.api.documents.IngestionPipeline", lambda: FakePipeline())

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("handbook.txt", b"leave policy\nsubmit request first", "text/plain")}
        data = {"title": "Employee Handbook"}
        response = await client.post("/api/v1/documents/upload", files=files, data=data)

    # Sync fallback processes the document, so we get 200 instead of 503
    assert response.status_code == 200
    data = response.json()
    assert data["document_id"]


@pytest.mark.asyncio
async def test_reindex_falls_back_to_sync_when_celery_down(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DOCUMENTS_STORAGE_DIR", str(tmp_path / "uploads"))

    class CreateOnlyTaskApp:
        @staticmethod
        def send_task(*args, **kwargs):
            return {"queued": True}

    class FailingTaskApp:
        @staticmethod
        def send_task(*args, **kwargs):
            raise ConnectionError("broker down")

    class FakePipeline:
        async def run(self, **kwargs):
            return {"status": "ok"}

    monkeypatch.setattr("app.api.documents.celery_app", CreateOnlyTaskApp)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("handbook.txt", b"leave policy\nsubmit request first", "text/plain")}
        data = {"title": "Employee Handbook"}
        upload_response = await client.post("/api/v1/documents/upload", files=files, data=data)
        document_id = upload_response.json()["document_id"]

    monkeypatch.setattr("app.api.documents.celery_app", FailingTaskApp)
    monkeypatch.setattr("app.api.documents.IngestionPipeline", lambda: FakePipeline())

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(f"/api/v1/documents/{document_id}/reindex")

    # Falls back to sync pipeline instead of returning 503
    assert response.status_code == 200
    assert response.json()["ok"] is True


@pytest.mark.asyncio
async def test_chat_reports_vector_store_unavailable_instead_of_silent_empty_retrieval(
    monkeypatch,
) -> None:
    class StubRouter:
        async def detect(self, text: str):
            from app.core.intent_router import IntentDetectionResult, IntentPolicy

            return IntentDetectionResult(
                intents=["general"],
                primary_intent="general",
                used_fallback=False,
                policies={
                    "general": IntentPolicy(memory_strategy="working_only", skill_whitelist=[])
                },
            )

    class StubWorkingMemory:
        def __init__(self, session_id: str, tenant_id: str | None = None) -> None:
            self.messages: list[tuple[str, str]] = []

        def add_message(self, role: str, content: str) -> None:
            self.messages.append((role, content))

        def get_messages(self):
            return [{"role": role, "content": content} for role, content in self.messages]

    class StubContextBuilder:
        def __init__(
            self, session_id: str, working_mem=None, tenant_id: str | None = None, **kwargs
        ) -> None:
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
            "/api/v1/chat",
            json={"session_id": "s-kb", "message": "请总结知识库文档", "use_retrieval": True},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "Knowledge base vector store is unavailable"
