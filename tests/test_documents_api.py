from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_documents_upload_list_detail_and_delete(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DOCUMENTS_STORAGE_DIR", str(tmp_path / "uploads"))

    class DummyTaskApp:
        @staticmethod
        def send_task(*args, **kwargs):
            return {"queued": True}

    monkeypatch.setattr("app.api.documents.celery_app", DummyTaskApp)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("handbook.txt", b"leave policy\nsubmit request first", "text/plain")}
        data = {"title": "Employee Handbook"}
        upload_response = await client.post("/api/documents/upload", files=files, data=data)
        assert upload_response.status_code == 200, upload_response.text
        uploaded = upload_response.json()
        assert uploaded["document_id"].startswith("doc_")
        assert uploaded["status"] in {"queued", "ready"}

        list_response = await client.get("/api/documents")
        assert list_response.status_code == 200
        listed = list_response.json()
        assert listed["total"] >= 1
        document_id = listed["items"][0]["document_id"]

        detail_response = await client.get(f"/api/documents/{document_id}")
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["title"] == "Employee Handbook"
        assert detail["original_filename"] == "handbook.txt"

        chunks_response = await client.get(f"/api/documents/{document_id}/chunks")
        assert chunks_response.status_code == 200
        chunks = chunks_response.json()
        assert "items" in chunks

        delete_response = await client.delete(f"/api/documents/{document_id}")
        assert delete_response.status_code == 200
        assert delete_response.json()["ok"] is True
