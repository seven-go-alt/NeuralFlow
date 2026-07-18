from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.session import get_db
from app.main import app


def _clear_api_cache() -> None:
    """Clear Redis API cache keys to prevent stale responses between tests."""
    try:
        import redis as sync_redis

        r = sync_redis.Redis(decode_responses=True)
        cursor = 0
        while True:
            cursor, keys = cast(
                tuple[int, list[Any]], r.scan(cursor, match="api_cache:*", count=100)
            )
            if keys:
                r.delete(*keys)
            if cursor == 0:
                break
        r.close()
    except Exception:
        pass


@pytest.fixture
def test_db():
    """Create a temporary file-based SQLite database for testing.

    Uses a temp file instead of in-memory SQLite to avoid cross-thread
    issues with TestClient (which runs requests in a thread pool).
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    engine = create_engine(
        f"sqlite:///{db_path}", echo=False, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    test_session_factory = sessionmaker(bind=engine)
    session = test_session_factory()

    def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db
    # Invalidate Redis API cache to avoid stale cached responses between tests
    _clear_api_cache()
    yield session
    session.close()
    engine.dispose()
    Path(db_path).unlink(missing_ok=True)
    app.dependency_overrides.clear()


@pytest.fixture
def client(test_db):
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_healthz_returns_ok(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "degraded")
        assert "database" in data


class TestDocumentFlow:
    def test_list_documents_empty(self, client):
        resp = client.get("/api/v1/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["items"] == []

    def test_list_documents_with_data(self, client, test_db):
        from app.documents.repository import DocumentRepository
        from app.documents.schemas import DocumentCreate

        repo = DocumentRepository(test_db)
        payload = DocumentCreate(
            tenant_id="public",
            owner_user_id="test",
            title="test.pdf",
            filename="test.pdf",
            original_filename="test.pdf",
            file_type="pdf",
            mime_type="application/pdf",
            size_bytes=1024,
            storage_path="/tmp/test.pdf",
            checksum_sha256="abc123",
        )
        repo.create_document(payload=payload, document_id="test-1")

        resp = client.get("/api/v1/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        assert len(data["items"]) >= 1
        item = data["items"][0]
        assert item["document_id"] == "test-1"
        assert item["filename"] == "test.pdf"
