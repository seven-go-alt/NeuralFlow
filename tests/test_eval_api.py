from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base  # noqa: E402
from app.db.models.eval_run import EvalRunORM  # noqa: F401 — register model before init_db
from app.db.models.rag_trace import RAGTraceORM  # noqa: F401 — register model before init_db


@pytest.fixture(scope="module")
def client():
    from app.main import app

    # StaticPool ensures all threads share the same in-memory SQLite connection
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )
    Base.metadata.create_all(engine)
    test_session_factory = sessionmaker(bind=engine)
    session = test_session_factory()

    from app.db.session import get_db

    def _override():
        yield session

    app.dependency_overrides[get_db] = _override
    yield TestClient(app)
    app.dependency_overrides.clear()
    session.close()


def test_list_eval_runs_empty(client) -> None:
    resp = client.get("/api/v1/eval/runs")
    assert resp.status_code == 200
    data = resp.json()
    assert data["runs"] == []


def test_get_eval_run_not_found(client) -> None:
    resp = client.get("/api/v1/eval/runs/nonexistent")
    assert resp.status_code == 404


def test_list_traces_empty(client) -> None:
    resp = client.get("/api/v1/traces")
    assert resp.status_code == 200
    data = resp.json()
    assert data["traces"] == []


def test_get_trace_not_found(client) -> None:
    resp = client.get("/api/v1/traces/nonexistent")
    assert resp.status_code == 404
