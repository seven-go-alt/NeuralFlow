from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestAPIEndpointsExist:
    """Verify all documented API endpoints return proper responses."""

    def test_documents_list_shape(self):
        resp = client.get("/api/v1/documents")
        assert resp.status_code in (200, 401)
        if resp.status_code == 200:
            data = resp.json()
            assert "items" in data
            assert "total" in data
            assert "page" in data
            assert "page_size" in data

    def test_eval_runs_shape(self):
        resp = client.get("/api/v1/eval/runs")
        assert resp.status_code in (200, 401)
        if resp.status_code == 200:
            data = resp.json()
            assert "runs" in data

    def test_traces_shape(self):
        resp = client.get("/api/v1/traces")
        assert resp.status_code in (200, 401)
        if resp.status_code == 200:
            data = resp.json()
            assert "traces" in data

    def test_skills_endpoint(self):
        resp = client.get("/api/v1/skills")
        assert resp.status_code in (200, 401)
        if resp.status_code == 200:
            data = resp.json()
            assert "skills" in data

    def test_models_endpoint(self):
        resp = client.get("/api/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert "current_model" in data
