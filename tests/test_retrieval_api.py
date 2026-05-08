from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.retrieval.schemas import RetrievalResponse, RetrievalResult


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
                    content="员工请假需要提前申请",
                    score=0.95,
                    metadata={"page_number": 2},
                    source={"title": "员工手册", "filename": "handbook.pdf", "page_number": 2},
                )
            ],
        )


@pytest.mark.asyncio
async def test_retrieval_search_api(monkeypatch) -> None:
    monkeypatch.setattr("app.api.retrieval.RetrievalService", StubRetrievalService)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/retrieval/search",
            json={"query": "请假制度", "top_k": 5, "score_threshold": 0.2, "filters": {}},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "请假制度"
    assert payload["results"][0]["document_id"] == "doc_1"
    assert payload["results"][0]["source"]["page_number"] == 2
