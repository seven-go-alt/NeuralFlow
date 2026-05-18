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
            "/api/v1/retrieval/search",
            json={"query": "请假制度", "top_k": 5, "score_threshold": 0.2, "filters": {}},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "请假制度"
    assert payload["results"][0]["document_id"] == "doc_1"
    assert payload["results"][0]["source"]["page_number"] == 2


@pytest.mark.asyncio
async def test_retrieval_service_build_where_tenant_only(monkeypatch) -> None:
    from app.retrieval.schemas import RetrievalFilters
    from app.retrieval.service import RetrievalService

    monkeypatch.setattr("app.retrieval.service.ChromaDocumentStore", lambda: None)
    service = RetrievalService(document_repo=None)
    where = service._build_where(tenant_id="public", filters=RetrievalFilters().model_dump())
    assert where == {"tenant_id": "public"}


@pytest.mark.asyncio
async def test_retrieval_service_build_where_with_document_id(monkeypatch) -> None:
    from app.retrieval.schemas import RetrievalFilters
    from app.retrieval.service import RetrievalService

    monkeypatch.setattr("app.retrieval.service.ChromaDocumentStore", lambda: None)
    service = RetrievalService(document_repo=None)
    where = service._build_where(
        tenant_id="public",
        filters=RetrievalFilters(document_ids=["doc_1"]).model_dump(),
    )
    assert where == {"$and": [{"tenant_id": "public"}, {"document_id": "doc_1"}]}


@pytest.mark.asyncio
async def test_retrieval_service_build_where_with_file_type(monkeypatch) -> None:
    from app.retrieval.schemas import RetrievalFilters
    from app.retrieval.service import RetrievalService

    monkeypatch.setattr("app.retrieval.service.ChromaDocumentStore", lambda: None)
    service = RetrievalService(document_repo=None)
    where = service._build_where(
        tenant_id="tenant-alpha",
        filters=RetrievalFilters(file_types=["pdf"]).model_dump(),
    )
    assert where == {"$and": [{"tenant_id": "tenant-alpha"}, {"file_type": "pdf"}]}


@pytest.mark.asyncio
async def test_retrieval_service_build_where_multiple_clauses(monkeypatch) -> None:
    from app.retrieval.schemas import RetrievalFilters
    from app.retrieval.service import RetrievalService

    monkeypatch.setattr("app.retrieval.service.ChromaDocumentStore", lambda: None)
    service = RetrievalService(document_repo=None)
    where = service._build_where(
        tenant_id="public",
        filters=RetrievalFilters(document_ids=["doc_1"], file_types=["pdf"]).model_dump(),
    )
    assert len(where["$and"]) == 3
    assert {"tenant_id": "public"} in where["$and"]
    assert {"document_id": "doc_1"} in where["$and"]
    assert {"file_type": "pdf"} in where["$and"]


def test_retrieval_service_dedupe(monkeypatch) -> None:
    from app.retrieval.schemas import RetrievalResult
    from app.retrieval.service import RetrievalService

    monkeypatch.setattr("app.retrieval.service.ChromaDocumentStore", lambda: None)
    service = RetrievalService(document_repo=None)
    results = [
        RetrievalResult(chunk_id="c1", document_id="d1", content="a", score=0.9),
        RetrievalResult(chunk_id="c1", document_id="d1", content="a", score=0.9),
        RetrievalResult(chunk_id="c2", document_id="d1", content="b", score=0.8),
    ]
    deduped = service._dedupe(results)
    assert len(deduped) == 2
    assert deduped[0].chunk_id == "c1"
    assert deduped[1].chunk_id == "c2"


@pytest.mark.asyncio
async def test_retrieval_service_score_threshold_filtering(monkeypatch) -> None:
    from app.retrieval.schemas import RetrievalFilters, RetrievalRequest
    from app.retrieval.service import RetrievalService

    class StubStore:
        async def query(self, query_text: str, top_k: int, where: dict | None = None) -> dict:
            return {
                "documents": [["low", "high"]],
                "metadatas": [[{"chunk_id": "c1"}, {"chunk_id": "c2"}]],
                "ids": [["c1", "c2"]],
                "distances": [[0.9, 0.01]],
            }

    monkeypatch.setattr("app.retrieval.service.ChromaDocumentStore", lambda: StubStore())

    service = RetrievalService(document_repo=None)
    request = RetrievalRequest(query="test", score_threshold=0.5, filters=RetrievalFilters())
    response = await service.search(tenant_id="public", request=request)

    # distance 0.9 → score ≈ 0.53 (passes threshold 0.5)
    # distance 0.01 → score ≈ 0.99 (passes threshold 0.5)
    assert len(response.results) == 2


@pytest.mark.asyncio
async def test_retrieval_service_search_parses_chroma_response(monkeypatch) -> None:
    from app.retrieval.schemas import RetrievalFilters, RetrievalRequest
    from app.retrieval.service import RetrievalService

    class StubStore:
        async def query(self, query_text: str, top_k: int, where: dict | None = None) -> dict:
            return {
                "documents": [["content1"]],
                "metadatas": [[{"chunk_id": "c1", "document_id": "d1", "title": "Doc1"}]],
                "ids": [["c1"]],
                "distances": [[0.1]],
            }

    monkeypatch.setattr("app.retrieval.service.ChromaDocumentStore", lambda: StubStore())

    service = RetrievalService(document_repo=None)
    request = RetrievalRequest(query="test", filters=RetrievalFilters())
    response = await service.search(tenant_id="public", request=request)

    assert len(response.results) == 1
    assert response.results[0].chunk_id == "c1"
    assert response.results[0].document_id == "d1"
    assert response.results[0].source["title"] == "Doc1"
    assert response.results[0].score == pytest.approx(0.909, rel=1e-3)
