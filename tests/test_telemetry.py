from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient
from prometheus_client import CollectorRegistry
import pytest

from app.middleware.telemetry import TelemetryMiddleware
from app.utils.observability import create_observability


@pytest.mark.asyncio
async def test_telemetry_middleware_injects_request_id_and_records_metrics() -> None:
    app = FastAPI()
    observability = create_observability(registry=CollectorRegistry())
    app.add_middleware(TelemetryMiddleware, observability=observability)

    @app.get("/ok")
    async def ok(request: Request) -> JSONResponse:
        request.state.intent = "general"
        request.state.session_id = "session-1"
        return JSONResponse({"status": "ok"})

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/ok")

    assert response.status_code == 200
    assert response.headers["x-request-id"]

    metrics = observability.render_metrics().decode("utf-8")
    assert 'endpoint="/ok"' in metrics
    assert 'intent="general"' in metrics
    assert "neuralflow_request_duration_seconds" in metrics


@pytest.mark.asyncio
async def test_telemetry_middleware_counts_unhandled_errors() -> None:
    app = FastAPI()
    observability = create_observability(registry=CollectorRegistry())
    app.add_middleware(TelemetryMiddleware, observability=observability)

    @app.get("/boom")
    async def boom() -> JSONResponse:
        raise RuntimeError("boom")

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/boom")

    assert response.status_code == 500
    assert response.headers["x-request-id"]

    metrics = observability.render_metrics().decode("utf-8")
    assert "neuralflow_errors_total" in metrics
    assert 'endpoint="/boom"' in metrics
