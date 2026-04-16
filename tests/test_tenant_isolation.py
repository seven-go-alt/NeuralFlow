from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient
import pytest

from app.middleware.tenant_isolation import TenantIsolationMiddleware


@pytest.mark.asyncio
async def test_tenant_middleware_injects_public_context_when_header_missing() -> None:
    app = FastAPI()
    app.add_middleware(TenantIsolationMiddleware)

    @app.get("/tenant")
    async def tenant_endpoint(request: Request) -> JSONResponse:
        tenant = request.state.tenant
        return JSONResponse(tenant.model_dump(mode="json"))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/tenant")

    assert response.status_code == 200
    assert response.json() == {
        "tenant_id": "public",
        "scope": ["public"],
        "roles": ["reader"],
        "subject": "anonymous",
        "source": "default",
    }


@pytest.mark.asyncio
async def test_tenant_middleware_rejects_scope_overreach_with_403() -> None:
    app = FastAPI()
    app.add_middleware(TenantIsolationMiddleware)

    @app.get("/tenant")
    async def tenant_endpoint(request: Request) -> JSONResponse:
        return JSONResponse({"tenant_id": request.state.tenant.tenant_id})

    headers = {
        "X-Tenant-ID": "tenant-a",
        "X-Tenant-Scope": "tenant-b,tenant-c",
        "X-Tenant-Roles": "reader",
    }
    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        response = await client.get("/tenant", headers=headers)

    assert response.status_code == 403
    assert response.json() == {"detail": "Tenant scope violation"}
