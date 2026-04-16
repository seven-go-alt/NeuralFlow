from __future__ import annotations

import os

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.models.tenant import TenantContext
from app.utils.observability import configure_structured_logging

logger = configure_structured_logging(
    logger_name="neuralflow.security",
    audit_log_path=os.getenv("NEURALFLOW_AUDIT_LOG_PATH", "/tmp/neuralflow_audit.log"),
)


class TenantIsolationMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, default_tenant_id: str = "public") -> None:
        super().__init__(app)
        self.default_tenant_id = default_tenant_id

    async def dispatch(self, request: Request, call_next):
        tenant = self._build_tenant_context(request)
        if not tenant.can_access(tenant.tenant_id):
            logger.warning(
                "tenant scope violation",
                extra={
                    "tenant_id": tenant.tenant_id,
                    "tenant_scope": tenant.scope,
                    "tenant_roles": tenant.roles,
                    "security_alert": True,
                    "path": request.url.path,
                },
            )
            return JSONResponse(status_code=403, content={"detail": "Tenant scope violation"})

        request.state.tenant = tenant
        request.state.tenant_id = tenant.tenant_id
        return await call_next(request)

    def _build_tenant_context(self, request: Request) -> TenantContext:
        tenant_id = (request.headers.get("X-Tenant-ID") or self.default_tenant_id).strip() or self.default_tenant_id
        scope = _parse_csv_header(request.headers.get("X-Tenant-Scope")) or [tenant_id]
        roles = _parse_csv_header(request.headers.get("X-Tenant-Roles")) or ["reader"]
        subject = (request.headers.get("X-Tenant-Subject") or "anonymous").strip() or "anonymous"
        source = "header" if request.headers.get("X-Tenant-ID") else "default"
        return TenantContext(
            tenant_id=tenant_id,
            scope=scope,
            roles=roles,
            subject=subject,
            source=source,
        )


def _parse_csv_header(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]
