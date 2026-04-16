from __future__ import annotations

import os
from time import perf_counter
from uuid import uuid4

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.utils.observability import Observability, configure_structured_logging, create_observability, set_log_context

logger = configure_structured_logging(
    logger_name="neuralflow.request",
    audit_log_path=os.getenv("NEURALFLOW_AUDIT_LOG_PATH", "/tmp/neuralflow_audit.log"),
)


class TelemetryMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, observability: Observability | None = None) -> None:
        super().__init__(app)
        self.observability = observability or create_observability()

    async def dispatch(self, request: Request, call_next):
        trace_id = request.headers.get("X-Request-ID") or str(uuid4())
        default_session_id = request.headers.get("X-Session-ID", "anonymous")
        endpoint = request.url.path
        request.state.trace_id = trace_id
        request.state.session_id = default_session_id
        set_log_context(session_id=default_session_id, trace_id=trace_id, intent="unknown")
        started_at = perf_counter()
        session_tracked = False
        if default_session_id != "anonymous":
            self.observability.active_sessions.inc()
            session_tracked = True
        logger.info("request started", extra={"session_id": default_session_id, "trace_id": trace_id, "intent": "unknown"})
        try:
            response = await call_next(request)
        except Exception:
            response = self._build_error_response(request, endpoint=endpoint, trace_id=trace_id, started_at=started_at)
        else:
            response = self._finalize_response(request, response, endpoint=endpoint, trace_id=trace_id, started_at=started_at)
        finally:
            if session_tracked:
                self.observability.active_sessions.dec()
        return response

    def _finalize_response(
        self,
        request: Request,
        response: Response,
        *,
        endpoint: str,
        trace_id: str,
        started_at: float,
    ) -> Response:
        session_id = getattr(request.state, "session_id", "anonymous")
        intent = getattr(request.state, "intent", "unknown")
        duration_ms = round((perf_counter() - started_at) * 1000, 3)
        duration_seconds = duration_ms / 1000
        set_log_context(session_id=session_id, trace_id=trace_id, intent=intent)
        self.observability.request_duration.labels(endpoint=endpoint, intent=intent).observe(duration_seconds)
        response.headers["X-Request-ID"] = trace_id
        logger.info(
            "request completed",
            extra={
                "session_id": session_id,
                "trace_id": trace_id,
                "intent": intent,
                "duration_ms": duration_ms,
                "status_code": response.status_code,
                "endpoint": endpoint,
            },
        )
        return response

    def _build_error_response(
        self,
        request: Request,
        *,
        endpoint: str,
        trace_id: str,
        started_at: float,
    ) -> JSONResponse:
        session_id = getattr(request.state, "session_id", "anonymous")
        intent = getattr(request.state, "intent", "unknown")
        duration_ms = round((perf_counter() - started_at) * 1000, 3)
        duration_seconds = duration_ms / 1000
        set_log_context(session_id=session_id, trace_id=trace_id, intent=intent)
        self.observability.request_duration.labels(endpoint=endpoint, intent=intent).observe(duration_seconds)
        self.observability.error_total.labels(endpoint=endpoint, intent=intent).inc()
        logger.exception(
            "request failed",
            extra={
                "session_id": session_id,
                "trace_id": trace_id,
                "intent": intent,
                "duration_ms": duration_ms,
                "endpoint": endpoint,
            },
        )
        response = JSONResponse(status_code=500, content={"detail": "Internal Server Error"})
        response.headers["X-Request-ID"] = trace_id
        return response
