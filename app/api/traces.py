from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.rag_trace import RAGTraceORM
from app.db.session import get_db
from app.utils.cache import ResponseCache

router = APIRouter(prefix="/api/v1/traces", tags=["traces"])

cache = ResponseCache()


@router.get(
    "",
    summary="List RAG traces",
    description="Return recent RAG pipeline execution traces, filterable by tenant and session. Each trace shows query, duration, and token count.",
)
async def list_traces(
    request: Request,
    tenant_id: str = Query(default="public"),
    limit: int = Query(default=50, le=200),
    db: Session = Depends(get_db),
):
    params = {"tenant_id": tenant_id, "limit": limit}
    cached = await cache.get(tenant_id, request.url.path, params)
    if cached:
        return cached
    records = (
        db.execute(
            select(RAGTraceORM)
            .where(RAGTraceORM.tenant_id == tenant_id)
            .order_by(RAGTraceORM.created_at.desc())
            .limit(limit)
        )
        .scalars()
        .all()
    )
    response_data = {
        "traces": [
            {
                "trace_id": r.trace_id,
                "session_id": r.session_id,
                "query": r.query,
                "total_duration_ms": r.total_duration_ms,
                "token_count": r.token_count,
                "created_at": r.created_at.isoformat(),
            }
            for r in records
        ]
    }
    await cache.set(tenant_id, request.url.path, response_data, params, ttl=30)
    return response_data


@router.get(
    "/{trace_id}",
    summary="Get trace details",
    description="Return a full RAG trace with the complete span tree showing timing for each pipeline stage.",
)
def get_trace(trace_id: str, db: Session = Depends(get_db)):
    record = db.execute(
        select(RAGTraceORM).where(RAGTraceORM.trace_id == trace_id)
    ).scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return {
        "trace_id": record.trace_id,
        "tenant_id": record.tenant_id,
        "session_id": record.session_id,
        "query": record.query,
        "answer": record.answer,
        "span_tree": record.span_tree_json,
        "total_duration_ms": record.total_duration_ms,
        "token_count": record.token_count,
        "created_at": record.created_at.isoformat(),
    }
