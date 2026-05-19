from __future__ import annotations

from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session as SASession

from app.db.models.rag_trace import RAGTraceORM
from app.observability.trace_manager import TraceManager


class TracePersister:
    def __init__(self, db: SASession) -> None:
        self._db = db

    def persist(
        self,
        trace: TraceManager,
        tenant_id: str,
        session_id: str,
        query: str,
        answer: str | None,
    ) -> str:
        root = trace.close()
        trace_id = root.trace_id or str(uuid4())
        record = RAGTraceORM(
            trace_id=trace_id,
            tenant_id=tenant_id,
            session_id=session_id,
            query=query,
            answer=answer,
            span_tree_json=trace.to_dict(),
            total_duration_ms=root.duration_ms,
        )
        self._db.add(record)
        self._db.commit()
        return trace_id

    def get_trace(self, trace_id: str) -> dict | None:
        result = self._db.execute(select(RAGTraceORM).where(RAGTraceORM.trace_id == trace_id))
        record = result.scalar_one_or_none()
        if record is None:
            return None
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

    def list_traces(self, tenant_id: str, limit: int = 50) -> list[dict]:
        result = self._db.execute(
            select(RAGTraceORM)
            .where(RAGTraceORM.tenant_id == tenant_id)
            .order_by(RAGTraceORM.created_at.desc())
            .limit(limit)
        )
        records = result.scalars().all()
        return [
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
