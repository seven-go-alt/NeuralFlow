from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class RAGTraceORM(Base):
    __tablename__ = "rag_traces"
    __table_args__ = (
        Index("ix_rag_traces_tenant_created", "tenant_id", "created_at"),
    )

    trace_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(128), index=True)
    session_id: Mapped[str] = mapped_column(String(128))
    query: Mapped[str] = mapped_column(Text)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    span_tree_json: Mapped[dict] = mapped_column(JSON)
    total_duration_ms: Mapped[float] = mapped_column(Float, default=0.0)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
