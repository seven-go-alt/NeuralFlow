from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class EvalRunORM(Base):
    __tablename__ = "eval_runs"
    __table_args__ = (Index("ix_eval_runs_created", "created_at"),)

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(128), index=True)
    dataset_name: Mapped[str] = mapped_column(String(255))
    total_cases: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    celery_task_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    config_snapshot_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    token_usage_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    per_case_results_json: Mapped[dict] = mapped_column(JSON)
    started_at: Mapped[datetime] = mapped_column(DateTime)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
