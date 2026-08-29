from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from kombu.exceptions import OperationalError
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.eval_run import EvalRunORM
from app.db.session import get_db
from app.evals.dataset_resolver import EvalDatasetError, resolve_eval_dataset
from app.evals.service import create_eval_run
from app.utils.cache import ResponseCache
from worker import celery_app

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/eval", tags=["eval"])
cache = ResponseCache()


class EvalRunRequest(BaseModel):
    dataset_id: str = Field(min_length=1, max_length=255)
    top_k: int = Field(default=5, ge=1, le=20)


class EvalRunSummary(BaseModel):
    run_id: str
    dataset_name: str
    total_cases: int
    status: str
    progress: int
    error_message: str | None = None
    retrieval_hit_rate: float
    citation_accuracy: float
    keyword_coverage: float
    average_latency_ms: float
    started_at: str
    completed_at: str | None
    answer_relevance: float | None = None
    answer_faithfulness: float | None = None
    answer_completeness: float | None = None


@router.post(
    "/run",
    summary="Queue an evaluation run",
    description="Queue a RAG evaluation on a controlled JSONL dataset. Poll the run detail endpoint for progress.",
)
async def trigger_eval_run(
    request: EvalRunRequest,
    http_request: Request,
    db: Session = Depends(get_db),
):
    tenant_id = getattr(http_request.state, "tenant_id", "public")
    try:
        resolve_eval_dataset(request.dataset_id)
    except EvalDatasetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    run = create_eval_run(
        db,
        tenant_id=tenant_id,
        dataset_name=request.dataset_id.removesuffix(".jsonl"),
        top_k=request.top_k,
    )
    try:
        task = celery_app.send_task(
            "neuralflow.run_eval",
            kwargs={
                "run_id": run.run_id,
                "tenant_id": tenant_id,
                "dataset_id": request.dataset_id,
                "top_k": request.top_k,
            },
        )
        run.celery_task_id = task.id
        db.commit()
    except (OperationalError, ConnectionError, OSError) as exc:
        run.status = "failed"
        run.error_message = "Evaluation worker is unavailable"
        db.commit()
        raise HTTPException(status_code=503, detail="Evaluation worker is unavailable") from exc

    await cache.invalidate(tenant_id, "/api/v1/eval/runs")
    return {
        "run_id": run.run_id,
        "status": run.status,
        "progress": run.progress,
        "total_cases": run.total_cases,
    }


@router.get(
    "/runs",
    summary="List evaluation runs",
    description="Return the 50 most recent evaluation runs for the current tenant.",
)
async def list_eval_runs(request: Request, db: Session = Depends(get_db)):
    tenant_id = getattr(request.state, "tenant_id", "public")
    cached = await cache.get(tenant_id, request.url.path)
    if cached:
        return cached
    records = (
        db.execute(
            select(EvalRunORM)
            .where(EvalRunORM.tenant_id == tenant_id)
            .order_by(EvalRunORM.created_at.desc())
            .limit(50)
        )
        .scalars()
        .all()
    )
    response_data = {"runs": [_serialize_summary(record) for record in records]}
    await cache.set(tenant_id, request.url.path, response_data, ttl=30)
    return response_data


@router.get(
    "/runs/{run_id}",
    summary="Get evaluation run details",
    description="Return full evaluation run details for the current tenant.",
)
def get_eval_run(run_id: str, request: Request, db: Session = Depends(get_db)):
    tenant_id = getattr(request.state, "tenant_id", "public")
    record = db.execute(
        select(EvalRunORM).where(
            EvalRunORM.run_id == run_id,
            EvalRunORM.tenant_id == tenant_id,
        )
    ).scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=404, detail="Eval run not found")
    return {
        "run_id": record.run_id,
        "dataset_name": record.dataset_name,
        "total_cases": record.total_cases,
        "status": record.status,
        "progress": record.progress,
        "error_message": record.error_message,
        "metrics": record.metrics_json,
        "per_case_results": record.per_case_results_json,
        "config_snapshot": record.config_snapshot_json,
        "token_usage": record.token_usage_json,
        "started_at": record.started_at.isoformat(),
        "completed_at": record.completed_at.isoformat() if record.completed_at else None,
    }


def _serialize_summary(record: EvalRunORM) -> dict:
    metrics = record.metrics_json or {}
    return EvalRunSummary(
        run_id=record.run_id,
        dataset_name=record.dataset_name,
        total_cases=record.total_cases,
        status=record.status,
        progress=record.progress,
        error_message=record.error_message,
        retrieval_hit_rate=float(metrics.get("retrieval_hit_rate", 0.0)),
        citation_accuracy=float(metrics.get("citation_accuracy", 0.0)),
        keyword_coverage=float(metrics.get("keyword_coverage", 0.0)),
        average_latency_ms=float(metrics.get("average_latency_ms", 0.0)),
        started_at=record.started_at.isoformat(),
        completed_at=record.completed_at.isoformat() if record.completed_at else None,
        answer_relevance=metrics.get("answer_relevance"),
        answer_faithfulness=metrics.get("answer_faithfulness"),
        answer_completeness=metrics.get("answer_completeness"),
    ).model_dump()
