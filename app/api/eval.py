from __future__ import annotations

import logging
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models.eval_run import EvalRunORM
from app.db.session import get_db
from app.evals.factories import (
    make_live_answer_eval_fn,
    make_live_answer_fn,
    make_live_retrieve_fn,
)
from app.evals.metrics import aggregate_metrics
from app.evals.runner import run_eval
from app.utils.cache import ResponseCache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/eval", tags=["eval"])

cache = ResponseCache()


class EvalRunRequest(BaseModel):
    dataset_path: str
    top_k: int = 5


class EvalRunSummary(BaseModel):
    run_id: str
    dataset_name: str
    total_cases: int
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
    summary="Trigger an evaluation run",
    description="Run RAG evaluation on a JSONL dataset. Runs retrieval and answer generation for each case, computes metrics (hit rate, citation accuracy, keyword coverage), and stores results in the database.",
)
async def trigger_eval_run(
    request: EvalRunRequest,
    db: Session = Depends(get_db),
):
    run_id = str(uuid4())
    dataset_name = request.dataset_path.rstrip("/").split("/")[-1].replace(".jsonl", "")
    started_at = datetime.utcnow()

    settings = get_settings()
    config_snapshot = {
        "embedding_model": settings.embedding_model,
        "embedding_provider": settings.embedding_provider,
        "litellm_model": settings.litellm_model,
        "top_k": request.top_k,
        "rag_advanced_enabled": settings.rag_advanced_enabled,
        "reranker_enabled": settings.cross_encoder_enabled,
        "chunking_strategy": settings.chunking_strategy,
    }

    retrieve_fn = make_live_retrieve_fn()
    answer_fn = make_live_answer_fn()
    answer_eval_fn = make_live_answer_eval_fn()

    results = await run_eval(
        cases_path=request.dataset_path,
        retrieve_fn=retrieve_fn,
        answer_fn=answer_fn,
        top_k=request.top_k,
        answer_eval_fn=answer_eval_fn,
    )
    metrics = aggregate_metrics(results)
    completed_at = datetime.utcnow()

    total_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}
    for r in results:
        if r.token_usage_json:
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                total_usage[k] = total_usage.get(k, 0) + r.token_usage_json.get(k, 0)
            total_usage["cost_usd"] = total_usage.get("cost_usd", 0.0) + r.token_usage_json.get("cost_usd", 0.0)

    record = EvalRunORM(
        run_id=run_id,
        tenant_id="public",
        dataset_name=dataset_name,
        total_cases=metrics.total_cases,
        metrics_json={
            "retrieval_hit_rate": metrics.retrieval_hit_rate,
            "citation_accuracy": metrics.citation_accuracy,
            "keyword_coverage": metrics.keyword_coverage,
            "average_latency_ms": metrics.average_latency_ms,
            "answer_relevance": metrics.average_answer_relevance,
            "answer_faithfulness": metrics.average_answer_faithfulness,
            "answer_completeness": metrics.average_answer_completeness,
        },
        per_case_results_json={"results": [r.__dict__ for r in results]},
        config_snapshot_json=config_snapshot,
        token_usage_json=total_usage,
        started_at=started_at,
        completed_at=completed_at,
    )
    db.add(record)
    db.commit()

    return {
        "run_id": run_id,
        "status": "completed",
        "total_cases": metrics.total_cases,
        "token_usage": total_usage,
    }


@router.get(
    "/runs",
    summary="List evaluation runs",
    description="Return the 50 most recent evaluation runs with summary metrics, ordered by creation time descending.",
)
async def list_eval_runs(request: Request, db: Session = Depends(get_db)):
    tenant_id = getattr(request.state, "tenant_id", "public")
    cached = await cache.get(tenant_id, request.url.path)
    if cached:
        return cached
    records = (
        db.execute(select(EvalRunORM).order_by(EvalRunORM.created_at.desc()).limit(50))
        .scalars()
        .all()
    )
    response_data = {
        "runs": [
            EvalRunSummary(
                run_id=r.run_id,
                dataset_name=r.dataset_name,
                total_cases=r.total_cases,
                retrieval_hit_rate=float(r.metrics_json.get("retrieval_hit_rate", 0.0)),
                citation_accuracy=float(r.metrics_json.get("citation_accuracy", 0.0)),
                keyword_coverage=float(r.metrics_json.get("keyword_coverage", 0.0)),
                average_latency_ms=float(r.metrics_json.get("average_latency_ms", 0.0)),
                started_at=r.started_at.isoformat(),
                completed_at=r.completed_at.isoformat() if r.completed_at else None,
                answer_relevance=r.metrics_json.get("answer_relevance"),
                answer_faithfulness=r.metrics_json.get("answer_faithfulness"),
                answer_completeness=r.metrics_json.get("answer_completeness"),
            )
            for r in records
        ]
    }
    await cache.set(tenant_id, request.url.path, response_data, ttl=30)
    return response_data


@router.get(
    "/runs/{run_id}",
    summary="Get evaluation run details",
    description="Return full evaluation run details including per-case results and aggregated metrics.",
)
def get_eval_run(run_id: str, db: Session = Depends(get_db)):
    record = db.execute(select(EvalRunORM).where(EvalRunORM.run_id == run_id)).scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=404, detail="Eval run not found")
    return {
        "run_id": record.run_id,
        "dataset_name": record.dataset_name,
        "total_cases": record.total_cases,
        "metrics": record.metrics_json,
        "per_case_results": record.per_case_results_json,
        "config_snapshot": record.config_snapshot_json,
        "token_usage": record.token_usage_json,
        "started_at": record.started_at.isoformat(),
        "completed_at": record.completed_at.isoformat() if record.completed_at else None,
    }
