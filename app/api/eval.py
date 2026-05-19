from __future__ import annotations

import logging
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.eval_run import EvalRunORM
from app.db.session import get_db
from app.evals.metrics import aggregate_metrics
from app.evals.runner import run_eval

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/eval", tags=["eval"])


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


@router.post("/run")
async def trigger_eval_run(
    request: EvalRunRequest,
    db: Session = Depends(get_db),
):
    run_id = str(uuid4())
    dataset_name = request.dataset_path.rstrip("/").split("/")[-1].replace(".jsonl", "")
    started_at = datetime.utcnow()

    results = await run_eval(
        cases_path=request.dataset_path,
        retrieve_fn=lambda q, k: [],
        answer_fn=lambda q, c: "eval answer stub",
        top_k=request.top_k,
    )
    metrics = aggregate_metrics(results)
    completed_at = datetime.utcnow()

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
        },
        per_case_results_json={"results": [r.__dict__ for r in results]},
        started_at=started_at,
        completed_at=completed_at,
    )
    db.add(record)
    db.commit()

    return {"run_id": run_id, "status": "completed", "total_cases": metrics.total_cases}


@router.get("/runs")
def list_eval_runs(db: Session = Depends(get_db)):
    records = (
        db.execute(select(EvalRunORM).order_by(EvalRunORM.created_at.desc()).limit(50))
        .scalars()
        .all()
    )
    return {
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


@router.get("/runs/{run_id}")
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
        "started_at": record.started_at.isoformat(),
        "completed_at": record.completed_at.isoformat() if record.completed_at else None,
    }
