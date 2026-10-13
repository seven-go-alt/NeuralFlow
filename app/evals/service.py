from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models.eval_run import EvalRunORM
from app.evals.factories import (
    make_live_answer_eval_fn,
    make_live_answer_fn,
    make_live_retrieve_fn,
)
from app.evals.metrics import aggregate_metrics
from app.evals.runner import run_eval


def create_eval_run(
    db: Session,
    *,
    tenant_id: str,
    dataset_name: str,
    top_k: int,
) -> EvalRunORM:
    settings = get_settings()
    run = EvalRunORM(
        run_id=str(uuid4()),
        tenant_id=tenant_id,
        dataset_name=dataset_name,
        total_cases=0,
        status="queued",
        progress=0,
        metrics_json={},
        per_case_results_json={"results": []},
        config_snapshot_json={
            "embedding_model": settings.embedding_model,
            "embedding_provider": settings.embedding_provider,
            "litellm_model": settings.litellm_model,
            "top_k": top_k,
            "rag_advanced_enabled": settings.rag_advanced_enabled,
            "reranker_enabled": settings.cross_encoder_enabled,
            "chunking_strategy": settings.chunking_strategy,
        },
        token_usage_json=None,
        started_at=datetime.utcnow(),
        completed_at=None,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def mark_eval_failed(db: Session, run_id: str, error: Exception | str) -> None:
    run = db.get(EvalRunORM, run_id)
    if run is None:
        return
    run.status = "failed"
    run.error_message = str(error)[:2000]
    run.completed_at = datetime.utcnow()
    db.commit()


async def execute_eval_run(
    db: Session,
    *,
    run_id: str,
    tenant_id: str,
    dataset_path: str | Path,
    top_k: int,
) -> EvalRunORM:
    run = db.get(EvalRunORM, run_id)
    if run is None or run.tenant_id != tenant_id:
        raise ValueError("evaluation run not found")

    run.status = "running"
    run.progress = 0
    run.started_at = datetime.utcnow()
    db.commit()

    try:
        results = await run_eval(
            cases_path=dataset_path,
            retrieve_fn=make_live_retrieve_fn(tenant_id=tenant_id),
            answer_fn=make_live_answer_fn(),
            top_k=top_k,
            answer_eval_fn=make_live_answer_eval_fn(),
        )
        metrics = aggregate_metrics(results)
        total_usage: dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        }
        for result in results:
            if result.token_usage_json:
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    total_usage[key] += result.token_usage_json.get(key, 0)
                total_usage["cost_usd"] += result.token_usage_json.get("cost_usd", 0.0)

        run.total_cases = metrics.total_cases
        run.progress = 100
        run.status = "completed"
        run.metrics_json = {
            "retrieval_hit_rate": metrics.retrieval_hit_rate,
            "citation_accuracy": metrics.citation_accuracy,
            "keyword_coverage": metrics.keyword_coverage,
            "average_latency_ms": metrics.average_latency_ms,
            "answer_relevance": metrics.average_answer_relevance,
            "answer_faithfulness": metrics.average_answer_faithfulness,
            "answer_completeness": metrics.average_answer_completeness,
        }
        run.per_case_results_json = {
            "results": [
                {
                    **asdict(result),
                    "retrieved_doc_ids": list(result.retrieved_doc_ids),
                    "retrieved_contents": list(result.retrieved_contents),
                }
                for result in results
            ]
        }
        run.token_usage_json = total_usage
        run.error_message = None
        run.completed_at = datetime.utcnow()
        db.commit()
        db.refresh(run)
        return run
    except Exception as exc:
        run.status = "failed"
        run.error_message = str(exc)[:2000]
        run.completed_at = datetime.utcnow()
        db.commit()
        raise
