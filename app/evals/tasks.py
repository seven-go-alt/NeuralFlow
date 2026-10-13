from __future__ import annotations

from celery import current_app

from app.db.session import SessionLocal
from app.evals.dataset_resolver import resolve_eval_dataset
from app.evals.service import execute_eval_run


@current_app.task(name="neuralflow.run_eval")
def run_eval_task(run_id: str, tenant_id: str, dataset_id: str, top_k: int = 5) -> dict[str, str]:
    """Execute a persisted evaluation run in a Celery worker."""
    import asyncio

    db = SessionLocal()
    try:
        dataset_path = resolve_eval_dataset(dataset_id)
        run = asyncio.run(
            execute_eval_run(
                db,
                run_id=run_id,
                tenant_id=tenant_id,
                dataset_path=dataset_path,
                top_k=top_k,
            )
        )
        return {"run_id": run.run_id, "status": run.status}
    except Exception as exc:
        from app.evals.service import mark_eval_failed

        mark_eval_failed(db, run_id, exc)
        raise
    finally:
        db.close()
