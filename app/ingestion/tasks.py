from __future__ import annotations

from celery import current_app

from app.ingestion.pipeline import IngestionPipeline


@current_app.task(name="neuralflow.ingest_document")
def ingest_document_task(
    tenant_id: str, document_id: str, embedding_model: str | None = None
) -> dict:
    import asyncio

    pipeline = IngestionPipeline()
    return asyncio.run(
        pipeline.run(tenant_id=tenant_id, document_id=document_id, embedding_model=embedding_model)
    )
