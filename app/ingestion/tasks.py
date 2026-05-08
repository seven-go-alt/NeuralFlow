from __future__ import annotations

from celery import shared_task

from app.ingestion.pipeline import IngestionPipeline


@shared_task(name="neuralflow.ingest_document")
def ingest_document_task(tenant_id: str, document_id: str, embedding_model: str = "text-embedding-3-small") -> dict:
    import asyncio

    pipeline = IngestionPipeline()
    return asyncio.run(pipeline.run(tenant_id=tenant_id, document_id=document_id, embedding_model=embedding_model))
