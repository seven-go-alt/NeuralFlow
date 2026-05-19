from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from kombu.exceptions import OperationalError
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.documents.enums import DocumentStatus
from app.documents.repository import DocumentRepository
from app.documents.schemas import (
    DocumentChunkRead,
    DocumentChunksResponse,
    DocumentListResponse,
    DocumentRead,
    DocumentUploadResponse,
)
from app.documents.service import DocumentService, DocumentValidationError
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.chroma_store import ChromaDocumentStore
from worker import celery_app

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


async def _enqueue_or_run_sync(*, tenant_id: str, document_id: str) -> None:
    """Try Celery first; fall back to synchronous processing for local dev."""
    try:
        celery_app.send_task(
            "neuralflow.ingest_document",
            kwargs={"tenant_id": tenant_id, "document_id": document_id},
        )
        logger.info("dispatched ingestion task for doc=%s via Celery", document_id)
        return
    except (OperationalError, ConnectionError, OSError):
        logger.info("Celery broker unavailable — processing doc=%s synchronously", document_id)

    pipeline = IngestionPipeline()
    try:
        await pipeline.run(tenant_id=tenant_id, document_id=document_id)
    except Exception as exc:
        logger.error("sync ingestion failed for doc=%s: %s", document_id, exc)
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {exc}") from exc


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    summary="Upload a document",
    description="Upload a document file (PDF, DOCX, MD, TXT) for ingestion. The file is saved, parsed, chunked, embedded, and indexed in ChromaDB. Optionally triggers multimodal extraction when enabled.",
)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    db: Session = Depends(get_db),
):
    tenant_id = getattr(request.state, "tenant_id", "public")
    owner_user_id = getattr(getattr(request.state, "tenant", None), "subject", "anonymous")
    service = DocumentService(db)
    try:
        record = await service.upload_document(
            tenant_id=tenant_id, owner_user_id=owner_user_id, upload=file, title=title
        )
    except DocumentValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await _enqueue_or_run_sync(tenant_id=tenant_id, document_id=record.document_id)
    refreshed = DocumentRepository(db).get_document(
        tenant_id=tenant_id, document_id=record.document_id
    )
    if refreshed is None:
        raise HTTPException(status_code=500, detail="Uploaded document missing after ingestion")
    return DocumentUploadResponse(
        document_id=refreshed.document_id,
        filename=refreshed.original_filename,
        status=DocumentStatus(refreshed.status),
        tenant_id=refreshed.tenant_id,
        owner_user_id=refreshed.owner_user_id,
        created_at=refreshed.created_at,
    )


@router.get("", response_model=DocumentListResponse)
def list_documents(
    request: Request,
    page: int = 1,
    page_size: int = 20,
    status: str | None = None,
    file_type: str | None = None,
    keyword: str | None = None,
    db: Session = Depends(get_db),
):
    tenant_id = getattr(request.state, "tenant_id", "public")
    repo = DocumentRepository(db)
    items, total = repo.list_documents(
        tenant_id=tenant_id,
        page=page,
        page_size=page_size,
        status=status,
        file_type=file_type,
        keyword=keyword,
    )
    return DocumentListResponse(
        items=[DocumentRead.model_validate(item) for item in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{document_id}", response_model=DocumentRead)
def get_document(document_id: str, request: Request, db: Session = Depends(get_db)):
    tenant_id = getattr(request.state, "tenant_id", "public")
    repo = DocumentRepository(db)
    record = repo.get_document(tenant_id=tenant_id, document_id=document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentRead.model_validate(record)


@router.get("/{document_id}/chunks", response_model=DocumentChunksResponse)
def list_document_chunks(document_id: str, request: Request, db: Session = Depends(get_db)):
    tenant_id = getattr(request.state, "tenant_id", "public")
    repo = DocumentRepository(db)
    items, total = repo.list_chunks(tenant_id=tenant_id, document_id=document_id)
    return DocumentChunksResponse(
        items=[DocumentChunkRead.model_validate(item) for item in items],
        total=total,
    )


@router.delete("/{document_id}")
def delete_document(document_id: str, request: Request, db: Session = Depends(get_db)):
    tenant_id = getattr(request.state, "tenant_id", "public")
    repo = DocumentRepository(db)
    ok = repo.soft_delete(tenant_id=tenant_id, document_id=document_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Document not found")
    ChromaDocumentStore(allow_in_memory=True).delete_document(
        tenant_id=tenant_id, document_id=document_id
    )
    return {"ok": True, "document_id": document_id}


@router.post("/{document_id}/reindex")
async def reindex_document(document_id: str, request: Request, db: Session = Depends(get_db)):
    tenant_id = getattr(request.state, "tenant_id", "public")
    repo = DocumentRepository(db)
    record = repo.get_document(tenant_id=tenant_id, document_id=document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")
    await _enqueue_or_run_sync(tenant_id=tenant_id, document_id=document_id)
    return {"ok": True, "document_id": document_id, "status": "queued"}
