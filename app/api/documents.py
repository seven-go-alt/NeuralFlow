from __future__ import annotations

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
from app.retrieval.chroma_store import ChromaDocumentStore
from worker import celery_app

router = APIRouter(prefix="/api/documents", tags=["documents"])


def _enqueue_ingestion(*, tenant_id: str, document_id: str) -> None:
    try:
        celery_app.send_task(
            "neuralflow.ingest_document",
            kwargs={"tenant_id": tenant_id, "document_id": document_id},
        )
    except (OperationalError, ConnectionError, OSError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Document ingestion queue is unavailable",
        ) from exc


@router.post("/upload", response_model=DocumentUploadResponse)
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
    _enqueue_ingestion(tenant_id=tenant_id, document_id=record.document_id)
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
    _enqueue_ingestion(tenant_id=tenant_id, document_id=document_id)
    return {"ok": True, "document_id": document_id, "status": "queued"}
