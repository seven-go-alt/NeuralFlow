from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.documents.repository import DocumentRepository
from app.retrieval.schemas import RetrievalRequest, RetrievalResponse
from app.retrieval.service import RetrievalService

router = APIRouter(prefix="/api/v1/retrieval", tags=["retrieval"])


@router.post(
    "/search",
    response_model=RetrievalResponse,
    summary="Search documents",
    description="Vector search across ingested document chunks. Supports filtering by document ID, file type, and content type (text, image_description, table_content). Returns ranked results with similarity scores and source metadata.",
)
async def search_documents(
    request: Request, payload: RetrievalRequest, db: Session = Depends(get_db)
):
    tenant_id = getattr(request.state, "tenant_id", "public")
    service = RetrievalService(document_repo=DocumentRepository(db))
    return await service.search(tenant_id=tenant_id, request=payload)
