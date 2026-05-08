from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.documents.repository import DocumentRepository
from app.retrieval.schemas import RetrievalRequest, RetrievalResponse
from app.retrieval.service import RetrievalService

router = APIRouter(prefix="/api/retrieval", tags=["retrieval"])


@router.post("/search", response_model=RetrievalResponse)
async def search_documents(request: Request, payload: RetrievalRequest, db: Session = Depends(get_db)):
    tenant_id = getattr(request.state, "tenant_id", "public")
    service = RetrievalService(document_repo=DocumentRepository(db))
    return await service.search(tenant_id=tenant_id, request=payload)
