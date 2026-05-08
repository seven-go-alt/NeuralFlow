from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import DocumentChunkORM, DocumentORM
from app.documents.enums import DocumentStatus
from app.documents.schemas import ChunkRecord, DocumentCreate


class DocumentRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create_document(self, payload: DocumentCreate, document_id: str) -> DocumentORM:
        record = DocumentORM(
            document_id=document_id,
            tenant_id=payload.tenant_id,
            owner_user_id=payload.owner_user_id,
            title=payload.title,
            filename=payload.filename,
            original_filename=payload.original_filename,
            file_type=payload.file_type,
            mime_type=payload.mime_type,
            size_bytes=payload.size_bytes,
            storage_path=payload.storage_path,
            checksum_sha256=payload.checksum_sha256,
            status=DocumentStatus.UPLOADED.value,
            metadata_json=payload.metadata_json,
            source_info_json=payload.source_info_json,
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def get_document(self, tenant_id: str, document_id: str) -> DocumentORM | None:
        stmt = select(DocumentORM).where(
            DocumentORM.tenant_id == tenant_id,
            DocumentORM.document_id == document_id,
            DocumentORM.deleted_at.is_(None),
        )
        return self.db.scalar(stmt)

    def list_documents(
        self,
        tenant_id: str,
        page: int,
        page_size: int,
        status: str | None = None,
        file_type: str | None = None,
        keyword: str | None = None,
    ) -> tuple[list[DocumentORM], int]:
        stmt = select(DocumentORM).where(DocumentORM.tenant_id == tenant_id, DocumentORM.deleted_at.is_(None))
        count_stmt = select(func.count()).select_from(DocumentORM).where(
            DocumentORM.tenant_id == tenant_id,
            DocumentORM.deleted_at.is_(None),
        )
        if status:
            stmt = stmt.where(DocumentORM.status == status)
            count_stmt = count_stmt.where(DocumentORM.status == status)
        if file_type:
            stmt = stmt.where(DocumentORM.file_type == file_type)
            count_stmt = count_stmt.where(DocumentORM.file_type == file_type)
        if keyword:
            like = f"%{keyword}%"
            stmt = stmt.where((DocumentORM.filename.ilike(like)) | (DocumentORM.title.ilike(like)))
            count_stmt = count_stmt.where((DocumentORM.filename.ilike(like)) | (DocumentORM.title.ilike(like)))
        stmt = stmt.order_by(DocumentORM.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
        items = list(self.db.scalars(stmt).all())
        total = int(self.db.scalar(count_stmt) or 0)
        return items, total

    def update_status(
        self,
        tenant_id: str,
        document_id: str,
        status: DocumentStatus,
        *,
        error_message: str | None = None,
        failed_stage: str | None = None,
        chunk_count: int | None = None,
        token_count: int | None = None,
        indexed: bool = False,
    ) -> DocumentORM | None:
        record = self.get_document(tenant_id=tenant_id, document_id=document_id)
        if record is None:
            return None
        record.status = status.value
        record.error_message = error_message
        record.failed_stage = failed_stage
        if chunk_count is not None:
            record.chunk_count = chunk_count
        if token_count is not None:
            record.token_count = token_count
        if indexed:
            record.indexed_at = datetime.utcnow()
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def soft_delete(self, tenant_id: str, document_id: str) -> bool:
        record = self.get_document(tenant_id=tenant_id, document_id=document_id)
        if record is None:
            return False
        record.status = DocumentStatus.DELETED.value
        record.deleted_at = datetime.utcnow()
        self.db.add(record)
        self.db.commit()
        return True

    def replace_chunks(self, tenant_id: str, document_id: str, chunks: list[ChunkRecord], embedding_model: str) -> None:
        self.db.query(DocumentChunkORM).filter(
            DocumentChunkORM.tenant_id == tenant_id,
            DocumentChunkORM.document_id == document_id,
        ).delete()
        for chunk in chunks:
            self.db.add(
                DocumentChunkORM(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    tenant_id=chunk.tenant_id,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    token_count=chunk.token_count,
                    page_number=chunk.page_number,
                    section_title=chunk.section_title,
                    metadata_json=chunk.metadata,
                    embedding_model=embedding_model,
                    embedding_status="ready" if chunk.embedding else "pending",
                )
            )
        self.db.commit()

    def list_chunks(self, tenant_id: str, document_id: str) -> tuple[list[DocumentChunkORM], int]:
        stmt = select(DocumentChunkORM).where(
            DocumentChunkORM.tenant_id == tenant_id,
            DocumentChunkORM.document_id == document_id,
        ).order_by(DocumentChunkORM.chunk_index.asc())
        items = list(self.db.scalars(stmt).all())
        return items, len(items)

    def get_distinct_document_ids(self, tenant_id: str) -> list[str]:
        stmt = select(DocumentORM.document_id).where(DocumentORM.tenant_id == tenant_id, DocumentORM.deleted_at.is_(None))
        return list(self.db.scalars(stmt).all())
