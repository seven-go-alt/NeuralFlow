from __future__ import annotations

import mimetypes
import uuid
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.documents.enums import DocumentFileType, DocumentStatus
from app.documents.repository import DocumentRepository
from app.documents.schemas import DocumentCreate
from app.documents.storage import LocalDocumentStorage

_ALLOWED_SUFFIXES = {
    ".pdf": DocumentFileType.PDF.value,
    ".md": DocumentFileType.MARKDOWN.value,
    ".markdown": DocumentFileType.MARKDOWN.value,
    ".txt": DocumentFileType.TXT.value,
    ".docx": DocumentFileType.DOCX.value,
}


class DocumentValidationError(ValueError):
    pass


class DocumentService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.repo = DocumentRepository(db)
        self.storage = LocalDocumentStorage()

    async def upload_document(
        self, tenant_id: str, owner_user_id: str, upload: UploadFile, title: str | None = None
    ):
        filename = upload.filename or "upload.bin"
        suffix = Path(filename).suffix.lower()
        file_type = _ALLOWED_SUFFIXES.get(suffix)
        if not file_type:
            raise DocumentValidationError(f"Unsupported file type: {suffix or 'unknown'}")

        document_id = f"doc_{uuid.uuid4().hex[:24]}"
        storage_path, size_bytes, checksum = await self.storage.save_upload(
            tenant_id=tenant_id, document_id=document_id, upload=upload
        )
        mime_type = (
            upload.content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        )
        payload = DocumentCreate(
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            title=title or Path(filename).stem,
            filename=Path(storage_path).name,
            original_filename=filename,
            file_type=file_type,
            mime_type=mime_type,
            size_bytes=size_bytes,
            storage_path=storage_path,
            checksum_sha256=checksum,
            metadata_json={},
            source_info_json={"original_filename": filename},
        )
        record = self.repo.create_document(payload=payload, document_id=document_id)
        self.repo.update_status(
            tenant_id=tenant_id, document_id=document_id, status=DocumentStatus.QUEUED
        )
        return record
