from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.documents.enums import DocumentStatus


class DocumentMetadata(BaseModel):
    tags: list[str] = Field(default_factory=list)
    external_id: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class DocumentCreate(BaseModel):
    tenant_id: str
    owner_user_id: str
    title: str | None = None
    filename: str
    original_filename: str
    file_type: str
    mime_type: str
    size_bytes: int
    storage_path: str
    checksum_sha256: str
    metadata_json: dict[str, Any] = Field(default_factory=dict)
    source_info_json: dict[str, Any] = Field(default_factory=dict)


class DocumentRead(BaseModel):
    document_id: str
    tenant_id: str
    owner_user_id: str
    title: str | None = None
    filename: str
    original_filename: str
    file_type: str
    mime_type: str
    size_bytes: int
    storage_path: str
    checksum_sha256: str
    status: DocumentStatus
    chunk_count: int = 0
    token_count: int | None = None
    metadata_json: dict[str, Any] = Field(default_factory=dict)
    source_info_json: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None
    failed_stage: str | None = None
    created_at: datetime
    updated_at: datetime
    indexed_at: datetime | None = None

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    items: list[DocumentRead]
    total: int
    page: int
    page_size: int


class DocumentChunkRead(BaseModel):
    chunk_id: str
    document_id: str
    tenant_id: str
    chunk_index: int
    content: str
    token_count: int
    page_number: int | None = None
    section_title: str | None = None
    metadata_json: dict[str, Any] = Field(default_factory=dict)
    embedding_model: str | None = None
    embedding_status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentChunksResponse(BaseModel):
    items: list[DocumentChunkRead]
    total: int


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: DocumentStatus
    tenant_id: str
    owner_user_id: str
    created_at: datetime


class ParsedSection(BaseModel):
    section_id: str
    content: str
    page_number: int | None = None
    heading: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    document_id: str
    tenant_id: str
    title: str | None = None
    source_type: str
    source_path: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    sections: list[ParsedSection] = Field(default_factory=list)
    extracted_text: str
    created_at: datetime


class ChunkRecord(BaseModel):
    chunk_id: str
    document_id: str
    tenant_id: str
    chunk_index: int
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None
    token_count: int
    page_number: int | None = None
    section_title: str | None = None
    created_at: datetime
