from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest

from app.documents.schemas import ParsedDocument, ParsedSection
from app.embeddings.service import EmbeddingService
from app.ingestion.chunking import RecursiveChunkSplitter
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.chroma_store import ChromaDocumentStore


class StubParser:
    def parse(
        self, document_id: str, tenant_id: str, source_path: str, title: str | None = None
    ) -> ParsedDocument:
        return ParsedDocument(
            document_id=document_id,
            tenant_id=tenant_id,
            title=title,
            source_type="txt",
            source_path=source_path,
            metadata={},
            sections=[
                ParsedSection(
                    section_id=f"{document_id}:s1",
                    content="员工请假需要提前申请。" * 40,
                    page_number=1,
                    heading="请假制度",
                    metadata={"page_number": 1},
                )
            ],
            extracted_text="员工请假需要提前申请。" * 40,
            created_at=datetime.utcnow(),
        )


class StubEmbeddingService(EmbeddingService):
    async def embed_texts(self, texts: list[str], model: str) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class StubStore(ChromaDocumentStore):
    def __init__(self) -> None:
        self.upserts: list[list[dict]] = []

    def upsert_chunks(self, chunks: list[dict]) -> None:
        self.upserts.append(chunks)


class DummyStore:
    def upsert_chunks(self, chunks: list[dict]) -> None:
        return None


def test_recursive_chunk_splitter_produces_chunk_metadata() -> None:
    splitter = RecursiveChunkSplitter(chunk_size=40, chunk_overlap=10)
    doc = ParsedDocument(
        document_id="doc_1",
        tenant_id="public",
        title="员工手册",
        source_type="txt",
        source_path="/tmp/handbook.txt",
        metadata={},
        sections=[
            ParsedSection(
                section_id="doc_1:s1",
                content="请假制度" * 100,
                page_number=2,
                heading="请假制度",
                metadata={"page_number": 2},
            )
        ],
        extracted_text="请假制度" * 100,
        created_at=datetime.utcnow(),
    )

    chunks = splitter.split(doc)

    assert len(chunks) > 1
    assert chunks[0].document_id == "doc_1"
    assert chunks[0].page_number == 2
    assert chunks[0].section_title == "请假制度"
    assert chunks[0].token_count > 0


@pytest.mark.asyncio
async def test_ingestion_pipeline_indexes_document(monkeypatch, tmp_path: Path) -> None:
    from app.db.session import SessionLocal, init_db
    from app.documents.enums import DocumentStatus
    from app.documents.repository import DocumentRepository
    from app.documents.schemas import DocumentCreate

    init_db()
    db = SessionLocal()
    repo = DocumentRepository(db)
    file_path = tmp_path / "policy.txt"
    file_path.write_text("员工请假需要提前申请。", encoding="utf-8")

    document_id = f"doc_test_ingest_{uuid4().hex[:8]}"
    record = repo.create_document(
        DocumentCreate(
            tenant_id="public",
            owner_user_id="tester",
            title="Policy",
            filename="policy.txt",
            original_filename="policy.txt",
            file_type="txt",
            mime_type="text/plain",
            size_bytes=file_path.stat().st_size,
            storage_path=str(file_path),
            checksum_sha256="abc123",
            metadata_json={},
            source_info_json={},
        ),
        document_id=document_id,
    )
    repo.update_status("public", record.document_id, DocumentStatus.QUEUED)
    db.close()

    stub_store = StubStore()
    monkeypatch.setattr("app.ingestion.pipeline.ParserFactory.create", lambda path: StubParser())
    monkeypatch.setattr(
        "app.ingestion.pipeline.ChromaDocumentStore", lambda *args, **kwargs: DummyStore()
    )

    pipeline = IngestionPipeline()
    pipeline.embedding_service = StubEmbeddingService()
    pipeline.store = stub_store

    result = await pipeline.run(
        tenant_id="public", document_id=document_id, embedding_model="test-embedding"
    )

    assert result["status"] == "ready"
    assert result["chunk_count"] >= 1
    assert len(stub_store.upserts) == 1
    assert stub_store.upserts[0][0]["metadata"]["document_id"] == document_id


@pytest.mark.asyncio
async def test_ingestion_pipeline_merges_document_metadata(monkeypatch, tmp_path: Path) -> None:
    from app.db.session import SessionLocal, init_db
    from app.documents.enums import DocumentStatus
    from app.documents.repository import DocumentRepository
    from app.documents.schemas import DocumentCreate

    init_db()
    db = SessionLocal()
    repo = DocumentRepository(db)
    file_path = tmp_path / "leave.md"
    file_path.write_text("# Annual Leave\n员工请假需要提前申请。", encoding="utf-8")

    document_id = f"doc_test_meta_{uuid4().hex[:8]}"
    repo.create_document(
        DocumentCreate(
            tenant_id="public",
            owner_user_id="tester",
            title="Leave Policy",
            filename="leave.md",
            original_filename="leave.md",
            file_type="md",
            mime_type="text/markdown",
            size_bytes=file_path.stat().st_size,
            storage_path=str(file_path),
            checksum_sha256="abc123",
            metadata_json={"canonical_doc_id": "doc_hr_leave", "eval_corpus": True},
            source_info_json={},
        ),
        document_id=document_id,
    )
    repo.update_status("public", document_id, DocumentStatus.QUEUED)
    db.close()

    stub_store = StubStore()
    monkeypatch.setattr("app.ingestion.pipeline.ParserFactory.create", lambda path: StubParser())
    monkeypatch.setattr(
        "app.ingestion.pipeline.ChromaDocumentStore", lambda *args, **kwargs: DummyStore()
    )
    pipeline = IngestionPipeline()
    pipeline.embedding_service = StubEmbeddingService()
    pipeline.store = stub_store

    await pipeline.run(
        tenant_id="public", document_id=document_id, embedding_model="test-embedding"
    )

    metadata = stub_store.upserts[0][0]["metadata"]
    assert metadata["canonical_doc_id"] == "doc_hr_leave"
    assert metadata["document_id"] == document_id  # 系统字段不被文档 metadata 覆盖
