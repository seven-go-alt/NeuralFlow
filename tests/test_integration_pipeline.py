"""Integration test: document pipeline end-to-end.

Tests real service composition (parsing -> chunking -> persistence)
without mocking internal services. Only external API calls (LLM, embedding)
are outside scope — this tests everything else.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.db.base import Base
from app.documents.enums import DocumentStatus
from app.documents.repository import DocumentRepository
from app.documents.schemas import DocumentCreate
from app.ingestion.chunking import RecursiveChunkSplitter
from app.ingestion.parser_factory import ParserFactory


@pytest.mark.asyncio
async def test_document_pipeline_full_cycle() -> None:
    """End-to-end: create temp file -> parse -> chunk -> persist -> verify.

    This validates the core document pipeline is correctly wired:
    - File parsing extracts meaningful content
    - Chunk splitting produces valid chunks with content and metadata
    - DB persistence retains all fields correctly
    - Status transitions work as expected
    """
    # --- Setup: in-memory SQLite ---
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(bind=engine)

    # --- Setup: temp file with realistic content ---
    content = """员工手册

第一章 总则

本手册适用于公司全体员工。

第二章 考勤制度

1. 上班时间为周一至周五 9:00-18:00。
2. 迟到或早退超过30分钟需提交说明。
3. 请假需提前在系统提交申请。

第三章 休假政策

1. 年假：入职满一年享有5天年假。
2. 病假：需提供医院证明，每年最多15天。
3. 事假：每年最多10天，需提前2天申请。

第四章 薪酬福利

1. 每月15日发放上月工资。
2. 年终奖根据绩效考核发放。
3. 五险一金按国家规定缴纳。"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        filepath = f.name

    doc_id = "doc_integration_test_001"
    tenant_id = "test_tenant"
    title = "员工手册"

    try:
        # --- Phase 1: Parse ---
        parser = ParserFactory.create(filepath)
        parsed = parser.parse(doc_id, tenant_id, filepath, title)
        assert parsed.document_id == doc_id
        assert parsed.tenant_id == tenant_id
        assert parsed.title == title
        assert parsed.source_type == "txt"
        assert parsed.extracted_text
        assert "考勤制度" in parsed.extracted_text
        assert "薪酬福利" in parsed.extracted_text
        assert len(parsed.sections) > 0

        # --- Phase 2: Chunk ---
        splitter = RecursiveChunkSplitter(chunk_size=200, chunk_overlap=30)
        chunks = splitter.split(parsed)
        assert len(chunks) > 0, "Should produce at least one chunk"
        for chunk in chunks:
            assert chunk.chunk_id.startswith("chk_")
            assert chunk.document_id == doc_id
            assert chunk.tenant_id == tenant_id
            assert chunk.content, "Chunk content should not be empty"
            assert chunk.token_count > 0, "Chunk token_count should be > 0"

        # Verify key content is in at least one chunk
        all_chunk_text = " ".join(c.content for c in chunks)
        assert "考勤制度" in all_chunk_text
        assert "年假" in all_chunk_text
        assert "年终奖" in all_chunk_text

        # --- Phase 3: Persist Document Record ---
        db = session_local()
        try:
            repo = DocumentRepository(db)
            record = repo.create_document(
                DocumentCreate(
                    tenant_id=tenant_id,
                    owner_user_id="test_user",
                    title=title,
                    filename="employee_handbook.txt",
                    original_filename="员工手册.txt",
                    file_type="txt",
                    mime_type="text/plain",
                    size_bytes=len(content.encode("utf-8")),
                    storage_path=filepath,
                    checksum_sha256="test_checksum",
                ),
                document_id=doc_id,
            )
            assert record.tenant_id == tenant_id
            assert record.title == title
            assert record.status == DocumentStatus.UPLOADED

            # --- Phase 4: Update status and chunk count ---
            repo.update_status(
                tenant_id,
                record.document_id,
                DocumentStatus.READY,
                chunk_count=len(chunks),
                token_count=sum(c.token_count for c in chunks),
                indexed=True,
            )
            updated = repo.get_document(tenant_id, record.document_id)
            assert updated is not None
            assert updated.status == DocumentStatus.READY
            assert updated.chunk_count == len(chunks)
            assert updated.token_count is not None and updated.token_count > 0

            # --- Phase 5: Persist chunks ---
            repo.replace_chunks(
                tenant_id=tenant_id,
                document_id=record.document_id,
                chunks=chunks,
                embedding_model="text-embedding-3-small",
            )
            stored_chunks, total = repo.list_chunks(tenant_id, record.document_id)
            assert total == len(chunks)
            assert len(stored_chunks) == len(chunks)
            assert stored_chunks[0].document_id == record.document_id

            # Verify chunk ordering
            for i, stored in enumerate(stored_chunks):
                assert stored.chunk_index == i

        finally:
            db.close()

    finally:
        Path(filepath).unlink(missing_ok=True)
