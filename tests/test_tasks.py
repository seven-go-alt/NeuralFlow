from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.ingestion.tasks import ingest_document_task


class TestIngestDocumentTask:
    @patch("app.ingestion.tasks.IngestionPipeline")
    def test_success(self, MockPipeline) -> None:
        pipeline_instance = MockPipeline.return_value
        pipeline_instance.run = AsyncMock(return_value={"status": "ok", "chunks": 5})

        result = ingest_document_task("t1", "d1", "text-embedding-3-small")
        assert result == {"status": "ok", "chunks": 5}
        MockPipeline.assert_called_once()
        pipeline_instance.run.assert_called_once_with(
            tenant_id="t1", document_id="d1", embedding_model="text-embedding-3-small"
        )

    @patch("app.ingestion.tasks.IngestionPipeline")
    def test_default_embedding_model(self, MockPipeline) -> None:
        pipeline_instance = MockPipeline.return_value
        pipeline_instance.run = AsyncMock(return_value={"status": "ok"})

        result = ingest_document_task("t1", "d1")
        assert result["status"] == "ok"
        pipeline_instance.run.assert_called_once_with(
            tenant_id="t1", document_id="d1", embedding_model="text-embedding-3-small"
        )
