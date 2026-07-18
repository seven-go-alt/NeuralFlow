from __future__ import annotations

from typing import Any

import pytest


class TestSanitizeChunkMetadata:
    """纯函数单测:空列表/None/嵌套 dict 被剔除,合规值保留."""

    @pytest.fixture(scope="class")
    def sanitize_fn(self) -> Any:
        from app.retrieval.chroma_store import sanitize_chunk_metadata

        return sanitize_chunk_metadata

    def test_removes_problematic_values(self, sanitize_fn: Any) -> None:
        metadata = {
            "tags": [],
            "owner": None,
            "nested": {"key": "val"},
            "canonical_doc_id": "doc_hr_leave",
            "tenant_id": "public",
        }
        result = sanitize_fn(metadata)
        assert "tags" not in result, "empty list should be dropped"
        assert "owner" not in result, "None should be dropped"
        assert "nested" not in result, "nested dict should be dropped"
        assert result["canonical_doc_id"] == "doc_hr_leave"
        assert result["tenant_id"] == "public"

    def test_keeps_valid_values(self, sanitize_fn: Any) -> None:
        metadata = {
            "canonical_doc_id": "doc_001",
            "tenant_id": "public",
            "chunk_index": 3,
            "score": 0.95,
            "is_active": True,
            "tags": ["hr", "policy"],
        }
        result = sanitize_fn(metadata)
        assert result == metadata, "all valid values should be kept unchanged"
        assert result["tags"] == ["hr", "policy"], "non-empty list should be kept"


@pytest.mark.asyncio
async def test_upsert_with_sanitized_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """集成测试:经过清洗的 metadata 能成功写入 ChromaDB 并能读出."""
    chromadb = pytest.importorskip("chromadb")
    from app.retrieval.chroma_store import ChromaDocumentStore

    monkeypatch.setattr(
        "app.retrieval.chroma_store.get_vector_client",
        lambda **kw: chromadb.EphemeralClient(),
    )

    store = ChromaDocumentStore(collection_name="test_sanitize")

    chunk = {
        "chunk_id": "chk_sanitize_test_001",
        "content": "员工请假需要提前申请。",
        "metadata": {
            "tags": [],
            "owner": None,
            "canonical_doc_id": "doc_hr_leave",
            "tenant_id": "public",
        },
        "embedding": [0.1, 0.2, 0.3],
    }
    store.upsert_chunks([chunk])

    result = store.collection.get(
        ids=["chk_sanitize_test_001"], include=["metadatas"]
    )
    assert len(result["ids"]) == 1, "record should exist after upsert"
    meta = result["metadatas"][0]
    assert isinstance(meta, dict)
    assert meta.get("canonical_doc_id") == "doc_hr_leave"
    assert "tags" not in meta, "empty tags list must not appear in ChromaDB"
    assert "owner" not in meta, "None owner must not appear in ChromaDB"
