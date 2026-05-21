from __future__ import annotations

from app.retrieval.hybrid_service import HybridRetrievalService
from app.retrieval.schemas import RetrievalFilters
from app.retrieval.service import RetrievalService


class _FakeVectorStore:
    async def query(self, **kwargs: object) -> dict[str, list]:
        return {"documents": [[]], "metadatas": [[]], "ids": [[]], "distances": [[]]}


class TestRetrievalFilters:
    def setup_method(self) -> None:
        self.service = RetrievalService.__new__(RetrievalService)
        self.hybrid = HybridRetrievalService.__new__(HybridRetrievalService)
        self.hybrid._vector_store = _FakeVectorStore()

    def test_content_types_excluded_when_empty(self) -> None:
        f = RetrievalFilters()
        where = self.service._build_where("t", f.model_dump())
        assert where == {"tenant_id": "t"}

    def test_tags_filter_mql(self) -> None:
        f = RetrievalFilters(tags=["ai", "ml"])
        where = self.service._build_where("t", f.model_dump())
        assert {"tags": {"$in": ["ai", "ml"]}} in where["$and"]

    def test_owner_filter_mql(self) -> None:
        f = RetrievalFilters(owner="alice")
        where = self.service._build_where("t", f.model_dump())
        assert {"owner": {"$eq": "alice"}} in where["$and"]

    def test_groups_filter_mql(self) -> None:
        f = RetrievalFilters(groups=["eng", "data"])
        where = self.service._build_where("t", f.model_dump())
        assert {"groups": {"$in": ["eng", "data"]}} in where["$and"]

    def test_all_new_filters_combined(self) -> None:
        f = RetrievalFilters(tags=["ai"], owner="bob", groups=["eng"])
        where = self.service._build_where("t", f.model_dump())
        assert len(where["$and"]) == 4

    def test_none_fields_excluded(self) -> None:
        f = RetrievalFilters(tags=None, owner=None, groups=None)
        where = self.service._build_where("t", f.model_dump())
        assert where == {"tenant_id": "t"}

    def test_all_existing_and_new_combined(self) -> None:
        f = RetrievalFilters(
            document_ids=["d1"],
            file_types=["pdf"],
            content_types=["text"],
            tags=["ai"],
            owner="alice",
            groups=["eng"],
        )
        where = self.service._build_where("t", f.model_dump())
        assert len(where["$and"]) == 7  # tenant + 6 filters

    def test_multi_document_ids(self) -> None:
        f = RetrievalFilters(document_ids=["d1", "d2"])
        where = self.service._build_where("t", f.model_dump())
        assert {"document_id": {"$in": ["d1", "d2"]}} in where["$and"]

    def test_multi_file_types(self) -> None:
        f = RetrievalFilters(file_types=["pdf", "docx"])
        where = self.service._build_where("t", f.model_dump())
        assert {"file_type": {"$in": ["pdf", "docx"]}} in where["$and"]

    def test_multi_content_types(self) -> None:
        f = RetrievalFilters(content_types=["text", "image_description"])
        where = self.service._build_where("t", f.model_dump())
        assert {"content_type": {"$in": ["text", "image_description"]}} in where["$and"]

    def test_tags_hybrid(self) -> None:
        f = RetrievalFilters(tags=["ai"])
        where = self.hybrid._build_where("t", f.model_dump())
        assert {"tags": {"$in": ["ai"]}} in where["$and"]

    def test_owner_hybrid(self) -> None:
        f = RetrievalFilters(owner="alice")
        where = self.hybrid._build_where("t", f.model_dump())
        assert {"owner": {"$eq": "alice"}} in where["$and"]

    def test_groups_hybrid(self) -> None:
        f = RetrievalFilters(groups=["eng"])
        where = self.hybrid._build_where("t", f.model_dump())
        assert {"groups": {"$in": ["eng"]}} in where["$and"]
