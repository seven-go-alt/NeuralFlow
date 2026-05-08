from __future__ import annotations

from app.rag.context_builder import RAGContextBuilder
from app.retrieval.schemas import RetrievalResult


def test_rag_context_builder_returns_context_and_citations() -> None:
    builder = RAGContextBuilder()
    results = [
        RetrievalResult(
            chunk_id="chk_1",
            document_id="doc_1",
            content="员工请假需要提前申请",
            score=0.93,
            metadata={"page_number": 2},
            source={"title": "员工手册", "filename": "handbook.pdf", "page_number": 2},
        ),
        RetrievalResult(
            chunk_id="chk_1",
            document_id="doc_1",
            content="员工请假需要提前申请",
            score=0.91,
            metadata={"page_number": 2},
            source={"title": "员工手册", "filename": "handbook.pdf", "page_number": 2},
        ),
    ]

    built = builder.build(query="请假制度", results=results)

    assert "[1] 员工手册" in built.context
    assert len(built.used_chunks) == 1
    assert built.citations[0]["document_id"] == "doc_1"
    assert built.citations[0]["page_number"] == 2
