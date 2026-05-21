from __future__ import annotations

from datetime import datetime

from app.documents.schemas import ParsedDocument, ParsedSection
from app.ingestion.chunking import MarkdownHeadingSplitter


def _doc(source_type: str = "md") -> ParsedDocument:
    return ParsedDocument(
        document_id="d1",
        tenant_id="t1",
        source_type=source_type,
        source_path="/dev/null",
        metadata={},
        sections=[],
        extracted_text="",
        created_at=datetime.utcnow(),
    )


class TestMarkdownHeadingSplitter:
    def test_simple_headings(self) -> None:
        md = "# Title\n\n" + "paragraph content\n" * 25 + "\n## Sub\n\n" + "sub details\n" * 25
        doc = _doc()
        doc.sections = [ParsedSection(section_id="s0", content=md, metadata={})]
        splitter = MarkdownHeadingSplitter(chunk_size=500)
        chunks = splitter.split(doc)
        assert len(chunks) >= 2

    def test_heading_chain_metadata(self) -> None:
        md = "# A\n\n" + "data\n" * 30 + "\n## B\n\n" + "info\n" * 30
        doc = _doc()
        doc.sections = [ParsedSection(section_id="s0", content=md, metadata={})]
        splitter = MarkdownHeadingSplitter(chunk_size=500)
        chunks = splitter.split(doc)
        chains = [c.metadata.get("heading_chain") for c in chunks]
        assert any(c and "A" in c for c in chains)

    def test_section_title(self) -> None:
        md = "# Top\n\n" + "content\n" * 30 + "\n## Middle\n\n" + "more\n" * 30
        doc = _doc()
        doc.sections = [ParsedSection(section_id="s0", content=md, metadata={})]
        splitter = MarkdownHeadingSplitter(chunk_size=500)
        chunks = splitter.split(doc)
        titles = [c.section_title for c in chunks]
        assert "Top" in titles or "Middle" in titles

    def test_no_headings_fallback(self) -> None:
        text = "plain text without any markdown headings " * 50
        doc = _doc()
        doc.sections = [ParsedSection(section_id="s0", content=text, metadata={})]
        splitter = MarkdownHeadingSplitter(chunk_size=200)
        chunks = splitter.split(doc)
        assert len(chunks) > 1

    def test_small_sections_merged(self) -> None:
        md = "# H1\n\ntiny\n\n## H2\n\nalso tiny"
        doc = _doc()
        doc.sections = [ParsedSection(section_id="s0", content=md, metadata={})]
        splitter = MarkdownHeadingSplitter(chunk_size=500, min_section_chars=200)
        chunks = splitter.split(doc)
        assert len(chunks) >= 1

    def test_large_section_subdivided(self) -> None:
        md = "# Big\n\n" + "word " * 2000
        doc = _doc()
        doc.sections = [ParsedSection(section_id="s0", content=md, metadata={})]
        splitter = MarkdownHeadingSplitter(chunk_size=500, max_section_chars=500)
        chunks = splitter.split(doc)
        assert len(chunks) > 1

    def test_heading_chain_is_none_for_plain_text(self) -> None:
        doc = _doc()
        doc.sections = [ParsedSection(section_id="s0", content="just some text", metadata={})]
        splitter = MarkdownHeadingSplitter()
        chunks = splitter.split(doc)
        for c in chunks:
            assert c.metadata.get("heading_chain") is None

    def test_token_count_populated(self) -> None:
        md = "# H\n\ncontent"
        doc = _doc()
        doc.sections = [ParsedSection(section_id="s0", content=md, metadata={})]
        splitter = MarkdownHeadingSplitter()
        chunks = splitter.split(doc)
        for c in chunks:
            assert c.token_count > 0

    def test_empty_document(self) -> None:
        doc = _doc()
        splitter = MarkdownHeadingSplitter()
        assert splitter.split(doc) == []

    def test_non_markdown_uses_recursive(self) -> None:
        doc = _doc(source_type="pdf")
        doc.sections = [ParsedSection(section_id="s0", content="some content", metadata={})]
        splitter = MarkdownHeadingSplitter()
        chunks = splitter.split(doc)
        assert len(chunks) == 1
        assert chunks[0].content == "some content"
