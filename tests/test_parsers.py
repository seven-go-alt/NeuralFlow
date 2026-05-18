from __future__ import annotations

from pathlib import Path

from app.ingestion.parsers import MarkdownParser, TXTParser


def test_txt_parser_reads_content(tmp_path: Path) -> None:
    file_path = tmp_path / "test.txt"
    file_path.write_text("hello world", encoding="utf-8")

    result = TXTParser().parse("doc_1", "public", str(file_path), title="Test")

    assert result.document_id == "doc_1"
    assert result.tenant_id == "public"
    assert result.title == "Test"
    assert result.source_type == "txt"
    assert result.extracted_text == "hello world"
    assert len(result.sections) == 1
    assert result.sections[0].content == "hello world"


def test_markdown_parser_produces_sections(tmp_path: Path) -> None:
    md_content = """# Title

Some intro text.

## Section 1

Content for section one.

## Section 2

Content for section two.
"""
    file_path = tmp_path / "test.md"
    file_path.write_text(md_content, encoding="utf-8")

    result = MarkdownParser().parse("doc_2", "tenant-a", str(file_path), title="Doc")

    assert result.source_type == "md"
    assert result.document_id == "doc_2"
    assert len(result.sections) >= 2


def test_markdown_parser_handles_plain_text_without_headings(tmp_path: Path) -> None:
    file_path = tmp_path / "noheadings.md"
    file_path.write_text("Just a plain text without any markdown headings.", encoding="utf-8")

    result = MarkdownParser().parse("doc_3", "public", str(file_path))

    assert len(result.sections) == 1
    assert "headings" in result.sections[0].content


def test_markdown_parser_inline_without_heading_sets_section_heading(tmp_path: Path) -> None:
    """First inline token becomes heading, subsequent inline tokens become content."""
    file_path = tmp_path / "inline_first.md"
    file_path.write_text("First paragraph before any heading.\n\nSecond line.", encoding="utf-8")

    result = MarkdownParser().parse("doc_4", "public", str(file_path))

    assert len(result.sections) >= 1
    assert result.sections[0].heading == "First paragraph before any heading."
    assert result.sections[0].content == "Second line."
