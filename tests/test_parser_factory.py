from __future__ import annotations

import pytest

from app.ingestion.parser_factory import ParserFactory
from app.ingestion.parsers import DOCXParser, MarkdownParser, PDFParser, TXTParser


class TestParserFactory:
    def test_pdf(self) -> None:
        parser = ParserFactory.create("doc.pdf")
        assert isinstance(parser, PDFParser)

    def test_md(self) -> None:
        parser = ParserFactory.create("readme.md")
        assert isinstance(parser, MarkdownParser)

    def test_markdown(self) -> None:
        parser = ParserFactory.create("doc.markdown")
        assert isinstance(parser, MarkdownParser)

    def test_txt(self) -> None:
        parser = ParserFactory.create("notes.txt")
        assert isinstance(parser, TXTParser)

    def test_docx(self) -> None:
        parser = ParserFactory.create("report.docx")
        assert isinstance(parser, DOCXParser)

    def test_upper_case_suffix(self) -> None:
        parser = ParserFactory.create("DOCUMENT.PDF")
        assert isinstance(parser, PDFParser)

    def test_path_with_dirs(self) -> None:
        parser = ParserFactory.create("/path/to/file.txt")
        assert isinstance(parser, TXTParser)

    def test_unsupported_suffix(self) -> None:
        with pytest.raises(ValueError, match="No parser for file suffix"):
            ParserFactory.create("file.csv")

    def test_no_suffix(self) -> None:
        with pytest.raises(ValueError, match="No parser for file suffix"):
            ParserFactory.create("Makefile")
