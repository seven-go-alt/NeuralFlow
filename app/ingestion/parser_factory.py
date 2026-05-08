from __future__ import annotations

from pathlib import Path

from app.ingestion.parsers import DOCXParser, MarkdownParser, PDFParser, TXTParser


class ParserFactory:
    @staticmethod
    def create(source_path: str):
        suffix = Path(source_path).suffix.lower()
        if suffix == ".pdf":
            return PDFParser()
        if suffix in {".md", ".markdown"}:
            return MarkdownParser()
        if suffix == ".txt":
            return TXTParser()
        if suffix == ".docx":
            return DOCXParser()
        raise ValueError(f"No parser for file suffix: {suffix}")
