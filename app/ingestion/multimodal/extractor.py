from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path

import fitz
from docx import Document as DocxDocument

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ExtractedImage:
    image_data: bytes
    page_number: int | None
    image_index: int
    format: str
    size_bytes: int


@dataclass(slots=True)
class ExtractedTable:
    markdown_text: str
    page_number: int | None
    table_index: int


class ImageExtractor:
    def __init__(self, max_size_mb: int = 5, max_images: int = 20) -> None:
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._max_images = max_images

    def extract_images(self, path: str, file_type: str) -> list[ExtractedImage]:
        if file_type == "pdf":
            return self._extract_from_pdf(path)
        if file_type == "docx":
            return self._extract_from_docx(path)
        return []

    def _extract_from_pdf(self, path: str) -> list[ExtractedImage]:
        images: list[ExtractedImage] = []
        doc = fitz.open(path)
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                for _img_index, img in enumerate(page.get_images(full=True)):
                    if len(images) >= self._max_images:
                        return images
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    try:
                        image_data = pix.tobytes()
                        if len(image_data) > self._max_size_bytes:
                            continue
                        ext = "png" if pix.n < 5 else "jpeg"
                        images.append(
                            ExtractedImage(
                                image_data=image_data,
                                page_number=page_num + 1,
                                image_index=len(images),
                                format=ext,
                                size_bytes=len(image_data),
                            )
                        )
                    finally:
                        pix = None
        finally:
            doc.close()
        return images

    def _extract_from_docx(self, path: str) -> list[ExtractedImage]:
        images: list[ExtractedImage] = []
        try:
            with zipfile.ZipFile(path, "r") as z:
                media_files = [n for n in z.namelist() if n.startswith("word/media/")]
                for idx, media_path in enumerate(media_files):
                    if len(images) >= self._max_images:
                        break
                    ext = Path(media_path).suffix.lstrip(".") or "png"
                    if ext.lower() in ("png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp"):
                        image_data = z.read(media_path)
                        if len(image_data) > self._max_size_bytes:
                            continue
                        images.append(
                            ExtractedImage(
                                image_data=image_data,
                                page_number=None,
                                image_index=idx,
                                format=ext,
                                size_bytes=len(image_data),
                            )
                        )
        except zipfile.BadZipFile:
            logger.warning("not a valid zip/docx file: %s", path)
        return images


class TableExtractor:
    def __init__(self, max_tables: int = 50) -> None:
        self._max_tables = max_tables

    def extract_tables(self, path: str, file_type: str) -> list[ExtractedTable]:
        if file_type == "pdf":
            return self._extract_from_pdf(path)
        if file_type == "docx":
            return self._extract_from_docx(path)
        return []

    def _extract_from_pdf(self, path: str) -> list[ExtractedTable]:
        tables: list[ExtractedTable] = []
        doc = fitz.open(path)
        try:
            for page_num in range(len(doc)):
                if len(tables) >= self._max_tables:
                    break
                page = doc[page_num]
                pdf_tables = page.find_tables()
                for table in pdf_tables:
                    if len(tables) >= self._max_tables:
                        break
                    data = table.extract()
                    if not data:
                        continue
                    lines = []
                    for row_idx, row in enumerate(data):
                        cells = [str(cell or "").strip() for cell in row]
                        lines.append("| " + " | ".join(cells) + " |")
                        if row_idx == 0:
                            lines.append("|" + "|".join("---" for _ in cells) + "|")
                    tables.append(
                        ExtractedTable(
                            markdown_text="\n".join(lines),
                            page_number=page_num + 1,
                            table_index=len(tables),
                        )
                    )
        finally:
            doc.close()
        return tables

    def _extract_from_docx(self, path: str) -> list[ExtractedTable]:
        tables: list[ExtractedTable] = []
        doc = DocxDocument(path)
        for idx, table in enumerate(doc.tables):
            if len(tables) >= self._max_tables:
                break
            lines = []
            for row_idx, row in enumerate(table.rows):
                cells = [cell.text.strip() for cell in row.cells]
                lines.append("| " + " | ".join(cells) + " |")
                if row_idx == 0:
                    lines.append("|" + "|".join("---" for _ in cells) + "|")
            tables.append(
                ExtractedTable(
                    markdown_text="\n".join(lines),
                    page_number=None,
                    table_index=idx,
                )
            )
        return tables
