from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pdfplumber
from docx import Document as DocxDocument
from PIL import Image as PILImage
from pypdf import PdfReader

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
        reader = PdfReader(path)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            if len(images) >= self._max_images:
                break
            resources = page.get("/Resources")
            if resources is None or "/XObject" not in resources:
                continue
            xobjects = resources["/XObject"]
            for obj_name in xobjects:
                xobj = xobjects[obj_name].get_object()
                if xobj["/Subtype"] != "/Image":
                    continue
                if len(images) >= self._max_images:
                    break

                raw_data = xobj.get_data()
                width = xobj["/Width"]
                height = xobj["/Height"]

                try:
                    image_data = self._convert_image_to_png(raw_data, width, height, xobj)
                except Exception:
                    logger.exception(
                        "Failed to process image %s on page %d", obj_name, page_num + 1
                    )
                    continue

                if image_data is None or len(image_data) > self._max_size_bytes:
                    continue

                images.append(
                    ExtractedImage(
                        image_data=image_data,
                        page_number=page_num + 1,
                        image_index=len(images),
                        format="png",
                        size_bytes=len(image_data),
                    )
                )
        return images

    @staticmethod
    def _convert_image_to_png(
        raw_data: bytes, width: int, height: int, xobj: object
    ) -> bytes | None:
        import io

        # DCTDecode (JPEG) stores JPEG bytes directly
        pdf_filter = xobj.get("/Filter")  # type: ignore[attr-defined]
        if pdf_filter == "/DCTDecode":
            pil_image: Any = PILImage.open(io.BytesIO(raw_data))
        else:
            color_space = xobj.get("/ColorSpace", "/DeviceRGB")  # type: ignore[attr-defined]
            mode = _colorspace_to_pil_mode(color_space)
            pil_image = PILImage.frombytes(mode, (width, height), raw_data)
            if mode == "CMYK":
                pil_image = pil_image.convert("RGB")

        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        return buf.getvalue()

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
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                if len(tables) >= self._max_tables:
                    break
                pdf_tables = page.extract_tables()
                for table_data in pdf_tables:
                    if len(tables) >= self._max_tables:
                        break
                    if not table_data:
                        continue
                    lines = []
                    for row_idx, row in enumerate(table_data):
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


def _colorspace_to_pil_mode(color_space: object) -> str:
    cs = str(color_space)
    if cs == "/DeviceGray":
        return "L"
    elif cs == "/DeviceCMYK":
        return "CMYK"
    else:
        return "RGB"
