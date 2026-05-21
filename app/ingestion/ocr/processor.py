from __future__ import annotations

import logging

from app.config import Settings
from app.documents.schemas import ParsedSection

logger = logging.getLogger(__name__)

IMAGE_FILE_TYPES = frozenset({"png", "jpg", "jpeg", "tiff"})


class OCRProcessor:
    def process(
        self,
        source_path: str,
        file_type: str,
        document_id: str,
        tenant_id: str,
        settings: Settings,
    ) -> list[ParsedSection]:
        if not settings.ocr_enabled:
            logger.info("OCR disabled, skipping", extra={"document_id": document_id})
            return []

        file_type = file_type.lower()

        if file_type == "pdf":
            sections = self._process_pdf(source_path, document_id, settings)
        elif file_type in IMAGE_FILE_TYPES:
            sections = self._process_image(source_path, file_type, document_id, settings)
        else:
            logger.warning("Unsupported file type for OCR: %s", file_type)
            return []

        logger.info(
            "OCR completed",
            extra={
                "document_id": document_id,
                "tenant_id": tenant_id,
                "page_count": len(sections),
            },
        )
        return sections

    def _process_pdf(
        self,
        source_path: str,
        document_id: str,
        settings: Settings,
    ) -> list[ParsedSection]:
        try:
            from pdf2image import convert_from_path
        except ImportError:
            logger.warning("pdf2image not installed, cannot OCR PDF")
            return []

        try:
            import pytesseract
        except ImportError:
            logger.warning("pytesseract not installed, cannot OCR PDF")
            return []

        logger.info("Converting PDF pages to images for OCR", extra={"path": source_path})

        try:
            images = convert_from_path(source_path, dpi=settings.ocr_dpi)
        except Exception as exc:
            logger.warning("Failed to convert PDF to images: %s", exc)
            return []

        sections: list[ParsedSection] = []
        for page_num, image in enumerate(images, start=1):
            try:
                text = pytesseract.image_to_string(image, lang=settings.ocr_language).strip()
            except Exception as exc:
                logger.warning("OCR failed for page %d: %s", page_num, exc)
                continue

            if not text:
                logger.warning("OCR returned empty text for page %d", page_num)
                continue

            sections.append(
                ParsedSection(
                    section_id=f"{document_id}:ocr_p{page_num}",
                    content=text,
                    page_number=page_num,
                    heading=f"OCR Page {page_num}",
                    metadata={
                        "page_number": page_num,
                        "ocr_source": "pdf",
                    },
                )
            )

        return sections

    def _process_image(
        self,
        source_path: str,
        file_type: str,
        document_id: str,
        settings: Settings,
    ) -> list[ParsedSection]:
        try:
            import pytesseract
        except ImportError:
            logger.warning("pytesseract not installed, cannot OCR image")
            return []

        try:
            from PIL import Image
        except ImportError:
            logger.warning("PIL not installed, cannot OCR image")
            return []

        logger.info("Running OCR on image", extra={"path": source_path})

        try:
            image = Image.open(source_path)
            text = pytesseract.image_to_string(image, lang=settings.ocr_language).strip()
        except Exception as exc:
            logger.warning("OCR failed for image %s: %s", source_path, exc)
            return []

        if not text:
            logger.warning("OCR returned empty text for image: %s", source_path)
            return []

        return [
            ParsedSection(
                section_id=f"{document_id}:ocr_img",
                content=text,
                page_number=1,
                heading="OCR Image",
                metadata={
                    "page_number": 1,
                    "ocr_source": "image",
                    "file_type": file_type,
                },
            )
        ]
