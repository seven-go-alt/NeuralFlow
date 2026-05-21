from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings, get_settings
from app.ingestion.ocr.processor import OCRProcessor


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    get_settings.cache_clear()


def test_ocr_skipped_when_disabled() -> None:
    settings = Settings()
    processor = OCRProcessor()
    result = processor.process(
        source_path="/fake/doc.pdf",
        file_type="pdf",
        document_id="doc1",
        tenant_id="t1",
        settings=settings,
    )
    assert result == []


def test_ocr_processes_pdf(monkeypatch) -> None:
    monkeypatch.setenv("OCR_ENABLED", "true")
    monkeypatch.setenv("OCR_LANGUAGE", "eng")
    monkeypatch.setenv("OCR_DPI", "300")
    settings = Settings()
    mock_image = MagicMock()

    with patch("pdf2image.convert_from_path", return_value=[mock_image]) as mock_convert:
        with patch("pytesseract.image_to_string", return_value="Extracted text") as mock_ocr:
            processor = OCRProcessor()
            sections = processor.process(
                source_path="/fake/doc.pdf",
                file_type="pdf",
                document_id="doc1",
                tenant_id="t1",
                settings=settings,
            )

    mock_convert.assert_called_once_with("/fake/doc.pdf", dpi=300)
    mock_ocr.assert_called_once_with(mock_image, lang="eng")
    assert len(sections) == 1
    assert sections[0].content == "Extracted text"
    assert sections[0].page_number == 1
    assert sections[0].section_id == "doc1:ocr_p1"
    assert sections[0].heading == "OCR Page 1"
    assert sections[0].metadata["ocr_source"] == "pdf"


def test_ocr_processes_png(monkeypatch) -> None:
    monkeypatch.setenv("OCR_ENABLED", "true")
    settings = Settings()
    mock_image = MagicMock()

    with patch("PIL.Image.open", return_value=mock_image):
        with patch("pytesseract.image_to_string", return_value="Image text"):
            processor = OCRProcessor()
            sections = processor.process(
                source_path="/fake/doc.png",
                file_type="png",
                document_id="doc1",
                tenant_id="t1",
                settings=settings,
            )

    assert len(sections) == 1
    assert sections[0].content == "Image text"
    assert sections[0].page_number == 1
    assert sections[0].section_id == "doc1:ocr_img"
    assert sections[0].heading == "OCR Image"
    assert sections[0].metadata["ocr_source"] == "image"


@pytest.mark.parametrize("img_type", ["jpg", "jpeg", "tiff"])
def test_ocr_processes_other_image_types(img_type: str, monkeypatch) -> None:
    monkeypatch.setenv("OCR_ENABLED", "true")
    settings = Settings()
    mock_image = MagicMock()

    with patch("PIL.Image.open", return_value=mock_image):
        with patch("pytesseract.image_to_string", return_value="Image text"):
            processor = OCRProcessor()
            sections = processor.process(
                source_path=f"/fake/doc.{img_type}",
                file_type=img_type,
                document_id="doc1",
                tenant_id="t1",
                settings=settings,
            )

    assert len(sections) == 1
    assert sections[0].metadata["file_type"] == img_type


def test_ocr_per_page_error_handling(monkeypatch) -> None:
    """A failing page should not fail the whole PDF OCR."""
    monkeypatch.setenv("OCR_ENABLED", "true")
    settings = Settings()
    mock_images = [MagicMock(), MagicMock(), MagicMock()]

    with patch("pdf2image.convert_from_path", return_value=mock_images):
        with patch(
            "pytesseract.image_to_string",
            side_effect=["Page 1 content", Exception("OCR engine failed"), "Page 3 content"],
        ):
            processor = OCRProcessor()
            sections = processor.process(
                source_path="/fake/doc.pdf",
                file_type="pdf",
                document_id="doc1",
                tenant_id="t1",
                settings=settings,
            )

    assert len(sections) == 2
    assert sections[0].content == "Page 1 content"
    assert sections[0].page_number == 1
    assert sections[1].content == "Page 3 content"
    assert sections[1].page_number == 3


def test_ocr_skipped_for_unsupported_file_type(monkeypatch) -> None:
    monkeypatch.setenv("OCR_ENABLED", "true")
    settings = Settings()
    processor = OCRProcessor()
    result = processor.process(
        source_path="/fake/doc.docx",
        file_type="docx",
        document_id="doc1",
        tenant_id="t1",
        settings=settings,
    )
    assert result == []


def test_ocr_config_defaults() -> None:
    settings = get_settings()
    assert settings.ocr_enabled is False
    assert settings.ocr_language == "eng"
    assert settings.ocr_dpi == 300


def test_ocr_handles_pdf_conversion_failure(monkeypatch) -> None:
    monkeypatch.setenv("OCR_ENABLED", "true")
    settings = Settings()

    with patch("pdf2image.convert_from_path", side_effect=RuntimeError("poppler not found")):
        processor = OCRProcessor()
        sections = processor.process(
            source_path="/fake/doc.pdf",
            file_type="pdf",
            document_id="doc1",
            tenant_id="t1",
            settings=settings,
        )

    assert sections == []


def test_ocr_handles_empty_text_from_page(monkeypatch) -> None:
    monkeypatch.setenv("OCR_ENABLED", "true")
    settings = Settings()
    mock_images = [MagicMock(), MagicMock()]

    with patch("pdf2image.convert_from_path", return_value=mock_images):
        with patch(
            "pytesseract.image_to_string",
            side_effect=["Valid text", ""],
        ):
            processor = OCRProcessor()
            sections = processor.process(
                source_path="/fake/doc.pdf",
                file_type="pdf",
                document_id="doc1",
                tenant_id="t1",
                settings=settings,
            )

    assert len(sections) == 1
    assert sections[0].content == "Valid text"


def test_ocr_handles_image_open_failure(monkeypatch) -> None:
    monkeypatch.setenv("OCR_ENABLED", "true")
    settings = Settings()

    with patch("PIL.Image.open", side_effect=FileNotFoundError("No such file")):
        processor = OCRProcessor()
        sections = processor.process(
            source_path="/fake/missing.png",
            file_type="png",
            document_id="doc1",
            tenant_id="t1",
            settings=settings,
        )

    assert sections == []


def test_ocr_handles_image_empty_text(monkeypatch) -> None:
    monkeypatch.setenv("OCR_ENABLED", "true")
    settings = Settings()
    mock_image = MagicMock()

    with patch("PIL.Image.open", return_value=mock_image):
        with patch("pytesseract.image_to_string", return_value="   "):
            processor = OCRProcessor()
            sections = processor.process(
                source_path="/fake/blank.png",
                file_type="png",
                document_id="doc1",
                tenant_id="t1",
                settings=settings,
            )

    assert sections == []
