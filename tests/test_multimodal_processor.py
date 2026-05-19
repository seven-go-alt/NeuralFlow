from __future__ import annotations

from datetime import datetime

import pytest

from app.documents.schemas import ParsedDocument, ParsedSection
from app.ingestion.multimodal.extractor import (
    ExtractedImage,
    ExtractedTable,
    ImageExtractor,
    TableExtractor,
)
from app.ingestion.multimodal.processor import MultimodalDocumentProcessor


class StubImageExtractor(ImageExtractor):
    def __init__(self, images: list | None = None) -> None:
        super().__init__()
        self._stub_images = (
            images
            if images is not None
            else [
                ExtractedImage(
                    image_data=b"\x89PNG\r\n\x1a\n",
                    page_number=1,
                    image_index=0,
                    format="png",
                    size_bytes=10,
                ),
            ]
        )

    def extract_images(self, path: str, file_type: str) -> list:
        return self._stub_images


class StubTableExtractor(TableExtractor):
    def __init__(self, tables: list | None = None) -> None:
        super().__init__()
        self._stub_tables = (
            tables
            if tables is not None
            else [
                ExtractedTable(
                    markdown_text="| A | B |\n|---|---|\n| 1 | 2 |", page_number=1, table_index=0
                ),
            ]
        )

    def extract_tables(self, path: str, file_type: str) -> list:
        return self._stub_tables


class StubVisionDescriber:
    async def describe_images(self, images: list) -> list[str]:
        return [f"Description of image {img.image_index}" for img in images]


@pytest.fixture
def parsed_doc():
    return ParsedDocument(
        document_id="doc1",
        tenant_id="t1",
        source_type="pdf",
        source_path="/fake/test.pdf",
        extracted_text="some text",
        created_at=datetime.utcnow(),
        sections=[
            ParsedSection(section_id="doc1:s1", content="text content", metadata={}),
        ],
    )


async def test_processor_creates_image_sections(parsed_doc) -> None:
    processor = MultimodalDocumentProcessor(
        image_extractor=StubImageExtractor(),
        table_extractor=StubTableExtractor(tables=[]),
        vision_describer=StubVisionDescriber(),
    )
    result = await processor.process("/fake/test.pdf", "pdf", parsed_doc)

    assert result.image_count == 1
    assert len(result.image_sections) == 1
    section = result.image_sections[0]
    assert "Description" in section.content
    assert section.metadata["content_type"] == "image_description"


async def test_processor_creates_table_sections(parsed_doc) -> None:
    processor = MultimodalDocumentProcessor(
        image_extractor=StubImageExtractor(images=[]),
        table_extractor=StubTableExtractor(),
        vision_describer=StubVisionDescriber(),
    )
    result = await processor.process("/fake/test.pdf", "pdf", parsed_doc)

    assert result.table_count == 1
    assert len(result.table_sections) == 1
    section = result.table_sections[0]
    assert "|" in section.content
    assert section.metadata["content_type"] == "table_content"


async def test_processor_empty_no_images_or_tables(parsed_doc) -> None:
    processor = MultimodalDocumentProcessor(
        image_extractor=StubImageExtractor(images=[]),
        table_extractor=StubTableExtractor(tables=[]),
        vision_describer=StubVisionDescriber(),
    )
    result = await processor.process("/fake/test.pdf", "pdf", parsed_doc)

    assert result.image_count == 0
    assert result.table_count == 0
    assert result.image_sections == []
    assert result.table_sections == []
