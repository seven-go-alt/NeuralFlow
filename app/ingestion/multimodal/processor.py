from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.documents.schemas import ParsedDocument, ParsedSection
from app.ingestion.multimodal.extractor import ImageExtractor, TableExtractor
from app.ingestion.multimodal.vision import VisionDescriber

logger = logging.getLogger(__name__)


@dataclass
class MultimodalExtractionResult:
    image_sections: list = field(default_factory=list)
    table_sections: list = field(default_factory=list)
    image_count: int = 0
    table_count: int = 0


class MultimodalDocumentProcessor:
    def __init__(
        self,
        image_extractor: ImageExtractor,
        table_extractor: TableExtractor,
        vision_describer: VisionDescriber,
    ) -> None:
        self._image_extractor = image_extractor
        self._table_extractor = table_extractor
        self._vision_describer = vision_describer

    async def process(
        self,
        source_path: str,
        file_type: str,
        parsed_doc: ParsedDocument,
    ) -> MultimodalExtractionResult:
        images = self._image_extractor.extract_images(source_path, file_type)
        tables = self._table_extractor.extract_tables(source_path, file_type)

        logger.info(
            "multimodal extraction: %d images, %d tables from %s",
            len(images),
            len(tables),
            source_path,
        )

        descriptions: list[str] = []
        if images:
            descriptions = await self._vision_describer.describe_images(images)

        image_sections = []
        for idx, desc in enumerate(descriptions):
            img = images[idx]
            image_sections.append(
                ParsedSection(
                    section_id=f"{parsed_doc.document_id}:img{idx}",
                    content=f"[Image {idx + 1}]: {desc}",
                    page_number=img.page_number,
                    heading=f"Image {idx + 1}",
                    metadata={
                        "content_type": "image_description",
                        "source_image_index": idx,
                        "image_format": img.format,
                    },
                )
            )

        table_sections = []
        for idx, table in enumerate(tables):
            table_sections.append(
                ParsedSection(
                    section_id=f"{parsed_doc.document_id}:tbl{idx}",
                    content=table.markdown_text,
                    page_number=table.page_number,
                    heading=f"Table {idx + 1}",
                    metadata={
                        "content_type": "table_content",
                        "source_table_index": idx,
                    },
                )
            )

        return MultimodalExtractionResult(
            image_sections=image_sections,
            table_sections=table_sections,
            image_count=len(images),
            table_count=len(tables),
        )
