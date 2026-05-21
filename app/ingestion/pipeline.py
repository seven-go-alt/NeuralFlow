from __future__ import annotations

import asyncio
import logging

from app.db.session import SessionLocal
from app.documents.enums import DocumentStatus
from app.documents.repository import DocumentRepository
from app.embeddings.service import EmbeddingService
from app.ingestion.chunking import MarkdownHeadingSplitter, RecursiveChunkSplitter
from app.ingestion.parser_factory import ParserFactory
from app.retrieval.chroma_store import ChromaDocumentStore

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self) -> None:
        self.embedding_service = EmbeddingService()
        self.chunk_splitter: RecursiveChunkSplitter | MarkdownHeadingSplitter = (
            RecursiveChunkSplitter()
        )
        self.store = ChromaDocumentStore()

    async def run(
        self, tenant_id: str, document_id: str, embedding_model: str = "text-embedding-3-small"
    ) -> dict:
        from app.config import get_settings

        settings = get_settings()
        db = SessionLocal()
        repo = DocumentRepository(db)
        try:
            record = repo.get_document(tenant_id=tenant_id, document_id=document_id)
            if record is None:
                raise ValueError(f"Document not found: {document_id}")
            repo.update_status(tenant_id, document_id, DocumentStatus.PARSING)
            parser = ParserFactory.create(record.storage_path)
            parsed = await asyncio.to_thread(
                parser.parse,
                document_id,
                tenant_id,
                record.storage_path,
                record.title,
            )

            # Multimodal extraction (images + tables) for PDF/DOCX
            if settings.multimodal_enabled and record.file_type in ("pdf", "docx"):
                from app.core.llm import LLMClient
                from app.ingestion.multimodal import (
                    ImageExtractor,
                    MultimodalDocumentProcessor,
                    TableExtractor,
                    VisionDescriber,
                )

                processor = MultimodalDocumentProcessor(
                    image_extractor=ImageExtractor(
                        max_size_mb=settings.multimodal_max_image_size_mb,
                        max_images=settings.multimodal_max_images,
                    ),
                    table_extractor=TableExtractor(max_tables=settings.multimodal_max_tables),
                    vision_describer=VisionDescriber(
                        llm_client=LLMClient(),
                        vision_model=settings.vision_model,
                        prompt_template=settings.vision_prompt_template,
                    ),
                )
                extraction = await processor.process(
                    source_path=record.storage_path,
                    file_type=record.file_type,
                    parsed_doc=parsed,
                )
                all_sections = (
                    parsed.sections + extraction.image_sections + extraction.table_sections
                )
                parsed.sections = all_sections
                parsed.metadata["image_count"] = extraction.image_count
                parsed.metadata["table_count"] = extraction.table_count

            repo.update_status(tenant_id, document_id, DocumentStatus.CHUNKING)
            if settings.chunking_strategy == "markdown_heading" and parsed.source_type in (
                "md",
                "markdown",
            ):
                self.chunk_splitter = MarkdownHeadingSplitter(
                    chunk_size=900,
                    chunk_overlap=120,
                    max_section_chars=settings.chunk_max_section_chars,
                    min_section_chars=settings.chunk_min_section_chars,
                )
            else:
                self.chunk_splitter = RecursiveChunkSplitter()
            chunks = self.chunk_splitter.split(parsed)
            repo.update_status(
                tenant_id,
                document_id,
                DocumentStatus.EMBEDDING,
                chunk_count=len(chunks),
                token_count=sum(chunk.token_count for chunk in chunks),
            )
            vectors = await self.embedding_service.embed_texts(
                [chunk.content for chunk in chunks], model=embedding_model
            )
            for chunk, vector in zip(chunks, vectors, strict=False):
                chunk.embedding = vector
                chunk.metadata.update(
                    {
                        "tenant_id": tenant_id,
                        "document_id": document_id,
                        "chunk_id": chunk.chunk_id,
                        "title": record.title,
                        "filename": record.original_filename,
                        "file_type": record.file_type,
                        "embedding_model": embedding_model,
                    }
                )
            repo.update_status(tenant_id, document_id, DocumentStatus.INDEXING)
            self.store.upsert_chunks([chunk.model_dump() for chunk in chunks])
            repo.replace_chunks(
                tenant_id=tenant_id,
                document_id=document_id,
                chunks=chunks,
                embedding_model=embedding_model,
            )
            repo.update_status(
                tenant_id,
                document_id,
                DocumentStatus.READY,
                chunk_count=len(chunks),
                token_count=sum(chunk.token_count for chunk in chunks),
                indexed=True,
            )
            logger.info(
                "document indexed",
                extra={
                    "tenant_id": tenant_id,
                    "document_id": document_id,
                    "chunk_count": len(chunks),
                },
            )
            return {
                "document_id": document_id,
                "chunk_count": len(chunks),
                "status": DocumentStatus.READY.value,
            }
        except Exception as exc:
            logger.exception(
                "document ingestion failed",
                extra={"tenant_id": tenant_id, "document_id": document_id},
            )
            repo.update_status(
                tenant_id,
                document_id,
                DocumentStatus.FAILED,
                error_message=str(exc),
                failed_stage="ingestion",
            )
            raise
        finally:
            db.close()
