from __future__ import annotations

import asyncio
import logging

from app.db.session import SessionLocal
from app.documents.enums import DocumentStatus
from app.documents.repository import DocumentRepository
from app.embeddings.service import EmbeddingService
from app.ingestion.chunking import RecursiveChunkSplitter
from app.ingestion.parser_factory import ParserFactory
from app.retrieval.chroma_store import ChromaDocumentStore

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self) -> None:
        self.embedding_service = EmbeddingService()
        self.chunk_splitter = RecursiveChunkSplitter()
        self.store = ChromaDocumentStore()

    async def run(self, tenant_id: str, document_id: str, embedding_model: str = "text-embedding-3-small") -> dict:
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
            repo.update_status(tenant_id, document_id, DocumentStatus.CHUNKING)
            chunks = self.chunk_splitter.split(parsed)
            repo.update_status(
                tenant_id,
                document_id,
                DocumentStatus.EMBEDDING,
                chunk_count=len(chunks),
                token_count=sum(chunk.token_count for chunk in chunks),
            )
            vectors = await self.embedding_service.embed_texts([chunk.content for chunk in chunks], model=embedding_model)
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
            repo.replace_chunks(tenant_id=tenant_id, document_id=document_id, chunks=chunks, embedding_model=embedding_model)
            repo.update_status(
                tenant_id,
                document_id,
                DocumentStatus.READY,
                chunk_count=len(chunks),
                token_count=sum(chunk.token_count for chunk in chunks),
                indexed=True,
            )
            logger.info("document indexed", extra={"tenant_id": tenant_id, "document_id": document_id, "chunk_count": len(chunks)})
            return {"document_id": document_id, "chunk_count": len(chunks), "status": DocumentStatus.READY.value}
        except Exception as exc:
            logger.exception("document ingestion failed", extra={"tenant_id": tenant_id, "document_id": document_id})
            repo.update_status(tenant_id, document_id, DocumentStatus.FAILED, error_message=str(exc), failed_stage="ingestion")
            raise
        finally:
            db.close()
