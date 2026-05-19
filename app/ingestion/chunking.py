from __future__ import annotations

import hashlib
from datetime import datetime

import tiktoken

from app.documents.schemas import ChunkRecord, ParsedDocument


class RecursiveChunkSplitter:
    def __init__(
        self, chunk_size: int = 900, chunk_overlap: int = 120, encoding_name: str = "cl100k_base"
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)

    def split(self, document: ParsedDocument) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        chunk_index = 0
        for section in document.sections:
            for text in self._split_text(section.content):
                token_count = len(self.encoding.encode(text))
                digest = hashlib.sha1(
                    f"{document.document_id}:{chunk_index}:{text[:64]}".encode()
                ).hexdigest()[:24]
                chunks.append(
                    ChunkRecord(
                        chunk_id=f"chk_{digest}",
                        document_id=document.document_id,
                        tenant_id=document.tenant_id,
                        chunk_index=chunk_index,
                        content=text,
                        metadata={
                            "source_path": document.source_path,
                            "source_type": document.source_type,
                            "content_type": section.metadata.get("content_type", "text"),
                            **section.metadata,
                        },
                        token_count=token_count,
                        page_number=section.page_number,
                        section_title=section.heading,
                        created_at=datetime.utcnow(),
                    )
                )
                chunk_index += 1
        return chunks

    def _split_text(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        tokens = self.encoding.encode(text)
        if len(tokens) <= self.chunk_size:
            return [text]
        chunks: list[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_text = self.encoding.decode(tokens[start:end]).strip()
            if chunk_text:
                chunks.append(chunk_text)
            if end >= len(tokens):
                break
            start = max(0, end - self.chunk_overlap)
        return chunks
