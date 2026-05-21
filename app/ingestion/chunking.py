from __future__ import annotations

import hashlib
import re
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
                            "document_id": document.document_id,
                            "tags": document.metadata.get("tags", []),
                            "owner": document.metadata.get("owner"),
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


class MarkdownHeadingSplitter:
    """Split markdown content by heading hierarchy, preserving heading chains in metadata."""

    HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def __init__(
        self,
        chunk_size: int = 900,
        chunk_overlap: int = 120,
        encoding_name: str = "cl100k_base",
        max_section_chars: int = 2000,
        min_section_chars: int = 100,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
        self._max_section_chars = max_section_chars
        self._min_section_chars = min_section_chars

    def split(self, document: ParsedDocument) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        chunk_index = 0

        raw_text = "\n\n".join(s.content for s in document.sections if s.content.strip())
        raw_sections = self._parse_headings(raw_text)
        raw_sections = self._merge_tiny_sections(raw_sections)
        raw_sections = self._subdivide_large_sections(raw_sections)

        for heading_chain, content in raw_sections:
            texts = self._split_text(content)
            for text in texts:
                token_count = len(self.encoding.encode(text))
                digest = hashlib.sha1(
                    f"{document.document_id}:{chunk_index}:{text[:64]}".encode()
                ).hexdigest()[:24]
                section_title = heading_chain.split(" > ")[-1] if heading_chain else None
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
                            "content_type": "text",
                            "heading_chain": heading_chain,
                            "document_id": document.document_id,
                            "tags": document.metadata.get("tags", []),
                            "owner": document.metadata.get("owner"),
                        },
                        token_count=token_count,
                        section_title=section_title,
                        created_at=datetime.utcnow(),
                    )
                )
                chunk_index += 1
        return chunks

    def _parse_headings(self, text: str) -> list[tuple[str | None, str]]:
        matches = list(self.HEADING_RE.finditer(text))
        if not matches:
            return [(None, text)]

        sections: list[tuple[str | None, str]] = []
        heading_chain: list[str] = []

        for i, m in enumerate(matches):
            level = len(m.group(1))
            heading_text = m.group(2).strip()

            while heading_chain and len(heading_chain) >= level:
                heading_chain.pop()
            if len(heading_chain) >= level:
                heading_chain[level - 1] = heading_text
            else:
                heading_chain.append(heading_text)
            full_chain = " > ".join(heading_chain) if heading_chain else None

            content_start = m.end()
            content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[content_start:content_end].strip()
            sections.append((full_chain, content))

        return sections

    def _merge_tiny_sections(
        self, sections: list[tuple[str | None, str]]
    ) -> list[tuple[str | None, str]]:
        if not sections:
            return sections
        merged: list[tuple[str | None, str]] = [sections[0]]
        for heading, content in sections[1:]:
            if len(content) < self._min_section_chars and merged:
                prev_heading, prev_content = merged[-1]
                merged[-1] = (prev_heading, prev_content + "\n\n" + content)
            else:
                merged.append((heading, content))
        return merged

    def _subdivide_large_sections(
        self, sections: list[tuple[str | None, str]]
    ) -> list[tuple[str | None, str]]:
        result: list[tuple[str | None, str]] = []
        for heading, content in sections:
            if len(content) <= self._max_section_chars:
                result.append((heading, content))
            else:
                subdivided = self._subdivide(heading or "", content)
                result.extend(subdivided)
        return result

    def _subdivide(self, heading: str, content: str) -> list[tuple[str | None, str]]:
        sub_matches = list(re.finditer(r"^#{4,}\s+(.+)$", content, re.MULTILINE))
        if sub_matches:
            parts: list[tuple[str | None, str]] = []
            for i, m in enumerate(sub_matches):
                start = m.end()
                end = sub_matches[i + 1].start() if i + 1 < len(sub_matches) else len(content)
                sub_heading = f"{heading} > {m.group(1).strip()}" if heading else m.group(1).strip()
                parts.append((sub_heading, content[start:end].strip()))
            return parts
        paras = re.split(r"\n\s*\n", content)
        groups: list[str] = []
        current = ""
        for para in paras:
            if not para.strip():
                continue
            if len(current) + len(para) < self._max_section_chars:
                current = (current + "\n\n" + para).strip()
            else:
                if current:
                    groups.append(current)
                current = para
        if current:
            groups.append(current)
        return [(heading, g) for g in groups]

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
