# Changelog

## 2026-05-08 - Enterprise Document RAG Upgrade

### Added
- Document upload API for PDF / Markdown / TXT / DOCX
- SQLAlchemy-backed document metadata storage
- Document status lifecycle and chunk metadata persistence
- Async ingestion pipeline via Celery
- Parser support for PDF / DOCX / Markdown / TXT
- Token-aware recursive chunk splitting with overlap
- OpenAI-compatible embedding provider abstraction
- Embedding cache layer
- ChromaDB-backed document retrieval API
- RAG context builder with citation generation
- `/chat` citations response payload
- `/chat/stream` retrieval and chunk SSE events
- Documents page, document detail page, and chunk viewer in Next.js frontend
- Runtime source visualization and assistant-message citation list
- Regression tests for documents, ingestion, retrieval, RAG chat, and RAG context builder
- Architecture documentation in `docs/rag-architecture.md`

### Changed
- NeuralFlow positioning upgraded from lightweight memory-first agent template to enterprise AI knowledge base agent platform
- Chat flow now supports automatic retrieval injection into prompt assembly
- Document upload flow now queues ingestion work instead of doing all parsing synchronously in request path
- Frontend document upload now refreshes document list automatically after success
- README rewritten around AI knowledge base / RAG system design and usage

### Tested
- `uv run pytest -q tests/test_rag_chat.py tests/test_rag_context_builder.py tests/test_streaming.py`
- `uv run pytest -q tests/test_documents_api.py tests/test_ingestion_pipeline.py tests/test_retrieval_api.py`
- `uv run python -m compileall app tests`
- `cd frontend && npm run typecheck`

### Notes
- Current embedding cache is process-local and can be upgraded to Redis later.
- Current metadata filters are intentionally conservative and can be extended with tags / ACL / document groups.
