# Retrieval Module

Multi-mode document retrieval pipeline supporting vector, keyword, and hybrid search.

## Components

### `KeywordStore`
- In-memory keyword index with CJK-aware tokenization
- TF-IDF-like scoring with term frequency and inverse query frequency
- Tenant-level isolation via `tenant_id` filter

### `heuristic_rerank()`
- Weighted reranking: vector score (50%) + keyword overlap (30%) + metadata signal (20%)
- Boosts results where query terms appear in content or title

### `HybridRetrievalService`
- Three modes: `vector`, `keyword`, `hybrid`
- Hybrid mode merges vector + keyword results, deduplicates, then reranks
- Supports `score_threshold` filtering

### Supporting Modules
- `chroma_store.py` — ChromaDB vector store client
- `schemas.py` — Pydantic models (`RetrievalRequest`, `RetrievalResult`, `RetrievalResponse`)
- `service.py` — Legacy `RetrievalService` (wraps ChromaDB directly)
