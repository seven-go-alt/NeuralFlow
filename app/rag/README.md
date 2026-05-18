# RAG Module

Retrieval-Augmented Generation: context building, quality guards, and prompt management.

## Components

### `RAGContextBuilder`
- Builds structured context from `RetrievalResult` list
- Deduplicates by content prefix, formats with citation labels `[1]`, `[2]`, etc.
- Applies `TokenBudgetManager` to trim context within token limits

### `CitationVerifier`
- `verify_citations(answer, citations)` — validates `[N]` markers in generated text
- Detects hallucinated citations (indices not in source metadata)
- Tracks unused citations and invalid references

### `NoAnswerPolicy`
- `decide_no_answer(query, results)` — evaluates retrieval confidence + term overlap
- Configurable thresholds and `empty_result_policy` (refuse/fallback)
- Returns decision with structured reason

### `PromptRegistry`
- Versioned prompt template registry with `render()` and `required_fields`
- Built-in templates: `rag_system` (with citations), `rag_system_no_citation`
- Supports `register()`, `get()`, `list_templates()`, `remove()`

### `RAGService`
- High-level orchestrator combining retrieval + context building
