# Spec: RAG + Citations Contract

## Status
Draft

## Goal
Define the minimum stable contract for NeuralFlow retrieval and citation behavior so backend, frontend, and evaluation tooling can evolve without silent drift.

## Why
The project already returns retrieval results and chat citations, but the rules are implicit and easy to break when changing prompt assembly, retrieval ranking, or chat response formatting.

A written contract prevents accidental regressions and gives Eval v1 something concrete to check.

## Scope
In scope:
- `/api/retrieval/search` request/response contract
- `/chat` retrieval-enabled behavior
- citation object minimum fields
- no-result / low-result fallback behavior
- optional debug fields that do not break existing consumers

Out of scope:
- frontend rendering details
- exact prompt wording
- ranking algorithm internals
- long-term ranking research

## Endpoint contract: `/api/retrieval/search`

### Request
Expected request fields:
- `query: string`
- `top_k: int`
- `score_threshold: float`
- `filters: object`

### Response
Response must include:
- `query`
- `results: []`

Each result must include:
- `chunk_id`
- `document_id`
- `content`
- `score`
- `metadata`
- `source`

`source` should contain, when available:
- `title`
- `filename`
- `page_number`

### Behavior
- results should be sorted by descending relevance score
- empty results are valid and must not be treated as transport failure
- score_threshold should be applied consistently server-side

## Endpoint contract: `/chat`

### Request
Current relevant request fields:
- `session_id`
- `message`
- `use_retrieval`

### Response
Current response must include:
- `reply`
- `citations` (may be empty)

### Retrieval-enabled behavior
When `use_retrieval=true` and retrieval returns supporting context:
- response should include non-empty citations when evidence is used
- citation order should match the constructed RAG context order
- reply should remain human-readable even if no citations exist

### No-result fallback behavior
When retrieval returns no relevant results:
- `/chat` must still return a valid reply
- `citations` may be empty
- the system should avoid fabricating specific sourced claims that appear grounded in nonexistent documents

## Citation contract
Each citation should contain at minimum:
- `index`
- `label`
- `document_id`
- `chunk_id`
- `page_number` when available

Additional metadata may be added, but these minimum fields should remain stable unless versioned.

## Compatibility requirement
Changes to citation shape or retrieval result shape must be reflected in:
- backend tests
- frontend consumers
- eval case expectations
- this spec

## Optional debug contract
If debug mode is enabled for internal evaluation or diagnostics, the response may include a `debug` object.

Possible debug sections:
- `token_budget`
- `retrieval`
- `timing`

Requirements:
- debug fields must be optional
- default client behavior must remain unchanged when debug is omitted
- debug must not be required by frontend production flow

## Acceptance criteria
- retrieval API schema remains stable and covered by tests
- chat response always includes `reply` and `citations`
- citations include minimum required fields
- no-result retrieval path is explicitly handled and testable
- any future contract change updates this spec and eval expectations
