# Spec: Eval v1

## Status
Draft

## Goal
Create a minimal, in-repo evaluation system that can run fixed cases against NeuralFlow's existing APIs, produce simple metrics, and compare against a baseline to detect regressions.

## Why
The project already has retrieval, chat, citations, and token-budget behavior, but lacks a stable way to answer:
- did retrieval quality regress?
- did citations disappear or degrade?
- did answer quality drift on known cases?
- did latency or token-budget trimming behavior materially worsen?

Eval v1 should close that gap without introducing a heavy external framework.

## Scope
In scope:
- fixed, versioned test cases stored in repo
- batch runner for `/api/retrieval/search` and `/chat`
- deterministic metrics first
- JSON report output
- baseline comparison for regressions
- optional debug capture for token-budget data

Out of scope for v1:
- external eval SaaS
- database-backed run history
- LLM judge as required gate
- broad UI benchmark tooling
- multi-tenant benchmark orchestration

## Existing assets to reuse
- `/chat`
- `/api/retrieval/search`
- citations returned by chat flows
- token budget logic in `app/core/token_budget.py`
- current tests under `tests/test_rag_chat.py`, `tests/test_retrieval_api.py`, `tests/test_token_budget.py`
- current lightweight evaluator utilities under `scripts/eval/evaluator.py`

## User stories
- As a developer, I can run a small fixed suite locally before merging retrieval/chat changes.
- As a maintainer, I can compare current results to a baseline and quickly see regressions.
- As a reviewer, I can inspect concrete per-case failures instead of only reading raw model outputs.

## Functional requirements

### 1. Fixed case definitions
The system must support versioned JSON case files for at least two suites:
- retrieval
- chat

Each case must have:
- stable `id`
- input payload
- expected assertions
- optional tags

### 2. Batch runner
The runner must:
- call live HTTP endpoints rather than internal functions by default
- support `retrieval`, `chat`, and `all` suites
- emit machine-readable JSON report files
- return non-zero exit on hard failure when configured

### 3. Metrics

#### Retrieval metrics
Per case:
- latency_ms
- result_count
- expected document hit
- expected chunk hit when specified
- expected title hit when specified

Suite summary:
- pass_rate
- avg_latency_ms
- document_hit_rate
- chunk_hit_rate

#### Chat metrics
Per case:
- latency_ms
- keyword_score
- forbidden keyword violations
- citation_count
- citation document/title/page hits when specified

Suite summary:
- pass_rate
- avg_latency_ms
- avg_keyword_score
- citation_coverage

### 4. Baseline comparison
The project must support comparing a current report against a checked-in baseline.

The comparison must highlight:
- case status flips (pass -> fail)
- pass-rate drop
- citation coverage drop
- keyword score drop beyond threshold
- latency increase beyond threshold

### 5. Optional token-budget debug capture
The runner may collect token-budget debug data when exposed by the API, but this must remain optional and not break the normal chat API contract for clients.

## Proposed case schemas

### Retrieval case
```json
{
  "id": "retrieval_leave_policy_001",
  "type": "retrieval",
  "query": "请假制度",
  "top_k": 5,
  "score_threshold": 0.2,
  "filters": {},
  "expected": {
    "min_results": 1,
    "must_hit_document_ids": ["doc_1"],
    "must_hit_chunk_ids": ["chk_1"],
    "must_hit_titles": ["员工手册"]
  },
  "tags": ["rag", "policy"]
}
```

### Chat case
```json
{
  "id": "chat_leave_policy_001",
  "type": "chat",
  "message": "请假制度是什么？",
  "session_id": "eval-chat-leave-policy-001",
  "use_retrieval": true,
  "expected": {
    "answer_keywords": ["请假", "申请"],
    "forbidden_keywords": ["无法回答"],
    "min_citations": 1,
    "must_cite_document_ids": ["doc_1"],
    "must_cite_titles": ["员工手册"],
    "must_reference_page_numbers": [2]
  },
  "tags": ["rag", "citations"]
}
```

## Report schema
Each run should produce a report with:
- suite name
- timestamp
- base URL
- git commit if available
- summary metrics
- per-case results

Each per-case result should contain:
- case_id
- status
- request snapshot
- response snapshot (trimmed if needed)
- metrics
- failures

## Baseline policy
Recommended initial thresholds:
- pass_rate must not decrease
- citation_coverage drop > 0.05 is a regression
- avg_keyword_score drop > 0.05 is a regression
- avg_latency_ms increase > 30% is a regression warning or failure depending on CLI flag

## CLI shape
Examples:
- `python scripts/eval/runner.py --suite retrieval`
- `python scripts/eval/runner.py --suite chat --base-url http://127.0.0.1:8000`
- `python scripts/eval/runner.py --suite all --report-out scripts/eval/reports/latest.json`
- `python scripts/eval/compare.py --current current.json --baseline scripts/eval/baselines/chat_baseline.json`

## Acceptance criteria
- repo contains fixed retrieval and chat case files
- runner can execute both suites against running service
- runner emits JSON report
- compare script can fail on regression
- no heavy external eval framework is required
