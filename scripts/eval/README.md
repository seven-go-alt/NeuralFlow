# Eval v1

Lightweight in-repo evaluation for NeuralFlow.

## Goals

- Run fixed retrieval/chat cases against the live HTTP API
- Produce JSON reports for inspection and CI artifacts
- Compare current results to a baseline to catch regressions
- Reuse existing `/chat`, `/api/retrieval/search`, citations, and token-budget related facilities

## Suites

- `retrieval` → `POST /api/retrieval/search`
- `chat` → `POST /chat`
- `all` → both suites

## Files

- `test_cases/retrieval_cases.json`
- `test_cases/chat_cases.json`
- `runner.py`
- `compare.py`
- `evaluator.py` (existing helper functions)

## Run

```bash
python scripts/eval/runner.py --suite retrieval --base-url http://127.0.0.1:8001
python scripts/eval/runner.py --suite chat --base-url http://127.0.0.1:8001
python scripts/eval/runner.py --suite all --base-url http://127.0.0.1:8001 --report-out scripts/eval/reports/latest.json
```

## Compare with baseline

```bash
python scripts/eval/compare.py \
  --current scripts/eval/reports/latest.json \
  --baseline scripts/eval/baselines/latest.json
```

## Notes

- These evals are intentionally deterministic-first.
- LLM judge scoring is optional and not part of the required CI gate.
- If the environment has no seeded RAG documents, retrieval/chat RAG cases may fail for environmental reasons rather than code regressions.
