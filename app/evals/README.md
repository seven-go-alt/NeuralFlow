# Eval Module

RAG evaluation framework with structured datasets, metrics computation, and report generation.

## Usage

```python
from app.evals.runner import run_eval, build_eval_report
from app.evals.metrics import aggregate_metrics

def retrieve(query: str, top_k: int) -> list[dict]:
    ...

def answer(query: str, context: str) -> str | None:
    ...

results = await run_eval("evals/rag_cases.jsonl", retrieve, answer)
metrics = aggregate_metrics(results)
report = build_eval_report(results, metrics)
print(report)
```

## Components

### `datasets.py`
- `EvalCase` dataclass: `id`, `question`, `expected_keywords`, `expected_doc_ids`, `should_answer`
- `load_cases(path)` — loads JSONL format, skips empty lines

### `metrics.py`
- `compute_retrieval_hit()` — whether expected doc IDs appear in retrieval results
- `compute_citation_match()` — whether expected doc IDs are mentioned in answer text
- `compute_keyword_coverage()` — ratio of expected keywords found in answer
- `compute_no_answer_correct()` — whether answer/refusal matches `should_answer`
- `aggregate_metrics()` — aggregates `CaseResult` list into `EvalMetrics`

### `runner.py`
- `run_eval(cases_path, retrieve_fn, answer_fn, top_k)` — runs full eval pipeline
- `build_eval_report(results, metrics)` — generates markdown report

### API evaluation runs
- `POST /api/v1/eval/run` accepts a controlled `dataset_id` (not an arbitrary filesystem path)
- Runs are queued through Celery on the `evals` queue and persist `queued/running/completed/failed` status
- `GET /api/v1/eval/runs` and `GET /api/v1/eval/runs/{run_id}` are tenant-scoped
- Configure `EVAL_DATASET_DIR`, `EVAL_MAX_DATASET_MB`, and `EVAL_MAX_CASES` for API dataset limits

Start a worker with the evaluation queue enabled:

```bash
celery -A worker.celery_app worker --queues documents,evals,celery
```

The local CLI intentionally keeps accepting explicit paths for offline development; the HTTP API does not.
