"""Retrieval smoke test for the seeded eval corpus.

For every positive eval case, runs top-k retrieval and checks whether
any result chunk carries the expected canonical_doc_id. Exits non-zero
below 90% hit rate.

Usage (from repo root):
    uv run python -m scripts.eval.smoke_retrieval
    uv run python -m scripts.eval.smoke_retrieval --top-k 5
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from app.config import get_settings
from app.db.session import SessionLocal, init_db
from app.documents.repository import DocumentRepository
from app.evals.datasets import EvalCase, load_cases
from app.retrieval.schemas import RetrievalRequest
from app.retrieval.service import RetrievalService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieval smoke test for eval corpus")
    parser.add_argument(
        "--datasets-dir",
        default="data/eval/datasets",
        help="Directory containing .jsonl dataset files (default: data/eval/datasets)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k results per query (default: 5)")
    parser.add_argument(
        "--tenant", default=None, help="Tenant id (default: settings.tenant_default_id)"
    )
    return parser.parse_args()


async def _main() -> int:
    args = parse_args()
    settings = get_settings()
    tenant_id = args.tenant or settings.tenant_default_id
    cases: list[EvalCase] = []
    for path in sorted(Path(args.datasets_dir).glob("*.jsonl")):
        cases.extend(load_cases(path))
    if not cases:
        print(f"error: no .jsonl files found in {args.datasets_dir}", file=sys.stderr)
        return 1
    positives = [case for case in cases if case.should_answer]

    init_db()
    db = SessionLocal()
    hits = 0
    misses: list[str] = []
    try:
        service = RetrievalService(document_repo=DocumentRepository(db))
        for case in positives:
            response = await service.search(
                tenant_id, RetrievalRequest(query=case.question, top_k=args.top_k)
            )
            found = {result.metadata.get("canonical_doc_id") for result in response.results}
            if found & set(case.expected_doc_ids):
                hits += 1
            else:
                got = sorted(str(item) for item in found if item)
                misses.append(f"{case.id} expected={list(case.expected_doc_ids)} got={got}")
    finally:
        db.close()

    for line in misses:
        print("MISS", line)
    rate = hits / len(positives) if positives else 0.0
    print(f"hit@{args.top_k}: {hits}/{len(positives)} = {rate:.1%}")
    return 0 if rate >= 0.9 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
