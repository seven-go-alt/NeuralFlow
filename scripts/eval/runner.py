from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from scripts.eval.evaluator import AgentEvaluator

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_URL = os.getenv("EVAL_BASE_URL", "http://127.0.0.1:8001")
DEFAULT_TIMEOUT = float(os.getenv("EVAL_HTTP_TIMEOUT", "30"))


@dataclass
class CaseResult:
    case_id: str
    suite: str
    status: str
    request: dict[str, Any]
    response: dict[str, Any]
    metrics: dict[str, Any]
    failures: list[str]
    tags: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "suite": self.suite,
            "status": self.status,
            "request": self.request,
            "response": self.response,
            "metrics": self.metrics,
            "failures": self.failures,
            "tags": self.tags,
        }


def load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT)
            .decode()
            .strip()
        )
    except Exception:
        return None


def contains_any(text: str, items: list[str]) -> bool:
    hay = text.lower()
    return any(item.lower() in hay for item in items)


def keyword_hits(text: str, items: list[str]) -> list[str]:
    hay = text.lower()
    return [item for item in items if item.lower() in hay]


async def run_retrieval_case(
    client: httpx.AsyncClient, base_url: str, case: dict[str, Any]
) -> CaseResult:
    payload = {
        "query": case["query"],
        "top_k": case.get("top_k", 5),
        "score_threshold": case.get("score_threshold", 0.0),
        "filters": case.get("filters", {}),
    }
    start = time.perf_counter()
    response = await client.post(f"{base_url}/api/retrieval/search", json=payload)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    failures: list[str] = []
    body: dict[str, Any] = {}

    if response.status_code != 200:
        failures.append(f"unexpected status {response.status_code}")
    else:
        body = response.json()

    results = body.get("results", []) if body else []
    titles = [((item.get("source") or {}).get("title") or "") for item in results]
    document_ids = [item.get("document_id") for item in results]
    chunk_ids = [item.get("chunk_id") for item in results]
    expected = case.get("expected", {})

    min_results = expected.get("min_results")
    max_results = expected.get("max_results")
    if min_results is not None and len(results) < min_results:
        failures.append(f"result_count {len(results)} < min_results {min_results}")
    if max_results is not None and len(results) > max_results:
        failures.append(f"result_count {len(results)} > max_results {max_results}")

    missing_doc_ids = [
        item for item in expected.get("must_hit_document_ids", []) if item not in document_ids
    ]
    missing_chunk_ids = [
        item for item in expected.get("must_hit_chunk_ids", []) if item not in chunk_ids
    ]
    missing_titles = [item for item in expected.get("must_hit_titles", []) if item not in titles]
    if missing_doc_ids:
        failures.append(f"missing document ids: {missing_doc_ids}")
    if missing_chunk_ids:
        failures.append(f"missing chunk ids: {missing_chunk_ids}")
    if missing_titles:
        failures.append(f"missing titles: {missing_titles}")

    metrics = {
        "latency_ms": latency_ms,
        "result_count": len(results),
        "document_hit": not missing_doc_ids if expected.get("must_hit_document_ids") else None,
        "chunk_hit": not missing_chunk_ids if expected.get("must_hit_chunk_ids") else None,
        "title_hit": not missing_titles if expected.get("must_hit_titles") else None,
        "top1_score": results[0].get("score") if results else None,
    }

    return CaseResult(
        case_id=case["id"],
        suite="retrieval",
        status="passed" if not failures else "failed",
        request=payload,
        response={"query": body.get("query"), "results": results[:5]},
        metrics=metrics,
        failures=failures,
        tags=case.get("tags", []),
    )


async def run_chat_case(
    client: httpx.AsyncClient, base_url: str, case: dict[str, Any]
) -> CaseResult:
    payload = {
        "session_id": case.get("session_id", f"eval-{case['id']}"),
        "message": case["message"],
        "use_retrieval": case.get("use_retrieval", True),
    }
    if case.get("retrieval_options"):
        payload["retrieval_options"] = case["retrieval_options"]

    start = time.perf_counter()
    response = await client.post(f"{base_url}/chat", json=payload)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    failures: list[str] = []
    body: dict[str, Any] = {}

    if response.status_code != 200:
        failures.append(f"unexpected status {response.status_code}")
    else:
        body = response.json()

    reply = body.get("reply", "") if body else ""
    citations = body.get("citations", []) if body else []
    expected = case.get("expected", {})

    answer_keywords = expected.get("answer_keywords", [])
    forbidden_keywords = expected.get("forbidden_keywords", [])
    hits = keyword_hits(reply, answer_keywords)
    keyword_score = AgentEvaluator.keyword_score(reply, answer_keywords)

    if answer_keywords and len(hits) < len(answer_keywords):
        failures.append(f"missing answer keywords: {[k for k in answer_keywords if k not in hits]}")
    if forbidden_keywords and contains_any(reply, forbidden_keywords):
        failures.append("forbidden keyword found in reply")

    min_citations = expected.get("min_citations")
    max_citations = expected.get("max_citations")
    if min_citations is not None and len(citations) < min_citations:
        failures.append(f"citation_count {len(citations)} < min_citations {min_citations}")
    if max_citations is not None and len(citations) > max_citations:
        failures.append(f"citation_count {len(citations)} > max_citations {max_citations}")

    citation_document_ids = [item.get("document_id") for item in citations]
    citation_titles = [item.get("label") for item in citations]
    citation_pages = [item.get("page_number") for item in citations]

    missing_cite_doc_ids = [
        item
        for item in expected.get("must_cite_document_ids", [])
        if item not in citation_document_ids
    ]
    missing_cite_titles = [
        item for item in expected.get("must_cite_titles", []) if item not in citation_titles
    ]
    missing_cite_pages = [
        item
        for item in expected.get("must_reference_page_numbers", [])
        if item not in citation_pages
    ]
    if missing_cite_doc_ids:
        failures.append(f"missing cited document ids: {missing_cite_doc_ids}")
    if missing_cite_titles:
        failures.append(f"missing cited titles: {missing_cite_titles}")
    if missing_cite_pages:
        failures.append(f"missing cited pages: {missing_cite_pages}")

    metrics = {
        "latency_ms": latency_ms,
        "keyword_score": keyword_score,
        "keyword_hits": hits,
        "citation_count": len(citations),
        "citation_hit_document": not missing_cite_doc_ids
        if expected.get("must_cite_document_ids")
        else None,
        "citation_hit_title": not missing_cite_titles if expected.get("must_cite_titles") else None,
        "citation_hit_page": not missing_cite_pages
        if expected.get("must_reference_page_numbers")
        else None,
    }

    return CaseResult(
        case_id=case["id"],
        suite="chat",
        status="passed" if not failures else "failed",
        request=payload,
        response={
            "intent": body.get("intent"),
            "reply": reply,
            "citations": citations,
        },
        metrics=metrics,
        failures=failures,
        tags=case.get("tags", []),
    )


def summarize(results: list[CaseResult], suite: str) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for item in results if item.status == "passed")
    failed = total - passed
    latencies: list[float] = []
    for item in results:
        v = item.metrics.get("latency_ms")
        if isinstance(v, (int, float)):
            latencies.append(v)
    avg_latency_ms = round(sum(latencies) / len(latencies), 2) if latencies else None

    summary: dict[str, Any] = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "avg_latency_ms": avg_latency_ms,
    }

    if suite in {"chat", "all"}:
        chat_results = [item for item in results if item.suite == "chat"]
        if chat_results:
            keyword_scores = [item.metrics.get("keyword_score", 0.0) for item in chat_results]
            citation_cases = [
                item for item in chat_results if item.metrics.get("citation_count") is not None
            ]
            citation_covered = [
                item for item in citation_cases if (item.metrics.get("citation_count") or 0) > 0
            ]
            summary["avg_keyword_score"] = round(sum(keyword_scores) / len(keyword_scores), 4)
            summary["citation_coverage"] = (
                round(len(citation_covered) / len(citation_cases), 4) if citation_cases else None
            )

    if suite in {"retrieval", "all"}:
        retrieval_results = [item for item in results if item.suite == "retrieval"]
        if retrieval_results:
            doc_checks = [
                item.metrics.get("document_hit")
                for item in retrieval_results
                if item.metrics.get("document_hit") is not None
            ]
            chunk_checks = [
                item.metrics.get("chunk_hit")
                for item in retrieval_results
                if item.metrics.get("chunk_hit") is not None
            ]
            title_checks = [
                item.metrics.get("title_hit")
                for item in retrieval_results
                if item.metrics.get("title_hit") is not None
            ]
            summary["document_hit_rate"] = (
                round(sum(1 for x in doc_checks if x) / len(doc_checks), 4) if doc_checks else None
            )
            summary["chunk_hit_rate"] = (
                round(sum(1 for x in chunk_checks if x) / len(chunk_checks), 4)
                if chunk_checks
                else None
            )
            summary["title_hit_rate"] = (
                round(sum(1 for x in title_checks if x) / len(title_checks), 4)
                if title_checks
                else None
            )

    return summary


async def run_suite(suite: str, base_url: str, timeout: float) -> dict[str, Any]:
    results: list[CaseResult] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        if suite in {"retrieval", "all"}:
            for case in load_cases(EVAL_DIR / "test_cases" / "retrieval_cases.json"):
                results.append(await run_retrieval_case(client, base_url, case))
        if suite in {"chat", "all"}:
            for case in load_cases(EVAL_DIR / "test_cases" / "chat_cases.json"):
                results.append(await run_chat_case(client, base_url, case))

    return {
        "suite": suite,
        "generated_at": datetime.now(UTC).isoformat(),
        "base_url": base_url,
        "git_commit": get_git_commit(),
        "summary": summarize(results, suite),
        "results": [item.to_dict() for item in results],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeuralFlow Eval v1 suites")
    parser.add_argument("--suite", choices=["retrieval", "chat", "all"], default="all")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--report-out", help="Optional path to write JSON report")
    parser.add_argument("--fail-on-case-failure", action="store_true")
    return parser.parse_args()


async def _main() -> int:
    args = parse_args()
    report = await run_suite(args.suite, args.base_url.rstrip("/"), args.timeout)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.report_out:
        out_path = Path(args.report_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")

    if args.fail_on_case_failure and report["summary"]["failed"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
