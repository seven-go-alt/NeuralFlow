from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.evals.datasets import load_cases

if TYPE_CHECKING:
    from app.rag.answer_evaluator import AnswerEvalResult
from app.evals.metrics import (
    CaseResult,
    EvalMetrics,
    compute_citation_match,
    compute_first_relevant_rank,
    compute_keyword_coverage,
    compute_no_answer_correct,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_retrieval_hit,
)

RetrieveFn = Callable[[str, int], list[dict[str, Any]]]
AnswerFn = Callable[[str, str], str | None]
AnswerEvalFn = Callable[[str, str, list[str]], "AnswerEvalResult | None"]


async def run_eval(
    cases_path: str | Path,
    retrieve_fn: RetrieveFn,
    answer_fn: AnswerFn,
    top_k: int = 5,
    answer_eval_fn: AnswerEvalFn | None = None,
) -> list[CaseResult]:
    cases = load_cases(cases_path)
    results: list[CaseResult] = []

    for case in cases:
        start = time.perf_counter()

        retrieved = retrieve_fn(case.question, top_k)
        retrieved_doc_ids = tuple(r.get("document_id", "") for r in retrieved)
        retrieved_contents = tuple(r.get("content", "") for r in retrieved)

        all_text = " ".join(retrieved_contents)
        answer = answer_fn(case.question, all_text)

        latency_ms = (time.perf_counter() - start) * 1000

        answer_eval: AnswerEvalResult | None = None
        if answer_eval_fn is not None and answer is not None:
            answer_eval = answer_eval_fn(case.question, answer, list(retrieved_contents))

        effective_k = len(retrieved_doc_ids) if len(retrieved_doc_ids) > 0 else top_k
        results.append(
            CaseResult(
                case_id=case.id,
                question=case.question,
                retrieved_doc_ids=retrieved_doc_ids,
                retrieved_contents=retrieved_contents,
                answer=answer,
                latency_ms=latency_ms,
                retrieval_hit=compute_retrieval_hit(retrieved_doc_ids, case.expected_doc_ids),
                citation_match=compute_citation_match(
                    answer, case.expected_doc_ids, retrieved_doc_ids
                ),
                keyword_coverage=compute_keyword_coverage(all_text, case.expected_keywords),
                no_answer_correct=compute_no_answer_correct(case.should_answer, answer),
                first_relevant_rank=compute_first_relevant_rank(
                    retrieved_doc_ids, case.expected_doc_ids
                ),
                precision_at_k=compute_precision_at_k(
                    retrieved_doc_ids, case.expected_doc_ids, effective_k
                ),
                recall_at_k=compute_recall_at_k(
                    retrieved_doc_ids, case.expected_doc_ids, effective_k
                ),
            )
        )

        if answer_eval is not None:
            results[-1].answer_relevance = answer_eval.relevance
            results[-1].answer_faithfulness = answer_eval.faithfulness
            results[-1].answer_completeness = answer_eval.completeness

    return results


def build_eval_report(
    results: list[CaseResult],
    metrics: EvalMetrics,
) -> str:
    lines: list[str] = []
    lines.append("# RAG Eval Report")
    lines.append("")
    lines.append(f"- **Total cases**: {metrics.total_cases}")
    lines.append(f"- **Retrieval Hit Rate**: {metrics.retrieval_hit_rate:.1%}")
    lines.append(f"- **Citation Accuracy**: {metrics.citation_accuracy:.1%}")
    lines.append(f"- **Keyword Coverage**: {metrics.keyword_coverage:.1%}")
    lines.append(f"- **No-Answer Accuracy**: {metrics.no_answer_accuracy:.1%}")
    lines.append(f"- **Average Latency**: {metrics.average_latency_ms:.1f} ms")
    lines.append(f"- **Mean Reciprocal Rank (MRR)**: {metrics.mean_reciprocal_rank:.4f}")
    lines.append(f"- **Avg Precision@k**: {metrics.average_precision_at_k:.1%}")
    lines.append(f"- **Avg Recall@k**: {metrics.average_recall_at_k:.1%}")
    lines.append("")
    lines.append("## Per-Case Details")
    lines.append("")
    lines.append(
        "| Case ID | Retrieval Hit | Citation Match | Keyword Cov | No-Answer Correct | Latency (ms) | Rank |"
    )
    lines.append(
        "|---------|--------------|---------------|-------------|-------------------|-------------|------|"
    )
    for r in results:
        no_ans = "✓" if r.no_answer_correct else "✗" if r.no_answer_correct is False else "—"
        rank_str = str(r.first_relevant_rank) if r.first_relevant_rank > 0 else "—"
        lines.append(
            f"| {r.case_id} "
            f"| {'✓' if r.retrieval_hit else '✗'} "
            f"| {'✓' if r.citation_match else '✗'} "
            f"| {r.keyword_coverage:.0%} "
            f"| {no_ans} "
            f"| {r.latency_ms:.1f} "
            f"| {rank_str} |"
        )
    lines.append("")
    return "\n".join(lines)
