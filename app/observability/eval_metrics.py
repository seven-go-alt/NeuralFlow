from __future__ import annotations

from typing import TYPE_CHECKING

from prometheus_client import Gauge, Histogram

if TYPE_CHECKING:
    from app.evals.metrics import EvalMetrics
    from app.rag.answer_evaluator import AnswerEvalResult

eval_retrieval_hit_rate = Gauge("rag_eval_retrieval_hit_rate", "Retrieval hit rate from latest eval run")
eval_citation_accuracy = Gauge("rag_eval_citation_accuracy", "Citation accuracy from latest eval run")
eval_answer_relevance = Gauge("rag_eval_answer_relevance", "Average answer relevance from latest eval run")
eval_answer_faithfulness = Gauge("rag_eval_answer_faithfulness", "Average answer faithfulness from latest eval run")
eval_answer_completeness = Gauge("rag_eval_answer_completeness", "Average answer completeness from latest eval run")
eval_answer_score = Histogram(
    "rag_eval_answer_score",
    "Answer evaluation scores",
    buckets=[0.2, 0.4, 0.6, 0.8, 1.0],
)


def record_eval_metrics(
    metrics: EvalMetrics,
    answer_scores: list[AnswerEvalResult],
) -> None:
    eval_retrieval_hit_rate.set(metrics.retrieval_hit_rate)
    eval_citation_accuracy.set(metrics.citation_accuracy)
    if answer_scores:
        avg_rel = sum(s.relevance for s in answer_scores) / len(answer_scores)
        avg_faith = sum(s.faithfulness for s in answer_scores) / len(answer_scores)
        avg_comp = sum(s.completeness for s in answer_scores) / len(answer_scores)
        eval_answer_relevance.set(avg_rel)
        eval_answer_faithfulness.set(avg_faith)
        eval_answer_completeness.set(avg_comp)
        for s in answer_scores:
            eval_answer_score.observe(s.overall)
