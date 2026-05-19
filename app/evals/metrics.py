from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EvalMetrics:
    total_cases: int = 0
    retrieval_hits: int = 0
    citation_matches: int = 0
    keyword_coverage_sum: float = 0.0
    no_answer_correct: int = 0
    no_answer_total: int = 0
    total_latency_ms: float = 0.0
    answer_relevance_sum: float = 0.0
    answer_faithfulness_sum: float = 0.0
    answer_completeness_sum: float = 0.0
    answer_count: int = 0

    @property
    def retrieval_hit_rate(self) -> float:
        if self.total_cases == 0:
            return 0.0
        return self.retrieval_hits / self.total_cases

    @property
    def citation_accuracy(self) -> float:
        if self.total_cases == 0:
            return 0.0
        return self.citation_matches / self.total_cases

    @property
    def keyword_coverage(self) -> float:
        if self.total_cases == 0:
            return 0.0
        return self.keyword_coverage_sum / self.total_cases

    @property
    def no_answer_accuracy(self) -> float:
        if self.no_answer_total == 0:
            return 1.0
        return self.no_answer_correct / self.no_answer_total

    @property
    def average_latency_ms(self) -> float:
        if self.total_cases == 0:
            return 0.0
        return self.total_latency_ms / self.total_cases

    @property
    def average_answer_relevance(self) -> float:
        if self.answer_count == 0:
            return 0.0
        return self.answer_relevance_sum / self.answer_count

    @property
    def average_answer_faithfulness(self) -> float:
        if self.answer_count == 0:
            return 0.0
        return self.answer_faithfulness_sum / self.answer_count

    @property
    def average_answer_completeness(self) -> float:
        if self.answer_count == 0:
            return 0.0
        return self.answer_completeness_sum / self.answer_count


@dataclass(slots=True)
class CaseResult:
    case_id: str
    question: str
    retrieved_doc_ids: tuple[str, ...]
    retrieved_contents: tuple[str, ...]
    answer: str | None
    latency_ms: float
    retrieval_hit: bool
    citation_match: bool
    keyword_coverage: float
    no_answer_correct: bool | None
    answer_relevance: float | None = None
    answer_faithfulness: float | None = None
    answer_completeness: float | None = None


def compute_retrieval_hit(
    retrieved_doc_ids: tuple[str, ...], expected_doc_ids: tuple[str, ...]
) -> bool:
    if not expected_doc_ids:
        return True
    return bool(set(expected_doc_ids) & set(retrieved_doc_ids))


def compute_citation_match(
    answer: str | None,
    expected_doc_ids: tuple[str, ...],
    retrieved_doc_ids: tuple[str, ...],
) -> bool:
    if not expected_doc_ids or not answer:
        return not expected_doc_ids
    cited_set = set(expected_doc_ids) & set(retrieved_doc_ids)
    if not cited_set:
        return False
    return any(doc_id.lower() in answer.lower() for doc_id in cited_set)


def compute_keyword_coverage(all_retrieved_text: str, expected_keywords: tuple[str, ...]) -> float:
    if not expected_keywords:
        return 1.0
    lower_text = all_retrieved_text.lower()
    matched = sum(1 for kw in expected_keywords if kw.lower() in lower_text)
    return matched / len(expected_keywords)


def compute_no_answer_correct(should_answer: bool, actual_answer: str | None) -> bool | None:
    if should_answer:
        return actual_answer is not None
    return actual_answer is None


def aggregate_metrics(results: list[CaseResult]) -> EvalMetrics:
    metrics = EvalMetrics(total_cases=len(results))
    for r in results:
        if r.retrieval_hit:
            metrics.retrieval_hits += 1
        if r.citation_match:
            metrics.citation_matches += 1
        metrics.keyword_coverage_sum += r.keyword_coverage
        metrics.total_latency_ms += r.latency_ms
        if r.no_answer_correct is not None:
            metrics.no_answer_total += 1
            if r.no_answer_correct:
                metrics.no_answer_correct += 1
        if r.answer_relevance is not None:
            metrics.answer_count += 1
            metrics.answer_relevance_sum += r.answer_relevance
            metrics.answer_faithfulness_sum += r.answer_faithfulness or 0.0
            metrics.answer_completeness_sum += r.answer_completeness or 0.0
    return metrics
