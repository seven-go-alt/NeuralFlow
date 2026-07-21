from __future__ import annotations

from pathlib import Path

import pytest

from app.evals.comparison import ComparisonResult, compare_runs, format_comparison_table
from app.evals.datasets import load_cases
from app.evals.metrics import (
    CaseResult,
    EvalMetrics,
    aggregate_metrics,
    compute_first_relevant_rank,
    compute_precision_at_k,
    compute_recall_at_k,
)
from app.evals.runner import run_eval

DATASET_DIR = Path(__file__).parent.parent / "data" / "eval" / "datasets"
DATASET_PATH = DATASET_DIR / "rag_quality_50.jsonl"
FINANCE_DATASET_PATH = DATASET_DIR / "rag_finance_30.jsonl"
TECHNICAL_DATASET_PATH = DATASET_DIR / "rag_technical_30.jsonl"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def test_dataset_loads_all_50_cases() -> None:
    cases = load_cases(str(DATASET_PATH))
    assert len(cases) == 50
    ids = {c.id for c in cases}
    assert len(ids) == 50, "Duplicate case IDs found"


def test_dataset_has_negative_tests() -> None:
    cases = load_cases(str(DATASET_PATH))
    no_answer = [c for c in cases if not c.should_answer]
    assert len(no_answer) >= 3, "Should have at least 3 negative test cases"
    for c in no_answer:
        assert not c.expected_doc_ids, f"Negative test {c.id} should not have expected_doc_ids"
        assert not c.expected_keywords, f"Negative test {c.id} should not have expected_keywords"


def test_dataset_has_keyword_heavy_cases() -> None:
    cases = load_cases(str(DATASET_PATH))
    keyword_heavy = [c for c in cases if len(c.expected_keywords) >= 4]
    assert len(keyword_heavy) >= 3, "Should have at least 3 keyword-heavy test cases"


def test_dataset_has_numerical_cases() -> None:
    cases = load_cases(str(DATASET_PATH))
    numerical = [
        c for c in cases if any(any(ch.isdigit() for ch in kw) for kw in c.expected_keywords)
    ]
    assert len(numerical) >= 3, "Should have at least 3 numerical reasoning test cases"


# ---------------------------------------------------------------------------
# Ranking metric helpers
# ---------------------------------------------------------------------------


class TestComputeFirstRelevantRank:
    def test_first_position(self) -> None:
        assert compute_first_relevant_rank(("d1", "d2", "d3"), ("d1",)) == 1

    def test_second_position(self) -> None:
        assert compute_first_relevant_rank(("d1", "d2", "d3"), ("d2",)) == 2

    def test_last_position(self) -> None:
        assert compute_first_relevant_rank(("d1", "d2", "d3"), ("d3",)) == 3

    def test_no_match(self) -> None:
        assert compute_first_relevant_rank(("d1", "d2"), ("d3",)) == 0

    def test_empty_expected(self) -> None:
        assert compute_first_relevant_rank(("d1",), ()) == 0

    def test_multi_expected_returns_lowest_rank(self) -> None:
        assert compute_first_relevant_rank(("d1", "d2", "d3"), ("d3", "d1")) == 1


class TestComputePrecisionAtK:
    def test_all_relevant(self) -> None:
        assert compute_precision_at_k(("d1", "d2"), ("d1", "d2"), 2) == 1.0

    def test_half_relevant(self) -> None:
        assert compute_precision_at_k(("d1", "d2", "d3"), ("d1",), 3) == pytest.approx(1.0 / 3)

    def test_none_relevant(self) -> None:
        assert compute_precision_at_k(("d1", "d2"), ("d3",), 2) == 0.0

    def test_empty_expected(self) -> None:
        assert compute_precision_at_k(("d1", "d2"), (), 2) == 1.0

    def test_k_is_zero(self) -> None:
        assert compute_precision_at_k((), ("d1",), 0) == 0.0

    def test_k_larger_than_retrieved(self) -> None:
        assert compute_precision_at_k(("d1",), ("d1",), 5) == 0.2


class TestComputeRecallAtK:
    def test_all_retrieved(self) -> None:
        assert compute_recall_at_k(("d1", "d2"), ("d1", "d2"), 2) == 1.0

    def test_half_retrieved(self) -> None:
        assert compute_recall_at_k(("d1",), ("d1", "d2"), 2) == 0.5

    def test_empty_expected(self) -> None:
        assert compute_recall_at_k(("d1",), (), 2) == 1.0

    def test_none_retrieved(self) -> None:
        assert compute_recall_at_k(("d3",), ("d1", "d2"), 2) == 0.0


# ---------------------------------------------------------------------------
# Full eval run with mock data
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_eval_with_mock_data() -> None:
    """Run the full 50-case dataset with mock retrieval and answer, verify metrics."""

    def mock_retrieve(query: str, top_k: int) -> list[dict]:
        return [
            {
                "document_id": "doc_hr_leave",
                "content": "Annual leave policy: 20 days for 5+ years.",
                "score": 0.95,
            },
            {
                "document_id": "doc_hr_sick",
                "content": "Sick leave: 3 days without medical cert.",
                "score": 0.88,
            },
            {
                "document_id": "doc_finance_expense",
                "content": "Expense reports over $5k need manager approval.",
                "score": 0.82,
            },
        ]

    def mock_answer(query: str, context: str) -> tuple[str | None, dict]:
        return f"Answer based on: {context[:80]}", {}

    results = await run_eval(str(DATASET_PATH), mock_retrieve, mock_answer, top_k=3)
    metrics = aggregate_metrics(results)

    assert metrics.total_cases == 50
    assert 0 <= metrics.retrieval_hit_rate <= 1.0
    assert 0 <= metrics.keyword_coverage <= 1.0
    assert metrics.average_latency_ms >= 0
    assert 0 <= metrics.mean_reciprocal_rank <= 1.0
    assert 0 <= metrics.average_precision_at_k <= 1.0
    assert 0 <= metrics.average_recall_at_k <= 1.0


@pytest.mark.asyncio
async def test_eval_empty_results() -> None:
    """Edge case: retrieval returns empty for all cases."""

    def empty_retrieve(query: str, top_k: int) -> list[dict]:
        return []

    def mock_answer(query: str, context: str) -> tuple[str | None, dict]:
        return "some answer", {}

    results = await run_eval(str(DATASET_PATH), empty_retrieve, mock_answer, top_k=3)

    results = await run_eval(str(DATASET_PATH), empty_retrieve, mock_answer, top_k=3)
    metrics = aggregate_metrics(results)

    assert metrics.total_cases == 50
    # 4 no-answer cases have empty expected_doc_ids, so they count as retrieval hits
    assert metrics.retrieval_hits == 4
    # 4 no-answer cases have empty expected_keywords, so keyword_coverage=1.0 for those
    assert metrics.keyword_coverage == 4.0 / 50.0
    assert metrics.mean_reciprocal_rank == 0.0
    # 4 no-answer cases have empty expected_doc_ids, so precision/recall=1.0 for those
    assert metrics.average_precision_at_k == 4.0 / 50.0
    assert metrics.average_recall_at_k == 4.0 / 50.0


@pytest.mark.asyncio
async def test_eval_no_answer_scenarios() -> None:
    """Verify that should_answer=False cases are handled correctly."""

    def retrieve_fn(query: str, top_k: int) -> list[dict]:
        return []

    def answer_fn(query: str, context: str) -> tuple[str | None, dict]:
        if (
            "favorite color" in query
            or "joke" in query
            or "Super Bowl" in query
            or "stock price" in query
        ):
            return None, {}
        return "some answer", {}

    results = await run_eval(str(DATASET_PATH), retrieve_fn, answer_fn, top_k=3)
    metrics = aggregate_metrics(results)

    # All no_answer_correct evaluations are non-None, so no_answer_total == total_cases
    assert metrics.no_answer_accuracy == 1.0


@pytest.mark.asyncio
async def test_eval_partial_keyword_coverage() -> None:
    """Edge case: only some keywords are covered in retrieved text."""

    def retrieve_fn(query: str, top_k: int) -> list[dict]:
        return [
            {"document_id": "doc_1", "content": "annual leave policy details.", "score": 0.9},
        ]

    def answer_fn(query: str, context: str) -> tuple[str | None, dict]:
        return "response", {}

    results = await run_eval(str(DATASET_PATH), retrieve_fn, answer_fn, top_k=1)
    metrics = aggregate_metrics(results)

    assert metrics.total_cases == 50
    assert metrics.keyword_coverage < 1.0


# ---------------------------------------------------------------------------
# aggregate_metrics with ranking fields
# ---------------------------------------------------------------------------


class TestAggregateMetricsWithRanking:
    def test_mrr_with_varying_ranks(self) -> None:
        results = [
            CaseResult(
                case_id="c1",
                question="q1",
                retrieved_doc_ids=("d1",),
                retrieved_contents=("c1",),
                answer="a",
                latency_ms=10.0,
                retrieval_hit=True,
                citation_match=True,
                keyword_coverage=1.0,
                no_answer_correct=True,
                first_relevant_rank=1,
                precision_at_k=1.0,
                recall_at_k=1.0,
            ),
            CaseResult(
                case_id="c2",
                question="q2",
                retrieved_doc_ids=("d1", "d2"),
                retrieved_contents=("c1", "c2"),
                answer="a",
                latency_ms=20.0,
                retrieval_hit=True,
                citation_match=True,
                keyword_coverage=0.5,
                no_answer_correct=True,
                first_relevant_rank=2,
                precision_at_k=0.5,
                recall_at_k=0.5,
            ),
            CaseResult(
                case_id="c3",
                question="q3",
                retrieved_doc_ids=("d3",),
                retrieved_contents=("c3",),
                answer=None,
                latency_ms=5.0,
                retrieval_hit=False,
                citation_match=False,
                keyword_coverage=0.0,
                no_answer_correct=None,
                first_relevant_rank=0,
                precision_at_k=0.0,
                recall_at_k=0.0,
            ),
        ]
        m = aggregate_metrics(results)
        assert m.total_cases == 3
        # MRR should be (1/1 + 1/2 + 0/1) / 3 = (1 + 0.5 + 0) / 3 = 0.5
        assert m.mean_reciprocal_rank == pytest.approx(0.5)
        # Avg precision@k: (1.0 + 0.5 + 0.0) / 3 = 0.5
        assert m.average_precision_at_k == pytest.approx(0.5)
        # Avg recall@k: (1.0 + 0.5 + 0.0) / 3 = 0.5
        assert m.average_recall_at_k == pytest.approx(0.5)

    def test_backward_compat_no_ranking_fields(self) -> None:
        """CaseResult without ranking fields (all defaults) should still work."""
        results = [
            CaseResult(
                case_id="c1",
                question="q1",
                retrieved_doc_ids=("d1",),
                retrieved_contents=("c1",),
                answer="a",
                latency_ms=10.0,
                retrieval_hit=True,
                citation_match=True,
                keyword_coverage=1.0,
                no_answer_correct=True,
            ),
        ]
        m = aggregate_metrics(results)
        assert m.total_cases == 1
        assert m.retrieval_hit_rate == 1.0


# ---------------------------------------------------------------------------
# A/B comparison
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compare_runs() -> None:
    """Verify the A/B comparison produces correct results."""

    def retrieve_a(query: str, top_k: int) -> list[dict]:
        return [
            {"document_id": "doc_hr_leave", "content": "Leave policy content.", "score": 0.9},
        ]

    def answer_a(query: str, context: str) -> tuple[str | None, dict]:
        return "answer A", {}

    def retrieve_b(query: str, top_k: int) -> list[dict]:
        return [
            {"document_id": "doc_hr_leave", "content": "Leave policy content.", "score": 0.95},
            {"document_id": "doc_hr_sick", "content": "Sick leave content.", "score": 0.85},
        ]

    def answer_b(query: str, context: str) -> tuple[str | None, dict]:
        return "answer B", {}

    result = await compare_runs(
        str(DATASET_PATH),
        baseline_retrieve_fn=retrieve_a,
        baseline_answer_fn=answer_a,
        experiment_retrieve_fn=retrieve_b,
        experiment_answer_fn=answer_b,
        top_k=3,
    )

    assert isinstance(result, ComparisonResult)
    assert result.total_cases == 50
    assert isinstance(result.deltas, dict)
    assert len(result.deltas) > 0
    assert isinstance(result.winners, dict)
    assert len(result.winners) > 0


def test_comparison_result_properties() -> None:
    """Verify ComparisonResult computed fields."""
    baseline = EvalMetrics(
        total_cases=50,
        retrieval_hits=40,
        mrr_sum=30.0,
        precision_at_k_sum=35.0,
        recall_at_k_sum=25.0,
        total_latency_ms=5000.0,
    )
    experiment = EvalMetrics(
        total_cases=50,
        retrieval_hits=45,
        mrr_sum=40.0,
        precision_at_k_sum=42.0,
        recall_at_k_sum=35.0,
        total_latency_ms=8000.0,
    )
    result = ComparisonResult(baseline=baseline, experiment=experiment)

    # Experiment retrieved more hits, so retrieval_hit_rate should be winner
    assert result.deltas["retrieval_hit_rate"] == pytest.approx(0.1)
    assert result.winners["retrieval_hit_rate"] == "experiment"

    # Baseline is faster (lower latency is better)
    assert result.winners["average_latency_ms"] == "baseline"


def test_format_comparison_table_output() -> None:
    """Verify format_comparison_table returns a valid non-empty string."""
    baseline = EvalMetrics(total_cases=50, retrieval_hits=40, total_latency_ms=5000.0)
    experiment = EvalMetrics(total_cases=50, retrieval_hits=45, total_latency_ms=8000.0)
    result = ComparisonResult(baseline=baseline, experiment=experiment)

    table = format_comparison_table(result)
    assert isinstance(table, str)
    assert len(table) > 0
    assert "A/B Comparison Results" in table
    assert "Retrieval Hit Rate" in table
    assert "Avg Latency" in table
    assert "baseline" in table or "experiment" in table or "tie" in table


# ---------------------------------------------------------------------------
# Multi-dataset regression tests
# ---------------------------------------------------------------------------


def test_finance_dataset_loads_all_30_cases() -> None:
    cases = load_cases(str(FINANCE_DATASET_PATH))
    assert len(cases) == 30
    ids = {c.id for c in cases}
    assert len(ids) == 30, "Duplicate case IDs in finance dataset"


def test_finance_dataset_has_negative_tests() -> None:
    cases = load_cases(str(FINANCE_DATASET_PATH))
    no_answer = [c for c in cases if not c.should_answer]
    assert len(no_answer) >= 1, "Finance dataset should have at least 1 negative test case"
    for c in no_answer:
        assert not c.expected_doc_ids, f"Negative test {c.id} should not have expected_doc_ids"
        assert not c.expected_keywords, f"Negative test {c.id} should not have expected_keywords"


def test_finance_dataset_has_keyword_heavy_cases() -> None:
    cases = load_cases(str(FINANCE_DATASET_PATH))
    keyword_heavy = [c for c in cases if len(c.expected_keywords) >= 4]
    assert len(keyword_heavy) >= 3, (
        "Finance dataset should have at least 3 keyword-heavy test cases"
    )


def test_technical_dataset_loads_all_30_cases() -> None:
    cases = load_cases(str(TECHNICAL_DATASET_PATH))
    assert len(cases) == 30
    ids = {c.id for c in cases}
    assert len(ids) == 30, "Duplicate case IDs in technical dataset"


def test_technical_dataset_has_negative_tests() -> None:
    cases = load_cases(str(TECHNICAL_DATASET_PATH))
    no_answer = [c for c in cases if not c.should_answer]
    assert len(no_answer) >= 1, "Technical dataset should have at least 1 negative test case"
    for c in no_answer:
        assert not c.expected_doc_ids, f"Negative test {c.id} should not have expected_doc_ids"
        assert not c.expected_keywords, f"Negative test {c.id} should not have expected_keywords"


def test_technical_dataset_has_keyword_heavy_cases() -> None:
    cases = load_cases(str(TECHNICAL_DATASET_PATH))
    keyword_heavy = [c for c in cases if len(c.expected_keywords) >= 4]
    assert len(keyword_heavy) >= 3, (
        "Technical dataset should have at least 3 keyword-heavy test cases"
    )


@pytest.mark.asyncio
async def test_finance_eval_with_mock_data() -> None:
    """Run the full 30-case finance dataset with mock retrieval and answer."""

    def mock_retrieve(query: str, top_k: int) -> list[dict]:
        return [
            {
                "document_id": "doc_fin_depreciation",
                "content": "Depreciation policy details.",
                "score": 0.95,
            },
            {
                "document_id": "doc_fin_journal",
                "content": "Journal entry guidelines.",
                "score": 0.88,
            },
        ]

    def mock_answer(query: str, context: str) -> tuple[str | None, dict]:
        return f"Answer based on: {context[:80]}", {}

    results = await run_eval(str(FINANCE_DATASET_PATH), mock_retrieve, mock_answer, top_k=3)
    metrics = aggregate_metrics(results)

    assert metrics.total_cases == 30
    assert 0 <= metrics.retrieval_hit_rate <= 1.0
    assert 0 <= metrics.keyword_coverage <= 1.0
    assert metrics.average_latency_ms >= 0
    assert 0 <= metrics.mean_reciprocal_rank <= 1.0
    assert 0 <= metrics.average_precision_at_k <= 1.0
    assert 0 <= metrics.average_recall_at_k <= 1.0


@pytest.mark.asyncio
async def test_technical_eval_with_mock_data() -> None:
    """Run the full 30-case technical dataset with mock retrieval and answer."""

    def mock_retrieve(query: str, top_k: int) -> list[dict]:
        return [
            {
                "document_id": "doc_tech_grpc",
                "content": "gRPC configuration details.",
                "score": 0.95,
            },
            {
                "document_id": "doc_tech_k8s_probe",
                "content": "Kubernetes probe settings.",
                "score": 0.88,
            },
        ]

    def mock_answer(query: str, context: str) -> tuple[str | None, dict]:
        return f"Answer based on: {context[:80]}", {}

    results = await run_eval(str(TECHNICAL_DATASET_PATH), mock_retrieve, mock_answer, top_k=3)
    metrics = aggregate_metrics(results)

    assert metrics.total_cases == 30
    assert 0 <= metrics.retrieval_hit_rate <= 1.0
    assert 0 <= metrics.keyword_coverage <= 1.0
    assert metrics.average_latency_ms >= 0
    assert 0 <= metrics.mean_reciprocal_rank <= 1.0
    assert 0 <= metrics.average_precision_at_k <= 1.0
    assert 0 <= metrics.average_recall_at_k <= 1.0
