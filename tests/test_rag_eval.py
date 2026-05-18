from __future__ import annotations

from pathlib import Path

import pytest

from app.evals.datasets import load_cases
from app.evals.metrics import (
    CaseResult,
    EvalMetrics,
    aggregate_metrics,
    compute_citation_match,
    compute_keyword_coverage,
    compute_no_answer_correct,
    compute_retrieval_hit,
)
from app.evals.runner import build_eval_report, run_eval


# --- datasets ---

def test_load_cases(tmp_path: Path) -> None:
    f = tmp_path / "cases.jsonl"
    f.write_text(
        '{"id": "c1", "question": "q?", "expected_keywords": ["k1"], "expected_doc_ids": ["d1"], "should_answer": true}\n'
        '{"id": "c2", "question": "q2?", "expected_keywords": [], "expected_doc_ids": [], "should_answer": false}\n'
    )
    cases = load_cases(str(f))
    assert len(cases) == 2
    assert cases[0].id == "c1"
    assert cases[0].expected_keywords == ("k1",)
    assert cases[0].should_answer is True
    assert cases[1].should_answer is False


def test_load_cases_skips_empty_lines(tmp_path: Path) -> None:
    f = tmp_path / "cases.jsonl"
    f.write_text(
        '{"id": "c1", "question": "q?", "expected_keywords": [], "expected_doc_ids": []}\n\n'
    )
    assert len(load_cases(str(f))) == 1


def test_load_cases_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_cases("/nonexistent/path.jsonl")


# --- metrics ---

class TestComputeRetrievalHit:
    def test_hit(self) -> None:
        assert compute_retrieval_hit(("d1", "d2"), ("d1",)) is True

    def test_miss(self) -> None:
        assert compute_retrieval_hit(("d3",), ("d1",)) is False

    def test_empty_expected(self) -> None:
        assert compute_retrieval_hit(("d1",), ()) is True


class TestComputeCitationMatch:
    def test_answer_mentions_doc_id(self) -> None:
        assert (
            compute_citation_match(
                "See doc_hr for policy",
                ("doc_hr",),
                ("doc_hr", "doc_tech"),
            )
            is True
        )

    def test_no_mention(self) -> None:
        assert (
            compute_citation_match(
                "Policy is clear",
                ("doc_hr",),
                ("doc_hr",),
            )
            is False
        )

    def test_empty_expected(self) -> None:
        assert compute_citation_match("Answer", (), ()) is True

    def test_no_answer(self) -> None:
        assert compute_citation_match(None, ("doc_hr",), ("doc_hr",)) is False


class TestComputeKeywordCoverage:
    def test_full_coverage(self) -> None:
        assert compute_keyword_coverage("hello world foo", ("hello", "world")) == 1.0

    def test_partial(self) -> None:
        assert (
            compute_keyword_coverage("hello world", ("hello", "missing"))
            == 0.5
        )

    def test_empty_keywords(self) -> None:
        assert compute_keyword_coverage("anything", ()) == 1.0

    def test_case_insensitive(self) -> None:
        assert (
            compute_keyword_coverage("Hello World", ("hello", "world")) == 1.0
        )

    def test_no_match(self) -> None:
        assert compute_keyword_coverage("hello", ("world",)) == 0.0


class TestComputeNoAnswerCorrect:
    def test_should_answer_has_answer(self) -> None:
        assert compute_no_answer_correct(True, "some answer") is True

    def test_should_answer_no_answer(self) -> None:
        assert compute_no_answer_correct(True, None) is False

    def test_should_not_answer_no_answer(self) -> None:
        assert compute_no_answer_correct(False, None) is True

    def test_should_not_answer_has_answer(self) -> None:
        assert compute_no_answer_correct(False, "some answer") is False


class TestAggregateMetrics:
    def test_empty(self) -> None:
        m = aggregate_metrics([])
        assert m.total_cases == 0
        assert m.retrieval_hit_rate == 0.0

    def test_all_pass(self) -> None:
        results = [
            CaseResult(
                case_id="c1",
                question="q1",
                retrieved_doc_ids=("d1",),
                retrieved_contents=("c1",),
                answer="see doc_hr",
                latency_ms=10.0,
                retrieval_hit=True,
                citation_match=True,
                keyword_coverage=1.0,
                no_answer_correct=True,
            ),
            CaseResult(
                case_id="c2",
                question="q2",
                retrieved_doc_ids=("d2",),
                retrieved_contents=("c2",),
                answer=None,
                latency_ms=20.0,
                retrieval_hit=False,
                citation_match=True,
                keyword_coverage=0.5,
                no_answer_correct=None,
            ),
        ]
        m = aggregate_metrics(results)
        assert m.total_cases == 2
        assert m.retrieval_hit_rate == 0.5
        assert m.citation_accuracy == 1.0
        assert m.keyword_coverage == 0.75
        assert pytest.approx(m.average_latency_ms) == 15.0
        assert m.no_answer_total == 1
        assert m.no_answer_accuracy == 1.0

    def test_smoke_properties(self) -> None:
        m = EvalMetrics()
        assert m.retrieval_hit_rate == 0.0
        assert m.citation_accuracy == 0.0
        assert m.keyword_coverage == 0.0
        assert m.no_answer_accuracy == 1.0
        assert m.average_latency_ms == 0.0


# --- runner ---

@pytest.mark.asyncio
async def test_run_eval_with_mock_fns(tmp_path: Path) -> None:
    f = tmp_path / "cases.jsonl"
    f.write_text(
        '{"id": "c1", "question": "年假?", "expected_keywords": ["年假"], "expected_doc_ids": ["doc_hr"], "should_answer": true}\n'
    )

    def retrieve_fn(query: str, top_k: int) -> list[dict]:
        return [
            {"document_id": "doc_hr", "content": "年假政策是15天", "score": 0.9},
        ]

    def answer_fn(query: str, context: str) -> str | None:
        return "doc_hr: 年假15天"

    results = await run_eval(str(f), retrieve_fn, answer_fn, top_k=5)
    assert len(results) == 1
    r = results[0]
    assert r.retrieval_hit is True
    assert r.citation_match is True
    assert r.keyword_coverage == 1.0
    assert r.no_answer_correct is True


@pytest.mark.asyncio
async def test_run_eval_should_not_answer(tmp_path: Path) -> None:
    f = tmp_path / "cases.jsonl"
    f.write_text(
        '{"id": "no_ans", "question": "天气?", "expected_keywords": [], "expected_doc_ids": [], "should_answer": false}\n'
    )

    def retrieve_fn(query: str, top_k: int) -> list[dict]:
        return []

    def answer_fn(query: str, context: str) -> str | None:
        return None

    results = await run_eval(str(f), retrieve_fn, answer_fn)
    assert len(results) == 1
    assert results[0].no_answer_correct is True
    assert results[0].retrieval_hit is True
    assert results[0].keyword_coverage == 1.0


# --- report ---

def test_build_eval_report() -> None:
    results = [
        CaseResult(
            case_id="c1",
            question="q1",
            retrieved_doc_ids=("d1",),
            retrieved_contents=("c1",),
            answer="a",
            latency_ms=5.0,
            retrieval_hit=True,
            citation_match=False,
            keyword_coverage=0.8,
            no_answer_correct=True,
        ),
    ]
    metrics = EvalMetrics(
        total_cases=1,
        retrieval_hits=1,
        citation_matches=0,
        keyword_coverage_sum=0.8,
        no_answer_correct=1,
        no_answer_total=1,
        total_latency_ms=5.0,
    )
    report = build_eval_report(results, metrics)
    assert "# RAG Eval Report" in report
    assert "c1" in report
