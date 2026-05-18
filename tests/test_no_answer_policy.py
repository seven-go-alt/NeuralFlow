from __future__ import annotations

import pytest

from app.rag.no_answer_policy import (
    NoAnswerDecision,
    decide_no_answer,
    evaluate_retrieval_confidence,
    has_query_context_overlap,
)
from app.retrieval.schemas import RetrievalResult


def _result(score: float, content: str = "some content") -> RetrievalResult:
    return RetrievalResult(
        chunk_id="c1",
        document_id="d1",
        content=content,
        score=score,
    )


class TestEvaluateRetrievalConfidence:
    def test_empty(self) -> None:
        assert evaluate_retrieval_confidence([]) == 0.0

    def test_single(self) -> None:
        assert evaluate_retrieval_confidence([_result(0.8)]) == 0.8

    def test_multiple(self) -> None:
        assert evaluate_retrieval_confidence([_result(0.9), _result(0.7)]) == 0.8


class TestHasQueryContextOverlap:
    def test_full_overlap(self) -> None:
        r = has_query_context_overlap("hello world", [_result(0.5, "hello world")])
        assert r == 1.0

    def test_partial(self) -> None:
        r = has_query_context_overlap("hello world missing", [_result(0.5, "hello")])
        assert r == pytest.approx(1.0 / 3.0)

    def test_empty_query(self) -> None:
        assert has_query_context_overlap("", [_result(0.5)]) == 0.0

    def test_empty_results(self) -> None:
        assert has_query_context_overlap("hello", []) == 0.0

    def test_no_match(self) -> None:
        assert has_query_context_overlap("python", [_result(0.5, "javascript")]) == 0.0

    def test_single_char_terms_ignored(self) -> None:
        assert has_query_context_overlap("a hello", [_result(0.5, "hello")]) == 1.0


class TestDecideNoAnswer:
    def test_should_answer(self) -> None:
        decision = decide_no_answer("hello", [_result(0.9, "hello world")])
        assert decision.should_answer is True

    def test_empty_results_refuse(self) -> None:
        decision = decide_no_answer("hello", [])
        assert decision.should_answer is False
        assert "No relevant documents" in decision.reason

    def test_empty_results_fallback(self) -> None:
        decision = decide_no_answer("hello", [], empty_result_policy="fallback")
        assert decision.should_answer is True

    def test_low_confidence(self) -> None:
        decision = decide_no_answer("hello", [_result(0.1, "hello world")], min_confidence=0.3)
        assert decision.should_answer is False
        assert "confidence" in decision.reason.lower()

    def test_low_overlap(self) -> None:
        decision = decide_no_answer(
            "python language",
            [_result(0.9, "irrelevant content about javascript")],
            min_overlap=0.5,
        )
        assert decision.should_answer is False
        assert "overlap" in decision.reason.lower()

    def test_low_both(self) -> None:
        decision = decide_no_answer(
            "python language",
            [_result(0.1, "irrelevant content")],
            min_confidence=0.3,
            min_overlap=0.5,
        )
        assert decision.should_answer is False
        assert "confidence" in decision.reason.lower()
        assert "overlap" in decision.reason.lower()

    def test_custom_thresholds(self) -> None:
        decision = decide_no_answer(
            "hello", [_result(0.5, "hello world")], min_confidence=0.6, min_overlap=0.5
        )
        assert decision.should_answer is False

    def test_type(self) -> None:
        decision = decide_no_answer("hello", [])
        assert isinstance(decision, NoAnswerDecision)
