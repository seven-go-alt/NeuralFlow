from __future__ import annotations

from app.rag.no_answer_policy import NoAnswerDecision
from app.rag.retrieval_grader import _combine_grade, grade_retrieval
from app.retrieval.schemas import RetrievalResult


class TestCombineGrade:
    def test_combine_uses_confidence_when_no_llm(self) -> None:
        decision = NoAnswerDecision(should_answer=True, reason="ok", confidence=0.8)
        result = _combine_grade(decision, llm_score=None)
        assert result.score == 0.8
        assert result.sufficient is True

    def test_combine_averages_llm_and_confidence(self) -> None:
        decision = NoAnswerDecision(should_answer=True, reason="ok", confidence=0.6)
        result = _combine_grade(decision, llm_score=0.8)
        assert result.score == 0.7
        assert result.sufficient is True

    def test_refuses_when_no_answer_policy_rejects(self) -> None:
        decision = NoAnswerDecision(
            should_answer=False,
            reason="Low confidence",
            confidence=0.1,
        )
        result = _combine_grade(decision, llm_score=None)
        assert result.sufficient is False
        assert "Low confidence" in result.reason

    def test_refuses_when_llm_score_too_low(self) -> None:
        decision = NoAnswerDecision(should_answer=True, reason="ok", confidence=0.6)
        result = _combine_grade(decision, llm_score=0.3)
        assert result.sufficient is False
        assert "LLM relevance score too low" in result.reason

    def test_details_contain_both_scores(self) -> None:
        decision = NoAnswerDecision(should_answer=True, reason="ok", confidence=0.7)
        result = _combine_grade(decision, llm_score=0.9)
        assert result.details["llm_score"] == 0.9
        assert result.details["no_answer_should_answer"] is True


class TestGradeRetrieval:
    async def test_empty_results_returns_not_sufficient(self) -> None:
        result = await grade_retrieval("test query", [], llm=None)
        assert result.sufficient is False

    async def test_grades_without_llm(self) -> None:
        results = [
            RetrievalResult(
                chunk_id="c1",
                document_id="d1",
                content="test content for query match",
                score=0.8,
            )
        ]
        result = await grade_retrieval("test", results, llm=None)
        assert isinstance(result.score, float)

    async def test_grades_with_llm(self) -> None:
        class FakeLLM:
            async def generate(self, prompt: str) -> str:
                return "4"

        results = [
            RetrievalResult(
                chunk_id="c1",
                document_id="d1",
                content="relevant content here",
                score=0.9,
            )
        ]
        result = await grade_retrieval("test query", results, llm=FakeLLM())
        assert result.score > 0
