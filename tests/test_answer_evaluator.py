from __future__ import annotations

from app.rag.answer_evaluator import AnswerEvalResult, evaluate_answer


class FakeLLM:
    def __init__(
        self,
        response: str = '{"relevance": 0.8, "faithfulness": 0.9, "completeness": 0.7, "reason": "good"}',
    ) -> None:
        self.response = response

    async def generate(self, prompt: str) -> str:  # noqa: ARG002
        return self.response


async def test_evaluate_answer_full_scores() -> None:
    result = await evaluate_answer(
        query="test query",
        answer="test answer content",
        context_chunks=["chunk1", "chunk2"],
        llm=FakeLLM(),
    )
    assert isinstance(result, AnswerEvalResult)
    assert result.relevance == 0.8
    assert result.faithfulness == 0.9
    assert result.completeness == 0.7
    assert abs(result.overall - 0.8) < 0.01
    assert result.reason == "good"


async def test_evaluate_answer_empty_answer() -> None:
    result = await evaluate_answer(query="test", answer="", context_chunks=[], llm=FakeLLM())
    assert result.relevance == 0.0
    assert result.reason == "empty answer"


async def test_evaluate_answer_clamps_values() -> None:
    llm = FakeLLM(
        response='{"relevance": 2.5, "faithfulness": -1.0, "completeness": 0.5, "reason": ""}'
    )
    result = await evaluate_answer(query="test", answer="answer", context_chunks=[], llm=llm)
    assert result.relevance == 1.0
    assert result.faithfulness == 0.0
    assert result.completeness == 0.5


async def test_evaluate_answer_parse_error() -> None:
    llm = FakeLLM(response="not json")
    result = await evaluate_answer(query="test", answer="answer", context_chunks=[], llm=llm)
    assert result.relevance == 0.0
    assert "parse error" in result.reason


async def test_evaluate_answer_llm_error() -> None:
    class FailingLLM:
        async def generate(self, prompt: str) -> str:  # noqa: ARG002
            msg = "llm unavailable"
            raise RuntimeError(msg)

    result = await evaluate_answer(
        query="test", answer="answer", context_chunks=[], llm=FailingLLM()
    )
    assert result.relevance == 0.0
    assert "llm error" in result.reason


async def test_from_llm_response_valid() -> None:
    result = AnswerEvalResult.from_llm_response(
        '{"relevance": 0.9, "faithfulness": 0.8, "completeness": 0.7, "reason": "ok"}'
    )
    assert result.relevance == 0.9
    assert result.faithfulness == 0.8
    assert result.completeness == 0.7
    assert abs(result.overall - 0.8) < 0.01


async def test_from_llm_response_no_json() -> None:
    result = AnswerEvalResult.from_llm_response("no json here")
    assert result.relevance == 0.0
    assert "parse error" in result.reason


async def test_zero_factory() -> None:
    result = AnswerEvalResult.zero("no data")
    assert result.relevance == 0.0
    assert result.faithfulness == 0.0
    assert result.completeness == 0.0
    assert result.overall == 0.0
    assert result.reason == "no data"
