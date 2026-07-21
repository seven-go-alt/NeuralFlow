from __future__ import annotations

import pytest

from app.evals.factories import make_live_answer_fn, make_live_retrieve_fn
from app.evals.metrics import aggregate_metrics
from app.evals.runner import run_eval

DATASET_PATH = "data/eval/datasets/rag_quality_50.jsonl"


class TestLiveFactories:
    """Unit tests for live factory functions — no real LLM/ChromaDB."""

    def test_make_live_answer_fn_returns_callable(self) -> None:
        """live answer fn should be callable."""
        fn = make_live_answer_fn()
        assert callable(fn)

    def test_make_live_retrieve_fn_is_callable(self) -> None:
        fn = make_live_retrieve_fn()
        assert callable(fn)


class TestLiveEvalWithMockPipeline:
    """E2E eval run with mock functions that match live signatures."""

    @pytest.mark.asyncio
    async def test_run_eval_with_token_usage(self) -> None:
        def mock_retrieve(query: str, top_k: int) -> list[dict]:
            return [
                {
                    "document_id": "doc_hr_leave",
                    "content": "Annual leave policy: 20 days for 5+ years.",
                    "score": 0.95,
                },
            ]

        def mock_answer(query: str, context: str) -> tuple[str | None, dict]:
            return "Mock answer", {
                "prompt_tokens": 150,
                "completion_tokens": 50,
                "total_tokens": 200,
            }

        def mock_judge(query: str, answer: str, chunks: list[str]) -> tuple:
            from app.rag.answer_evaluator import AnswerEvalResult

            return (
                AnswerEvalResult(
                    relevance=0.9,
                    faithfulness=0.8,
                    completeness=0.7,
                    overall=0.8,
                    reason="mock",
                ),
                {"prompt_tokens": 200, "completion_tokens": 30, "total_tokens": 230},
            )

        results = await run_eval(
            DATASET_PATH,
            mock_retrieve,
            mock_answer,
            top_k=3,
            answer_eval_fn=mock_judge,
        )
        metrics = aggregate_metrics(results)

        assert metrics.total_cases == 50
        for r in results:
            assert r.token_usage_json is not None
            assert r.token_usage_json["total_tokens"] == 430  # 200 + 230
        assert metrics.average_answer_relevance == pytest.approx(0.9)
        assert metrics.average_answer_faithfulness == pytest.approx(0.8)
        assert metrics.average_answer_completeness == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_run_eval_skip_judge(self) -> None:
        """answer_eval_fn=None should work without Judge."""

        def mock_retrieve(query: str, top_k: int) -> list[dict]:
            return [
                {"document_id": "doc_hr_leave", "content": "Leave policy.", "score": 0.9},
            ]

        def mock_answer(query: str, context: str) -> tuple[str | None, dict]:
            return "Answer", {"total_tokens": 100}

        results = await run_eval(
            DATASET_PATH,
            mock_retrieve,
            mock_answer,
            top_k=3,
            answer_eval_fn=None,
        )
        metrics = aggregate_metrics(results)
        assert metrics.total_cases == 50
        assert metrics.answer_count == 0  # no Judge, so answer_count stays 0
