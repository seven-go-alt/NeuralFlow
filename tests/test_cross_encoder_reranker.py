from __future__ import annotations

from unittest.mock import patch
from typing import Any

import pytest

from app.config import get_settings
from app.retrieval.reranker import CrossEncoderReranker, _sigmoid, heuristic_rerank
from app.retrieval.schemas import RetrievalResult


def _chunk(content: str, score: float = 0.5) -> RetrievalResult:
    return RetrievalResult(
        chunk_id="c1",
        document_id="d1",
        content=content,
        score=score,
        metadata={},
        source={},
    )


class TestCrossEncoderReranker:
    def test_empty_chunks(self) -> None:
        r = CrossEncoderReranker(enabled=True)
        assert r.rerank("query", []) == []

    def test_fallback_when_disabled(self) -> None:
        r = CrossEncoderReranker(enabled=False)
        chunks = [_chunk("hello"), _chunk("world")]
        result = r.rerank("test", chunks)
        assert len(result) == 2

    @patch("app.retrieval.reranker._load_model", return_value=None)
    def test_fallback_when_model_unavailable(self, mock_load: object) -> None:
        r = CrossEncoderReranker(enabled=True)
        chunks = [_chunk("hello"), _chunk("world")]
        result = r.rerank("test", chunks)
        assert len(result) == 2

    @patch("app.retrieval.reranker._load_model")
    def test_fallback_when_inference_fails(self, mock_load: Any) -> None:
        class FailingModel:
            def predict(self, pairs: object) -> object:
                raise RuntimeError("inference failed")

        mock_load.return_value = FailingModel()
        r = CrossEncoderReranker(enabled=True)
        chunks = [_chunk("hello"), _chunk("world")]
        result = r.rerank("test", chunks)
        assert len(result) == 2

    @patch("app.retrieval.reranker._load_model")
    def test_top_k_argument(self, mock_load: Any) -> None:
        import numpy as np

        class FakeModel:
            def predict(self, pairs: list[tuple[str, str]]) -> object:
                return np.array([1.0, 0.5, 0.0])

        mock_load.return_value = FakeModel()
        r = CrossEncoderReranker(enabled=True)
        chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
        result = r.rerank("q", chunks, top_k=2)
        assert len(result) == 2

    def test_sigmoid_correctness(self) -> None:
        assert _sigmoid(0.0) == pytest.approx(0.5, rel=0.01)
        assert _sigmoid(10.0) > 0.999
        assert _sigmoid(-10.0) < 0.001
        assert _sigmoid(1000.0) == pytest.approx(1.0, rel=0.01)
        assert _sigmoid(-1000.0) == pytest.approx(0.0, abs=1e-10)


class TestHeuristicReranker:
    def test_empty_results(self) -> None:
        assert heuristic_rerank([], "query") == []

    def test_basic_rerank(self) -> None:
        chunks = [_chunk("apple banana", 0.5), _chunk("cherry date", 0.3)]
        result = heuristic_rerank(chunks, "apple")
        assert len(result) == 2
        assert result[0].content == "apple banana"


class TestConfigDefaults:
    def test_default_values(self) -> None:
        s = get_settings()
        assert s.cross_encoder_enabled is True
        assert "MiniLM" in s.reranker_model
        assert s.reranker_top_k == 5
