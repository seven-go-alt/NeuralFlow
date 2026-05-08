from __future__ import annotations

import pytest

from app.core.token_budget import ContextSegment, TokenBudgetManager


class FakeEncoder:
    """Stateless word-level encoder: each unique word gets a deterministic token ID."""

    _word_to_id: dict[str, int] = {}

    def encode(self, text: str) -> list[int]:
        for word in text.split():
            if word not in FakeEncoder._word_to_id:
                FakeEncoder._word_to_id[word] = len(FakeEncoder._word_to_id)
        return [FakeEncoder._word_to_id[w] for w in text.split()]

    def decode(self, tokens: list[int]) -> str:
        id_to_word = {v: k for k, v in FakeEncoder._word_to_id.items()}
        return " ".join(id_to_word[t] for t in tokens)


@pytest.fixture(autouse=True)
def _reset_encoder():
    FakeEncoder._word_to_id = {}
    yield
    FakeEncoder._word_to_id = {}


def test_count_tokens_uses_encoder_result() -> None:
    manager = TokenBudgetManager(
        encoding_name="test",
        soft_limit_tokens=20,
        hard_limit_tokens=30,
        encoder=FakeEncoder(),
    )

    assert manager.count_tokens("one two three") == 3


def test_trim_context_keeps_higher_priority_segments_when_over_budget() -> None:
    manager = TokenBudgetManager(
        encoding_name="test",
        soft_limit_tokens=5,
        hard_limit_tokens=5,
        encoder=FakeEncoder(),
    )
    segments = [
        ContextSegment(name="system", text="sys keep", priority=0),
        ContextSegment(name="recent_chat", text="recent keep", priority=1),
        ContextSegment(name="retrieved_memory", text="memory drop", priority=2),
        ContextSegment(name="early_chat", text="early drop", priority=3),
    ]

    trimmed = manager.trim_context(segments)

    assert trimmed.dropped_segment_names == ["early_chat", "retrieved_memory"]
    assert [segment.name for segment in trimmed.segments] == ["system", "recent_chat"]
    assert trimmed.token_before_trim == 8
    assert trimmed.token_after_trim == 4
    assert trimmed.soft_limit_exceeded
    assert trimmed.hard_limit_exceeded


def test_trim_context_truncates_single_oversized_segment_at_hard_limit() -> None:
    manager = TokenBudgetManager(
        encoding_name="test",
        soft_limit_tokens=5,
        hard_limit_tokens=6,
        encoder=FakeEncoder(),
    )
    segments = [
        ContextSegment(name="system", text="sys intro", priority=0),
        ContextSegment(name="recent_chat", text="one two three four five six seven", priority=1),
    ]

    trimmed = manager.trim_context(segments)

    assert [segment.name for segment in trimmed.segments] == ["system", "recent_chat"]
    assert trimmed.segments[1].text == "one two three four"
    assert trimmed.token_after_trim == 6
    assert trimmed.hard_limit_exceeded


def test_trim_context_marks_soft_limit_without_dropping_when_below_hard_limit() -> None:
    manager = TokenBudgetManager(
        encoding_name="test",
        soft_limit_tokens=4,
        hard_limit_tokens=8,
        encoder=FakeEncoder(),
    )
    segments = [
        ContextSegment(name="system", text="sys keep", priority=0),
        ContextSegment(name="recent_chat", text="recent keep now", priority=1),
    ]

    trimmed = manager.trim_context(segments)

    assert [segment.name for segment in trimmed.segments] == ["system", "recent_chat"]
    assert trimmed.token_before_trim == 5
    assert trimmed.token_after_trim == 5
    assert trimmed.soft_limit_exceeded
    assert not trimmed.hard_limit_exceeded
    assert trimmed.dropped_segment_names == []
