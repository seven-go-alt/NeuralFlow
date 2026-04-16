from __future__ import annotations

from app.core.token_budget import ContextSegment, TokenBudgetManager


class FakeEncoder:
    def encode(self, text: str) -> list[int]:
        return text.split()


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
