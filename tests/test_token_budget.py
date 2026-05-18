from __future__ import annotations

import pytest

from app.core.token_budget import (
    ContextSegment,
    FallbackTokenEncoder,
    TokenBudgetManager,
)


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


def test_count_tokens_with_empty_text_returns_zero() -> None:
    manager = TokenBudgetManager(
        encoding_name="test",
        soft_limit_tokens=10,
        hard_limit_tokens=20,
        encoder=FakeEncoder(),
    )
    assert manager.count_tokens("") == 0
    assert manager.count_tokens("   ") == 0


def test_token_budget_init_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError, match="positive"):
        TokenBudgetManager(soft_limit_tokens=0, hard_limit_tokens=10)
    with pytest.raises(ValueError, match="positive"):
        TokenBudgetManager(soft_limit_tokens=-1, hard_limit_tokens=10)
    with pytest.raises(ValueError, match="cannot exceed"):
        TokenBudgetManager(soft_limit_tokens=20, hard_limit_tokens=10)


def test_fallback_token_encoder_empty_text() -> None:
    encoder = FallbackTokenEncoder()
    assert encoder.encode("") == []
    assert encoder.decode([1, 2, 3]) == "x x x"


def test_find_truncation_candidate_empty_segments() -> None:
    manager = TokenBudgetManager(
        encoding_name="test",
        soft_limit_tokens=10,
        hard_limit_tokens=20,
        encoder=FakeEncoder(),
    )
    result = manager.trim_context([])
    assert result.dropped_segment_names == []
    assert result.token_before_trim == 0


def test_fallback_encoder_whitespace_only() -> None:
    """Whitespace-only text: split() returns [], falls to char-level tokens."""
    encoder = FallbackTokenEncoder()
    tokens = encoder.encode("   ")
    assert tokens == [0, 1, 2]  # 3 chars = 3 tokens


def test_fallback_encoder_with_words() -> None:
    """Non-empty words get tokenized by word index."""
    encoder = FallbackTokenEncoder()
    tokens = encoder.encode("hello world")
    assert tokens == [0, 1]  # 2 words = 2 tokens


def test_get_encoder_fallback_when_tiktoken_fails(monkeypatch) -> None:
    import tiktoken

    def failing_get_encoding(name: str) -> object:
        raise RuntimeError("tiktoken unavailable")

    monkeypatch.setattr(tiktoken, "get_encoding", failing_get_encoding)

    manager = TokenBudgetManager(
        encoding_name="test",
        soft_limit_tokens=10,
        hard_limit_tokens=20,
    )
    count = manager.count_tokens("hello world")

    assert isinstance(manager._encoder, FallbackTokenEncoder)
    assert count == 2  # FallbackTokenEncoder: 1 token per word


class _CharEncoder:
    """Character-level encoder — 1 char = 1 token, for precise token control."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


def test_truncate_text_returns_empty_when_max_tokens_zero() -> None:
    manager = TokenBudgetManager(
        encoding_name="test",
        soft_limit_tokens=10,
        hard_limit_tokens=20,
        encoder=_CharEncoder(),
    )
    assert manager._truncate_text("hello", 0) == ""
    assert manager._truncate_text("hello", -1) == ""


def test_truncate_text_returns_original_when_fits() -> None:
    manager = TokenBudgetManager(
        encoding_name="test",
        soft_limit_tokens=10,
        hard_limit_tokens=20,
        encoder=_CharEncoder(),
    )
    result = manager._truncate_text("hello", 10)
    assert result == "hello"


def test_truncate_text_truncates_via_encoder() -> None:
    manager = TokenBudgetManager(
        encoding_name="test",
        soft_limit_tokens=10,
        hard_limit_tokens=20,
        encoder=_CharEncoder(),
    )
    result = manager._truncate_text("hello world", 5)
    assert result == "hello"


class _FailingDecoder:
    """Encoder where decode always fails — tests word-level fallback."""

    def encode(self, text: str) -> list[int]:
        return list(range(max(1, len(text))))

    def decode(self, tokens: list[int]) -> str:
        raise RuntimeError("decode error")


def test_truncate_text_decode_failure_falls_back_to_words() -> None:
    manager = TokenBudgetManager(
        encoding_name="test",
        soft_limit_tokens=10,
        hard_limit_tokens=20,
        encoder=_FailingDecoder(),
    )
    result = manager._truncate_text("hello world foo bar baz", 3)
    assert result == "hello world foo"


def test_truncate_text_decode_failure_no_words_falls_to_slice() -> None:
    manager = TokenBudgetManager(
        encoding_name="test",
        soft_limit_tokens=10,
        hard_limit_tokens=20,
        encoder=_FailingDecoder(),
    )
    result = manager._truncate_text("   ", 1)
    assert result == " "


def test_find_truncation_candidate_empty_returns_none() -> None:
    manager = TokenBudgetManager(
        encoding_name="test",
        soft_limit_tokens=10,
        hard_limit_tokens=20,
        encoder=_CharEncoder(),
    )
    assert manager._find_truncation_candidate([]) is None


def test_trim_context_drops_segment_when_allowed_tokens_zero() -> None:
    """When a required segment gets 0 allowed_tokens, it's dropped."""
    manager = TokenBudgetManager(
        encoding_name="test",
        soft_limit_tokens=1,
        hard_limit_tokens=1,
        encoder=_CharEncoder(),
    )
    segments = [
        ContextSegment(name="a", text="aa", priority=0, required=True),
        ContextSegment(name="b", text="bb", priority=5, required=True),
    ]
    result = manager.trim_context(segments)

    # "b" is dropped inside _truncate_last_resort (allowed_tokens == 0)
    assert [s.name for s in result.segments] == ["a"]
    assert result.segments[0].text == "a"  # "aa" truncated to fit 1 token
    assert result.token_after_trim == 1


def test_count_tokens_caches_encoder() -> None:
    """_get_encoder result is cached, not recreated each call."""
    manager = TokenBudgetManager(
        encoding_name="test",
        soft_limit_tokens=10,
        hard_limit_tokens=20,
        encoder=FakeEncoder(),
    )
    encoder1 = manager._get_encoder()
    encoder2 = manager._get_encoder()
    assert encoder1 is encoder2
