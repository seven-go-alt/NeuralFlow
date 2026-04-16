from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)


class TokenEncoder(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: list[int]) -> str: ...


@dataclass(slots=True, frozen=True)
class ContextSegment:
    name: str
    text: str
    priority: int
    required: bool = False


@dataclass(slots=True, frozen=True)
class TokenTrimResult:
    segments: list[ContextSegment]
    token_before_trim: int
    token_after_trim: int
    soft_limit_exceeded: bool
    hard_limit_exceeded: bool
    dropped_segment_names: list[str]

    @property
    def trimmed_text(self) -> str:
        return "\n---\n".join(segment.text for segment in self.segments if segment.text)


class TokenBudgetManager:
    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        soft_limit_tokens: int = 6000,
        hard_limit_tokens: int = 8000,
        encoder: TokenEncoder | None = None,
    ) -> None:
        if soft_limit_tokens <= 0 or hard_limit_tokens <= 0:
            raise ValueError("Token limits must be positive")
        if soft_limit_tokens > hard_limit_tokens:
            raise ValueError("soft_limit_tokens cannot exceed hard_limit_tokens")
        self.encoding_name = encoding_name
        self.soft_limit_tokens = soft_limit_tokens
        self.hard_limit_tokens = hard_limit_tokens
        self._encoder = encoder

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self._get_encoder().encode(text))

    def trim_context(self, segments: list[ContextSegment]) -> TokenTrimResult:
        normalized = [segment for segment in segments if segment.text]
        token_before_trim = self._total_tokens(normalized)
        soft_limit_exceeded = token_before_trim > self.soft_limit_tokens
        hard_limit_exceeded = token_before_trim > self.hard_limit_tokens

        if soft_limit_exceeded:
            logger.warning(
                "token soft limit exceeded",
                extra={
                    "token_before_trim": token_before_trim,
                    "soft_limit_tokens": self.soft_limit_tokens,
                },
            )

        kept = list(normalized)
        dropped_segment_names: list[str] = []
        while kept and self._total_tokens(kept) > self.hard_limit_tokens:
            drop_index = self._find_drop_candidate(kept)
            if drop_index is None:
                break
            dropped_segment_names.append(kept[drop_index].name)
            kept.pop(drop_index)

        if self._total_tokens(kept) > self.hard_limit_tokens:
            kept = self._truncate_last_resort(kept)

        token_after_trim = self._total_tokens(kept)
        return TokenTrimResult(
            segments=kept,
            token_before_trim=token_before_trim,
            token_after_trim=token_after_trim,
            soft_limit_exceeded=soft_limit_exceeded,
            hard_limit_exceeded=hard_limit_exceeded,
            dropped_segment_names=dropped_segment_names,
        )

    def _get_encoder(self) -> TokenEncoder:
        if self._encoder is not None:
            return self._encoder
        import tiktoken

        self._encoder = tiktoken.get_encoding(self.encoding_name)
        return self._encoder

    def _total_tokens(self, segments: list[ContextSegment]) -> int:
        return sum(self.count_tokens(segment.text) for segment in segments)

    def _find_drop_candidate(self, segments: list[ContextSegment]) -> int | None:
        candidates = [
            (index, segment)
            for index, segment in enumerate(segments)
            if not segment.required
        ]
        if not candidates or len(segments) <= 2:
            return None
        candidates.sort(key=lambda item: (item[1].priority, item[0]), reverse=True)
        return candidates[0][0]

    def _truncate_last_resort(self, segments: list[ContextSegment]) -> list[ContextSegment]:
        trimmed = list(segments)
        while trimmed and self._total_tokens(trimmed) > self.hard_limit_tokens:
            index = self._find_truncation_candidate(trimmed)
            if index is None:
                break
            segment = trimmed[index]
            allowed_tokens = max(
                0,
                self.hard_limit_tokens - self._total_tokens(trimmed[:index] + trimmed[index + 1 :]),
            )
            if allowed_tokens == 0:
                trimmed.pop(index)
                continue
            new_text = self._truncate_text(segment.text, allowed_tokens)
            if not new_text:
                trimmed.pop(index)
                continue
            trimmed[index] = ContextSegment(
                name=segment.name,
                text=new_text,
                priority=segment.priority,
                required=segment.required,
            )
            if self.count_tokens(trimmed[index].text) <= allowed_tokens:
                break
        return trimmed

    def _find_truncation_candidate(self, segments: list[ContextSegment]) -> int | None:
        if not segments:
            return None
        ranked = sorted(
            enumerate(segments),
            key=lambda item: (item[1].priority, item[1].required, self.count_tokens(item[1].text)),
            reverse=True,
        )
        return ranked[0][0]

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        encoder = self._get_encoder()
        encoded = encoder.encode(text)
        if len(encoded) <= max_tokens:
            return text
        truncated_tokens = encoded[:max_tokens]
        if hasattr(encoder, "decode"):
            try:
                decoded = encoder.decode(truncated_tokens).strip()
                if decoded:
                    return decoded
            except Exception:
                logger.debug("token decode failed, falling back to word truncation", exc_info=True)
        words = text.split()
        if not words:
            return text[:max_tokens]
        return " ".join(words[:max_tokens]).strip()
