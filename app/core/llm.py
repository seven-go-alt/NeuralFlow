from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from litellm import acompletion

from app.config import get_settings


class LLMClient:
    def __init__(self, model: str | None = None) -> None:
        settings = get_settings()
        self.model = model or settings.litellm_model

    async def generate(self, prompt: str) -> str:
        response = await acompletion(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是 NeuralFlow 助手。"},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def stream_generate(self, prompt: str, include_thinking: bool = False) -> AsyncIterator[dict[str, str]]:
        stream = await acompletion(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是 NeuralFlow 助手。"},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        async for chunk in stream:
            thinking = self._extract_thinking(chunk)
            if include_thinking and thinking:
                yield {"event": "thinking", "data": thinking}
            delta = self._extract_delta(chunk)
            if delta:
                yield {"event": "message", "data": delta}

    def _extract_delta(self, chunk: Any) -> str:
        choice = self._first_choice(chunk)
        if choice is None:
            return ""
        delta = getattr(choice, "delta", None)
        if delta is None and isinstance(choice, dict):
            delta = choice.get("delta")
        if delta is None:
            return ""
        if isinstance(delta, dict):
            return str(delta.get("content") or "")
        return str(getattr(delta, "content", "") or "")

    def _extract_thinking(self, chunk: Any) -> str:
        choice = self._first_choice(chunk)
        if choice is None:
            return ""
        delta = getattr(choice, "delta", None)
        if delta is None and isinstance(choice, dict):
            delta = choice.get("delta")
        if isinstance(delta, dict):
            return str(delta.get("reasoning_content") or delta.get("reasoning") or "")
        return str(getattr(delta, "reasoning_content", "") or getattr(delta, "reasoning", "") or "")

    def _first_choice(self, chunk: Any) -> Any | None:
        choices = getattr(chunk, "choices", None)
        if choices is None and isinstance(chunk, dict):
            choices = chunk.get("choices")
        if not choices:
            return None
        return choices[0]
