from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from litellm import acompletion

from app.config import get_settings

logger = logging.getLogger(__name__)


def build_rule_based_fallback_reply(prompt: str, error: Exception | None = None) -> str:
    lines = [line.strip() for line in prompt.splitlines() if line.strip()]
    snippets = lines[-3:] if lines else [prompt.strip() or "未提供上下文"]
    summary = "；".join(snippet[:120] for snippet in snippets if snippet)
    error_hint = f"（原因：{error}）" if error else ""
    return f"离线兜底摘要{error_hint}：当前外部 LLM 不可用。我先基于已有上下文给出简要总结：{summary}"


class LLMClient:
    def __init__(self, model: str | None = None) -> None:
        settings = get_settings()
        self.model = model or settings.litellm_model
        self.fallback_model = settings.ollama_fallback_model
        self.offline_fallback_enabled = settings.offline_fallback_enabled

    async def generate(self, prompt: str) -> str:
        try:
            return await self._generate_once(prompt, model=self.model)
        except Exception as primary_exc:
            if not self.offline_fallback_enabled:
                raise
            if self.fallback_model:
                try:
                    logger.warning("primary llm failed, falling back to offline model", exc_info=primary_exc)
                    return await self._generate_once(prompt, model=self.fallback_model)
                except Exception as fallback_exc:
                    logger.warning("fallback llm failed, returning rule-based summary", exc_info=fallback_exc)
                    return build_rule_based_fallback_reply(prompt, error=fallback_exc)
            logger.warning("primary llm failed, returning rule-based summary", exc_info=primary_exc)
            return build_rule_based_fallback_reply(prompt, error=primary_exc)

    async def stream_generate(self, prompt: str, include_thinking: bool = False) -> AsyncIterator[dict[str, str]]:
        try:
            async for chunk in self._stream_once(prompt, model=self.model, include_thinking=include_thinking):
                yield chunk
            return
        except Exception as primary_exc:
            if not self.offline_fallback_enabled:
                raise
            if self.fallback_model:
                try:
                    logger.warning("primary stream llm failed, falling back to offline model", exc_info=primary_exc)
                    async for chunk in self._stream_once(
                        prompt,
                        model=self.fallback_model,
                        include_thinking=include_thinking,
                    ):
                        yield chunk
                    return
                except Exception as fallback_exc:
                    logger.warning("fallback stream llm failed, returning rule-based summary", exc_info=fallback_exc)
                    yield {"event": "message", "data": build_rule_based_fallback_reply(prompt, error=fallback_exc)}
                    return
            logger.warning("primary stream llm failed, returning rule-based summary", exc_info=primary_exc)
            yield {"event": "message", "data": build_rule_based_fallback_reply(prompt, error=primary_exc)}

    async def _generate_once(self, prompt: str, model: str) -> str:
        response = await acompletion(
            model=model,
            messages=self._build_messages(prompt),
        )
        return response.choices[0].message.content or ""

    async def _stream_once(self, prompt: str, model: str, include_thinking: bool = False) -> AsyncIterator[dict[str, str]]:
        stream = await acompletion(
            model=model,
            messages=self._build_messages(prompt),
            stream=True,
        )
        async for chunk in stream:
            thinking = self._extract_thinking(chunk)
            if include_thinking and thinking:
                yield {"event": "thinking", "data": thinking}
            delta = self._extract_delta(chunk)
            if delta:
                yield {"event": "message", "data": delta}

    def _build_messages(self, prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": "你是 NeuralFlow 助手。"},
            {"role": "user", "content": prompt},
        ]

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
