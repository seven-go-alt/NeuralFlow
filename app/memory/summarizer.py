from __future__ import annotations

import logging

from app.core.llm import LLMClient

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = """请将以下对话历史压缩为一段结构化摘要，包含：
1. 主题：对话讨论的核心话题
2. 关键信息：用户提到的重要事实、需求或偏好
3. 决策/结论：达成的共识或下一步行动

请用中文输出，控制在 200 字以内。只输出摘要内容，不要加标题或前缀。

对话历史：
{history_text}"""


class Summarizer:
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm = llm_client

    def summarize(self, history_text: str) -> str:
        text = history_text.strip()
        if not text:
            return "空会话，无需归档。"
        if len(text) <= 240:
            return f"摘要: {text}"
        return f"摘要: {text[:237]}..."

    def summarize_messages(self, session_id: str, messages: list[dict[str, str]]) -> str:
        if not messages:
            return f"session={session_id}\n对话为空。"

        lines = [f"session={session_id}", "conversation:"]
        for message in messages:
            role = message.get("role", "unknown")
            content = message.get("content", "").strip()
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    async def summarize_messages_async(self, session_id: str, messages: list[dict[str, str]]) -> str:
        """使用 LLM 生成结构化摘要。LLM 不可用时回退到规则截断。"""
        if not messages:
            return f"session={session_id}\n对话为空。"

        history_text = "\n".join(
            f"{m.get('role', '?')}: {m.get('content', '')}" for m in messages
        )

        if self._llm is not None:
            try:
                prompt = SUMMARY_PROMPT.format(history_text=history_text)
                summary = await self._llm.generate(prompt)
                if summary and not summary.startswith("离线兜底摘要"):
                    return f"session={session_id}\n{summary.strip()}"
            except Exception as exc:
                logger.warning("LLM summarization failed, falling back to truncation: %s", exc)

        return self.summarize_messages(session_id, messages)
