from __future__ import annotations

import inspect
import logging
from typing import Any

from app.config import get_settings
from app.core.token_budget import ContextSegment, TokenBudgetManager
from app.memory.long_term import LongTermMemory
from app.memory.working import WorkingMemory

logger = logging.getLogger(__name__)


class ContextBuilder:
    def __init__(
        self,
        session_id: str,
        working_mem: WorkingMemory | None = None,
        long_mem: LongTermMemory | None = None,
        token_budget_manager: TokenBudgetManager | None = None,
        tenant_id: str = "public",
    ) -> None:
        settings = get_settings()
        self.session_id = session_id
        self.settings = settings
        self.tenant_id = tenant_id or "public"
        self.working_mem = working_mem or WorkingMemory(session_id=session_id, tenant_id=self.tenant_id)
        self.long_mem = long_mem or LongTermMemory(tenant_id=self.tenant_id)
        self.token_budget_manager = token_budget_manager or TokenBudgetManager(
            encoding_name=settings.token_budget_encoding,
            soft_limit_tokens=settings.max_context_tokens_soft,
            hard_limit_tokens=settings.max_context_tokens,
        )
        self.last_trim_metadata = {
            "token_before_trim": 0,
            "token_after_trim": 0,
            "soft_limit_exceeded": False,
            "hard_limit_exceeded": False,
            "dropped_segment_names": [],
        }

    async def build_prompt(
        self,
        user_query: str,
        intent: str,
        memory_strategy: str | None = None,
        skill_whitelist: list[str] | None = None,
        skill_results: list[dict[str, object]] | None = None,
    ) -> str:
        segments: list[ContextSegment] = [
            ContextSegment(name="system_prompt", text="你是一个智能助手。", priority=0, required=True)
        ]

        effective_skill_whitelist = skill_whitelist
        if effective_skill_whitelist is None and intent == "coding":
            effective_skill_whitelist = ["python", "filesystem"]
        if effective_skill_whitelist:
            segments.append(
                ContextSegment(
                    name="available_skills",
                    text="当前可用技能: " + ", ".join(effective_skill_whitelist),
                    priority=1,
                )
            )

        if skill_results:
            rendered_results = []
            for item in skill_results:
                skill_name = str(item.get("skill", "unknown"))
                rendered_results.append(f"- {skill_name}: {item.get('result')}")
            segments.append(
                ContextSegment(
                    name="skill_results",
                    text="技能执行结果:\n" + "\n".join(rendered_results),
                    priority=1,
                )
            )

        effective_memory_strategy = memory_strategy
        if effective_memory_strategy is None:
            effective_memory_strategy = (
                "long_term" if intent in {"query_history", "personal_preference"} else "working_only"
            )

        if effective_memory_strategy == "long_term":
            memories = await self._search_long_term_memories(user_query)
            if memories:
                segments.append(
                    ContextSegment(
                        name="retrieved_memory",
                        text="相关历史记忆:\n" + "\n".join(memories),
                        priority=2,
                    )
                )

        early_chat, recent_chat = self._split_working_memory(self.working_mem.get_messages())
        if early_chat:
            segments.append(ContextSegment(name="early_chat", text=f"较早对话:\n{early_chat}", priority=3))
        segments.append(ContextSegment(name="recent_chat", text=f"当前对话:\n{recent_chat}", priority=1))
        segments.append(ContextSegment(name="user_query", text=f"用户问题:\n{user_query}", priority=0, required=True))

        trim_result = self.token_budget_manager.trim_context(segments)
        self.last_trim_metadata = {
            "token_before_trim": trim_result.token_before_trim,
            "token_after_trim": trim_result.token_after_trim,
            "soft_limit_exceeded": trim_result.soft_limit_exceeded,
            "hard_limit_exceeded": trim_result.hard_limit_exceeded,
            "dropped_segment_names": trim_result.dropped_segment_names,
        }
        logger.info("context trimmed", extra=self.last_trim_metadata)
        return trim_result.trimmed_text

    async def _search_long_term_memories(self, user_query: str) -> list[str]:
        search_method = self.long_mem.search
        search_kwargs = {"top_k": self.settings.vector_search_default_top_k}

        signature = inspect.signature(search_method)
        if "session_id" in signature.parameters:
            search_kwargs["session_id"] = self.session_id

        return await search_method(user_query, **search_kwargs)

    def _split_working_memory(self, messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if len(messages) <= 1:
            return [], messages
        configured_limit = max(1, self.settings.token_budget_recent_messages)
        recent_limit = min(configured_limit, len(messages) - 1)
        return messages[:-recent_limit], messages[-recent_limit:]
