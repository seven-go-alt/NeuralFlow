from __future__ import annotations

from app.memory.long_term import LongTermMemory
from app.memory.working import WorkingMemory


class ContextBuilder:
    def __init__(
        self,
        session_id: str,
        working_mem: WorkingMemory | None = None,
        long_mem: LongTermMemory | None = None,
    ) -> None:
        self.working_mem = working_mem or WorkingMemory(session_id=session_id)
        self.long_mem = long_mem or LongTermMemory()

    async def build_prompt(
        self,
        user_query: str,
        intent: str,
        memory_strategy: str | None = None,
        skill_whitelist: list[str] | None = None,
        skill_results: list[dict[str, object]] | None = None,
    ) -> str:
        context_parts: list[str] = ["你是一个智能助手。"]

        effective_skill_whitelist = skill_whitelist
        if effective_skill_whitelist is None and intent == "coding":
            effective_skill_whitelist = ["python", "filesystem"]

        if effective_skill_whitelist:
            context_parts.append("当前可用技能: " + ", ".join(effective_skill_whitelist))

        if skill_results:
            rendered_results = []
            for item in skill_results:
                skill_name = str(item.get("skill", "unknown"))
                result = item.get("result")
                rendered_results.append(f"- {skill_name}: {result}")
            context_parts.append("技能执行结果:\n" + "\n".join(rendered_results))

        effective_memory_strategy = memory_strategy
        if effective_memory_strategy is None:
            effective_memory_strategy = (
                "long_term" if intent in {"query_history", "personal_preference"} else "working_only"
            )

        if effective_memory_strategy == "long_term":
            memories = await self.long_mem.search(user_query, top_k=3)
            if memories:
                context_parts.append("相关历史记忆:\n" + "\n".join(memories))

        recent_chat = self.working_mem.get_messages()
        context_parts.append(f"当前对话:\n{recent_chat}")
        context_parts.append(f"用户问题:\n{user_query}")

        return "\n---\n".join(context_parts)
