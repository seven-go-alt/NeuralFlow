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

    async def build_prompt(self, user_query: str, intent: str) -> str:
        context_parts: list[str] = ["你是一个智能助手。"]

        if intent == "coding":
            context_parts.append("当前可用技能: Python代码解释器, 文件读写工具。")

        if intent in {"query_history", "personal_preference"}:
            memories = await self.long_mem.search(user_query, top_k=3)
            if memories:
                context_parts.append("相关历史记忆:\n" + "\n".join(memories))

        recent_chat = self.working_mem.get_messages()
        context_parts.append(f"当前对话:\n{recent_chat}")
        context_parts.append(f"用户问题:\n{user_query}")

        return "\n---\n".join(context_parts)
