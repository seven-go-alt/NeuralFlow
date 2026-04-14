import json
from typing import Any

import redis

from app.config import get_settings
from app.memory.base import MemoryStore
from app.utils.redis_client import get_redis_client


class WorkingMemory(MemoryStore):
    def __init__(
        self,
        session_id: str,
        max_turns: int | None = None,
        client: redis.Redis | None = None,
    ) -> None:
        settings = get_settings()
        self.client = client or get_redis_client()
        self.key = f"session:{session_id}:history"
        self.max_turns = max_turns or settings.working_memory_max_turns

    def add(self, role: str, content: str) -> None:
        self.add_message(role=role, content=content)

    def add_message(self, role: str, content: str) -> None:
        payload = {"role": role, "content": content}
        self.client.lpush(self.key, json.dumps(payload, ensure_ascii=False))
        self.client.ltrim(self.key, 0, self.max_turns - 1)

    def get_messages(self) -> list[dict[str, Any]]:
        raw_data = self.client.lrange(self.key, 0, -1)
        return [json.loads(item) for item in reversed(raw_data)]

    def pop_all_messages(self) -> list[dict[str, Any]]:
        messages = self.get_messages()
        self.client.delete(self.key)
        return messages
