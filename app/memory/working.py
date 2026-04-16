import json
import logging
from typing import Any

import redis

from app.config import get_settings
from app.memory.base import MemoryStore
from app.utils.redis_client import get_redis_client

logger = logging.getLogger(__name__)


class WorkingMemory(MemoryStore):
    def __init__(
        self,
        session_id: str,
        max_turns: int | None = None,
        archive_batch_size: int = 4,
        client: redis.Redis | None = None,
    ) -> None:
        settings = get_settings()
        self.client = client or get_redis_client()
        self.key = f"session:{session_id}:history"
        self.archive_key = f"session:{session_id}:archive"
        self.max_turns = max_turns or settings.working_memory_max_turns
        self.archive_batch_size = archive_batch_size
        self._fallback_enabled = False
        self._fallback_history: list[dict[str, str]] = []
        self._fallback_archive: list[dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        self.add_message(role=role, content=content)

    def add_message(self, role: str, content: str) -> None:
        payload = {"role": role, "content": content}
        if self._fallback_enabled:
            self._add_message_fallback(payload)
            return

        try:
            self.client.lpush(self.key, json.dumps(payload, ensure_ascii=False))

            raw_items = self.client.lrange(self.key, 0, -1)
            overflow = raw_items[self.max_turns :]
            if overflow:
                for item in reversed(overflow):
                    self.client.lpush(self.archive_key, item)

            self.client.ltrim(self.key, 0, self.max_turns - 1)
        except redis.RedisError as exc:
            self._enable_fallback(exc)
            self._add_message_fallback(payload)

    def get_messages(self) -> list[dict[str, Any]]:
        if self._fallback_enabled:
            return [item.copy() for item in self._fallback_history]

        try:
            raw_data = self.client.lrange(self.key, 0, -1)
            return [json.loads(item) for item in reversed(raw_data)]
        except redis.RedisError as exc:
            self._enable_fallback(exc)
            return [item.copy() for item in self._fallback_history]

    def pop_all_messages(self) -> list[dict[str, Any]]:
        messages = self.get_messages()
        if self._fallback_enabled:
            self._fallback_history.clear()
            return messages

        try:
            self.client.delete(self.key)
        except redis.RedisError as exc:
            self._enable_fallback(exc)
            self._fallback_history.clear()
        return messages

    def pop_archive_batch(self, batch_size: int | None = None) -> list[dict[str, Any]]:
        limit = batch_size or self.archive_batch_size
        if limit <= 0:
            return []
        if self._fallback_enabled:
            return [item.copy() for item in self._fallback_archive[:limit]]

        try:
            raw_data = self.client.lrange(self.archive_key, 0, limit - 1)
            return [json.loads(item) for item in reversed(raw_data)]
        except redis.RedisError as exc:
            self._enable_fallback(exc)
            return [item.copy() for item in self._fallback_archive[:limit]]

    def clear_archive_batch(self, batch_size: int | None = None) -> None:
        limit = batch_size or self.archive_batch_size
        if limit <= 0:
            return
        if self._fallback_enabled:
            del self._fallback_archive[:limit]
            return

        try:
            self.client.ltrim(self.archive_key, limit, -1)
        except redis.RedisError as exc:
            self._enable_fallback(exc)
            del self._fallback_archive[:limit]

    def _enable_fallback(self, exc: Exception) -> None:
        if self._fallback_enabled:
            return
        logger.warning("redis unavailable for working memory, using in-process fallback: %s", exc)
        self._fallback_enabled = True

    def _add_message_fallback(self, payload: dict[str, str]) -> None:
        self._fallback_history.append(payload.copy())
        overflow_count = max(0, len(self._fallback_history) - self.max_turns)
        if overflow_count:
            overflow = self._fallback_history[:overflow_count]
            self._fallback_archive.extend(item.copy() for item in overflow)
            del self._fallback_history[:overflow_count]
