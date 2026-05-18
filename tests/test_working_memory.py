from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import redis

from app.memory.working import WorkingMemory


class MockRedis:
    """In-memory mock of Redis list operations."""

    def __init__(self) -> None:
        self._store: dict[str, list[str]] = {}

    def lpush(self, key: str, value: str) -> int:
        if key not in self._store:
            self._store[key] = []
        self._store[key].insert(0, value)
        return 1

    def ltrim(self, key: str, start: int, end: int) -> None:
        if key in self._store:
            if end == -1:
                self._store[key] = self._store[key][start:]
            else:
                self._store[key] = self._store[key][start : end + 1]

    def lrange(self, key: str, start: int, end: int) -> list[str]:
        items = self._store.get(key, [])
        if end == -1:
            return items[start:]
        return items[start : end + 1]

    def delete(self, key: str) -> int:
        return 1 if self._store.pop(key, None) is not None else 0


@pytest.fixture
def mock_redis_client() -> MockRedis:
    return MockRedis()


class TestWorkingMemory:
    def test_add_and_get_messages(self, mock_redis_client: MockRedis) -> None:
        wm = WorkingMemory("session-1", max_turns=10, client=mock_redis_client)
        wm.add("user", "hello")
        wm.add("assistant", "hi there")
        messages = wm.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_add_message_alias(self, mock_redis_client: MockRedis) -> None:
        wm = WorkingMemory("session-1", max_turns=10, client=mock_redis_client)
        wm.add_message("user", "hello")
        assert len(wm.get_messages()) == 1

    def test_max_turns_overflow(self, mock_redis_client: MockRedis) -> None:
        wm = WorkingMemory(
            "session-1", max_turns=2, archive_batch_size=10, client=mock_redis_client
        )
        wm.add("user", "msg1")
        wm.add("user", "msg2")
        wm.add("user", "msg3")
        messages = wm.get_messages()
        assert len(messages) == 2

    def test_pop_all_messages(self, mock_redis_client: MockRedis) -> None:
        wm = WorkingMemory("session-1", max_turns=10, client=mock_redis_client)
        wm.add("user", "hello")
        popped = wm.pop_all_messages()
        assert len(popped) == 1
        assert wm.get_messages() == []

    def test_pop_archive_batch(self, mock_redis_client: MockRedis) -> None:
        wm = WorkingMemory("session-1", max_turns=2, archive_batch_size=5, client=mock_redis_client)
        wm.add("user", "m1")
        wm.add("user", "m2")
        wm.add("user", "m3")
        batch = wm.pop_archive_batch(batch_size=5)
        assert len(batch) >= 1

    def test_clear_archive_batch(self, mock_redis_client: MockRedis) -> None:
        wm = WorkingMemory("session-1", max_turns=2, archive_batch_size=5, client=mock_redis_client)
        wm.add("user", "m1")
        wm.add("user", "m2")
        wm.add("user", "m3")
        wm.clear_archive_batch(batch_size=5)

    def test_negative_batch_size_noop(self, mock_redis_client: MockRedis) -> None:
        wm = WorkingMemory("session-1", client=mock_redis_client)
        assert wm.pop_archive_batch(batch_size=0) == []
        wm.clear_archive_batch(batch_size=0)

    def test_redis_error_triggers_fallback(self) -> None:
        failing_client = MagicMock()
        failing_client.lpush.side_effect = redis.RedisError("connection refused")

        wm = WorkingMemory("session-1", max_turns=10, client=failing_client)
        wm.add("user", "hello")
        messages = wm.get_messages()
        assert len(messages) == 1
        assert messages[0]["content"] == "hello"

    def test_fallback_archive(self) -> None:
        failing_client = MagicMock()
        failing_client.lpush.side_effect = redis.RedisError("offline")

        wm = WorkingMemory("session-1", max_turns=1, archive_batch_size=10, client=failing_client)
        wm.add("user", "m1")
        wm.add("user", "m2")
        batch = wm.pop_archive_batch()
        assert len(batch) >= 1

    def test_fallback_clear_archive(self) -> None:
        failing_client = MagicMock()
        failing_client.lpush.side_effect = redis.RedisError("offline")

        wm = WorkingMemory("session-1", max_turns=1, archive_batch_size=10, client=failing_client)
        wm.add("user", "m1")
        wm.add("user", "m2")
        wm.clear_archive_batch()

    def test_pop_all_fallback(self) -> None:
        failing_client = MagicMock()
        failing_client.lpush.side_effect = redis.RedisError("offline")

        wm = WorkingMemory("session-1", max_turns=10, client=failing_client)
        wm.add("user", "hello")
        popped = wm.pop_all_messages()
        assert len(popped) == 1

    def test_fallback_get_then_pop_all(self) -> None:
        wm = WorkingMemory("session-1", max_turns=10)
        wm._fallback_enabled = True
        wm._fallback_history.append({"role": "user", "content": "hi"})
        popped = wm.pop_all_messages()
        assert len(popped) == 1
        assert wm._fallback_history == []

    def test_tenant_id_in_key(self, mock_redis_client: MockRedis) -> None:
        wm = WorkingMemory("session-1", tenant_id="t1", client=mock_redis_client)
        assert "tenant:t1" in wm.key

    def test_tenant_id_default(self, mock_redis_client: MockRedis) -> None:
        wm = WorkingMemory("session-1", client=mock_redis_client)
        assert "public" in wm.key
