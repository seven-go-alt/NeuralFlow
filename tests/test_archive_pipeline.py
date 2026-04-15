import pytest

from app.core.context import ContextBuilder
from app.memory.working import WorkingMemory
from app.utils.vector_client import InMemoryVectorClient


class FakeRedis:
    def __init__(self) -> None:
        self.storage: dict[str, list[str]] = {}

    def lpush(self, key: str, value: str) -> None:
        self.storage.setdefault(key, []).insert(0, value)

    def ltrim(self, key: str, start: int, end: int) -> None:
        items = self.storage.get(key, [])
        normalized_end = None if end == -1 else end + 1
        self.storage[key] = items[start:normalized_end]

    def lrange(self, key: str, start: int, end: int):
        items = self.storage.get(key, [])
        normalized_end = None if end == -1 else end + 1
        return items[start:normalized_end]

    def delete(self, key: str) -> None:
        self.storage.pop(key, None)


class FakeLongTermMemory:
    def __init__(self) -> None:
        self.queries: list[tuple[str, int]] = []

    async def search(self, query: str, top_k: int = 3):
        self.queries.append((query, top_k))
        return ["历史记忆：用户更偏好模块化设计。"]


@pytest.mark.asyncio
async def test_context_builder_uses_memory_strategy_and_skill_whitelist() -> None:
    builder = ContextBuilder(
        session_id="demo",
        working_mem=type("WM", (), {"get_messages": lambda self: [{"role": "user", "content": "hello"}]})(),
        long_mem=FakeLongTermMemory(),
    )

    prompt = await builder.build_prompt(
        user_query="帮我回忆一下之前的设计",
        intent="query_history",
        memory_strategy="long_term",
        skill_whitelist=["memory", "planner"],
    )

    assert "相关历史记忆" in prompt
    assert "当前可用技能" in prompt
    assert "memory" in prompt
    assert "planner" in prompt


def test_working_memory_moves_overflow_messages_to_archive_queue() -> None:
    redis_client = FakeRedis()
    memory = WorkingMemory(session_id="demo", max_turns=2, archive_batch_size=4, client=redis_client)

    memory.add_message("user", "u1")
    memory.add_message("assistant", "a1")
    memory.add_message("user", "u2")
    memory.add_message("assistant", "a2")

    live_messages = memory.get_messages()
    archived_batch = memory.pop_archive_batch(batch_size=4)

    assert live_messages == [
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    assert archived_batch == [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]
