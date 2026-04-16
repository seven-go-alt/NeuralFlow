import pytest

from app.core.context import ContextBuilder


class FakeWorkingMemory:
    def get_messages(self):
        return [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]


class FakeLongTermMemory:
    def __init__(self):
        self.queries = []

    async def search(self, query: str, top_k: int = 3):
        self.queries.append((query, top_k))
        return ["用户喜欢简洁的回答"]


@pytest.mark.asyncio
async def test_context_builder_skips_long_term_lookup_for_default_intent() -> None:
    long_mem = FakeLongTermMemory()
    builder = ContextBuilder(
        session_id="demo",
        working_mem=FakeWorkingMemory(),
        long_mem=long_mem,
    )

    prompt = await builder.build_prompt(user_query="你好", intent="general")

    assert "相关历史记忆" not in prompt
    assert long_mem.queries == []
    assert "当前对话" in prompt


@pytest.mark.asyncio
async def test_context_builder_loads_long_term_lookup_for_history_intent() -> None:
    long_mem = FakeLongTermMemory()
    builder = ContextBuilder(
        session_id="demo",
        working_mem=FakeWorkingMemory(),
        long_mem=long_mem,
    )

    prompt = await builder.build_prompt(user_query="我之前喜欢什么风格？", intent="personal_preference")

    assert "相关历史记忆" in prompt
    assert long_mem.queries == [("我之前喜欢什么风格？", 3)]


@pytest.mark.asyncio
async def test_context_builder_uses_registered_default_skill_names_for_coding_intent() -> None:
    builder = ContextBuilder(
        session_id="demo",
        working_mem=FakeWorkingMemory(),
        long_mem=FakeLongTermMemory(),
    )

    prompt = await builder.build_prompt(user_query="帮我看一下这个 Python 报错", intent="coding")

    assert "当前可用技能: python, filesystem" in prompt
