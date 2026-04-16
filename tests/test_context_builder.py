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

    async def search(self, query: str, top_k: int = 3, session_id: str | None = None):
        self.queries.append((query, top_k, session_id))
        return ["用户喜欢简洁的回答"]


class FakeTokenBudgetManager:
    def __init__(self) -> None:
        self.segments = []

    def trim_context(self, segments):
        self.segments = segments

        class Result:
            token_before_trim = 120
            token_after_trim = 80
            soft_limit_exceeded = True
            hard_limit_exceeded = True
            dropped_segment_names = ["retrieved_memory", "early_chat"]
            trimmed_text = "你是一个智能助手。\n---\n当前对话:\n[{'role': 'assistant', 'content': 'Hello'}]\n---\n用户问题:\n你好"

        return Result()


class LegacyLongTermMemory:
    def __init__(self):
        self.queries = []

    async def search(self, query: str, top_k: int = 3):
        self.queries.append((query, top_k))
        return ["旧接口返回的历史记忆"]


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
    assert long_mem.queries == [("我之前喜欢什么风格？", 3, "demo")]


@pytest.mark.asyncio
async def test_context_builder_supports_legacy_long_term_search_signature() -> None:
    long_mem = LegacyLongTermMemory()
    builder = ContextBuilder(
        session_id="demo",
        working_mem=FakeWorkingMemory(),
        long_mem=long_mem,
    )

    prompt = await builder.build_prompt(user_query="帮我回忆一下之前的设计", intent="query_history")

    assert "相关历史记忆" in prompt
    assert long_mem.queries == [("帮我回忆一下之前的设计", 3)]


@pytest.mark.asyncio
async def test_context_builder_uses_registered_default_skill_names_for_coding_intent() -> None:
    builder = ContextBuilder(
        session_id="demo",
        working_mem=FakeWorkingMemory(),
        long_mem=FakeLongTermMemory(),
    )

    prompt = await builder.build_prompt(user_query="帮我看一下这个 Python 报错", intent="coding")

    assert "当前可用技能: python, filesystem" in prompt


@pytest.mark.asyncio
async def test_context_builder_applies_token_budget_and_exposes_trim_metrics() -> None:
    long_mem = FakeLongTermMemory()
    token_budget = FakeTokenBudgetManager()
    builder = ContextBuilder(
        session_id="demo",
        working_mem=FakeWorkingMemory(),
        long_mem=long_mem,
        token_budget_manager=token_budget,
    )

    prompt = await builder.build_prompt(user_query="你好", intent="query_history")

    assert prompt == "你是一个智能助手。\n---\n当前对话:\n[{'role': 'assistant', 'content': 'Hello'}]\n---\n用户问题:\n你好"
    assert [segment.name for segment in token_budget.segments] == [
        "system_prompt",
        "retrieved_memory",
        "early_chat",
        "recent_chat",
        "user_query",
    ]
    assert builder.last_trim_metadata == {
        "token_before_trim": 120,
        "token_after_trim": 80,
        "soft_limit_exceeded": True,
        "hard_limit_exceeded": True,
        "dropped_segment_names": ["retrieved_memory", "early_chat"],
    }
