import pytest

from app.memory.summarizer import Summarizer
from worker import compress_and_archive


class FakeLongTermMemory:
    def __init__(self) -> None:
        self.saved: list[tuple[str, dict]] = []

    def save_summary(self, summary: str, metadata: dict) -> str:
        self.saved.append((summary, metadata))
        return "archive-1"


class FakeWorkingMemory:
    def __init__(self, messages):
        self.messages = messages
        self.cleared = False

    def pop_archive_batch(self, batch_size: int):
        assert batch_size == 4
        return self.messages

    def clear_archive_batch(self, batch_size: int):
        assert batch_size == 4
        self.cleared = True


def test_summarizer_formats_structured_conversation_summary() -> None:
    summarizer = Summarizer()

    summary = summarizer.summarize_messages(
        session_id="demo-session",
        messages=[
            {"role": "user", "content": "我想做一个 FastAPI + Redis 的 Agent。"},
            {"role": "assistant", "content": "可以先从短期记忆和长期记忆拆分开始。"},
        ],
    )

    assert "demo-session" in summary
    assert "user: 我想做一个 FastAPI + Redis 的 Agent。" in summary
    assert "assistant: 可以先从短期记忆和长期记忆拆分开始。" in summary


def test_compress_and_archive_saves_summary_with_metadata() -> None:
    fake_store = FakeLongTermMemory()

    result = compress_and_archive(
        session_id="session-42",
        messages=[
            {"role": "user", "content": "之前我们讨论了 ChromaDB。"},
            {"role": "assistant", "content": "还讨论了 Celery 异步摘要。"},
        ],
        long_term_memory=fake_store,
    )

    assert result == "Archived summary for session=session-42"
    assert len(fake_store.saved) == 1
    summary, metadata = fake_store.saved[0]
    assert "session-42" in summary
    assert metadata == {
        "session_id": "session-42",
        "message_count": 2,
        "source": "archive",
    }


def test_summarize_empty_returns_no_content() -> None:
    assert Summarizer().summarize("") == "空会话，无需归档。"


def test_summarize_short_text() -> None:
    result = Summarizer().summarize("hello world")
    assert result == "摘要: hello world"


def test_summarize_long_text_truncates() -> None:
    long_text = "a" * 300
    result = Summarizer().summarize(long_text)
    assert result == "摘要: " + "a" * 237 + "..."


def test_summarize_messages_empty() -> None:
    result = Summarizer().summarize_messages("s1", [])
    assert result == "session=s1\n对话为空。"


@pytest.mark.asyncio
async def test_summarize_messages_async_empty() -> None:
    result = await Summarizer().summarize_messages_async("s1", [])
    assert result == "session=s1\n对话为空。"


@pytest.mark.asyncio
async def test_summarize_messages_async_llm_fallback(monkeypatch) -> None:
    from app.core.llm import LLMClient

    async def fake_generate(self: LLMClient, prompt: str) -> str:
        raise RuntimeError("LLM unavailable")

    monkeypatch.setattr(LLMClient, "generate", fake_generate)

    s = Summarizer(llm_client=LLMClient())
    result = await s.summarize_messages_async(
        "s1",
        [{"role": "user", "content": "hi"}],
    )
    assert "session=s1" in result
    assert "user: hi" in result


@pytest.mark.asyncio
async def test_summarize_messages_async_llm_success(monkeypatch) -> None:
    from app.core.llm import LLMClient

    async def fake_generate(self: LLMClient, prompt: str) -> str:
        return "用户问了关于性能的问题"

    monkeypatch.setattr(LLMClient, "generate", fake_generate)

    s = Summarizer(llm_client=LLMClient())
    result = await s.summarize_messages_async(
        "s1",
        [{"role": "user", "content": "帮我优化性能"}],
    )
    assert "session=s1" in result
    assert "用户问了关于性能的问题" in result


@pytest.mark.asyncio
async def test_summarize_messages_async_llm_returns_fallback_summary(monkeypatch) -> None:
    from app.core.llm import LLMClient

    async def fake_generate(self: LLMClient, prompt: str) -> str:
        return "离线兜底摘要（原因：LLM不可用）"

    monkeypatch.setattr(LLMClient, "generate", fake_generate)

    s = Summarizer(llm_client=LLMClient())
    result = await s.summarize_messages_async(
        "s1",
        [{"role": "user", "content": "hi"}],
    )
    assert "session=s1" in result
    assert "user: hi" in result  # Falls back to sync summarize_messages
