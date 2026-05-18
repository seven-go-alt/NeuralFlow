from __future__ import annotations

import pytest

from app.core.llm import LLMClient, build_rule_based_fallback_reply


class FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = type("Message", (), {"content": content})()


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [FakeChoice(content)]


@pytest.mark.asyncio
async def test_llm_client_falls_back_to_ollama_when_primary_provider_fails(monkeypatch) -> None:
    calls: list[str] = []

    async def fake_acompletion(*, model: str, messages: list[dict], stream: bool = False, **kwargs):
        calls.append(model)
        if model == "primary-model":
            raise RuntimeError("invalid openai key")
        return FakeResponse("ollama reply")

    monkeypatch.setattr("app.core.llm.acompletion", fake_acompletion)

    client = LLMClient(model="primary-model")
    client.fallback_model = "ollama/qwen2.5:7b"
    client.offline_fallback_enabled = True

    reply = await client.generate("你好")

    assert reply == "ollama reply"
    assert calls == ["primary-model", "ollama/qwen2.5:7b"]


@pytest.mark.asyncio
async def test_llm_client_returns_rule_based_summary_when_all_models_fail(monkeypatch) -> None:
    async def fake_acompletion(*, model: str, messages: list[dict], stream: bool = False, **kwargs):
        raise RuntimeError(f"{model} unavailable")

    monkeypatch.setattr("app.core.llm.acompletion", fake_acompletion)

    client = LLMClient(model="primary-model")
    client.fallback_model = "ollama/qwen2.5:7b"
    client.offline_fallback_enabled = True

    reply = await client.generate("用户问：帮我总结 Redis 和 Chroma 的隔离差异")

    assert "离线兜底摘要" in reply
    assert "Redis" in reply
    assert "Chroma" in reply


def test_build_rule_based_fallback_reply_with_error() -> None:
    prompt = "用户问：介绍一下请假制度"
    result = build_rule_based_fallback_reply(prompt, error=RuntimeError("LLM connection refused"))
    assert "离线兜底摘要" in result
    assert "LLM connection refused" in result
    assert "请假制度" in result


def test_build_rule_based_fallback_reply_without_error() -> None:
    result = build_rule_based_fallback_reply("帮我总结一下")
    assert "离线兜底摘要" in result
    assert "帮我总结一下" in result


@pytest.mark.asyncio
async def test_generate_raises_when_fallback_disabled(monkeypatch) -> None:
    async def fake_acompletion(**kwargs):
        raise RuntimeError("primary failed")

    monkeypatch.setattr("app.core.llm.acompletion", fake_acompletion)

    client = LLMClient(model="primary-model")
    client.offline_fallback_enabled = False

    with pytest.raises(RuntimeError, match="primary failed"):
        await client.generate("hello")


@pytest.mark.asyncio
async def test_generate_fallback_no_model_returns_rule_based(monkeypatch) -> None:
    """When fallback_model is not set but fallback is enabled, return rule-based reply."""

    async def fake_acompletion(**kwargs):
        raise RuntimeError("primary failed")

    monkeypatch.setattr("app.core.llm.acompletion", fake_acompletion)

    client = LLMClient(model="primary-model")
    client.offline_fallback_enabled = True
    client.fallback_model = None

    reply = await client.generate("test query")
    assert "离线兜底摘要" in reply


@pytest.mark.asyncio
async def test_stream_generate_raises_when_fallback_disabled(monkeypatch) -> None:
    """stream_generate raises when primary fails and offline_fallback_enabled=False."""

    async def fake_stream_once(self, prompt, model, include_thinking=False):
        raise RuntimeError("stream failed")
        yield  # pragma: no cover

    monkeypatch.setattr(LLMClient, "_stream_once", fake_stream_once)

    client = LLMClient(model="primary-model")
    client.offline_fallback_enabled = False

    with pytest.raises(RuntimeError, match="stream failed"):
        async for _ in client.stream_generate("hello"):
            pass


@pytest.mark.asyncio
async def test_stream_generate_fallback_no_model(monkeypatch) -> None:
    """stream_generate falls back to rule-based when no fallback model is set."""

    async def fake_stream_once(self, prompt, model, include_thinking=False):
        raise RuntimeError("stream failed")
        yield  # pragma: no cover

    monkeypatch.setattr(LLMClient, "_stream_once", fake_stream_once)

    client = LLMClient(model="primary-model")
    client.offline_fallback_enabled = True
    client.fallback_model = None

    chunks = [chunk async for chunk in client.stream_generate("test query")]
    assert len(chunks) == 1
    assert chunks[0]["event"] == "message"
    assert "离线兜底摘要" in chunks[0]["data"]


@pytest.mark.asyncio
async def test_generate_once_passes_api_base_and_key(monkeypatch) -> None:
    """_generate_once passes api_base and api_key to acompletion when configured."""

    captured: dict = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        msg = type("Msg", (), {"content": "ok"})()
        choice = type("Choice", (), {"message": msg})()
        return type("Resp", (), {"choices": [choice]})()

    monkeypatch.setattr("app.core.llm.acompletion", fake_acompletion)

    client = LLMClient(model="test-model")
    client.api_base = "https://custom.api/v1"
    client.api_key = "sk-custom"

    reply = await client._generate_once("prompt", "test-model")

    assert reply == "ok"
    assert captured["api_base"] == "https://custom.api/v1"
    assert captured["api_key"] == "sk-custom"


def test_extract_delta_from_dict_choice_edge_case() -> None:
    client = LLMClient(model="test-model")
    # Dict-style choice where delta is None
    chunk = {"choices": [{"delta": None}]}
    assert client._extract_delta(chunk) == ""

    class ChoiceWithContent:
        delta = {"content": None}

    class ChunkWithNoneContent:
        choices = [ChoiceWithContent()]

    assert client._extract_delta(ChunkWithNoneContent()) == ""


def test_llm_client_first_choice_returns_none() -> None:
    client = LLMClient(model="test-model")

    class EmptyChunk:
        choices = []

    assert client._first_choice(EmptyChunk()) is None

    class DictChunk:
        choices = None

    assert client._first_choice(DictChunk()) is None

    assert client._first_choice(object()) is None


def test_llm_client_extract_delta_edge_cases() -> None:
    client = LLMClient(model="test-model")

    # Dict delta with content
    class DictChoice:
        delta = {"content": "hello"}

    class DictChunk:
        choices = [DictChoice()]

    assert client._extract_delta(DictChunk()) == "hello"

    # No delta
    class NoDeltaChoice:
        delta = None

    class NoDeltaChunk:
        choices = [NoDeltaChoice()]

    assert client._extract_delta(NoDeltaChunk()) == ""

    # Dict-style chunk
    dict_chunk = {"choices": [{"delta": {"content": "from dict"}}]}
    assert client._extract_delta(dict_chunk) == "from dict"


@pytest.mark.asyncio
async def test_stream_generate_falls_back_to_fallback_model(monkeypatch) -> None:
    """stream_generate yields fallback model chunks when primary _stream_once fails."""

    async def fake_stream_once(self, prompt, model, include_thinking=False):
        if "primary" in model:
            raise RuntimeError("primary stream failed")
        for chunk in [
            {"event": "message", "data": "fallback "},
            {"event": "message", "data": "reply"},
        ]:
            yield chunk

    monkeypatch.setattr(LLMClient, "_stream_once", fake_stream_once)

    client = LLMClient(model="primary-model")
    client.fallback_model = "ollama/qwen2.5:7b"
    client.offline_fallback_enabled = True

    chunks = [chunk async for chunk in client.stream_generate("hello")]
    assert len(chunks) == 2
    assert chunks[0]["data"] == "fallback "
    assert chunks[1]["data"] == "reply"


@pytest.mark.asyncio
async def test_stream_generate_rule_based_when_all_streams_fail(monkeypatch) -> None:
    """stream_generate yields rule-based reply when both primary and fallback fail."""

    async def fake_stream_once(self, prompt, model, include_thinking=False):
        raise RuntimeError(f"{model} stream failed")
        yield  # pragma: no cover

    monkeypatch.setattr(LLMClient, "_stream_once", fake_stream_once)

    client = LLMClient(model="primary-model")
    client.fallback_model = "ollama/qwen2.5:7b"
    client.offline_fallback_enabled = True

    chunks = [chunk async for chunk in client.stream_generate("test query")]
    assert len(chunks) == 1
    assert chunks[0]["event"] == "message"
    assert "离线兜底摘要" in chunks[0]["data"]


def test_llm_client_extract_thinking() -> None:
    client = LLMClient(model="test-model")

    # reasoning_content from object delta
    class ThinkingDelta:
        reasoning_content = "let me think..."

    class ThinkingChoice:
        delta = ThinkingDelta()

    class ThinkingChunk:
        choices = [ThinkingChoice()]

    thinking = client._extract_thinking(ThinkingChunk())
    assert thinking == "let me think..."

    # reasoning from dict delta
    dict_chunk: dict = {"choices": [{"delta": {"reasoning": "step by step..."}}]}
    thinking = client._extract_thinking(dict_chunk)
    assert thinking == "step by step..."

    # No thinking content
    class NoThinkingDelta:
        content = "direct reply"

    class NoThinkingChoice:
        delta = NoThinkingDelta()

    class NoThinkingChunk:
        choices = [NoThinkingChoice()]

    assert client._extract_thinking(NoThinkingChunk()) == ""


class _StreamChunk:
    """Simulate a litellm streaming chunk with choice delta."""

    def __init__(self, content: str = "", reasoning: str = "") -> None:
        delta = type("Delta", (), {"content": content, "reasoning_content": reasoning})()
        choice = type("Choice", (), {"delta": delta})()
        self.choices = [choice]


@pytest.mark.asyncio
async def test_stream_once_yields_message_chunks(monkeypatch) -> None:
    chunks = [_StreamChunk("hello "), _StreamChunk("world")]

    async def fake_stream():
        for c in chunks:
            yield c

    async def fake_acompletion(**kwargs):
        assert kwargs.get("stream") is True
        return fake_stream()

    monkeypatch.setattr("app.core.llm.acompletion", fake_acompletion)

    client = LLMClient(model="test-model")
    collected = [chunk async for chunk in client._stream_once("prompt", "test-model")]

    assert collected == [
        {"event": "message", "data": "hello "},
        {"event": "message", "data": "world"},
    ]


@pytest.mark.asyncio
async def test_stream_once_with_thinking(monkeypatch) -> None:
    chunks = [_StreamChunk("answer", "let me think..."), _StreamChunk(" more")]

    async def fake_stream():
        for c in chunks:
            yield c

    async def fake_acompletion(**kwargs):
        return fake_stream()

    monkeypatch.setattr("app.core.llm.acompletion", fake_acompletion)

    client = LLMClient(model="test-model")
    collected = [
        chunk async for chunk in client._stream_once("prompt", "test-model", include_thinking=True)
    ]

    assert collected == [
        {"event": "thinking", "data": "let me think..."},
        {"event": "message", "data": "answer"},
        {"event": "message", "data": " more"},
    ]


@pytest.mark.asyncio
async def test_stream_once_skips_empty_delta(monkeypatch) -> None:
    async def fake_stream():
        yield _StreamChunk("")
        yield _StreamChunk("real")

    async def fake_acompletion(**kwargs):
        return fake_stream()

    monkeypatch.setattr("app.core.llm.acompletion", fake_acompletion)

    client = LLMClient(model="test-model")
    collected = [chunk async for chunk in client._stream_once("prompt", "test-model")]

    assert collected == [{"event": "message", "data": "real"}]


def test_extract_delta_empty_choices() -> None:
    """_extract_delta returns empty string when chunk has no choices."""
    client = LLMClient(model="test-model")

    class EmptyChunk:
        choices = []

    assert client._extract_delta(EmptyChunk()) == ""


def test_extract_thinking_empty_choices() -> None:
    """_extract_thinking returns empty string when chunk has no choices."""
    client = LLMClient(model="test-model")

    class EmptyChunk:
        choices = []

    assert client._extract_thinking(EmptyChunk()) == ""


@pytest.mark.asyncio
async def test_stream_generate_success(monkeypatch) -> None:
    """stream_generate yields chunks from primary _stream_once."""

    async def fake_stream_once(self, prompt: str, model: str, include_thinking: bool = False):
        yield {"event": "message", "data": "hello "}
        yield {"event": "message", "data": "world"}

    monkeypatch.setattr(LLMClient, "_stream_once", fake_stream_once)

    client = LLMClient(model="test-model")
    chunks = [chunk async for chunk in client.stream_generate("test")]

    assert chunks == [
        {"event": "message", "data": "hello "},
        {"event": "message", "data": "world"},
    ]


@pytest.mark.asyncio
async def test_stream_generate_primary_fails_fallback_succeeds(monkeypatch) -> None:
    """stream_generate falls back to fallback model when primary fails."""

    async def fake_stream_once(self, prompt: str, model: str, include_thinking: bool = False):
        count = getattr(self, "_stream_call_count", 0) + 1
        self._stream_call_count = count
        if count == 1:
            raise RuntimeError(f"{model} failed")
            yield  # pragma: no cover
        yield {"event": "message", "data": "fallback reply by " + model}

    monkeypatch.setattr(LLMClient, "_stream_once", fake_stream_once)

    client = LLMClient(model="primary-model")
    client.offline_fallback_enabled = True
    client.fallback_model = "fallback-model"

    chunks = [chunk async for chunk in client.stream_generate("test")]
    assert len(chunks) == 1
    assert "fallback reply" in chunks[0]["data"]


@pytest.mark.asyncio
async def test_stream_generate_both_models_fail(monkeypatch) -> None:
    """Both primary and fallback fail, returns rule-based summary."""

    async def fake_stream_once(self, prompt: str, model: str, include_thinking: bool = False):
        raise RuntimeError(f"{model} failed")
        yield  # pragma: no cover

    monkeypatch.setattr(LLMClient, "_stream_once", fake_stream_once)

    client = LLMClient(model="primary-model")
    client.offline_fallback_enabled = True
    client.fallback_model = "fallback-model"

    chunks = [chunk async for chunk in client.stream_generate("test")]
    assert len(chunks) == 1
    assert chunks[0]["event"] == "message"
    assert "离线兜底摘要" in chunks[0]["data"]
