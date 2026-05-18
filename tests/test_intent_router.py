import pytest
from fastapi.testclient import TestClient

from app.core.intent_router import IntentDetectionResult, IntentPolicy, IntentRouter
from app.main import app


class FakeLLMClassifier:
    def __init__(self, response: list[str] | Exception) -> None:
        self.response = response
        self.calls: list[str] = []

    async def classify(self, text: str) -> list[str]:
        self.calls.append(text)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@pytest.mark.asyncio
async def test_intent_router_detects_multiple_rule_based_intents() -> None:
    router = IntentRouter(
        llm_classifier=FakeLLMClassifier(["general"]),
        keyword_rules={
            "query_history": ["之前", "历史"],
            "coding": ["代码", "bug"],
        },
        policy_map={
            "query_history": IntentPolicy(memory_strategy="long_term", skill_whitelist=["memory"]),
            "coding": IntentPolicy(memory_strategy="working_only", skill_whitelist=["python"]),
            "general": IntentPolicy(memory_strategy="working_only", skill_whitelist=[]),
        },
    )

    result = await router.detect("我之前写过的代码还有 bug 吗？")

    assert result.intents == ["query_history", "coding"]
    assert result.primary_intent == "query_history"
    assert result.used_fallback is False
    assert result.policies["coding"].skill_whitelist == ["python"]


@pytest.mark.asyncio
async def test_intent_router_falls_back_to_llm_when_rules_miss() -> None:
    llm = FakeLLMClassifier(["planning", "coding"])
    router = IntentRouter(
        llm_classifier=llm,
        keyword_rules={"coding": ["代码"]},
        policy_map={
            "planning": IntentPolicy(memory_strategy="working_only", skill_whitelist=["planner"]),
            "coding": IntentPolicy(memory_strategy="working_only", skill_whitelist=["python"]),
            "general": IntentPolicy(memory_strategy="working_only", skill_whitelist=[]),
        },
    )

    result = await router.detect("帮我规划一下这个功能怎么拆分")

    assert result.intents == ["planning", "coding"]
    assert result.primary_intent == "planning"
    assert result.used_fallback is True
    assert llm.calls == ["帮我规划一下这个功能怎么拆分"]


@pytest.mark.asyncio
async def test_intent_router_returns_general_when_llm_fails() -> None:
    router = IntentRouter(
        llm_classifier=FakeLLMClassifier(RuntimeError("llm down")),
        keyword_rules={"coding": ["代码"]},
        policy_map={
            "general": IntentPolicy(memory_strategy="working_only", skill_whitelist=[]),
            "coding": IntentPolicy(memory_strategy="working_only", skill_whitelist=["python"]),
        },
    )

    result = await router.detect("随便聊聊今天吃什么")

    assert result == IntentDetectionResult(
        intents=["general"],
        primary_intent="general",
        used_fallback=False,
        policies={"general": IntentPolicy(memory_strategy="working_only", skill_whitelist=[])},
    )


def test_intent_router_rejects_unknown_default_intent() -> None:
    with pytest.raises(ValueError, match="Unknown default intent"):
        IntentRouter(
            llm_classifier=FakeLLMClassifier(["general"]),
            keyword_rules={"coding": ["代码"]},
            policy_map={
                "general": IntentPolicy(memory_strategy="working_only", skill_whitelist=[]),
                "coding": IntentPolicy(memory_strategy="working_only", skill_whitelist=["python"]),
            },
            default_intent="planning",
        )


def test_intent_detect_endpoint_returns_structured_result(monkeypatch) -> None:
    class StubRouter:
        async def detect(self, text: str) -> IntentDetectionResult:
            assert text == "帮我查一下之前的代码"
            return IntentDetectionResult(
                intents=["query_history", "coding"],
                primary_intent="query_history",
                used_fallback=False,
                policies={
                    "query_history": IntentPolicy(
                        memory_strategy="long_term", skill_whitelist=["memory"]
                    ),
                    "coding": IntentPolicy(
                        memory_strategy="working_only", skill_whitelist=["python"]
                    ),
                },
            )

    monkeypatch.setattr("app.main.intent_router", StubRouter())
    client = TestClient(app)

    response = client.post("/api/v1/intent/detect", json={"message": "帮我查一下之前的代码"})

    assert response.status_code == 200
    assert response.json() == {
        "intents": ["query_history", "coding"],
        "primary_intent": "query_history",
        "used_fallback": False,
        "policies": {
            "query_history": {
                "memory_strategy": "long_term",
                "skill_whitelist": ["memory"],
            },
            "coding": {
                "memory_strategy": "working_only",
                "skill_whitelist": ["python"],
            },
        },
    }


def test_dedupe_preserve_order() -> None:
    from app.core.intent_router import _dedupe_preserve_order

    assert _dedupe_preserve_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]
    assert _dedupe_preserve_order([]) == []
    assert _dedupe_preserve_order(["x"]) == ["x"]


def test_parse_llm_intents() -> None:
    from app.core.intent_router import _parse_llm_intents

    assert _parse_llm_intents('["coding", "planning"]') == ["coding", "planning"]
    assert _parse_llm_intents('["general"]') == ["general"]


def test_parse_llm_intents_with_code_fence() -> None:
    from app.core.intent_router import _parse_llm_intents

    result = _parse_llm_intents('```json\n["query_history"]\n```')
    assert result == ["query_history"]


def test_parse_llm_intents_fail() -> None:
    import json

    from app.core.intent_router import _parse_llm_intents

    with pytest.raises(json.JSONDecodeError):
        _parse_llm_intents("not json")


def test_build_policy_map() -> None:
    from app.core.intent_router import IntentPolicy, _build_policy_map

    raw = {
        "coding": {"memory_strategy": "working_only", "skill_whitelist": ["python"]},
        "query_history": {"memory_strategy": "long_term", "skill_whitelist": ["memory"]},
    }
    policies = _build_policy_map(raw)
    assert isinstance(policies["coding"], IntentPolicy)
    assert policies["coding"].memory_strategy == "working_only"
    assert policies["query_history"].skill_whitelist == ["memory"]


def test_build_policy_map_adds_general() -> None:
    from app.core.intent_router import _build_policy_map

    policies = _build_policy_map({})
    assert "general" in policies
    assert policies["general"].memory_strategy == "working_only"
    assert policies["general"].skill_whitelist == []


def test_intent_router_build_result_normalizes_unknown_to_general() -> None:
    from app.core.intent_router import IntentPolicy, IntentRouter

    router = IntentRouter(
        llm_classifier=FakeLLMClassifier(["general"]),
        keyword_rules={},
        policy_map={
            "general": IntentPolicy(memory_strategy="working_only", skill_whitelist=[]),
        },
    )
    result = router._build_result(["unknown_intent"], used_fallback=False)
    assert result.intents == ["general"]
    assert result.primary_intent == "general"


class FakeEmbeddingClassifier:
    def __init__(self, response: list[str] | Exception) -> None:
        self.response = response

    async def classify(self, text: str) -> list[str]:
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@pytest.mark.asyncio
async def test_intent_router_empty_text_returns_default() -> None:
    router = IntentRouter(
        llm_classifier=FakeLLMClassifier(["general"]),
        keyword_rules={"_dummy": ["__no_match__"]},
        policy_map={
            "general": IntentPolicy(memory_strategy="working_only", skill_whitelist=[]),
            "_dummy": IntentPolicy(memory_strategy="working_only", skill_whitelist=[]),
        },
    )
    result = await router.detect("")
    assert result.intents == ["general"]
    assert result.primary_intent == "general"
    assert result.used_fallback is False


@pytest.mark.asyncio
async def test_intent_router_uses_embedding_fallback() -> None:
    router = IntentRouter(
        llm_classifier=FakeLLMClassifier(["general"]),
        keyword_rules={"_dummy": ["__no_match__"]},
        policy_map={
            "general": IntentPolicy(memory_strategy="working_only", skill_whitelist=[]),
            "query_history": IntentPolicy(memory_strategy="long_term", skill_whitelist=["memory"]),
            "_dummy": IntentPolicy(memory_strategy="working_only", skill_whitelist=[]),
        },
        embedding_fallback_enabled=True,
        embedding_classifier=FakeEmbeddingClassifier(["query_history"]),
    )
    result = await router.detect("hello world")
    assert "query_history" in result.intents
    assert result.used_fallback is True


@pytest.mark.asyncio
async def test_intent_router_embedding_fallback_fails_continues_to_llm() -> None:
    llm = FakeLLMClassifier(["planning"])
    router = IntentRouter(
        llm_classifier=llm,
        keyword_rules={"_dummy": ["__no_match__"]},
        policy_map={
            "general": IntentPolicy(memory_strategy="working_only", skill_whitelist=[]),
            "planning": IntentPolicy(memory_strategy="working_only", skill_whitelist=["planner"]),
            "_dummy": IntentPolicy(memory_strategy="working_only", skill_whitelist=[]),
        },
        embedding_fallback_enabled=True,
        embedding_classifier=FakeEmbeddingClassifier(RuntimeError("embedding down")),
    )
    result = await router.detect("hello world")
    assert "planning" in result.intents
    assert result.used_fallback is True


@pytest.mark.asyncio
async def test_intent_router_embedding_returns_empty_skips_to_llm() -> None:
    llm = FakeLLMClassifier(["coding"])
    router = IntentRouter(
        llm_classifier=llm,
        keyword_rules={"_dummy": ["__no_match__"]},
        policy_map={
            "general": IntentPolicy(memory_strategy="working_only", skill_whitelist=[]),
            "coding": IntentPolicy(memory_strategy="working_only", skill_whitelist=["python"]),
            "_dummy": IntentPolicy(memory_strategy="working_only", skill_whitelist=[]),
        },
        embedding_fallback_enabled=True,
        embedding_classifier=FakeEmbeddingClassifier([]),
    )
    result = await router.detect("hello world")
    assert "coding" in result.intents
    assert result.used_fallback is True


@pytest.mark.asyncio
async def test_intent_router_llm_fallback_disabled_returns_default() -> None:
    router = IntentRouter(
        llm_classifier=FakeLLMClassifier(["general"]),
        keyword_rules={"_dummy": ["__no_match__"]},
        policy_map={
            "general": IntentPolicy(memory_strategy="working_only", skill_whitelist=[]),
            "_dummy": IntentPolicy(memory_strategy="working_only", skill_whitelist=[]),
        },
        llm_fallback_enabled=False,
    )
    result = await router.detect("查一下记录")
    assert result.intents == ["general"]
    assert result.primary_intent == "general"
    assert result.used_fallback is False


def test_match_keywords_case_insensitive() -> None:
    router = IntentRouter(
        llm_classifier=FakeLLMClassifier(["general"]),
        keyword_rules={"coding": ["代码", "BUG"]},
        policy_map={
            "general": IntentPolicy(memory_strategy="working_only", skill_whitelist=[]),
            "coding": IntentPolicy(memory_strategy="working_only", skill_whitelist=["python"]),
        },
    )
    assert "coding" in router._match_keywords("修复了一个Bug")
    assert "coding" in router._match_keywords("找到代码的bug")
    assert router._match_keywords("随便聊聊吃什么") == []


def test_parse_llm_intents_not_list_raises_value_error() -> None:
    from app.core.intent_router import _parse_llm_intents

    with pytest.raises(ValueError, match="must return a JSON array"):
        _parse_llm_intents('{"key": "value"}')

    with pytest.raises(ValueError, match="must return a JSON array"):
        _parse_llm_intents('"string"')


@pytest.mark.asyncio
async def test_litellm_intent_classifier(monkeypatch) -> None:
    from app.core.intent_router import LiteLLMIntentClassifier
    from app.core.llm import LLMClient

    async def fake_generate(self: LLMClient, prompt: str) -> str:
        return '["coding", "planning"]'

    monkeypatch.setattr(LLMClient, "generate", fake_generate)

    classifier = LiteLLMIntentClassifier(llm_client=LLMClient(model="test"))
    intents = await classifier.classify("写代码")
    assert intents == ["coding", "planning"]


@pytest.mark.asyncio
async def test_litellm_intent_classifier_returns_general(monkeypatch) -> None:
    from app.core.intent_router import LiteLLMIntentClassifier
    from app.core.llm import LLMClient

    async def fake_generate(self: LLMClient, prompt: str) -> str:
        return '["general"]'

    monkeypatch.setattr(LLMClient, "generate", fake_generate)

    classifier = LiteLLMIntentClassifier(llm_client=LLMClient(model="test"))
    intents = await classifier.classify("随便聊聊")
    assert intents == ["general"]


def test_intent_router_build_result_mixed_known_unknown() -> None:
    from app.core.intent_router import IntentPolicy, IntentRouter

    router = IntentRouter(
        llm_classifier=FakeLLMClassifier(["general"]),
        keyword_rules={},
        policy_map={
            "general": IntentPolicy(memory_strategy="working_only", skill_whitelist=[]),
            "coding": IntentPolicy(memory_strategy="working_only", skill_whitelist=["python"]),
        },
    )
    result = router._build_result(["coding", "unknown_intent", "general"], used_fallback=True)
    assert result.intents == ["coding", "general"]
    assert result.primary_intent == "coding"
    assert result.used_fallback is True
    assert result.policies["coding"].skill_whitelist == ["python"]


@pytest.mark.asyncio
async def test_embedding_classifier_matches_intent(monkeypatch) -> None:
    from app.core.intent_router import INTENT_EXAMPLES, EmbeddingIntentClassifier

    async def fake_get_embedding(self: EmbeddingIntentClassifier, text: str) -> list[float] | None:
        if text == "查一下之前的历史记录":
            return [1.0, 0.0]
        if text in INTENT_EXAMPLES["query_history"]:
            return [0.9, 0.1]
        if text in INTENT_EXAMPLES["coding"]:
            return [0.0, 1.0]
        if text in INTENT_EXAMPLES["planning"]:
            return [-1.0, 0.0]
        return [0.0, 0.0]

    monkeypatch.setattr(EmbeddingIntentClassifier, "_get_embedding", fake_get_embedding)

    classifier = EmbeddingIntentClassifier(model="test")
    result = await classifier.classify("查一下之前的历史记录")
    assert result == ["query_history"]


@pytest.mark.asyncio
async def test_embedding_classifier_no_match(monkeypatch) -> None:
    from app.core.intent_router import EmbeddingIntentClassifier

    async def fake_get_embedding(self: EmbeddingIntentClassifier, text: str) -> list[float] | None:
        if text == "完全无关的内容":
            return [0.0, 100.0]
        return [1.0, 0.0]

    monkeypatch.setattr(EmbeddingIntentClassifier, "_get_embedding", fake_get_embedding)

    classifier = EmbeddingIntentClassifier(model="test")
    result = await classifier.classify("完全无关的内容")
    assert result == []


@pytest.mark.asyncio
async def test_embedding_classifier_get_embedding_fails(monkeypatch) -> None:
    from app.core.intent_router import EmbeddingIntentClassifier

    async def fake_get_embedding(self: EmbeddingIntentClassifier, text: str) -> list[float] | None:
        return None

    monkeypatch.setattr(EmbeddingIntentClassifier, "_get_embedding", fake_get_embedding)

    classifier = EmbeddingIntentClassifier(model="test")
    result = await classifier.classify("anything")
    assert result == []


def test_cosine_similarity_zero_vectors() -> None:
    from app.core.intent_router import EmbeddingIntentClassifier

    assert EmbeddingIntentClassifier._cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert EmbeddingIntentClassifier._cosine_similarity([1.0, 0.0], [0.0, 0.0]) == 0.0
    assert EmbeddingIntentClassifier._cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0


def test_cosine_similarity_normal() -> None:
    from app.core.intent_router import EmbeddingIntentClassifier

    sim = EmbeddingIntentClassifier._cosine_similarity([1.0, 0.0], [2.0, 0.0])
    assert sim == pytest.approx(1.0)

    sim = EmbeddingIntentClassifier._cosine_similarity([1.0, 0.0], [0.0, 1.0])
    assert sim == pytest.approx(0.0)

    sim = EmbeddingIntentClassifier._cosine_similarity([1.0, 0.0], [-1.0, 0.0])
    assert sim == pytest.approx(-1.0)


@pytest.mark.asyncio
async def test_embedding_get_embedding_success(monkeypatch) -> None:
    from app.core.intent_router import EmbeddingIntentClassifier

    async def fake_aembedding(**kwargs: object) -> object:
        class FakeResponse:
            data = [{"embedding": [0.1, 0.2, 0.3]}]

        return FakeResponse()

    monkeypatch.setattr("litellm.aembedding", fake_aembedding)

    classifier = EmbeddingIntentClassifier(model="test")
    classifier.api_base = None
    classifier.api_key = None

    result = await classifier._get_embedding("hello")
    assert result == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_embedding_get_embedding_failure(monkeypatch) -> None:
    from app.core.intent_router import EmbeddingIntentClassifier

    async def fake_aembedding(**kwargs: object) -> object:
        raise RuntimeError("API error")

    monkeypatch.setattr("litellm.aembedding", fake_aembedding)

    classifier = EmbeddingIntentClassifier(model="test")
    classifier.api_base = None
    classifier.api_key = None

    result = await classifier._get_embedding("hello")
    assert result is None


@pytest.mark.asyncio
async def test_embedding_get_embedding_with_api_base_and_key(monkeypatch) -> None:
    from app.core.intent_router import EmbeddingIntentClassifier

    captured: dict[str, object] = {}

    async def fake_aembedding(**kwargs: object) -> object:
        captured.update(kwargs)

        class FakeResponse:
            data = [{"embedding": [0.1, 0.2, 0.3]}]

        return FakeResponse()

    monkeypatch.setattr("litellm.aembedding", fake_aembedding)

    classifier = EmbeddingIntentClassifier(model="test")
    classifier.api_base = "https://custom.api/v1"
    classifier.api_key = "sk-custom"

    result = await classifier._get_embedding("hello")
    assert result == [0.1, 0.2, 0.3]
    assert captured["api_base"] == "https://custom.api/v1"
    assert captured["api_key"] == "sk-custom"


@pytest.mark.asyncio
async def test_embedding_classifier_all_examples_fail(monkeypatch) -> None:
    from app.core.intent_router import INTENT_EXAMPLES, EmbeddingIntentClassifier

    all_examples: set[str] = set(
        INTENT_EXAMPLES["query_history"] + INTENT_EXAMPLES["coding"] + INTENT_EXAMPLES["planning"]
    )

    async def fake_get_embedding(self: EmbeddingIntentClassifier, text: str) -> list[float] | None:
        if text in all_examples:
            return None
        return [1.0, 0.0]

    monkeypatch.setattr(EmbeddingIntentClassifier, "_get_embedding", fake_get_embedding)

    classifier = EmbeddingIntentClassifier(model="test")
    result = await classifier.classify("some text")
    assert result == []
