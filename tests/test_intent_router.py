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
