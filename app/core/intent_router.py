from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Protocol

from app.config import get_settings
from app.core.llm import LLMClient

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class IntentPolicy:
    memory_strategy: str
    skill_whitelist: list[str]


@dataclass(slots=True, frozen=True)
class IntentDetectionResult:
    intents: list[str]
    primary_intent: str
    used_fallback: bool
    policies: dict[str, IntentPolicy]


class IntentLLMClassifier(Protocol):
    async def classify(self, text: str) -> list[str]: ...


class LiteLLMIntentClassifier:
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    async def classify(self, text: str) -> list[str]:
        prompt = (
            "请根据用户输入识别 1 到 3 个意图，并只返回 JSON 数组。"
            "可选意图包括：general, query_history, coding, planning。"
            '如果无法判断，返回 ["general"]。\n'
            f"用户输入：{text}"
        )
        raw_output = await self.llm_client.generate(prompt)
        return _parse_llm_intents(raw_output)


class IntentRouter:
    def __init__(
        self,
        llm_classifier: IntentLLMClassifier | None = None,
        keyword_rules: dict[str, list[str]] | None = None,
        policy_map: dict[str, IntentPolicy] | None = None,
        default_intent: str | None = None,
        llm_fallback_enabled: bool | None = None,
    ) -> None:
        settings = get_settings()
        self.default_intent = default_intent or settings.intent_default
        self.keyword_rules = keyword_rules or settings.intent_keyword_rules
        self.policy_map = policy_map or _build_policy_map(settings.intent_policy_map)
        if self.default_intent not in self.policy_map:
            raise ValueError(f"Unknown default intent: {self.default_intent}")
        self.llm_fallback_enabled = (
            settings.intent_llm_fallback_enabled
            if llm_fallback_enabled is None
            else llm_fallback_enabled
        )
        self.llm_classifier = llm_classifier or LiteLLMIntentClassifier()

    async def detect(self, user_query: str) -> IntentDetectionResult:
        normalized_text = user_query.strip()
        if not normalized_text:
            return self._build_result([self.default_intent], used_fallback=False)

        matched_intents = self._match_keywords(normalized_text)
        if matched_intents:
            logger.info(
                "intent detection matched rules",
                extra={
                    "component": "intent_router",
                    "primary_intent": matched_intents[0],
                    "intent_count": len(matched_intents),
                },
            )
            return self._build_result(matched_intents, used_fallback=False)

        if not self.llm_fallback_enabled:
            return self._build_result([self.default_intent], used_fallback=False)

        try:
            llm_intents = await self.llm_classifier.classify(normalized_text)
        except Exception as exc:
            logger.warning(
                "intent detection fallback failed",
                extra={"component": "intent_router", "error": str(exc)},
            )
            return self._build_result([self.default_intent], used_fallback=False)

        intents = _dedupe_preserve_order(llm_intents) or [self.default_intent]
        logger.info(
            "intent detection used llm fallback",
            extra={
                "component": "intent_router",
                "primary_intent": intents[0],
                "intent_count": len(intents),
            },
        )
        return self._build_result(intents, used_fallback=True)

    def _match_keywords(self, text: str) -> list[str]:
        matched: list[str] = []
        lowered = text.casefold()
        for intent, keywords in self.keyword_rules.items():
            if any(keyword.casefold() in lowered for keyword in keywords):
                matched.append(intent)
        return _dedupe_preserve_order(matched)

    def _build_result(self, intents: list[str], used_fallback: bool) -> IntentDetectionResult:
        normalized = [intent for intent in intents if intent in self.policy_map]
        if not normalized:
            normalized = [self.default_intent]
        policies = {
            intent: self.policy_map.get(intent, self.policy_map[self.default_intent])
            for intent in normalized
        }
        return IntentDetectionResult(
            intents=normalized,
            primary_intent=normalized[0],
            used_fallback=used_fallback,
            policies=policies,
        )


def _build_policy_map(raw_map: dict[str, dict[str, object]]) -> dict[str, IntentPolicy]:
    policies: dict[str, IntentPolicy] = {}
    for intent, raw_policy in raw_map.items():
        memory_strategy = str(raw_policy.get("memory_strategy", "working_only"))
        raw_whitelist = raw_policy.get("skill_whitelist", [])
        skill_whitelist = [str(item) for item in raw_whitelist] if isinstance(raw_whitelist, list) else []
        policies[intent] = IntentPolicy(
            memory_strategy=memory_strategy,
            skill_whitelist=skill_whitelist,
        )
    if "general" not in policies:
        policies["general"] = IntentPolicy(memory_strategy="working_only", skill_whitelist=[])
    return policies


def _parse_llm_intents(raw_output: str) -> list[str]:
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    parsed = json.loads(cleaned)
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    raise ValueError("LLM intent classifier must return a JSON array")


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped
