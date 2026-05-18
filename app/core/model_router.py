from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class ModelProfile:
    """Profile for a model with capability and cost metadata."""

    model_id: str
    provider: str = "openai"
    capabilities: tuple[str, ...] = ()
    cost_tier: int = 1  # 1=cheapest, 5=most expensive
    latency_tier: int = 1  # 1=fastest, 5=slowest
    context_window: int = 8192
    description: str = ""

    def supports_intent(self, intent: str) -> bool:
        return intent in self.capabilities


_INTENT_MODEL_MAP: dict[str, list[str]] = {
    "general": ["gpt-4o-mini", "gpt-4o"],
    "query_history": ["gpt-4o-mini", "gpt-4o"],
    "coding": ["gpt-4o", "claude-sonnet-4-20250514", "gpt-4o-mini"],
    "planning": ["gpt-4o", "claude-sonnet-4-20250514"],
}

DEFAULT_MODEL_PROFILES: dict[str, ModelProfile] = {
    "gpt-4o-mini": ModelProfile(
        model_id="gpt-4o-mini",
        provider="openai",
        capabilities=("general", "query_history"),
        cost_tier=1,
        latency_tier=1,
        context_window=128000,
        description="Fast, cheap model for simple queries",
    ),
    "gpt-4o": ModelProfile(
        model_id="gpt-4o",
        provider="openai",
        capabilities=("general", "coding", "planning", "query_history"),
        cost_tier=3,
        latency_tier=3,
        context_window=128000,
        description="Strong general-purpose model",
    ),
    "claude-sonnet-4-20250514": ModelProfile(
        model_id="claude-sonnet-4-20250514",
        provider="anthropic",
        capabilities=("coding", "planning"),
        cost_tier=4,
        latency_tier=4,
        context_window=200000,
        description="Best for complex coding and planning tasks",
    ),
}


class ModelRouter:
    """Route requests to optimal models based on intent, cost, and latency constraints."""

    def __init__(
        self,
        profiles: dict[str, ModelProfile] | None = None,
        intent_map: dict[str, list[str]] | None = None,
        default_model: str = "gpt-4o-mini",
    ) -> None:
        self._profiles = profiles or dict(DEFAULT_MODEL_PROFILES)
        self._intent_map = intent_map or dict(_INTENT_MODEL_MAP)
        self._default_model = default_model

    def select(self, intent: str, *, cost_max: int = 5, latency_max: int = 5) -> str:
        """Select the best model for the given intent within constraints."""
        candidates = self._intent_map.get(intent) or [self._default_model]
        valid = [
            m
            for m in candidates
            if m in self._profiles
            and self._profiles[m].cost_tier <= cost_max
            and self._profiles[m].latency_tier <= latency_max
        ]
        return valid[0] if valid else self._default_model

    def fallback_chain(self, intent: str) -> list[str]:
        """Return ordered fallback chain for the given intent."""
        candidates = self._intent_map.get(intent) or [self._default_model]
        return [m for m in candidates if m in self._profiles] or [self._default_model]

    def get_profile(self, model_id: str) -> ModelProfile | None:
        return self._profiles.get(model_id)

    def list_profiles(self) -> list[dict[str, Any]]:
        return [
            {
                "model_id": p.model_id,
                "provider": p.provider,
                "capabilities": list(p.capabilities),
                "cost_tier": p.cost_tier,
                "latency_tier": p.latency_tier,
                "context_window": p.context_window,
                "description": p.description,
            }
            for p in self._profiles.values()
        ]

    def register_profile(self, profile: ModelProfile, intents: list[str] | None = None) -> None:
        self._profiles[profile.model_id] = profile
        if intents:
            for intent in intents:
                if intent not in self._intent_map:
                    self._intent_map[intent] = []
                if profile.model_id not in self._intent_map[intent]:
                    self._intent_map[intent].append(profile.model_id)
