from __future__ import annotations

import pytest

from app.core.model_router import ModelProfile, ModelRouter


class TestModelProfile:
    def test_supports_intent(self) -> None:
        p = ModelProfile(model_id="test", capabilities=("coding", "planning"))
        assert p.supports_intent("coding") is True
        assert p.supports_intent("general") is False

    def test_defaults(self) -> None:
        p = ModelProfile(model_id="test")
        assert p.provider == "openai"
        assert p.cost_tier == 1
        assert p.context_window == 8192


class TestModelRouter:
    def test_select_by_intent(self) -> None:
        router = ModelRouter()
        model = router.select("coding")
        assert model == "gpt-4o"

    def test_select_general_uses_cheapest(self) -> None:
        router = ModelRouter()
        model = router.select("general")
        assert model == "gpt-4o-mini"

    def test_select_with_cost_constraint(self) -> None:
        router = ModelRouter()
        model = router.select("coding", cost_max=1)
        # gpt-4o-mini has cost_tier=1
        assert model == "gpt-4o-mini"

    def test_select_with_latency_constraint(self) -> None:
        router = ModelRouter()
        model = router.select("coding", latency_max=2)
        assert model == "gpt-4o-mini"  # fast models only

    def test_select_unknown_intent_uses_default(self) -> None:
        router = ModelRouter(default_model="gpt-4o-mini")
        model = router.select("unknown_intent")
        assert model == "gpt-4o-mini"

    def test_select_no_valid_candidates(self) -> None:
        router = ModelRouter(default_model="gpt-4o-mini")
        model = router.select("coding", cost_max=0, latency_max=0)
        assert model == "gpt-4o-mini"

    def test_fallback_chain(self) -> None:
        router = ModelRouter()
        chain = router.fallback_chain("coding")
        assert chain == ["gpt-4o", "claude-sonnet-4-20250514", "gpt-4o-mini"]

    def test_fallback_chain_unknown_intent(self) -> None:
        router = ModelRouter(default_model="gpt-4o-mini")
        chain = router.fallback_chain("unknown")
        assert chain == ["gpt-4o-mini"]

    def test_get_profile(self) -> None:
        router = ModelRouter()
        p = router.get_profile("gpt-4o")
        assert p is not None
        assert p.provider == "openai"

    def test_get_profile_nonexistent(self) -> None:
        router = ModelRouter()
        assert router.get_profile("nonexistent") is None

    def test_list_profiles(self) -> None:
        router = ModelRouter()
        profiles = router.list_profiles()
        assert len(profiles) >= 3
        model_ids = {p["model_id"] for p in profiles}
        assert "gpt-4o-mini" in model_ids

    def test_register_profile_with_intents(self) -> None:
        router = ModelRouter()
        p = ModelProfile(model_id="custom-model", capabilities=("custom",))
        router.register_profile(p, intents=["custom"])
        assert router.select("custom") == "custom-model"
        assert router.get_profile("custom-model") is p

    def test_register_profile_without_intents(self) -> None:
        router = ModelRouter()
        p = ModelProfile(model_id="standalone")
        router.register_profile(p)
        assert router.get_profile("standalone") is p

    def test_custom_config(self) -> None:
        profiles = {
            "fast": ModelProfile(model_id="fast", cost_tier=1, latency_tier=1),
            "powerful": ModelProfile(model_id="powerful", cost_tier=5, latency_tier=5),
        }
        intent_map = {"default": ["fast", "powerful"]}
        router = ModelRouter(profiles=profiles, intent_map=intent_map)
        assert router.select("default") == "fast"
        assert router.select("default", cost_max=5, latency_max=5) == "fast"
