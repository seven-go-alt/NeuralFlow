from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_INTENT_KEYWORD_RULES = {
    "query_history": ["之前", "历史", "偏好", "记得", "上次"],
    "coding": ["代码", "bug", "接口", "函数", "部署"],
    "planning": ["方案", "规划", "拆分", "路线图", "设计"],
}

DEFAULT_INTENT_POLICIES = {
    "general": {
        "memory_strategy": "working_only",
        "skill_whitelist": [],
    },
    "query_history": {
        "memory_strategy": "long_term",
        "skill_whitelist": ["memory"],
    },
    "coding": {
        "memory_strategy": "working_only",
        "skill_whitelist": ["python", "filesystem"],
    },
    "planning": {
        "memory_strategy": "working_only",
        "skill_whitelist": ["planner"],
    },
}


class Settings(BaseSettings):
    app_name: str = "NeuralFlow"
    app_env: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    working_memory_max_turns: int = 10

    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection: str = "conversation_memory"

    litellm_model: str = "gpt-4o-mini"
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    mcp_base_url: str = "http://localhost:9000"
    token_budget_encoding: str = "cl100k_base"
    max_context_tokens_soft: int = 6000
    max_context_tokens: int = 8000
    token_budget_recent_messages: int = 4
    vector_search_cache_ttl_seconds: int = 300
    vector_search_default_top_k: int = 3
    stream_thinking_enabled: bool = False

    intent_default: str = "general"
    intent_llm_fallback_enabled: bool = True
    intent_keyword_rules_json: str = Field(
        default_factory=lambda: json.dumps(DEFAULT_INTENT_KEYWORD_RULES, ensure_ascii=False),
        alias="INTENT_KEYWORD_RULES_JSON",
    )
    intent_policy_map_json: str = Field(
        default_factory=lambda: json.dumps(DEFAULT_INTENT_POLICIES, ensure_ascii=False),
        alias="INTENT_POLICY_MAP_JSON",
    )

    celery_broker_url_override: str | None = Field(default=None, alias="CELERY_BROKER_URL")
    celery_result_backend_override: str | None = Field(
        default=None,
        alias="CELERY_RESULT_BACKEND",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def celery_broker_url(self) -> str:
        return self.celery_broker_url_override or self.redis_url

    @property
    def celery_result_backend(self) -> str:
        if self.celery_result_backend_override:
            return self.celery_result_backend_override
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db + 1}"

    @property
    def chroma_api_url(self) -> str:
        return f"http://{self.chroma_host}:{self.chroma_port}"

    @property
    def intent_keyword_rules(self) -> dict[str, list[str]]:
        return _load_json_mapping(self.intent_keyword_rules_json)

    @property
    def intent_policy_map(self) -> dict[str, dict[str, Any]]:
        return _load_json_mapping(self.intent_policy_map_json)


def _load_json_mapping(raw_value: str) -> dict[str, Any]:
    value = json.loads(raw_value)
    if not isinstance(value, dict):
        raise ValueError("Intent config must decode to a JSON object")
    return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
