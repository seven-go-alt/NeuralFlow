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
        "skill_whitelist": ["python_exec", "file_read", "file_write", "file_list", "terminal"],
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
    public_base_url: str | None = Field(default=None, alias="PUBLIC_BASE_URL")
    cors_allow_origins: str = Field(default="*", alias="CORS_ALLOW_ORIGINS")

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    working_memory_max_turns: int = 10

    chroma_host: str = "127.0.0.1"
    chroma_port: int = 8000
    chroma_collection: str = "conversation_memory"
    document_chroma_collection: str = "document_knowledge"

    database_url: str = Field(default="sqlite:///./data/neuralflow.db", alias="DATABASE_URL")
    db_pool_size: int = Field(default=5, alias="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=10, alias="DB_MAX_OVERFLOW")
    documents_storage_dir: str = Field(default="data/uploads", alias="DOCUMENTS_STORAGE_DIR")
    document_max_upload_mb: int = 50
    rag_default_top_k: int = 5
    rag_score_threshold: float = 0.0
    embedding_model: str = "text-embedding-3-small"

    litellm_model: str = "gpt-4o-mini"
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    llm_api_base: str | None = Field(default=None, alias="LLM_API_BASE")
    llm_api_key: str | None = Field(default=None, alias="LLM_API_KEY")
    embedding_api_base: str | None = Field(default=None, alias="EMBEDDING_API_BASE")
    embedding_api_key: str | None = Field(default=None, alias="EMBEDDING_API_KEY")
    offline_fallback_enabled: bool = True
    ollama_fallback_model: str | None = "ollama/qwen2.5:7b"
    celery_worker_pool: str = Field(default="solo", alias="CELERY_WORKER_POOL")

    mcp_base_url: str = "http://localhost:9000"
    mcp_code_server_url: str | None = Field(default=None, alias="MCP_CODE_SERVER_URL")
    mcp_filesystem_server_url: str | None = Field(default=None, alias="MCP_FILESYSTEM_SERVER_URL")
    mcp_timeout_seconds: float = 15.0
    mcp_retry_attempts: int = 3
    mcp_retry_backoff_seconds: float = 0.5

    llm_request_timeout_seconds: int = 120
    llm_stream_timeout_seconds: int = 300
    embedding_request_timeout_seconds: int = 30
    chroma_request_timeout_seconds: int = 15
    vector_search_timeout_seconds: int = 15
    external_http_timeout_seconds: int = 30

    terminal_timeout_seconds: float = Field(default=30.0, alias="TERMINAL_TIMEOUT_SECONDS")
    terminal_working_dir: str | None = Field(default=None, alias="TERMINAL_WORKING_DIR")
    terminal_enabled: bool = Field(default=True, alias="TERMINAL_ENABLED")

    tenant_default_id: str = "public"

    auth_enabled: bool = False
    auth_admin_username: str = "admin"
    auth_admin_password: str = "admin123"
    auth_jwt_secret: str = "neuralflow-dev-secret-change-in-prod"

    token_budget_encoding: str = "cl100k_base"
    max_context_tokens_soft: int = 6000
    max_context_tokens: int = 8000
    token_budget_recent_messages: int = 4
    vector_search_cache_ttl_seconds: int = 300
    vector_search_default_top_k: int = 3
    stream_thinking_enabled: bool = False
    streaming_eval_enabled: bool = False

    rag_advanced_enabled: bool = False
    rag_use_multi_query: bool = False
    rag_use_hyde: bool = False
    rag_max_corrections: int = 2

    multimodal_enabled: bool = False
    vision_model: str = "gpt-4o"
    vision_prompt_template: str = Field(
        default="Describe this image in detail, focusing on content relevant to document understanding.",
        alias="VISION_PROMPT_TEMPLATE",
    )
    multimodal_max_images: int = Field(default=20, alias="MULTIMODAL_MAX_IMAGES")
    multimodal_max_image_size_mb: int = Field(default=5, alias="MULTIMODAL_MAX_IMAGE_SIZE_MB")
    multimodal_max_tables: int = Field(default=50, alias="MULTIMODAL_MAX_TABLES")

    # Cross-encoder reranker
    cross_encoder_enabled: bool = Field(default=True, alias="CROSS_ENCODER_ENABLED")
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL"
    )
    reranker_top_k: int = Field(default=5, alias="RERANKER_TOP_K")
    reranker_heuristic_weights_json: str = Field(
        default='{"vector_weight": 0.5, "keyword_weight": 0.3, "metadata_weight": 0.2}',
        alias="RERANKER_HEURISTIC_WEIGHTS_JSON",
    )

    # BM25 configuration for hybrid memory retrieval
    bm25_k1: float = Field(default=1.5, alias="BM25_K1")
    bm25_b: float = Field(default=0.75, alias="BM25_B")

    # Chunking strategy
    chunking_strategy: str = Field(default="fixed", alias="CHUNKING_STRATEGY")
    chunk_max_section_chars: int = Field(default=2000, alias="CHUNK_MAX_SECTION_CHARS")
    chunk_min_section_chars: int = Field(default=100, alias="CHUNK_MIN_SECTION_CHARS")

    # OCR pipeline for scanned documents and images
    ocr_enabled: bool = Field(default=False, alias="OCR_ENABLED")
    ocr_language: str = Field(default="eng", alias="OCR_LANGUAGE")
    ocr_dpi: int = Field(default=300, alias="OCR_DPI")

    # Signed document preview URLs
    document_signed_url_ttl: int = Field(default=3600, alias="DOCUMENT_SIGNED_URL_TTL")
    document_signed_url_secret: str = Field(
        default="neuralflow-default-signing-secret", alias="DOCUMENT_SIGNED_URL_SECRET"
    )

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

    rate_limit_max_requests: int = Field(default=100, alias="RATE_LIMIT_MAX_REQUESTS")
    rate_limit_window_seconds: int = Field(default=60, alias="RATE_LIMIT_WINDOW_SECONDS")

    sentry_dsn: str | None = Field(default=None, alias="SENTRY_DSN")
    sentry_traces_sample_rate: float = Field(default=0.1, alias="SENTRY_TRACES_SAMPLE_RATE")

    celery_broker_url_override: str | None = Field(default=None, alias="CELERY_BROKER_URL")
    celery_result_backend_override: str | None = Field(default=None, alias="CELERY_RESULT_BACKEND")

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

    @property
    def reranker_heuristic_weights(self) -> dict[str, float]:
        return _load_json_mapping(self.reranker_heuristic_weights_json)


def _load_json_mapping(raw_value: str) -> dict[str, Any]:
    value = json.loads(raw_value)
    if not isinstance(value, dict):
        raise ValueError("Config value must decode to a JSON object")
    return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
