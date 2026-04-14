from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
