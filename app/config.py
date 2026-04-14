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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
