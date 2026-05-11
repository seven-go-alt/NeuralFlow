from __future__ import annotations

from app.config import Settings


def test_settings_support_separate_embedding_and_cors_configuration(monkeypatch) -> None:
    monkeypatch.setenv("LLM_API_BASE", "https://chat.example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "chat-key")
    monkeypatch.setenv("EMBEDDING_API_BASE", "https://embed.example.com/v1")
    monkeypatch.setenv("EMBEDDING_API_KEY", "embed-key")
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "https://app.example.com,https://admin.example.com")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://app.example.com")
    monkeypatch.setenv("CELERY_WORKER_POOL", "prefork")

    settings = Settings()

    assert settings.llm_api_base == "https://chat.example.com/v1"
    assert settings.embedding_api_base == "https://embed.example.com/v1"
    assert settings.embedding_api_key == "embed-key"
    assert settings.cors_allow_origins == "https://app.example.com,https://admin.example.com"
    assert settings.public_base_url == "https://app.example.com"
    assert settings.celery_worker_pool == "prefork"
