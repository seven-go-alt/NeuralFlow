from app.config import Settings, get_settings


def test_settings_exposes_redis_and_chroma_helpers(monkeypatch) -> None:
    monkeypatch.delenv("REDIS_HOST", raising=False)
    monkeypatch.delenv("REDIS_PORT", raising=False)
    monkeypatch.delenv("REDIS_DB", raising=False)
    monkeypatch.delenv("CHROMA_HOST", raising=False)
    monkeypatch.delenv("CHROMA_PORT", raising=False)
    monkeypatch.delenv("CELERY_BROKER_URL", raising=False)
    monkeypatch.delenv("CELERY_RESULT_BACKEND", raising=False)
    get_settings.cache_clear()

    settings = Settings()

    assert settings.redis_url == "redis://localhost:6379/0"
    assert settings.celery_broker_url == "redis://localhost:6379/0"
    assert settings.celery_result_backend == "redis://localhost:6379/1"
    assert settings.chroma_api_url == "http://localhost:8000"
    assert not hasattr(settings, "mongo_host")


def test_settings_helpers_follow_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("REDIS_HOST", "redis")
    monkeypatch.setenv("REDIS_PORT", "6380")
    monkeypatch.setenv("REDIS_DB", "2")
    monkeypatch.setenv("CHROMA_HOST", "chroma")
    monkeypatch.setenv("CHROMA_PORT", "8001")
    monkeypatch.delenv("CELERY_BROKER_URL", raising=False)
    monkeypatch.delenv("CELERY_RESULT_BACKEND", raising=False)
    get_settings.cache_clear()

    settings = Settings()

    assert settings.redis_url == "redis://redis:6380/2"
    assert settings.celery_broker_url == "redis://redis:6380/2"
    assert settings.celery_result_backend == "redis://redis:6380/3"
    assert settings.chroma_api_url == "http://chroma:8001"
