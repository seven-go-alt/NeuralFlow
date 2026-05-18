from __future__ import annotations

import jwt
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.auth import require_auth
from app.main import app


def test_login_success(monkeypatch) -> None:
    monkeypatch.setattr("app.auth.get_settings", _settings_with_auth)
    client = TestClient(app)

    response = client.post(
        "/api/v1/auth/token",
        json={"username": "admin", "password": "admin123"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["tenant_id"] == "test-tenant"


def test_login_invalid_credentials(monkeypatch) -> None:
    monkeypatch.setattr("app.auth.get_settings", _settings_with_auth)
    client = TestClient(app)

    response = client.post(
        "/api/v1/auth/token",
        json={"username": "admin", "password": "wrong"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid credentials"


def test_login_wrong_username(monkeypatch) -> None:
    monkeypatch.setattr("app.auth.get_settings", _settings_with_auth)
    client = TestClient(app)

    response = client.post(
        "/api/v1/auth/token",
        json={"username": "hacker", "password": "admin123"},
    )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_require_auth_disabled_returns_anonymous(monkeypatch) -> None:
    monkeypatch.setattr("app.auth.get_settings", _settings_no_auth)

    result = await require_auth(authorization=None)
    assert result == {"sub": "anonymous", "tenant_id": "test-tenant"}


@pytest.mark.asyncio
async def test_require_auth_missing_header(monkeypatch) -> None:
    monkeypatch.setattr("app.auth.get_settings", _settings_with_auth)

    with pytest.raises(HTTPException) as exc:
        await require_auth(authorization=None)
    assert exc.value.status_code == 401
    assert "Missing" in exc.value.detail


@pytest.mark.asyncio
async def test_require_auth_invalid_scheme(monkeypatch) -> None:
    monkeypatch.setattr("app.auth.get_settings", _settings_with_auth)

    with pytest.raises(HTTPException) as exc:
        await require_auth(authorization="Basic xxx")
    assert exc.value.status_code == 401
    assert "Invalid Authorization scheme" in exc.value.detail


@pytest.mark.asyncio
async def test_require_auth_expired_token(monkeypatch) -> None:
    monkeypatch.setattr("app.auth.get_settings", _settings_with_auth)

    from datetime import datetime, timedelta

    expired = jwt.encode(
        {
            "sub": "admin",
            "exp": datetime.utcnow() - timedelta(hours=1),
        },
        "test-secret",
        algorithm="HS256",
    )

    with pytest.raises(HTTPException) as exc:
        await require_auth(authorization=f"Bearer {expired}")
    assert exc.value.status_code == 401
    assert "Token expired" in exc.value.detail


@pytest.mark.asyncio
async def test_require_auth_invalid_signature(monkeypatch) -> None:
    monkeypatch.setattr("app.auth.get_settings", _settings_with_auth)

    bad_token = jwt.encode(
        {"sub": "admin", "exp": 9999999999},
        "wrong-secret",
        algorithm="HS256",
    )

    with pytest.raises(HTTPException) as exc:
        await require_auth(authorization=f"Bearer {bad_token}")
    assert exc.value.status_code == 401
    assert "Invalid token" in exc.value.detail


def _settings_no_auth():
    class FakeSettings:
        auth_enabled = False
        tenant_default_id = "test-tenant"
        auth_jwt_secret = "test-secret"
        auth_admin_username = "admin"
        auth_admin_password = "admin123"
        llm_api_base = ""
        llm_api_key = ""
        ollama_fallback_model = ""
        offline_fallback_enabled = False
        litellm_model = "gpt-4"
        embedding_api_base = ""
        embedding_api_key = ""
        openai_api_key = ""

    return FakeSettings()


def _settings_with_auth():
    s = _settings_no_auth()
    s.auth_enabled = True
    return s
