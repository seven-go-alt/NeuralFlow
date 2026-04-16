from __future__ import annotations

import os

from httpx import ASGITransport, AsyncClient
import pytest

os.environ.setdefault("ADMIN_SECRET_KEY", "test-admin-key")

from app.main import app, config_manager  # noqa: E402


@pytest.mark.asyncio
async def test_admin_config_requires_secret_key() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.patch("/admin/config", json={"max_context_tokens": 4096})

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_admin_config_patch_updates_runtime_config() -> None:
    await config_manager.reset()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.patch(
            "/admin/config",
            headers={"X-Admin-Secret": "test-admin-key"},
            json={"max_context_tokens": 4096, "stream_thinking_enabled": True},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["config"]["max_context_tokens"] == 4096
    assert payload["config"]["stream_thinking_enabled"] is True
    assert payload["audit_entry"]["source_ip"] == "127.0.0.1"


@pytest.mark.asyncio
async def test_admin_config_patch_rejects_invalid_values_without_mutating_state() -> None:
    await config_manager.reset()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        invalid_response = await client.patch(
            "/admin/config",
            headers={"X-Admin-Secret": "test-admin-key"},
            json={"max_context_tokens_soft": 9000, "max_context_tokens": 1000},
        )
        current_response = await client.get("/admin/config", headers={"X-Admin-Secret": "test-admin-key"})

    assert invalid_response.status_code == 422
    assert current_response.status_code == 200
    assert current_response.json()["config"]["max_context_tokens"] == 8000
