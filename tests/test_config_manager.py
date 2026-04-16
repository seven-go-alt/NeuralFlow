from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from app.config_manager import ConfigManager, RuntimeConfig


@pytest.mark.asyncio
async def test_config_manager_updates_runtime_config_and_records_audit() -> None:
    manager = ConfigManager(initial=RuntimeConfig())

    updated = await manager.update(
        {"max_context_tokens": 4096, "stream_thinking_enabled": True},
        source_ip="127.0.0.1",
        actor="admin",
    )

    assert updated.max_context_tokens == 4096
    assert updated.stream_thinking_enabled is True
    audit_entries = await manager.list_audit_entries()
    assert len(audit_entries) == 1
    assert audit_entries[0].source_ip == "127.0.0.1"
    assert audit_entries[0].changes["max_context_tokens"] == {"old": 8000, "new": 4096}


@pytest.mark.asyncio
async def test_config_manager_rolls_back_on_invalid_update() -> None:
    manager = ConfigManager(initial=RuntimeConfig())
    original = await manager.get_snapshot()

    with pytest.raises(ValidationError):
        await manager.update(
            {"max_context_tokens_soft": 9000, "max_context_tokens": 1000},
            source_ip="127.0.0.1",
            actor="admin",
        )

    current = await manager.get_snapshot()
    assert current == original
    assert await manager.list_audit_entries() == []


@pytest.mark.asyncio
async def test_config_manager_keeps_concurrent_reads_consistent() -> None:
    manager = ConfigManager(initial=RuntimeConfig())

    before = await manager.get_snapshot()
    update_task = asyncio.create_task(
        manager.update({"vector_search_cache_ttl_seconds": 120}, source_ip="127.0.0.1", actor="admin")
    )
    during = await manager.get_snapshot()
    await update_task
    after = await manager.get_snapshot()

    assert before.vector_search_cache_ttl_seconds == 300
    assert during.vector_search_cache_ttl_seconds in {300, 120}
    assert after.vector_search_cache_ttl_seconds == 120
