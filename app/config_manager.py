from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.config import get_settings
from app.utils.observability import configure_structured_logging

logger = configure_structured_logging(logger_name="neuralflow.audit")


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    working_memory_max_turns: int = Field(default=10, ge=1, le=100)
    max_context_tokens_soft: int = Field(default=6000, ge=256, le=200000)
    max_context_tokens: int = Field(default=8000, ge=256, le=200000)
    token_budget_recent_messages: int = Field(default=4, ge=1, le=50)
    vector_search_cache_ttl_seconds: int = Field(default=300, ge=0, le=86400)
    vector_search_default_top_k: int = Field(default=3, ge=1, le=20)
    stream_thinking_enabled: bool = False
    intent_llm_fallback_enabled: bool = True
    litellm_model: str = Field(default="gpt-4o-mini", min_length=1)
    model_routing_strategy: str = Field(default="primary")

    @model_validator(mode="after")
    def validate_token_limits(self) -> "RuntimeConfig":
        if self.max_context_tokens_soft > self.max_context_tokens:
            raise ValueError("max_context_tokens_soft cannot exceed max_context_tokens")
        return self


class ConfigAuditEntry(BaseModel):
    timestamp: datetime
    actor: str
    source_ip: str
    changes: dict[str, dict[str, Any]]


class AsyncRWLock:
    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._readers = 0
        self._writer = False

    def read_lock(self) -> "_ReadLock":
        return _ReadLock(self)

    def write_lock(self) -> "_WriteLock":
        return _WriteLock(self)

    async def acquire_read(self) -> None:
        async with self._condition:
            await self._condition.wait_for(lambda: not self._writer)
            self._readers += 1

    async def release_read(self) -> None:
        async with self._condition:
            self._readers -= 1
            if self._readers == 0:
                self._condition.notify_all()

    async def acquire_write(self) -> None:
        async with self._condition:
            await self._condition.wait_for(lambda: not self._writer and self._readers == 0)
            self._writer = True

    async def release_write(self) -> None:
        async with self._condition:
            self._writer = False
            self._condition.notify_all()


class _ReadLock:
    def __init__(self, lock: AsyncRWLock) -> None:
        self._lock = lock

    async def __aenter__(self) -> None:
        await self._lock.acquire_read()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._lock.release_read()


class _WriteLock:
    def __init__(self, lock: AsyncRWLock) -> None:
        self._lock = lock

    async def __aenter__(self) -> None:
        await self._lock.acquire_write()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._lock.release_write()


class ConfigManager:
    def __init__(self, initial: RuntimeConfig | None = None) -> None:
        self._initial = initial or runtime_config_from_settings()
        self._current = self._initial.model_copy(deep=True)
        self._audit_entries: list[ConfigAuditEntry] = []
        self._lock = AsyncRWLock()

    async def get_snapshot(self) -> RuntimeConfig:
        async with self._lock.read_lock():
            return self._current.model_copy(deep=True)

    async def list_audit_entries(self) -> list[ConfigAuditEntry]:
        async with self._lock.read_lock():
            return [entry.model_copy(deep=True) for entry in self._audit_entries]

    async def update(self, patch: Mapping[str, Any], *, source_ip: str, actor: str) -> RuntimeConfig:
        async with self._lock.write_lock():
            current_data = self._current.model_dump()
            normalized_patch = _normalize_patch(current_data, dict(patch))
            merged = {**current_data, **normalized_patch}
            try:
                candidate = RuntimeConfig.model_validate(merged)
            except ValidationError:
                logger.warning(
                    "runtime config update rolled back",
                    extra={"source_ip": source_ip, "actor": actor, "changes": normalized_patch},
                )
                raise
            changes = _compute_changes(current_data, candidate.model_dump())
            self._current = candidate
            if changes:
                entry = ConfigAuditEntry(
                    timestamp=datetime.now(UTC),
                    actor=actor,
                    source_ip=source_ip,
                    changes=changes,
                )
                self._audit_entries.append(entry)
                logger.info(
                    "runtime config updated",
                    extra={"source_ip": source_ip, "actor": actor, "changes": changes},
                )
            return self._current.model_copy(deep=True)

    async def reset(self) -> None:
        async with self._lock.write_lock():
            self._current = self._initial.model_copy(deep=True)
            self._audit_entries.clear()


def runtime_config_from_settings() -> RuntimeConfig:
    settings = get_settings()
    return RuntimeConfig(
        working_memory_max_turns=settings.working_memory_max_turns,
        max_context_tokens_soft=settings.max_context_tokens_soft,
        max_context_tokens=settings.max_context_tokens,
        token_budget_recent_messages=settings.token_budget_recent_messages,
        vector_search_cache_ttl_seconds=settings.vector_search_cache_ttl_seconds,
        vector_search_default_top_k=settings.vector_search_default_top_k,
        stream_thinking_enabled=settings.stream_thinking_enabled,
        intent_llm_fallback_enabled=settings.intent_llm_fallback_enabled,
        litellm_model=settings.litellm_model,
        model_routing_strategy="primary",
    )


def _compute_changes(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    changes: dict[str, dict[str, Any]] = {}
    for key, value in new.items():
        if old.get(key) != value:
            changes[key] = {"old": old.get(key), "new": value}
    return changes


def _normalize_patch(current: Mapping[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(patch)
    new_hard_limit = normalized.get("max_context_tokens")
    has_soft_limit = "max_context_tokens_soft" in normalized
    if isinstance(new_hard_limit, int) and not has_soft_limit:
        current_soft_limit = current.get("max_context_tokens_soft")
        if isinstance(current_soft_limit, int) and current_soft_limit > new_hard_limit:
            normalized["max_context_tokens_soft"] = new_hard_limit
    return normalized
